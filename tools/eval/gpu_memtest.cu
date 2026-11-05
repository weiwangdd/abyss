// Device memory test of each GPU, e.g. after a power cut or a BIOS change: allocates all but 1 GiB of the free memory
// and, for ITERS rounds, writes and verifies patterns over all of it (each word's index hashed with the round's seed,
// its complement, all zeros, all ones, 0x55..., 0xAA...), counting the words that differ. With "p2p", then copies 1 GiB
// of the hashed pattern from each GPU to the other with cudaMemcpyPeer, once without peer access (staged through host
// memory) and once with it (directly between the GPUs), checks it on the receiving GPU and prints the count of wrong
// words with the first few (byte offset, expected, got, xor).
//
//     nvcc -O2 -o gpu_memtest gpu_memtest.cu
//     ./gpu_memtest [ITERS] [p2p]
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(2);                                                                             \
        }                                                                                        \
    } while (0)

static const int n_kinds   = 6;
static const int max_samples = 16;

__device__ __forceinline__ uint32_t hash32(uint64_t i, uint32_t seed) {
    uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ull + seed;
    x ^= x >> 31;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 29;
    return (uint32_t) x;
}

__device__ __forceinline__ uint32_t pattern(uint64_t i, uint32_t seed, int kind) {
    switch (kind) {
        case 0:  return hash32(i, seed);
        case 1:  return ~hash32(i, seed);
        case 2:  return 0u;
        case 3:  return 0xFFFFFFFFu;
        case 4:  return 0x55555555u;
        default: return 0xAAAAAAAAu;
    }
}

__global__ void fill(uint32_t * p, uint64_t n, uint32_t seed, int kind) {
    for (uint64_t i = blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; i < n; i += (uint64_t) gridDim.x * blockDim.x) {
        p[i] = pattern(i, seed, kind);
    }
}

struct result {
    unsigned long long n_bad;
    unsigned int       n_samples;
    unsigned long long idx[max_samples];
    uint32_t           expected[max_samples];
    uint32_t           got[max_samples];
};

__global__ void check(const uint32_t * p, uint64_t n, uint32_t seed, int kind, result * r) {
    for (uint64_t i = blockIdx.x * (uint64_t) blockDim.x + threadIdx.x; i < n; i += (uint64_t) gridDim.x * blockDim.x) {
        const uint32_t v = p[i];
        const uint32_t e = pattern(i, seed, kind);
        if (v != e) {
            atomicAdd(&r->n_bad, 1ull);
            const unsigned int k = atomicAdd(&r->n_samples, 1u);
            if (k < max_samples) {
                r->idx[k]      = i;
                r->expected[k] = e;
                r->got[k]      = v;
            }
        }
    }
}

// checks n words of p on the current device against the pattern; prints and returns the count of wrong words
static unsigned long long verify(const uint32_t * p, uint64_t n, uint32_t seed, int kind, result * r, bool print) {
    CK(cudaMemset(r, 0, sizeof(result)));
    check<<<4096, 256>>>(p, n, seed, kind, r);
    CK(cudaDeviceSynchronize());
    result h;
    CK(cudaMemcpy(&h, r, sizeof(h), cudaMemcpyDeviceToHost));
    if (print && h.n_bad) {
        const unsigned int m = h.n_samples < max_samples ? h.n_samples : max_samples;
        for (unsigned int k = 0; k < m; ++k) {
            printf("      offset %12llu: expected %08x got %08x xor %08x\n", h.idx[k]*4, h.expected[k], h.got[k],
                h.expected[k] ^ h.got[k]);
        }
    }
    return h.n_bad;
}

int main(int argc, char ** argv) {
    const int  iters = argc > 1 ? atoi(argv[1]) : 3;
    const bool p2p   = argc > 2 && strcmp(argv[2], "p2p") == 0;

    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    int bad = 0;

    for (int d = 0; d < n_dev; ++d) {
        CK(cudaSetDevice(d));
        size_t free_b = 0, total_b = 0;
        CK(cudaMemGetInfo(&free_b, &total_b));
        const size_t bytes = (free_b - (1ull << 30)) & ~((size_t) (1 << 20) - 1);
        const uint64_t n = bytes/4;
        uint32_t * p = nullptr;
        result   * r = nullptr;
        CK(cudaMalloc(&p, bytes));
        CK(cudaMalloc(&r, sizeof(result)));
        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0));
        CK(cudaEventCreate(&t1));
        CK(cudaEventRecord(t0));
        unsigned long long n_bad = 0;
        for (int it = 0; it < iters; ++it) {
            for (int kind = 0; kind < n_kinds; ++kind) {
                const uint32_t seed = 0x1234567u*(it + 1) + kind;
                fill<<<4096, 256>>>(p, n, seed, kind);
                CK(cudaGetLastError());
                n_bad += verify(p, n, seed, kind, r, n_bad == 0);
            }
        }
        CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        printf("GPU%d: %.1f GiB tested, %d rounds of %d patterns in %.1f s: %llu wrong words\n", d, bytes/1073741824.0,
            iters, n_kinds, ms/1000.0f, n_bad);
        bad += n_bad != 0;
        CK(cudaFree(p));
        CK(cudaFree(r));
    }

    if (p2p && n_dev >= 2) {
        const size_t   bytes = 1ull << 30;
        const uint64_t n     = bytes/4;
        uint32_t * buf[2];
        result   * res[2];
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            CK(cudaMalloc(&buf[d], bytes));
            CK(cudaMalloc(&res[d], sizeof(result)));
        }
        for (int peer = 0; peer < 2; ++peer) {
            if (peer) {
                for (int d = 0; d < 2; ++d) {
                    CK(cudaSetDevice(d));
                    CK(cudaDeviceEnablePeerAccess(1 - d, 0));
                }
            }
            for (int src = 0; src < 2; ++src) {
                const int      dst  = 1 - src;
                const uint32_t seed = 0xC0FFEEu + 2*src + peer;
                CK(cudaSetDevice(src));
                fill<<<4096, 256>>>(buf[src], n, seed, 0);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpyPeer(buf[dst], dst, buf[src], src, bytes));
                CK(cudaDeviceSynchronize());
                CK(cudaSetDevice(dst));
                // the source still holds the pattern, so a wrong copy is the copy's
                const unsigned long long n_bad = verify(buf[dst], n, seed, 0, res[dst], true);
                printf("GPU%d -> GPU%d, 1 GiB, %s: %llu wrong words\n", src, dst,
                    peer ? "peer access (direct)" : "no peer access (through host memory)", n_bad);
                bad += n_bad != 0;
            }
        }
    }
    return bad ? 1 : 0;
}
