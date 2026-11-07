// Latency of the all-reduce of two partial results across the two GPUs that tensor parallelism of a decode step needs
// after a layer's attention and after its FFN: each GPU writes its partial of S bytes, each adds the other's into its
// own, and the next step starts from the sum. Runs a chain of STEPS such steps, the GPUs ordered only by events on their
// streams (no host synchronization inside the chain), and prints the time per step for each size and way of moving the
// partials:
//   local  no exchange: each GPU adds its own partial (the kernels and launches alone, the floor)
//   copy   cudaMemcpyAsync of the other's partial on each GPU's own stream, a peer copy by the copy engines with peer
//          access on (cudaMemcpyPeerAsync does the same but cannot be captured into a graph), then the add
//   pull   the add kernel reads the other GPU's partial through peer access (loads over PCIe)
//   push   a kernel writes the partial into the other GPU's receive buffer as well (stores over PCIe), then each adds
// each issued from the host step by step ("eager") and captured once into a CUDA graph of both GPUs and replayed
// ("graph"). The partials are double-buffered, so the event of a step also covers the buffer the next step overwrites.
// The sums are checked at the end: with acc = (acc + 1) / 2 + (acc + 2) / 2 each step, both GPUs must hold 1.5 * STEPS
// everywhere (a silent peer-to-peer failure, as with the IOMMU on, shows as wrong sums).
//
//     nvcc -O2 -arch=sm_120 -o p2p_allreduce p2p_allreduce.cu
//     ./p2p_allreduce [STEPS (default 2000)] [sizes in bytes (default 20480 81920 1048576 41943040)]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            exit(2);                                                                             \
        }                                                                                        \
    } while (0)

enum mode { LOCAL, COPY, PULL, PUSH };
static const char * mode_name[] = { "local", "copy", "pull", "push" };

// the partial of GPU d: half of (acc + d + 1); with PUSH it is also stored into the other GPU's receive buffer
__global__ void k_part(float * part, float * remote, const float * acc, float d1, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = 0.5f * (acc[i] + d1);
        part[i] = v;
        if (remote) {
            remote[i] = v;
        }
    }
}

// acc = own partial + the other one (LOCAL: an own partial as stand-in for the other, the same work)
__global__ void k_add(float * acc, const float * part, const float * other, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        acc[i] = part[i] + other[i];
    }
}

struct gpu {
    cudaStream_t s;
    cudaEvent_t  ready[2];
    float *      acc;
    float *      part[2];
    float *      recv[2];
};

static gpu g[2];

// one step of the chain on both GPUs, parity b of the double buffers
static void step(mode m, int b, int n, size_t bytes) {
    const int threads = 256, blocks = (n + threads - 1) / threads;
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        float * remote = m == PUSH ? g[1 - d].recv[b] : nullptr;
        k_part<<<blocks, threads, 0, g[d].s>>>(g[d].part[b], remote, g[d].acc, (float) (d + 1), n);
        CK(cudaEventRecord(g[d].ready[b], g[d].s));
    }
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        const float * other = g[d].part[b];
        if (m != LOCAL) {
            CK(cudaStreamWaitEvent(g[d].s, g[1 - d].ready[b], 0));
        }
        if (m == COPY) {
            CK(cudaMemcpyAsync(g[d].recv[b], g[1 - d].part[b], bytes, cudaMemcpyDefault, g[d].s));
            other = g[d].recv[b];
        } else if (m == PULL) {
            other = g[1 - d].part[b];
        } else if (m == PUSH) {
            other = g[d].recv[b];
        }
        k_add<<<blocks, threads, 0, g[d].s>>>(g[d].acc, g[d].part[b], other, n);
    }
}

static void reset(size_t bytes) {
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        CK(cudaMemsetAsync(g[d].acc, 0, bytes, g[d].s));
        CK(cudaStreamSynchronize(g[d].s));
    }
}

static void sync_both() {
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        CK(cudaStreamSynchronize(g[d].s));
    }
}

// wrong elements of acc on both GPUs after steps steps (LOCAL computes acc + 1 + ... differently: acc = 2 * part)
static long check(mode m, int steps, int n) {
    std::vector<float> h(n);
    long bad = 0;
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        CK(cudaMemcpy(h.data(), g[d].acc, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost));
        // LOCAL: acc = acc + d + 1 each step
        const float want = m == LOCAL ? (float) (d + 1) * steps : 1.5f * steps;
        for (int i = 0; i < n; i++) {
            bad += h[i] != want;
        }
    }
    return bad;
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    const int steps = argc > 1 ? atoi(argv[1]) : 2000;
    std::vector<size_t> sizes;
    for (int i = 2; i < argc; i++) {
        sizes.push_back((size_t) atoll(argv[i]));
    }
    if (sizes.empty()) {
        sizes = { 20480, 81920, 1 << 20, 40 << 20 };
    }
    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    if (n_dev < 2) {
        fprintf(stderr, "needs two GPUs\n");
        return 1;
    }
    int can01 = 0, can10 = 0;
    CK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    printf("peer access 0->1 %d, 1->0 %d\n", can01, can10);
    const size_t max_bytes = *std::max_element(sizes.begin(), sizes.end());
    for (int d = 0; d < 2; d++) {
        CK(cudaSetDevice(d));
        CK(cudaDeviceEnablePeerAccess(1 - d, 0));
        CK(cudaStreamCreateWithFlags(&g[d].s, cudaStreamNonBlocking));
        for (int b = 0; b < 2; b++) {
            CK(cudaEventCreateWithFlags(&g[d].ready[b], cudaEventDisableTiming));
            CK(cudaMalloc(&g[d].part[b], max_bytes));
            CK(cudaMalloc(&g[d].recv[b], max_bytes));
        }
        CK(cudaMalloc(&g[d].acc, max_bytes));
    }

    printf("%10s %6s %10s %10s %8s\n", "bytes", "mode", "eager us", "graph us", "wrong");
    for (size_t bytes : sizes) {
        const int n = (int) (bytes / sizeof(float));
        const int s = bytes >= (8 << 20) ? std::max(20, steps / 20) : steps;
        for (mode m : { LOCAL, COPY, PULL, PUSH }) {
            // eager: the host issues every step
            reset(bytes);
            for (int i = 0; i < 20; i++) {
                step(m, i & 1, n, bytes);  // warm-up
            }
            sync_both();
            reset(bytes);
            double t0 = now_us();
            for (int i = 0; i < s; i++) {
                step(m, i & 1, n, bytes);
            }
            sync_both();
            const double eager = (now_us() - t0) / s;
            long bad = check(m, s, n);

            // graph: the same chain captured once from GPU0's stream, GPU1's stream forked in and joined by events
            double graph = -1;
            reset(bytes);
            cudaEvent_t fork, join;
            CK(cudaSetDevice(0));
            CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
            CK(cudaSetDevice(1));
            CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
            CK(cudaSetDevice(0));
            cudaGraph_t gr = nullptr;
            cudaGraphExec_t ge = nullptr;
            bool ok = cudaStreamBeginCapture(g[0].s, cudaStreamCaptureModeRelaxed) == cudaSuccess;
            if (ok) {
                CK(cudaEventRecord(fork, g[0].s));
                CK(cudaSetDevice(1));
                CK(cudaStreamWaitEvent(g[1].s, fork, 0));
                for (int i = 0; i < s; i++) {
                    step(m, i & 1, n, bytes);
                }
                CK(cudaSetDevice(1));
                CK(cudaEventRecord(join, g[1].s));
                CK(cudaSetDevice(0));
                CK(cudaStreamWaitEvent(g[0].s, join, 0));
                ok = cudaStreamEndCapture(g[0].s, &gr) == cudaSuccess &&
                     cudaGraphInstantiate(&ge, gr, 0) == cudaSuccess;
            }
            if (ok) {
                CK(cudaGraphLaunch(ge, g[0].s));  // warm-up
                CK(cudaStreamSynchronize(g[0].s));
                reset(bytes);
                CK(cudaSetDevice(0));
                double t1 = now_us();
                CK(cudaGraphLaunch(ge, g[0].s));
                CK(cudaStreamSynchronize(g[0].s));
                graph = (now_us() - t1) / s;
                bad += check(m, s, n);
            } else {
                cudaGetLastError();
                fprintf(stderr, "capture of %s failed: %s\n", mode_name[m], cudaGetErrorString(cudaGetLastError()));
            }
            if (ge) {
                cudaGraphExecDestroy(ge);
            }
            if (gr) {
                cudaGraphDestroy(gr);
            }
            printf("%10zu %6s %10.2f %10.2f %8ld\n", bytes, mode_name[m], eager, graph, bad);
            fflush(stdout);
        }
    }
    return 0;
}
