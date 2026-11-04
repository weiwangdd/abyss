// How long a decode step waits for its Engram rows when they are not in the page cache: 24 random rows from each of the
// two tables (as one token gathers them), read the way the server maps them (mmap, MADV_RANDOM), four ways per
// iteration, each on fresh random rows: A touch them one after another (the CPU get_rows of a decode, where the
// faults wait one by one), B MADV_WILLNEED on all of them first and then touch, C 16 threads touching them at once,
// D MADV_WILLNEED, then touch 100 us later (the time the graph launch takes before the get_rows runs). mincore counts
// the rows not resident before each read.
//
// With "idle" after the six table arguments it instead reads one row the page cache does not hold after the disk was idle
// for 20 ms to 3 s (5 times each), to see what a power state of the NVMe costs the first read after a pause.
//
//     gcc -O2 -pthread -o engram_reads engram_reads.c
//     ./engram_reads <shard> <offset> <rows> <shard> <offset> <rows> [iterations, default 30]
//     ./engram_reads <shard> <offset> <rows> <shard> <offset> <rows> idle
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define NT      2   // tables
#define PER     24  // rows per table and token
#define N       (NT*PER)
#define ROW     136 // bytes per row (256 MXFP4 values)
#define THREADS 16

typedef struct {
    const uint8_t * base;
    size_t          off;
    size_t          nrows;
} table_t;

static table_t         tables[NT];
static const uint8_t * ptrs[N];
static volatile uint64_t sink;
static uint64_t        rng = 88172645463325252ull;
static long            page;

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e6 + ts.tv_nsec/1e3;
}

static uint64_t next(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}

static void pick(void) {
    for (int e = 0; e < NT; e++) {
        for (int k = 0; k < PER; k++) {
            ptrs[e*PER + k] = tables[e].base + tables[e].off + (next() % tables[e].nrows)*ROW;
        }
    }
}

static int cold(void) {
    int n = 0;
    for (int i = 0; i < N; i++) {
        unsigned char v;
        if (mincore((void *) ((uintptr_t) ptrs[i] & ~(uintptr_t) (page - 1)), page, &v) == 0 && !(v & 1)) {
            n++;
        }
    }
    return n;
}

static void touch(int i0, int i1) {
    uint64_t s = 0;
    for (int i = i0; i < i1; i++) {
        s += ptrs[i][0] + ptrs[i][ROW - 1];
    }
    sink += s;
}

static void willneed(void) {
    for (int i = 0; i < N; i++) {
        const uintptr_t p = (uintptr_t) ptrs[i] & ~(uintptr_t) (page - 1);
        const uintptr_t q = ((uintptr_t) ptrs[i] + ROW - 1) & ~(uintptr_t) (page - 1);
        madvise((void *) p, q - p + page, MADV_WILLNEED);
    }
}

typedef struct {
    int i0, i1;
} span_t;

static void * worker(void * arg) {
    const span_t * s = arg;
    touch(s->i0, s->i1);
    return NULL;
}

static int cmp(const void * a, const void * b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return x < y ? -1 : x > y;
}

static void report(const char * name, double * t, int * c, int n) {
    double sum = 0, cs = 0;
    for (int i = 0; i < n; i++) {
        sum += t[i];
        cs += c[i];
    }
    qsort(t, n, sizeof(double), cmp);
    printf("%-44s mean %7.1f us, median %7.1f, p90 %7.1f, max %7.1f; %.1f of %d rows not resident\n", name, sum/n,
        t[n/2], t[9*n/10], t[n - 1], cs/n, N);
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <shard> <offset> <rows> <shard> <offset> <rows> [iterations]\n", argv[0]);
        return 1;
    }
    page = sysconf(_SC_PAGESIZE);
    for (int e = 0; e < NT; e++) {
        const int fd = open(argv[1 + 3*e], O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) {
            perror(argv[1 + 3*e]);
            return 1;
        }
        void * base = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            perror("mmap");
            return 1;
        }
        madvise(base, st.st_size, MADV_RANDOM);
        tables[e].base  = base;
        tables[e].off   = strtoull(argv[2 + 3*e], NULL, 10);
        tables[e].nrows = strtoull(argv[3 + 3*e], NULL, 10);
    }
    rng ^= (uint64_t) now_us();
    if (argc > 7 && strcmp(argv[7], "idle") == 0) {
        const int idle_ms[] = { 20, 50, 90, 110, 150, 300, 1000, 3000 };
        for (size_t k = 0; k < sizeof(idle_ms)/sizeof(idle_ms[0]); k++) {
            double t[5];
            for (int r = 0; r < 5; r++) {
                // a row that is not resident, then the pause, then the read
                do {
                    pick();
                    unsigned char v;
                    mincore((void *) ((uintptr_t) ptrs[0] & ~(uintptr_t) (page - 1)), page, &v);
                    if (!(v & 1)) {
                        break;
                    }
                } while (1);
                usleep(idle_ms[k]*1000);
                const double t0 = now_us();
                touch(0, 1);
                t[r] = now_us() - t0;
            }
            printf("idle %5d ms: first read %7.0f %7.0f %7.0f %7.0f %7.0f us\n", idle_ms[k], t[0], t[1], t[2], t[3], t[4]);
        }
        return 0;
    }
    const int iters = argc > 7 ? atoi(argv[7]) : 30;

    double ta[iters], tb[iters], tbm[iters], tc[iters], td[iters];
    int    ca[iters], cb[iters], cc[iters], cd[iters];
    for (int it = 0; it < iters; it++) {
        pick();
        ca[it] = cold();
        double t0 = now_us();
        touch(0, N);
        ta[it] = now_us() - t0;

        pick();
        cb[it] = cold();
        t0 = now_us();
        willneed();
        const double t1 = now_us();
        touch(0, N);
        tb[it]  = now_us() - t0;
        tbm[it] = t1 - t0;

        pick();
        cc[it] = cold();
        t0 = now_us();
        pthread_t th[THREADS];
        span_t    sp[THREADS];
        for (int k = 0; k < THREADS; k++) {
            sp[k].i0 = N*k/THREADS;
            sp[k].i1 = N*(k + 1)/THREADS;
            pthread_create(&th[k], NULL, worker, &sp[k]);
        }
        for (int k = 0; k < THREADS; k++) {
            pthread_join(th[k], NULL);
        }
        tc[it] = now_us() - t0;

        pick();
        cd[it] = cold();
        t0 = now_us();
        willneed();
        while (now_us() - t0 < 100) {
        }
        const double t2 = now_us();
        touch(0, N);
        td[it] = now_us() - t2;

        usleep(20000); // a token apart, as in a decode
    }
    report("A touch one after another", ta, ca, iters);
    report("B MADV_WILLNEED all, then touch (total)", tb, cb, iters);
    report("  of which the MADV_WILLNEED calls", tbm, cb, iters);
    report("C 16 threads touching", tc, cc, iters);
    report("D MADV_WILLNEED, touch 100 us later (touch)", td, cd, iters);
    return 0;
}
