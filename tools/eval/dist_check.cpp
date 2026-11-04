// The dist sampler with and without ABYSS_DIST_VEC (mx-llama.cpp/0029), run once each way: 129,280 random logits (normal,
// sd 3, with a few peaks, new ones every 10 draws), temperature 1.0, a fixed seed, ITERS draws; prints the time per
// draw and writes the selected token of each draw and the probabilities of the first draw to OUT, for comparing the
// two runs (dist_check.py). Needs no model.
//
//     g++ -O2 -std=c++17 dist_check.cpp -I <fork>/include -I <fork>/ggml/include -L <build>/bin -lllama \
//         -Wl,-rpath,<build>/bin -o dist_check
//     ABYSS_DIST_VEC=0 ./dist_check off.txt; ABYSS_DIST_VEC=1 ./dist_check on.txt; python3 dist_check.py off.txt on.txt
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s OUT [ITERS]\n", argv[0]);
        return 1;
    }
    const int n_vocab = 129280;
    const int iters   = argc > 2 ? atoi(argv[2]) : 2000;

    std::mt19937                    rng(1);
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<float>              logits(n_vocab);
    std::vector<llama_token_data>   cur(n_vocab);

    llama_sampler * smpl = llama_sampler_init_dist(1234);
    FILE * f = fopen(argv[1], "w");
    if (!f) {
        perror(argv[1]);
        return 1;
    }

    double total_ms = 0.0;
    for (int it = 0; it < iters; ++it) {
        if (it % 10 == 0) {
            for (auto & x : logits) {
                x = nd(rng);
            }
            for (int k = 0; k < 5; ++k) {
                logits[rng() % n_vocab] += 12.0f;
            }
        }
        for (int i = 0; i < n_vocab; ++i) {
            cur[i] = llama_token_data{ i, logits[i], 0.0f };
        }
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };

        const auto t0 = std::chrono::steady_clock::now();
        llama_sampler_apply(smpl, &cur_p);
        const auto t1 = std::chrono::steady_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        fprintf(f, "s %d\n", cur_p.data[cur_p.selected].id);
        if (it == 0) {
            for (int i = 0; i < n_vocab; ++i) {
                fprintf(f, "p %.9g\n", cur[i].p);
            }
        }
    }
    fclose(f);
    llama_sampler_free(smpl);
    printf("%d draws over %d tokens: %.3f ms per draw\n", iters, n_vocab, total_ms/iters);
    return 0;
}
