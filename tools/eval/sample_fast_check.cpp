// The token the server's sampling chain picks with and without ABYSS_SAMPLE_FAST (mx-llama.cpp/0035), for chains built
// as common_sampler_init builds them for the production settings: temperature 1.0, top-p 1.0, top-k 0, min-p 0, the
// penalties, DRY, XTC, typical and top-n-sigma off, ending in dist; the same with temperature 0, ending in greedy; and
// with top-p 0.95, which the fast path must leave to the chain. Two chains of each with the same seed, one applied to
// the candidates as common_sampler_sample does (building them from the logits first), the other through
// llama_sampler_chain_sample_logits; 129,280 random logits (normal, sd 3, with a few peaks, new ones every 10 draws),
// ITERS draws. Prints the draws whose tokens differ, the draws the fast path declined and the time per draw of each
// path, with the caches flushed before each (FLUSH_MB of writes, default 128) as the expert computations between two
// tokens do. Exits with 1 when a token differs. Needs no model.
//
//     g++ -O2 -std=c++17 sample_fast_check.cpp -I <fork>/include -I <fork>/ggml/include -L <build>/bin -lllama \
//         -Wl,-rpath,<build>/bin -o sample_fast_check
//     ABYSS_SAMPLE_FAST=1 ABYSS_DIST_VEC=1 ./sample_fast_check [ITERS] [FLUSH_MB]
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// the chain of common_sampler_init, in the order of the default samplers
static llama_sampler * make_chain(float temp, float top_p, uint32_t seed, int32_t n_vocab) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, 64, 1.0f, 0.0f, 0.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_dry(nullptr, 0.0f, 1.75f, 2, -1, nullptr, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(-1.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(0));
    llama_sampler_chain_add(chain, llama_sampler_init_typical(1.0f, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(0.0f, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_xtc(0.0f, 0.1f, 0, seed));
    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
    llama_sampler_chain_add(chain, temp == 0.0f ? llama_sampler_init_greedy() : llama_sampler_init_dist(seed));
    return chain;
}

int main(int argc, char ** argv) {
    const int    n_vocab = 129280;
    const int    iters   = argc > 1 ? atoi(argv[1]) : 2000;
    const size_t n_flush = (size_t) (argc > 2 ? atoi(argv[2]) : 128) << 20;

    struct config {
        const char * name;
        float        temp;
        float        top_p;
    };
    const config configs[] = {
        { "temperature 1.0", 1.0f, 1.0f  },
        { "temperature 0",   0.0f, 1.0f  },
        { "top-p 0.95",      1.0f, 0.95f },
    };

    std::vector<float>            logits(n_vocab);
    std::vector<llama_token_data> cur(n_vocab);
    std::vector<char>             junk(n_flush);
    auto flush = [&] {
        for (size_t i = 0; i < junk.size(); i += 64) {
            junk[i]++;
        }
    };
    using clock = std::chrono::steady_clock;

    int n_bad = 0;
    for (const auto & c : configs) {
        std::mt19937                    rng(1);
        std::normal_distribution<float> nd(0.0f, 3.0f);

        llama_sampler * a = make_chain(c.temp, c.top_p, 1234, n_vocab);
        llama_sampler * b = make_chain(c.temp, c.top_p, 1234, n_vocab);

        int    n_diff = 0, n_declined = 0;
        double ms_a = 0.0, ms_b = 0.0;
        for (int it = 0; it < iters; ++it) {
            if (it % 10 == 0) {
                for (auto & x : logits) {
                    x = nd(rng);
                }
                for (int k = 0; k < 5; ++k) {
                    logits[rng() % n_vocab] += 12.0f;
                }
            }

            flush();
            const auto t0 = clock::now();
            for (int i = 0; i < n_vocab; ++i) {
                cur[i] = llama_token_data{ i, logits[i], 0.0f };
            }
            llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
            llama_sampler_apply(a, &cur_p);
            const llama_token id_a = cur_p.data[cur_p.selected].id;
            const auto t1 = clock::now();

            flush();
            const auto t2 = clock::now();
            llama_token id_b = llama_sampler_chain_sample_logits(b, logits.data(), n_vocab);
            const auto t3 = clock::now();
            if (id_b == LLAMA_TOKEN_NULL) {
                // the chain then, as common_sampler_sample_fast falls back to it
                n_declined++;
                for (int i = 0; i < n_vocab; ++i) {
                    cur[i] = llama_token_data{ i, logits[i], 0.0f };
                }
                llama_token_data_array cur_b = { cur.data(), cur.size(), -1, false };
                llama_sampler_apply(b, &cur_b);
                id_b = cur_b.data[cur_b.selected].id;
            }

            ms_a += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_b += std::chrono::duration<double, std::milli>(t3 - t2).count();
            n_diff += id_a != id_b;

            llama_sampler_accept(a, id_a);
            llama_sampler_accept(b, id_b);
        }
        llama_sampler_free(a);
        llama_sampler_free(b);

        printf("%-16s %d draws: %d tokens differ, %d declined; %.3f ms per draw with the chain, %.3f with the fast "
            "path\n", c.name, iters, n_diff, n_declined, ms_a/iters, ms_b/iters);
        n_bad += n_diff;
    }
    return n_bad ? 1 : 0;
}
