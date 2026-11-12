// Decodes a random prompt in one or more llama_decode() calls, then generates greedily one token at a time, and writes
// the logits after every call and every generated token to a file. Run it twice, with and without ABYSS_CED_PREFILL=1
// (the switch is read once per process), and compare the files with ced_compare.py: the CED-aware prefill skips work
// nothing reads, so the logits and everything generated after the prompt must stay the same.
//
// usage: ced_check <model.gguf> <out.bin> [--calls a,b,c] [--ubatch U] [--gen G] [--seed S] [--ngl N] [--ctx C]
//                  [--outputs k]        mark every k-th token of a call as an output too (their logits are not saved)
//                  [--host-experts L]   keep the experts of layers L and up in host memory
//                  [--moe-cache MiB]    device cache for those experts, [--cpu-share f] share of its misses left to the CPU
//                  [--ts a,b]           the shares of the layers on each device, like -ts
#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> [--calls a,b,c] [--ubatch U] [--gen G] [--seed S] [--ngl N] [--ctx C] [--outputs k]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];

    std::vector<int> calls = { 300 };
    int n_ubatch = 64;
    int n_gen    = 24;
    int seed     = 1;
    int ngl      = 0;
    int n_ctx    = 0;
    int out_step = 0;
    int host_from = -1;
    int moe_cache = 0;
    float cpu_share = 0.0f;
    std::vector<float> tensor_split;

    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        const char * v = argv[i + 1];
        if (a == "--calls") {
            calls.clear();
            for (const char * p = v; *p != 0;) {
                calls.push_back(atoi(p));
                p = strchr(p, ',');
                if (p == nullptr) {
                    break;
                }
                ++p;
            }
        } else if (a == "--ubatch") {
            n_ubatch = atoi(v);
        } else if (a == "--gen") {
            n_gen = atoi(v);
        } else if (a == "--seed") {
            seed = atoi(v);
        } else if (a == "--ngl") {
            ngl = atoi(v);
        } else if (a == "--ctx") {
            n_ctx = atoi(v);
        } else if (a == "--outputs") {
            out_step = atoi(v);
        } else if (a == "--host-experts") {
            host_from = atoi(v);
        } else if (a == "--moe-cache") {
            moe_cache = atoi(v);
        } else if (a == "--cpu-share") {
            cpu_share = (float) atof(v);
        } else if (a == "--ts") {
            for (const char * p = v; *p != 0;) {
                tensor_split.push_back((float) atof(p));
                p = strchr(p, ',');
                if (p == nullptr) {
                    break;
                }
                ++p;
            }
        } else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 1;
        }
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    if (!tensor_split.empty()) {
        tensor_split.resize(llama_max_devices(), 0.0f);
        mparams.tensor_split = tensor_split.data();
    }

    // the experts of layers host_from and up stay in host memory, like -ot "blk\.(..)\.ffn_(gate|up|down)_exps\.=CPU"
    std::string host_pattern;
    llama_model_tensor_buft_override overrides[2] = { { nullptr, nullptr }, { nullptr, nullptr } };
    if (host_from >= 0) {
        host_pattern = "blk\\.(";
        for (int il = host_from; il < 64; ++il) {
            host_pattern += (il > host_from ? "|" : "") + std::to_string(il);
        }
        host_pattern += ")\\.ffn_(gate|up|down)_exps\\.";
        overrides[0] = { host_pattern.c_str(), ggml_backend_cpu_buffer_type() };
        mparams.tensor_buft_overrides = overrides;
    }
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    int n_max = 0;
    int n_all = n_gen;
    for (int c : calls) {
        n_max  = c > n_max ? c : n_max;
        n_all += c;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_max;
    cparams.n_ubatch        = n_ubatch;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    if (moe_cache > 0) {
        cparams.moe_cache_size      = (size_t) moe_cache*1024*1024;
        cparams.moe_cache_cpu_share = cpu_share;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }
    if (n_all > (int) llama_n_ctx(ctx)) {
        fprintf(stderr, "%d tokens do not fit the context of %u\n", n_all, llama_n_ctx(ctx));
        return 1;
    }

    FILE * out = fopen(out_path, "wb");
    if (out == nullptr) {
        fprintf(stderr, "cannot write %s\n", out_path);
        return 1;
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, n_vocab - 1);

    llama_batch batch = llama_batch_init(n_max, 0, 1);
    llama_pos   pos   = 0;
    int         rows  = 0;

    const auto decode = [&](const std::vector<llama_token> & tokens, const char * what) {
        batch.n_tokens = (int32_t) tokens.size();
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i]     = tokens[i];
            batch.pos[i]       = pos++;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = i == batch.n_tokens - 1 || (out_step > 0 && i % out_step == out_step - 1);
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "llama_decode failed with %d on %s of %d tokens\n", ret, what, batch.n_tokens);
            exit(1);
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fwrite(logits, sizeof(float), n_vocab, out);
        ++rows;

        llama_token best = 0;
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > logits[best]) {
                best = v;
            }
        }
        if (batch.n_tokens > 1) {
            printf("%s: %d tokens in %.1f ms, next token %d\n", what, batch.n_tokens, ms, best);
        }
        return best;
    };

    llama_token next = 0;
    for (int c : calls) {
        std::vector<llama_token> tokens(c);
        for (auto & t : tokens) {
            t = dist(rng);
        }
        next = decode(tokens, "call");
    }

    std::string gen;
    for (int i = 0; i < n_gen; ++i) {
        gen += " " + std::to_string(next);
        next = decode({ next }, "gen");
    }
    printf("generated:%s\n", gen.c_str());
    printf("%d logits rows of %d\n", rows, n_vocab);

    fclose(out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
