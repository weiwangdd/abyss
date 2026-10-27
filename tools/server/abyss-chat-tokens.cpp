// abyss: what the server's chat path makes of chat requests, loading only the vocabulary (no weights, no GPU). For each
// request, an OpenAI chat body per line on stdin, it prints one JSON line with the rendered prompt, the special tokens
// the template wrote, the tokens the server submits with ABYSS_PLAIN_INPUT=1 ("plain") and without it ("parsed").
// It takes the server's arguments:
//
//   llama-abyss-chat-tokens -m model.gguf --jinja --chat-template-file deepseek-v4.1.jinja --no-reasoning-preserve < requests.jsonl

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "server-common.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

// milliseconds since t0
static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }
    llama_backend_init();

    llama_model_params mparams = common_model_params_to_llama(params);
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load the vocabulary from %s\n", params.model.path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // the server's chat parameters, as server_context builds them
    server_chat_params opt;
    opt.use_jinja                = params.use_jinja;
    opt.prefill_assistant        = params.prefill_assistant;
    opt.reasoning_format         = params.reasoning_format;
    opt.chat_template_kwargs     = params.default_template_kwargs;
    opt.tmpls                    = common_chat_templates_init(model, params.chat_template);
    opt.allow_image              = false;
    opt.allow_audio              = false;
    opt.allow_video              = false;
    opt.enable_thinking          = params.enable_reasoning != 0 && params.use_jinja &&
                                   common_chat_templates_support_enable_thinking(opt.tmpls.get());
    opt.reasoning_budget         = params.sampling.reasoning_budget_tokens;
    opt.reasoning_budget_message = params.sampling.reasoning_budget_message;
    opt.media_path               = params.media_path;
    opt.force_pure_content       = params.force_pure_content_parser;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        json out;
        try {
            json body = json::parse(line);
            std::vector<raw_buffer> files;
            common_chat_session session;
            auto t0 = std::chrono::steady_clock::now();
            json data = oaicompat_chat_params_parse(vocab, body, opt, files, session);
            const double ms_render = ms_since(t0);
            const std::string prompt = data.at("prompt");
            json special = json::array();
            for (const auto & s : session.prompt_special()) {
                json item = json::array();
                item.push_back(s.start);
                item.push_back(s.end);
                item.push_back(s.token);
                special.push_back(std::move(item));
            }
            out["prompt"]  = prompt;
            out["known"]   = session.prompt_special_known();
            out["special"] = special;
            out["ms_render"] = ms_render;
            t0 = std::chrono::steady_clock::now();
            out["parsed"]    = tokenize_mixed(vocab, prompt, true, true);
            out["ms_parsed"] = ms_since(t0);
            if (session.prompt_special_known()) {
                t0 = std::chrono::steady_clock::now();
                out["plain"]    = tokenize_chat_prompt_special(vocab, prompt, special, true);
                out["ms_plain"] = ms_since(t0);
            }
        } catch (const std::exception & e) {
            out = {{"error", e.what()}};
        }
        std::cout << out.dump() << "\n" << std::flush;
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
