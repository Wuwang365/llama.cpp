#include "get-model.h"

#include "llama.h"
#include "../src/llama-model.h"

#include <cstdio>
#include <string>

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 128;
    cparams.n_batch = 128;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = llama_vocab_bos(vocab);
    if (token == LLAMA_TOKEN_NULL) {
        token = 0;
    }

    if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
        std::fprintf(stderr, "initial decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::string tensor_name;
    if (model->get_tensor("blk.0.attn_q.weight") != nullptr) {
        tensor_name = "blk.0.attn_q.weight";
    } else {
        for (const auto & [name, tensor] : llama_internal_get_tensor_map(model)) {
            if (tensor->view_src == nullptr && ggml_nbytes(tensor) > 0) {
                tensor_name = name;
                break;
            }
        }
    }

    if (tensor_name.empty()) {
        std::fprintf(stderr, "failed to find a main weight tensor\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (!model->is_tensor_loaded(tensor_name.c_str())) {
        std::fprintf(stderr, "tensor should be loaded before unload: %s\n", tensor_name.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::string err_msg;
    size_t bytes_freed = 0;
    if (!model->unload_tensor(tensor_name.c_str(), err_msg, &bytes_freed)) {
        std::fprintf(stderr, "failed to unload tensor %s: %s\n", tensor_name.c_str(), err_msg.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (bytes_freed == 0) {
        std::fprintf(stderr, "expected to free memory for tensor %s\n", tensor_name.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (model->is_tensor_loaded(tensor_name.c_str())) {
        std::fprintf(stderr, "tensor still appears loaded after unload: %s\n", tensor_name.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    err_msg.clear();
    if (model->unload_tensor(tensor_name.c_str(), err_msg, nullptr)) {
        std::fprintf(stderr, "duplicate unload unexpectedly succeeded for %s\n", tensor_name.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (err_msg.find("already unloaded") == std::string::npos) {
        std::fprintf(stderr, "unexpected duplicate-unload message: %s\n", err_msg.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
        std::fprintf(stderr, "decode after unload failed for %s\n", tensor_name.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (!model->is_tensor_loaded(tensor_name.c_str())) {
        std::fprintf(stderr, "tensor did not reload on demand: %s\n", tensor_name.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    bytes_freed = 0;
    err_msg.clear();
    if (!model->unload_tensor(tensor_name.c_str(), err_msg, &bytes_freed)) {
        std::fprintf(stderr, "tensor did not become unloadable again after reload: %s\n", err_msg.c_str());
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::printf("weight unload/reload test passed for %s\n", tensor_name.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
