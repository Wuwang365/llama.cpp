#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #x); \
        return 1; \
    } \
} while (0)

static std::string weight_name_at(const llama_model * model, int32_t i) {
    const int32_t len = llama_model_weight_name_by_index(model, i, nullptr, 0);
    if (len < 0) {
        return {};
    }
    std::vector<char> name((size_t) len + 1);
    llama_model_weight_name_by_index(model, i, name.data(), name.size());
    return name.data();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = false;
    mparams.parallel_load = true;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    ASSERT_TRUE(model != nullptr);

    const int32_t n_weights = llama_model_weight_count(model);
    ASSERT_TRUE(n_weights > 0);

    std::string name;
    for (int32_t i = 0; i < n_weights; ++i) {
        size_t nbytes = 0;
        bool loaded = false;
        if (llama_model_weight_info_by_index(model, i, &nbytes, nullptr, &loaded) && loaded && nbytes > 0) {
            name = weight_name_at(model, i);
            break;
        }
    }
    ASSERT_TRUE(!name.empty());

    size_t bytes_freed = 0;
    ASSERT_TRUE(llama_model_unload_tensor(model, name.c_str(), &bytes_freed));
    ASSERT_TRUE(bytes_freed > 0);
    ASSERT_TRUE(!llama_model_is_tensor_loaded(model, name.c_str()));
    ASSERT_TRUE(!llama_model_unload_tensor(model, name.c_str(), nullptr));

    ASSERT_TRUE(llama_model_ensure_tensors_ready(model));
    ASSERT_TRUE(llama_model_is_tensor_loaded(model, name.c_str()));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 64;
    cparams.n_batch = 16;
    cparams.n_ubatch = 16;
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    ASSERT_TRUE(ctx != nullptr);

    ASSERT_TRUE(llama_model_unload_tensor(model, name.c_str(), &bytes_freed));
    ASSERT_TRUE(!llama_model_is_tensor_loaded(model, name.c_str()));

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "Hello";
    const int n_prompt = -llama_tokenize(vocab, prompt, strlen(prompt), nullptr, 0, true, true);
    ASSERT_TRUE(n_prompt > 0);
    std::vector<llama_token> tokens((size_t) n_prompt);
    ASSERT_TRUE(llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true) == n_prompt);

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    int32_t ret = llama_model_has_encoder(model) ? llama_encode(ctx, batch) : llama_decode(ctx, batch);
    ASSERT_TRUE(ret == 0);
    ASSERT_TRUE(llama_model_is_tensor_loaded(model, name.c_str()));

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
