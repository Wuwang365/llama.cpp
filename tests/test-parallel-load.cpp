#include "llama.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #x); \
        return 1; \
    } \
} while (0)

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static void touch_file(const std::string & path) {
    std::ofstream f(path, std::ios::binary);
    f << "gate";
}

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
    mparams.async_io_load = true;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    ASSERT_TRUE(model != nullptr);

    std::string layer_weight;
    for (int32_t i = 0; i < llama_model_weight_count(model); ++i) {
        size_t nbytes = 0;
        int32_t layer = -1;
        bool loaded = false;
        if (llama_model_weight_info_by_index(model, i, &nbytes, &layer, &loaded) && loaded && nbytes > 0 && layer >= 0 && layer != INT32_MAX) {
            layer_weight = weight_name_at(model, i);
            break;
        }
    }
    ASSERT_TRUE(!layer_weight.empty());

    const std::string gate_path = std::string(argv[1]) + ".parallel-load-gate";

    size_t bytes_freed = 0;
    ASSERT_TRUE(llama_model_unload_tensor(model, layer_weight.c_str(), &bytes_freed));
    ASSERT_TRUE(bytes_freed > 0);
    ASSERT_TRUE(!llama_model_is_tensor_loaded(model, layer_weight.c_str()));

    touch_file(gate_path);
    set_env_var("LLAMA_WEIGHT_TEST_WAIT_FILE", gate_path.c_str());
    llama_model_start_async_tensors_load(model);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(!llama_model_is_tensor_loaded(model, layer_weight.c_str()));
    std::remove(gate_path.c_str());
    llama_model_wait_async_tensors_load(model);
    ASSERT_TRUE(llama_model_is_tensor_loaded(model, layer_weight.c_str()));

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 64;
    cparams.n_batch = 16;
    cparams.n_ubatch = 16;
    cparams.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    ASSERT_TRUE(ctx != nullptr);

    ASSERT_TRUE(llama_model_unload_tensor(model, layer_weight.c_str(), &bytes_freed));
    ASSERT_TRUE(!llama_model_is_tensor_loaded(model, layer_weight.c_str()));

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "Hello";
    const int n_prompt = -llama_tokenize(vocab, prompt, strlen(prompt), nullptr, 0, true, true);
    ASSERT_TRUE(n_prompt > 0);
    std::vector<llama_token> tokens((size_t) n_prompt);
    ASSERT_TRUE(llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true) == n_prompt);

    touch_file(gate_path);
    std::atomic<bool> finished{false};
    int32_t decode_ret = -100;
    std::thread decode_thread([&] {
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        decode_ret = llama_model_has_encoder(model) ? llama_encode(ctx, batch) : llama_decode(ctx, batch);
        finished.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(!finished.load(std::memory_order_acquire));
    ASSERT_TRUE(!llama_model_is_tensor_loaded(model, layer_weight.c_str()));

    std::remove(gate_path.c_str());
    decode_thread.join();
    ASSERT_TRUE(finished.load(std::memory_order_acquire));
    ASSERT_TRUE(decode_ret == 0);
    ASSERT_TRUE(llama_model_is_tensor_loaded(model, layer_weight.c_str()));

    set_env_var("LLAMA_WEIGHT_TEST_WAIT_FILE", nullptr);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
