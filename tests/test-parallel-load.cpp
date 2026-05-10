#include "get-model.h"

#include "llama.h"
#include "../src/llama-model.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <mutex>
#include <string>
#include <thread>

struct parallel_load_probe {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered  = false;
    bool released = false;
    std::thread::id callback_thread;
};

static bool parallel_load_progress(float progress, void * user_data) {
    auto * probe = static_cast<parallel_load_probe *>(user_data);

    std::unique_lock<std::mutex> lock(probe->mutex);
    if (!probe->entered) {
        probe->entered = true;
        probe->callback_thread = std::this_thread::get_id();
        probe->cv.notify_all();

        probe->cv.wait(lock, [&] {
            return probe->released;
        });
    }

    (void) progress;
    return true;
}

static bool wait_for_parallel_loader(parallel_load_probe & probe) {
    std::unique_lock<std::mutex> lock(probe.mutex);
    return probe.cv.wait_for(lock, std::chrono::seconds(10), [&] {
        return probe.entered;
    });
}

static void release_parallel_loader(parallel_load_probe & probe) {
    {
        std::lock_guard<std::mutex> lock(probe.mutex);
        probe.released = true;
    }
    probe.cv.notify_all();
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    parallel_load_probe probe;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = false;
    mparams.parallel_load = true;
    mparams.progress_callback = parallel_load_progress;
    mparams.progress_callback_user_data = &probe;

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

    if (!wait_for_parallel_loader(probe)) {
        std::fprintf(stderr, "background tensor loader did not start\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = llama_vocab_bos(vocab);
    if (token == LLAMA_TOKEN_NULL) {
        token = 0;
    }

    auto decode_future = std::async(std::launch::async, [&] {
        return llama_decode(ctx, llama_batch_get_one(&token, 1));
    });

    if (decode_future.wait_for(std::chrono::milliseconds(100)) != std::future_status::timeout) {
        std::fprintf(stderr, "decode finished while background load was intentionally blocked\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    release_parallel_loader(probe);

    const int ret = decode_future.get();
    if (ret != 0) {
        std::fprintf(stderr, "decode failed after background load completed: %d\n", ret);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (probe.callback_thread == std::thread::id() || probe.callback_thread == std::this_thread::get_id()) {
        std::fprintf(stderr, "progress callback did not run on a background thread\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    std::printf("parallel load/decode synchronization test passed\n");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
