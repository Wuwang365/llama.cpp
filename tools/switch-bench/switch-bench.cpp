#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#endif

struct bench_params {
    int runs = 5;
    int layer_begin = -1;
    int layer_end = -1;
    bool unload_global = false;
    bool unload_output = false;
    bool preload_eager_base = false;
    bool drop_cache = false;
    std::string format = "md";
    std::string output;
};

struct bench_row {
    int run = 0;
    double ttft_ms = 0.0;
    double prefill_ms = 0.0;
    uint64_t epoch_delta = 0;
    size_t bytes_freed = 0;
    size_t tensors_unloaded = 0;
    llama_model_weight_load_metrics metrics = {};
};

static int64_t now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count();
}

static void print_usage(int, char ** argv) {
    fprintf(stderr,
            "usage: %s -m MODEL [options]\n"
            "\n"
            "switch-bench options:\n"
            "  --runs N                         repeat count (default: 5)\n"
            "  --unload-layer-window A:B         unload layers in [A, B]\n"
            "  --unload-global                   unload global tensors\n"
            "  --unload-output                   unload output tensors\n"
            "  --preload-eager-base              restore all weights before each unload window\n"
            "  --drop-cache                      best-effort page-cache hint before each run\n"
            "  --output-format md|csv|jsonl      output format (default: md)\n"
            "  --output PATH                     write results to PATH\n",
            argv[0]);
}

static bool parse_window(const char * text, int & begin, int & end) {
    return sscanf(text, "%d:%d", &begin, &end) == 2 && begin >= 0 && end >= begin;
}

static std::vector<char *> strip_switch_args(int argc, char ** argv, bench_params & bparams) {
    std::vector<char *> out;
    out.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * name) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value\n", name);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--runs") {
            bparams.runs = std::max(1, atoi(need_value("--runs")));
        } else if (arg == "--unload-layer-window") {
            if (!parse_window(need_value("--unload-layer-window"), bparams.layer_begin, bparams.layer_end)) {
                fprintf(stderr, "invalid --unload-layer-window, expected A:B\n");
                std::exit(1);
            }
        } else if (arg == "--unload-global") {
            bparams.unload_global = true;
        } else if (arg == "--unload-output") {
            bparams.unload_output = true;
        } else if (arg == "--preload-eager-base") {
            bparams.preload_eager_base = true;
        } else if (arg == "--drop-cache") {
            bparams.drop_cache = true;
        } else if (arg == "--output-format") {
            bparams.format = need_value("--output-format");
        } else if (arg == "--output") {
            bparams.output = need_value("--output");
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            std::exit(0);
        } else {
            out.push_back(argv[i]);
        }
    }

    return out;
}

static void best_effort_drop_cache(const std::string & model_path) {
#if defined(__linux__)
    if (!model_path.empty()) {
        FILE * f = fopen(model_path.c_str(), "rb");
        if (f) {
            (void) posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
            fclose(f);
        }
    }
#else
    (void) model_path;
#endif
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

static bool should_unload(int32_t layer, const bench_params & bparams) {
    if (layer == -1) {
        return bparams.unload_global;
    }
    if (layer == INT32_MAX) {
        return bparams.unload_output;
    }
    return bparams.layer_begin >= 0 && layer >= bparams.layer_begin && layer <= bparams.layer_end;
}

static bool unload_selected(llama_model * model, const bench_params & bparams, size_t & bytes_freed, size_t & tensors_unloaded) {
    bytes_freed = 0;
    tensors_unloaded = 0;
    std::vector<std::string> names;

    const int32_t n_weights = llama_model_weight_count(model);
    for (int32_t i = 0; i < n_weights; ++i) {
        size_t nbytes = 0;
        int32_t layer = -1;
        bool loaded = false;
        if (!llama_model_weight_info_by_index(model, i, &nbytes, &layer, &loaded) || !loaded || nbytes == 0) {
            continue;
        }
        if (should_unload(layer, bparams)) {
            names.push_back(weight_name_at(model, i));
        }
    }

    for (const auto & name : names) {
        size_t freed = 0;
        if (llama_model_unload_tensor(model, name.c_str(), &freed)) {
            bytes_freed += freed;
            tensors_unloaded++;
        }
    }

    return !names.empty();
}

static void write_rows(const bench_params & bparams, const std::vector<bench_row> & rows) {
    std::ostringstream out;

    if (bparams.format == "csv") {
        out << "run,ttft_ms,prefill_ms,epoch_delta,tensors_unloaded,bytes_freed,ready_ms,read_ms,upload_ms,read_mib,upload_mib,reload_groups,queues,upload_batches\n";
        for (const auto & r : rows) {
            out << r.run << ',' << r.ttft_ms << ',' << r.prefill_ms << ',' << r.epoch_delta << ','
                << r.tensors_unloaded << ',' << r.bytes_freed << ','
                << r.metrics.ready_us / 1000.0 << ',' << r.metrics.read_wall_us / 1000.0 << ','
                << r.metrics.upload_set_us / 1000.0 << ','
                << r.metrics.read_bytes / 1048576.0 << ',' << r.metrics.upload_bytes / 1048576.0 << ','
                << r.metrics.reload_group_count << ',' << r.metrics.queue_count << ',' << r.metrics.upload_batches << '\n';
        }
    } else if (bparams.format == "jsonl") {
        for (const auto & r : rows) {
            out << "{\"run\":" << r.run
                << ",\"ttft_ms\":" << r.ttft_ms
                << ",\"prefill_ms\":" << r.prefill_ms
                << ",\"epoch_delta\":" << r.epoch_delta
                << ",\"tensors_unloaded\":" << r.tensors_unloaded
                << ",\"bytes_freed\":" << r.bytes_freed
                << ",\"ready_ms\":" << r.metrics.ready_us / 1000.0
                << ",\"read_ms\":" << r.metrics.read_wall_us / 1000.0
                << ",\"upload_ms\":" << r.metrics.upload_set_us / 1000.0
                << ",\"read_mib\":" << r.metrics.read_bytes / 1048576.0
                << ",\"upload_mib\":" << r.metrics.upload_bytes / 1048576.0
                << ",\"reload_groups\":" << r.metrics.reload_group_count
                << ",\"queues\":" << r.metrics.queue_count
                << ",\"upload_batches\":" << r.metrics.upload_batches
                << "}\n";
        }
    } else {
        out << "| run | TTFT ms | prefill ms | reload groups | read MiB | upload MiB | upload sync ms | unloaded tensors |\n";
        out << "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
        for (const auto & r : rows) {
            out << "| " << r.run
                << " | " << r.ttft_ms
                << " | " << r.prefill_ms
                << " | " << r.metrics.reload_group_count
                << " | " << r.metrics.read_bytes / 1048576.0
                << " | " << r.metrics.upload_bytes / 1048576.0
                << " | " << r.metrics.upload_sync_us / 1000.0
                << " | " << r.tensors_unloaded
                << " |\n";
        }
    }

    if (bparams.output.empty()) {
        fputs(out.str().c_str(), stdout);
    } else {
        std::ofstream f(bparams.output);
        f << out.str();
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();

    bench_params bparams;
    std::vector<char *> common_argv = strip_switch_args(argc, argv, bparams);

    common_params params;
    params.n_predict = 1;
    params.prompt = "Hello";
    params.use_mmap = false;
    params.parallel_load = true;
    params.async_io_load = true;
    params.load_micro_stats = true;

    if (!common_params_parse((int) common_argv.size(), common_argv.data(), params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to initialize model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = params.prompt.empty() ? "Hello" : params.prompt;
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }
    std::vector<llama_token> tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), true, true) != n_prompt) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return 1;
    }

    std::vector<bench_row> rows;
    rows.reserve((size_t) bparams.runs);

    for (int run = 1; run <= bparams.runs; ++run) {
        if (bparams.preload_eager_base) {
            llama_model_ensure_tensors_ready(model);
        }
        if (bparams.drop_cache) {
            best_effort_drop_cache(params.model.path);
        }

        bench_row row;
        row.run = run;
        unload_selected(model, bparams, row.bytes_freed, row.tensors_unloaded);

        const uint64_t epoch_before = llama_model_weight_epoch(model);
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_perf_context_reset(ctx);

        const int64_t t0 = now_us();
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        const int ret = llama_model_has_encoder(model) ? llama_encode(ctx, batch) : llama_decode(ctx, batch);
        const int64_t t1 = now_us();
        if (ret != 0) {
            fprintf(stderr, "decode failed on run %d: %d\n", run, ret);
            return 1;
        }

        row.ttft_ms = (t1 - t0) / 1000.0;
        row.prefill_ms = row.ttft_ms;
        row.epoch_delta = llama_model_weight_epoch(model) - epoch_before;
        llama_model_weight_last_load_metrics(model, &row.metrics);
        rows.push_back(row);
    }

    write_rows(bparams, rows);

    llama_backend_free();
    return 0;
}
