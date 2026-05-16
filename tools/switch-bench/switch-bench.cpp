#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-model.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

enum class output_format {
    md,
    csv,
    jsonl,
};

struct switch_params {
    int reps          = 7;
    int discard_first = 1;
    int unload_layer_start = 0;
    int unload_layers = 0;

    bool unload_global = false;
    bool unload_output = false;
    bool progress      = false;
    bool drop_caches_before_switch = false;
    bool preload_before_switch = false;

    output_format output = output_format::md;
};

struct unload_result {
    int    tensors = 0;
    int    groups  = 0;
    size_t bytes   = 0;
};

struct sample_result {
    int    rep = 0;
    int    n_prompt = 0;
    int    n_gen_tail = 0;
    int    unloaded_tensors = 0;
    int    unloaded_groups = 0;
    size_t unloaded_bytes = 0;

    double weight_prepare_ms = 0.0;
    double weight_io_ms = 0.0;
    double weight_upload_ms = 0.0;
    double weight_ready_wall_ms = 0.0;
    double weight_wait_during_decode_ms = 0.0;
    double switch_to_first_token_ms = 0.0;
    double decode_tail_ms = 0.0;

    size_t loaded_tensors = 0;
    size_t loaded_groups = 0;
    size_t read_bytes = 0;
    size_t upload_bytes = 0;
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -ngl 99 -npp 512 -r 7 --unload-layers 4 --async-io-load --no-mmap --output-format md\n", argv[0]);
    LOG("\ncustom switch-bench options:\n");
    LOG("  -r, --repetitions N              number of measured switch rounds (default: 7)\n");
    LOG("  --discard-first N                samples to discard in summary stats (default: 1)\n");
    LOG("  --unload-layer-start N           first layer group to unload before each round (default: 0)\n");
    LOG("  --unload-layers N                before each round, unload one tensor from N layer groups (default: 0)\n");
    LOG("  --unload-global                  also mark the global weight group unloaded before each round\n");
    LOG("  --unload-output                  also mark the output weight group unloaded before each round\n");
    LOG("  --drop-caches-before-switch      run sync/drop_caches after unload and before switch timing\n");
    LOG("  --preload-before-switch          synchronously ready all model weights before decode, included in switch timing\n");
    LOG("  --output-format md|csv|jsonl     output format (default: md)\n");
    LOG("  --progress                       print per-round progress to stderr\n");
    LOG("\ncommon benchmark options still apply, including -m, -ngl, -npp, -ntg, -c, -b, -ub, --mmap, --async-io-load, --load-micro-stats, --warmup.\n");
    LOG("\n");
}

static bool parse_output_format(const std::string & value, output_format & out) {
    if (value == "md") {
        out = output_format::md;
        return true;
    }
    if (value == "csv") {
        out = output_format::csv;
        return true;
    }
    if (value == "jsonl") {
        out = output_format::jsonl;
        return true;
    }
    return false;
}

static bool extract_switch_args(int argc, char ** argv, switch_params & sparams, std::vector<std::string> & common_args) {
    common_args.clear();
    common_args.emplace_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s: expected value for %s\n", __func__, name);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "-r" || arg == "--repetitions") {
            sparams.reps = std::stoi(require_value(arg.c_str()));
        } else if (arg == "--discard-first") {
            sparams.discard_first = std::stoi(require_value(arg.c_str()));
        } else if (arg == "--unload-layer-start") {
            sparams.unload_layer_start = std::stoi(require_value(arg.c_str()));
        } else if (arg == "--unload-layers") {
            sparams.unload_layers = std::stoi(require_value(arg.c_str()));
        } else if (arg == "--unload-global") {
            sparams.unload_global = true;
        } else if (arg == "--unload-output") {
            sparams.unload_output = true;
        } else if (arg == "--drop-caches-before-switch") {
            sparams.drop_caches_before_switch = true;
        } else if (arg == "--preload-before-switch") {
            sparams.preload_before_switch = true;
        } else if (arg == "--progress") {
            sparams.progress = true;
        } else if (arg == "--output-format") {
            const std::string value = require_value(arg.c_str());
            if (!parse_output_format(value, sparams.output)) {
                fprintf(stderr, "%s: invalid --output-format '%s'\n", __func__, value.c_str());
                return false;
            }
        } else {
            common_args.push_back(arg);
        }
    }

    if (sparams.reps <= 0) {
        fprintf(stderr, "%s: --repetitions must be > 0\n", __func__);
        return false;
    }
    if (sparams.discard_first < 0) {
        fprintf(stderr, "%s: --discard-first must be >= 0\n", __func__);
        return false;
    }
    if (sparams.unload_layers < 0) {
        fprintf(stderr, "%s: --unload-layers must be >= 0\n", __func__);
        return false;
    }
    if (sparams.unload_layer_start < 0) {
        fprintf(stderr, "%s: --unload-layer-start must be >= 0\n", __func__);
        return false;
    }

    return true;
}

static int tensor_layer_index(const char * name) {
    const char * blk = std::strstr(name, "blk.");
    if (blk == nullptr) {
        return -1;
    }

    const char * cur = blk + 4;
    if (*cur < '0' || *cur > '9') {
        return -1;
    }

    int il = 0;
    while (*cur >= '0' && *cur <= '9') {
        il = il * 10 + (*cur - '0');
        ++cur;
    }

    return *cur == '.' ? il : -1;
}

static bool tensor_name_is_output(const char * name) {
    return std::strncmp(name, "output", 6) == 0 ||
           std::strncmp(name, "enc.output", 10) == 0 ||
           std::strncmp(name, "dec.output", 10) == 0 ||
           std::strncmp(name, "cls", 3) == 0 ||
           std::strncmp(name, "dense", 5) == 0;
}

static unload_result unload_selected_weight_groups(llama_model * model, const switch_params & sparams) {
    unload_result result;
    std::set<int> unloaded_layers;
    bool unloaded_global = false;
    bool unloaded_output = false;

    for (const auto & name_tensor : llama_internal_get_tensor_map(model)) {
        const std::string & name = name_tensor.first;
        const ggml_tensor * tensor = name_tensor.second;
        if (tensor->view_src != nullptr) {
            continue;
        }

        bool selected = false;
        const int il = tensor_layer_index(name.c_str());
        if (il >= sparams.unload_layer_start &&
                il < sparams.unload_layer_start + sparams.unload_layers &&
                unloaded_layers.insert(il).second) {
            selected = true;
        } else if (il < 0 && sparams.unload_output && tensor_name_is_output(name.c_str()) && !unloaded_output) {
            unloaded_output = true;
            selected = true;
        } else if (il < 0 && sparams.unload_global && !tensor_name_is_output(name.c_str()) && !unloaded_global) {
            unloaded_global = true;
            selected = true;
        }

        if (!selected) {
            continue;
        }

        size_t bytes_freed = 0;
        std::string err_msg;
        if (model->unload_tensor(name.c_str(), err_msg, &bytes_freed)) {
            result.tensors++;
            result.bytes += bytes_freed;
        } else {
            fprintf(stderr, "%s: warning: failed to unload '%s': %s\n", __func__, name.c_str(), err_msg.c_str());
        }
    }

    result.groups = (int) unloaded_layers.size() + (unloaded_global ? 1 : 0) + (unloaded_output ? 1 : 0);
    return result;
}

static bool preload_selected_weight_groups(llama_model * model, const switch_params & sparams) {
    std::string err_msg;

    if (sparams.unload_global && !model->ensure_global_tensors_ready(err_msg)) {
        fprintf(stderr, "%s: failed to preload global tensors: %s\n", __func__, err_msg.c_str());
        return false;
    }

    const int layer_end = sparams.unload_layer_start + sparams.unload_layers;
    for (int il = sparams.unload_layer_start; il < layer_end; ++il) {
        if (!model->ensure_layer_tensors_ready(il, err_msg)) {
            fprintf(stderr, "%s: failed to preload layer %d tensors: %s\n", __func__, il, err_msg.c_str());
            return false;
        }
    }

    if (sparams.unload_output && !model->ensure_output_tensors_ready(err_msg)) {
        fprintf(stderr, "%s: failed to preload output tensors: %s\n", __func__, err_msg.c_str());
        return false;
    }

    return true;
}

static bool drop_caches_before_switch() {
    const char * cmd =
        "(sync; echo 3 > /proc/sys/vm/drop_caches) 2>/dev/null || "
        "su -c 'sync; echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1";
    return std::system(cmd) == 0;
}

static llama_tokens make_prompt_tokens(const common_params & params, const llama_vocab * vocab, int n_prompt) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_tokens tokens;

    if (!params.prompt.empty()) {
        tokens = common_tokenize(vocab, params.prompt, llama_vocab_get_add_bos(vocab), true);
        if (n_prompt > 0 && (int) tokens.size() > n_prompt) {
            tokens.resize(n_prompt);
        }
    }

    if (tokens.empty()) {
        const int n = std::max(1, n_prompt);
        tokens.resize(n);
        tokens[0] = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        for (int i = 1; i < n; ++i) {
            tokens[i] = std::rand() % n_vocab;
        }
    } else if (n_prompt > 0) {
        while ((int) tokens.size() < n_prompt) {
            tokens.push_back(std::rand() % n_vocab);
        }
    }

    return tokens;
}

static bool decode_tokens(llama_context * ctx, const llama_tokens & tokens, int n_batch) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);

    for (int i = 0; i < (int) tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, i == (int) tokens.size() - 1);
    }

    for (int i = 0; i < batch.n_tokens; i += n_batch) {
        const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);
        llama_batch batch_view = {
            n_tokens,
            batch.token    + i,
            nullptr,
            batch.pos      + i,
            batch.n_seq_id + i,
            batch.seq_id   + i,
            batch.logits   + i,
        };

        const int ret = llama_decode(ctx, batch_view);
        if (ret != 0) {
            fprintf(stderr, "%s: llama_decode failed, ret = %d\n", __func__, ret);
            llama_batch_free(batch);
            return false;
        }
    }

    llama_synchronize(ctx);
    llama_batch_free(batch);
    return true;
}

static bool decode_one(llama_context * ctx, llama_token token, int pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, token, pos, { 0 }, true);
    const int ret = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    llama_batch_free(batch);
    if (ret != 0) {
        fprintf(stderr, "%s: llama_decode failed, ret = %d\n", __func__, ret);
        return false;
    }
    return true;
}

static sample_result run_one(
        llama_model * model,
        llama_context * ctx,
        llama_sampler * sampler,
        const common_params & params,
        const switch_params & sparams,
        const llama_tokens & prompt_tokens,
        int n_gen_tail,
        int rep) {
    sample_result result;
    result.rep = rep;
    result.n_prompt = (int) prompt_tokens.size();
    result.n_gen_tail = n_gen_tail;

    llama_memory_clear(llama_get_memory(ctx), false);
    llama_sampler_reset(sampler);

    const int64_t t_prepare_start = llama_time_us();
    const unload_result unload = unload_selected_weight_groups(model, sparams);
    const int64_t t_prepare_end = llama_time_us();

    result.unloaded_tensors = unload.tensors;
    result.unloaded_groups  = unload.groups;
    result.unloaded_bytes   = unload.bytes;
    result.weight_prepare_ms = (t_prepare_end - t_prepare_start) / 1000.0;

    if (sparams.drop_caches_before_switch && !drop_caches_before_switch()) {
        fprintf(stderr, "%s: warning: failed to drop page cache before switch\n", __func__);
    }

    model->reset_weight_load_metrics();

    const int64_t t_switch_start = llama_time_us();
    if (sparams.preload_before_switch) {
        if (!preload_selected_weight_groups(model, sparams)) {
            std::exit(1);
        }
    }
    if (!decode_tokens(ctx, prompt_tokens, params.n_batch)) {
        std::exit(1);
    }
    const int64_t t_switch_end = llama_time_us();
    result.switch_to_first_token_ms = (t_switch_end - t_switch_start) / 1000.0;

    llama_token token = llama_sampler_sample(sampler, ctx, -1);
    llama_sampler_accept(sampler, token);

    const int64_t t_tail_start = llama_time_us();
    for (int i = 0; i < n_gen_tail; ++i) {
        if (!decode_one(ctx, token, result.n_prompt + i)) {
            std::exit(1);
        }
        token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, token);
    }
    const int64_t t_tail_end = llama_time_us();
    result.decode_tail_ms = (t_tail_end - t_tail_start) / 1000.0;

    const llama_weight_load_metrics metrics = model->get_weight_load_metrics();
    result.weight_io_ms = metrics.read_wall_ns / 1e6;
    result.weight_upload_ms = (metrics.upload_set_ns + metrics.upload_sync_ns) / 1e6;
    result.weight_ready_wall_ms = metrics.ready_wall_ns / 1e6;
    result.weight_wait_during_decode_ms = metrics.wait_ns / 1e6;
    result.loaded_tensors = metrics.n_tensors;
    result.loaded_groups = metrics.n_groups_loaded;
    result.read_bytes = metrics.read_bytes;
    result.upload_bytes = metrics.upload_bytes;

    return result;
}

static double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t idx = std::min(values.size() - 1, (size_t) std::ceil(p * values.size()) - 1);
    return values[idx];
}

static std::vector<double> kept_switch_samples(const std::vector<sample_result> & results, int discard_first) {
    std::vector<double> values;
    for (int i = std::min(discard_first, (int) results.size()); i < (int) results.size(); ++i) {
        values.push_back(results[i].switch_to_first_token_ms);
    }
    return values;
}

static void print_csv_header() {
    printf("rep,n_prompt,n_gen_tail,unloaded_groups,unloaded_tensors,unloaded_bytes,weight_prepare_ms,weight_io_ms,weight_upload_ms,weight_ready_wall_ms,weight_wait_during_decode_ms,switch_to_first_token_ms,decode_tail_ms,loaded_groups,loaded_tensors,read_bytes,upload_bytes\n");
}

static void print_csv_row(const sample_result & r) {
    printf("%d,%d,%d,%d,%d,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%zu,%zu,%zu,%zu\n",
            r.rep, r.n_prompt, r.n_gen_tail, r.unloaded_groups, r.unloaded_tensors, r.unloaded_bytes,
            r.weight_prepare_ms, r.weight_io_ms, r.weight_upload_ms, r.weight_ready_wall_ms, r.weight_wait_during_decode_ms,
            r.switch_to_first_token_ms, r.decode_tail_ms, r.loaded_groups, r.loaded_tensors, r.read_bytes, r.upload_bytes);
}

static void print_jsonl_row(const sample_result & r) {
    printf("{\"rep\":%d,\"n_prompt\":%d,\"n_gen_tail\":%d,\"unloaded_groups\":%d,\"unloaded_tensors\":%d,\"unloaded_bytes\":%zu,"
           "\"weight_prepare_ms\":%.3f,\"weight_io_ms\":%.3f,\"weight_upload_ms\":%.3f,\"weight_ready_wall_ms\":%.3f,\"weight_wait_during_decode_ms\":%.3f,"
           "\"switch_to_first_token_ms\":%.3f,\"decode_tail_ms\":%.3f,\"loaded_groups\":%zu,\"loaded_tensors\":%zu,\"read_bytes\":%zu,\"upload_bytes\":%zu}\n",
            r.rep, r.n_prompt, r.n_gen_tail, r.unloaded_groups, r.unloaded_tensors, r.unloaded_bytes,
            r.weight_prepare_ms, r.weight_io_ms, r.weight_upload_ms, r.weight_ready_wall_ms, r.weight_wait_during_decode_ms,
            r.switch_to_first_token_ms, r.decode_tail_ms, r.loaded_groups, r.loaded_tensors, r.read_bytes, r.upload_bytes);
}

static void print_md_header(const common_params & params, llama_model * model, double model_load_ms, double context_init_ms) {
    char model_desc[256];
    llama_model_desc(model, model_desc, sizeof(model_desc));

    printf("# llama-switch-bench\n\n");
    printf("- model: `%s`\n", params.model.path.c_str());
    printf("- model_desc: `%s`\n", model_desc);
    printf("- model_size: %.2f MiB\n", llama_model_size(model) / 1024.0 / 1024.0);
    printf("- model_params: %.2f B\n", llama_model_n_params(model) / 1e9);
    printf("- n_gpu_layers: %d\n", params.n_gpu_layers);
    printf("- mmap: %s\n", params.use_mmap ? "true" : "false");
    printf("- async_io_load: %s\n", params.async_io_load ? "true" : "false");
    printf("- parallel_load: %s\n", params.parallel_load ? "true" : "false");
    printf("- context_init_ms: %.3f\n", context_init_ms);
    printf("- initial_model_load_ms: %.3f\n\n", model_load_ms);
    printf("| rep | n_prompt | unloaded groups | unloaded tensors | weight prepare ms | weight IO ms | weight upload ms | weight ready wall ms | weight wait/decode ms | switch_to_first_token ms | tail decode ms |\n");
    printf("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
}

static void print_md_row(const sample_result & r) {
    printf("| %d | %d | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |\n",
            r.rep, r.n_prompt, r.unloaded_groups, r.unloaded_tensors,
            r.weight_prepare_ms, r.weight_io_ms, r.weight_upload_ms, r.weight_ready_wall_ms, r.weight_wait_during_decode_ms,
            r.switch_to_first_token_ms, r.decode_tail_ms);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    switch_params sparams;
    std::vector<std::string> common_arg_storage;
    if (!extract_switch_args(argc, argv, sparams, common_arg_storage)) {
        return 1;
    }

    std::vector<char *> common_argv;
    common_argv.reserve(common_arg_storage.size());
    for (std::string & arg : common_arg_storage) {
        common_argv.push_back(arg.data());
    }

    common_params params;
    common_init();

    if (!common_params_parse((int) common_argv.size(), common_argv.data(), params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    if (params.n_pp.empty()) {
        params.n_pp.push_back(512);
    }
    if (params.n_tg.empty()) {
        params.n_tg.push_back(0);
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    const int64_t t_model_start = llama_time_us();
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    const int64_t t_model_end = llama_time_us();
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        llama_backend_free();
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    const int64_t t_context_start = llama_time_us();
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    const int64_t t_context_end = llama_time_us();
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    if (params.warmup) {
        const llama_tokens warmup_tokens = make_prompt_tokens(params, vocab, std::min(16, std::max(1, params.n_pp.front())));
        if (sparams.progress) {
            fprintf(stderr, "llama-switch-bench: warmup\n");
        }
        if (!decode_tokens(ctx, warmup_tokens, params.n_batch)) {
            llama_sampler_free(sampler);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        llama_memory_clear(llama_get_memory(ctx), false);
        model->reset_weight_load_metrics();
    }

    const double model_load_ms = (t_model_end - t_model_start) / 1000.0;
    const double context_init_ms = (t_context_end - t_context_start) / 1000.0;

    if (sparams.output == output_format::csv) {
        print_csv_header();
    } else if (sparams.output == output_format::md) {
        print_md_header(params, model, model_load_ms, context_init_ms);
    }

    for (const int n_prompt : params.n_pp) {
        for (const int n_gen_tail : params.n_tg) {
            const llama_tokens prompt_tokens = make_prompt_tokens(params, vocab, n_prompt);
            std::vector<sample_result> results;
            results.reserve(sparams.reps);

            for (int rep = 0; rep < sparams.reps; ++rep) {
                if (sparams.progress) {
                    fprintf(stderr, "llama-switch-bench: prompt %d, tail %d, round %d/%d\n",
                            (int) prompt_tokens.size(), n_gen_tail, rep + 1, sparams.reps);
                }

                sample_result result = run_one(model, ctx, sampler, params, sparams, prompt_tokens, n_gen_tail, rep);
                results.push_back(result);

                if (sparams.output == output_format::csv) {
                    print_csv_row(result);
                } else if (sparams.output == output_format::jsonl) {
                    print_jsonl_row(result);
                } else {
                    print_md_row(result);
                }
                fflush(stdout);
            }

            if (sparams.output == output_format::md) {
                const std::vector<double> kept = kept_switch_samples(results, sparams.discard_first);
                printf("\n");
                printf("summary n_prompt=%d n_gen_tail=%d discard_first=%d: median_switch_to_first_token_ms=%.3f, p90_switch_to_first_token_ms=%.3f\n\n",
                        (int) prompt_tokens.size(), n_gen_tail, std::min(sparams.discard_first, (int) results.size()),
                        percentile(kept, 0.50), percentile(kept, 0.90));
            }
        }
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
