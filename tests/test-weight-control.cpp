#include "get-model.h"
#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void require_true(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "%s\n", msg);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params params = llama_model_default_params();
    params.use_mmap = true;

    llama_model * model = llama_model_load_from_file(model_path, params);
    require_true(model != nullptr, "failed to load model");

    const int32_t n_weights = llama_model_weight_count(model);
    require_true(n_weights > 0, "expected model to report at least one weight");

    std::vector<std::string> names;
    names.reserve((size_t) n_weights);
    for (int32_t i = 0; i < n_weights; ++i) {
        char name[256];
        const int32_t len = llama_model_weight_name(model, i, name, sizeof(name));
        require_true(len > 0, "expected non-empty weight name");
        require_true(name[0] != '\0', "expected null-terminated weight name");
        names.emplace_back(name);
    }

    require_true(!names.empty(), "expected at least one listed weight");

    const llama_weight_unload_result missing =
        llama_model_unload_weight(model, "__missing_weight_for_test__");
    require_true(missing == LLAMA_WEIGHT_UNLOAD_NOT_FOUND, "expected missing weight to return not-found");

    llama_model_free(model);
    llama_backend_free();

    return EXIT_SUCCESS;
}
