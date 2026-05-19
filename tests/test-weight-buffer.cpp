#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #x); \
        return 1; \
    } \
} while (0)

int main(void) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    ASSERT_TRUE(ctx != nullptr);

    ggml_tensor * weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * view   = ggml_view_1d(ctx, weight, 4, 0);
    ggml_tensor * input  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_tensor * add    = ggml_add(ctx, weight, input);

    ggml_backend_buffer_t weight_buf = ggml_backend_buft_alloc_buffer(
            ggml_backend_cpu_buffer_type(), ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), weight));
    ASSERT_TRUE(weight_buf != nullptr);
    ggml_backend_buffer_set_usage(weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ASSERT_TRUE(ggml_backend_tensor_alloc(weight_buf, weight, ggml_backend_buffer_get_base(weight_buf)) == GGML_STATUS_SUCCESS);
    ASSERT_TRUE(weight->weight_buffer == weight_buf);

    ASSERT_TRUE(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
    ASSERT_TRUE(view->buffer == weight_buf);
    ASSERT_TRUE(view->weight_buffer == nullptr);

    ggml_backend_buffer_t compute_buf = ggml_backend_buft_alloc_buffer(
            ggml_backend_cpu_buffer_type(), ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), add));
    ASSERT_TRUE(compute_buf != nullptr);
    ASSERT_TRUE(ggml_backend_tensor_alloc(compute_buf, add, ggml_backend_buffer_get_base(compute_buf)) == GGML_STATUS_SUCCESS);
    ASSERT_TRUE(add->weight_buffer == nullptr);

    ggml_backend_buffer_free(compute_buf);
    ggml_backend_buffer_free(weight_buf);
    ggml_free(ctx);
    return 0;
}
