#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml.h>

int main() {
    ggml_init_params params = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    GGML_ASSERT(ctx);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    GGML_ASSERT(buft != nullptr);

    ggml_tensor * w0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
    ggml_tensor * w1 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
    ggml_tensor * c0 = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
    GGML_ASSERT(w0 && w1 && c0);

    const size_t w0_size = ggml_backend_buft_get_alloc_size(buft, w0);
    const size_t w1_size = ggml_backend_buft_get_alloc_size(buft, w1);
    const size_t c0_size = ggml_backend_buft_get_alloc_size(buft, c0);

    ggml_backend_buffer_ptr buf_w0 { ggml_backend_buft_alloc_buffer(buft, w0_size) };
    ggml_backend_buffer_ptr buf_w1 { ggml_backend_buft_alloc_buffer(buft, w1_size) };
    ggml_backend_buffer_ptr buf_c0 { ggml_backend_buft_alloc_buffer(buft, c0_size) };
    GGML_ASSERT(buf_w0 && buf_w1 && buf_c0);

    ggml_backend_buffer_set_usage(buf_w0.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(buf_w1.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(buf_c0.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);

    GGML_ASSERT(ggml_backend_tensor_alloc(buf_w0.get(), w0, ggml_backend_buffer_get_base(buf_w0.get())) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_tensor_alloc(buf_w1.get(), w1, ggml_backend_buffer_get_base(buf_w1.get())) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(ggml_backend_tensor_alloc(buf_c0.get(), c0, ggml_backend_buffer_get_base(buf_c0.get())) == GGML_STATUS_SUCCESS);

    // allocated weight tensors keep a copied backend buffer handle
    GGML_ASSERT(w0->weight_buffer == w0->buffer);
    GGML_ASSERT(w1->weight_buffer == w1->buffer);
    GGML_ASSERT(c0->weight_buffer == nullptr);

    // strict 1:1 expectation for main tensors: distinct tensors can map to distinct buffers
    GGML_ASSERT(w0->buffer != w1->buffer);

    // view tensors should never carry the weight_buffer handle
    ggml_tensor * v0 = ggml_view_1d(ctx.get(), w0, 16, 0);
    GGML_ASSERT(v0);
    GGML_ASSERT(ggml_backend_view_init(v0) == GGML_STATUS_SUCCESS);
    GGML_ASSERT(v0->buffer == w0->buffer);
    GGML_ASSERT(v0->weight_buffer == nullptr);

    return 0;
}
