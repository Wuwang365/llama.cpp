# Parallel model weight loading

## Goal

This change lets llama.cpp overlap model weight loading with the runtime work that happens after a `llama_context` is created. When `llama_model_params::parallel_load` is enabled, model metadata and tensor descriptors are still loaded synchronously, but concrete tensor allocation and weight data upload are started by a background thread after context initialization has finished.

The feature is exposed in the common CLI as:

```bash
--parallel-load
--no-parallel-load
```

The default remains `--no-parallel-load`, so existing callers keep the previous first-inference lazy-load behavior unless they opt in.

## Design

The existing lazy tensor path already creates tensor metadata and zero-sized placeholder buffers during `llama_model::load_tensors()`. Before this change, the first `llama_decode()` or `llama_encode()` call synchronously loaded all deferred weights through `ensure_tensors_ready()`.

The new flow is:

1. `llama_model::load_tensors()` keeps deferring concrete tensor allocation and data load.
2. `llama_context` finishes backend setup, KV allocation, scheduler reserve, and sampler vocabulary initialization.
3. `llama_context` calls `llama_model::start_async_tensors_load()`.
4. A background thread calls `ensure_tensors_ready()`.
5. If inference reaches `ensure_tensors_ready()` before the background loader finishes, it synchronizes on the same model mutex and observes either the completed load or the stored error.

The loader thread is joined from `llama_model::~llama_model()` via `wait_async_tensors_load()` so model destruction cannot race with in-flight tensor upload.

## Vulkan fix

While validating on Pixel 8, Vulkan crashed in the deferred first-load path. The cause was that the lazy implementation uses one backend buffer per tensor, but `llama_model_loader::load_all_data()` assumes its mmap device path can map each GGUF file index to one large model buffer. With per-tensor Vulkan buffers, that old assumption can try to allocate later tensors into the first tensor's small buffer.

The fix is to load lazy tensors per tensor for the initial full load as well:

```cpp
llama_load_all_tensor_data_per_tensor(...)
```

This uses the same safe primitive as the later on-demand reload path:

```cpp
llama_load_tensor_data(...)
```

That keeps Vulkan, CPU, mmap, and non-mmap paths consistent under the per-tensor lazy allocation model.

## Tests

Added `tests/test-parallel-load.cpp`.

The test enables `parallel_load`, blocks the progress callback on the background loader, starts `llama_decode()` on another thread, verifies decode is waiting while the loader is blocked, releases the loader, then verifies decode completes successfully.

Local verification:

```bash
cmake --build build --target test-parallel-load test-weight-unload -j$(nproc)
ctest --test-dir build -R 'test-parallel-load|test-weight-unload' --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 3
```

## Pixel 8 Vulkan verification

Build:

```bash
cmake --build build-android-vulkan-pixel --target llama-cli -j$(nproc)
```

Device:

```text
Vulkan0: Mali-G715 (7287 MiB, 7287 MiB free)
```

Model:

```text
/home/wuwang/workspace/model_weights/qwen3-1.7b-gguf/Qwen3-1.7B-Q8_0.gguf
```

Run command on device:

```bash
cd /data/local/tmp/llama-parallel
LD_LIBRARY_PATH=. ./llama-cli \
  -m Qwen3-1.7B-Q8_0.gguf \
  -ngl 99 \
  --parallel-load \
  --no-warmup \
  --single-turn \
  --no-display-prompt \
  -c 512 \
  -n 4 \
  -p "Hello"
```

Observed result:

```text
Vulkan0 (Mali-G715): model 1743 MiB, context 56 MiB, compute 300 MiB
Prompt: 1.4 t/s
Generation: 9.7 t/s
```

The same path also ran with a derived Q4_0 copy of the 1.7B model:

```text
Vulkan0 (Mali-G715): model 999 MiB, context 56 MiB, compute 300 MiB
Prompt: 2.6 t/s
Generation: 10.8 t/s
```
