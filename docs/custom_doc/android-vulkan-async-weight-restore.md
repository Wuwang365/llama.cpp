# Android Vulkan 异步权重恢复

本文说明当前分支新增的 Android Vulkan 权重卸载与异步恢复能力。

## 目标

在 Android Vulkan 上支持加载后卸载部分模型权重，并在下一次推理时自动恢复。当前推荐策略是保留：

```text
token_embd.weight + blk.0..3
```

卸载：

```text
blk.4..N + output
```

这样可以在首请求前释放约 1.2 GiB 权重内存，并让首 token 延迟接近未卸载 baseline。

## CLI 参数

通用参数：

```text
--parallel-load / --no-parallel-load
--async-io-load / --no-async-io-load
--load-micro-stats / --no-load-micro-stats
--unload-all-after-load
--unload-after-load-fraction F
```

交互命令：

```text
/list_weight
/unload <tensor_name>
```

`/list_weight` 会列出权重 tensor 的 layer、大小和 resident 状态；`/unload` 会释放指定主权重 tensor 的 concrete backend buffer，下一次推理会自动恢复。

## C API

新增模型权重接口：

```cpp
llama_model_weight_count()
llama_model_weight_name_by_index()
llama_model_weight_info_by_index()
llama_model_unload_tensor()
llama_model_unload_all_tensors()
llama_model_unload_tensor_fraction()
llama_model_is_tensor_loaded()
llama_model_ensure_tensors_ready()
llama_model_ensure_global_tensors_ready()
llama_model_prefetch_unloaded_tensors()
llama_model_start_async_tensors_load()
llama_model_wait_async_tensors_load()
llama_model_weight_epoch()
llama_model_weight_last_load_metrics()
```

## 环境变量

Android Vulkan 推荐：

```text
GGML_VK_PREFER_HOST_MEMORY=1
LLAMA_VK_HOST_VISIBLE_DIRECT_SET=1
LLAMA_ASYNC_IO_QUEUES=2
LLAMA_WEIGHT_PREFIX_BARRIER=1
```

其他调试变量：

```text
LLAMA_WEIGHT_BARRIER_TARGET=8
LLAMA_WEIGHT_BARRIER_STRIDE=N
LLAMA_WEIGHT_WAIT_DEBUG=1
LLAMA_WEIGHT_LOAD_TRACE=1
LLAMA_WEIGHT_UNLOAD_KEEP_REGEX='(^token_embd\.weight$|^blk\.[0-3]\.)'
```

默认 Android IO queue 为 2，并 clamp 到 1..4。实测 queues=2 最稳定；queues=3/4 可能提高 read 吞吐，但不一定改善 TTFT。

## 行为与兼容性

默认模型加载行为保持原样：初始仍完整加载权重。只有显式调用 unload API、CLI 命令或使用 server/bench 卸载参数后，权重才进入 residency manager。

卸载只支持主权重 tensor：

- 主权重 tensor 通过 `ggml_tensor::weight_buffer` 记录 concrete backend buffer。
- view tensor 不能独立卸载，且 `weight_buffer == nullptr`。
- compute tensor 不持有 `weight_buffer`。
- 卸载后使用 placeholder buffer 保持 graph metadata 可用。

decode/encode 时：

1. 检查 `weight_epoch`，变化时同步 scheduler 并重建 reserve。
2. 先恢复 global 权重。
3. 在 reserve 前启动后台 prefetch。
4. 安装 prefix-only layer barrier。
5. scheduler eval boundary 不强制同步，仅用于切分 command submission。

当前默认只在首个 unloaded layer 前放一个 barrier。例如保留 `blk.0..3` 时，barrier 位于 `l_out-3`，output 合并进最后恢复 batch，不单独等待。

## switch-bench

新增工具：

```text
tools/switch-bench
```

常用参数：

```text
--runs N
--unload-layer-window A:B
--unload-global
--unload-output
--preload-eager-base
--drop-cache
--output-format md|csv|jsonl
--output PATH
```

Android 推荐测试命令：

```bash
LD_LIBRARY_PATH=. \
GGML_VK_PREFER_HOST_MEMORY=1 \
LLAMA_VK_HOST_VISIBLE_DIRECT_SET=1 \
LLAMA_ASYNC_IO_QUEUES=2 \
LLAMA_WEIGHT_PREFIX_BARRIER=1 \
./llama-switch-bench \
  -m /data/local/tmp/gguf/Qwen3-1.7B-Q8_0.gguf \
  -p "$PROMPT" \
  -c 4096 -ngl 99 \
  --runs 1 \
  --unload-layer-window 4:27 \
  --unload-output \
  --no-mmap \
  --parallel-load \
  --async-io-load \
  --load-micro-stats \
  --no-warmup \
  --drop-cache \
  --output-format csv \
  --output result.csv
```

每次独立执行前建议额外执行：

```bash
adb shell 'su -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
```

## 已验证结果

测试模型：

```text
Qwen3-1.7B-Q8_0.gguf
```

100-token prompt，Android Vulkan，保留 `token_embd + blk.0..3`，卸载 `blk.4..27 + output`：

| 场景 | TTFT |
| --- | ---: |
| 未卸载权重 | 3003.25 ms |
| 卸载后优化恢复 | 3232.20 ms |

卸载收益：

```text
265 tensors
1224.41 MiB
```

恢复统计：

```text
ready      1273.44 ms
read wall   826.38 ms
upload set   95.38 ms
upload sync   0.00 ms
reload groups 3
```

5 轮 formal P50：

```text
loaded P50 = 3214.81 ms
after  P50 = 3489.48 ms
ratio       = 1.085x
```

## 测试

本地测试：

```bash
cmake --build build --target llama-switch-bench llama-completion -j$(nproc)
ctest --test-dir build -R 'test-weight-buffer|test-weight-unload|test-parallel-load' --output-on-failure
```

Android Vulkan 构建：

```bash
cmake --build build-android-vulkan-arm64-v8a-api35 --target llama-switch-bench llama-completion -j$(nproc)
```

注意：连续压测会触发手机 thermal/cpufreq 限制。记录性能时应检查：

```bash
adb shell dumpsys thermalservice
```

如果 `Thermal Status` 非 0 或 cooling device 已限频，TTFT 绝对值会明显漂移。
