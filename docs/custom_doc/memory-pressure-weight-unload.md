# 基于内存压力的分时分量权重释放策略

本文说明 llama.cpp 中的 `memory pressure weight unload policy`。该特性默认关闭，用于 Android/Linux 上根据 `/proc/pressure/memory` 的 PSI 指标，在安全点分批释放可卸载的模型权重，从而降低 LLM 常驻权重对其他应用保活的影响。释放后的权重仍复用现有异步恢复路径，下一次推理会在 prefill 前后自动恢复。

## 适用场景

- Android Vulkan，推荐搭配 `--no-mmap --parallel-load --async-io-load --load-micro-stats`。
- 需要让 LLM 与前台或近期后台应用共存，观察 LMKD/OOM 风险。
- 需要先 dry-run 校准 PSI 阈值，再启用真实权重释放。

该策略不在算子执行中途释放正在被 graph 使用的权重；server/CLI 集成只在主循环安全点执行真实 unload。PSI 文件不可用或格式异常时，策略会记录错误并退化为不释放，不影响推理主流程。

## CLI 参数

默认值：

```text
--no-memory-pressure-unload
--memory-pressure-path /proc/pressure/memory
--memory-pressure-interval-ms 500
--memory-pressure-cooldown-ms 1000
--memory-pressure-step-mib 128
--memory-pressure-max-fraction 0.80
--memory-pressure-some-avg10-thold 1.0
--memory-pressure-some-avg60-thold 2.0
--memory-pressure-full-avg10-thold 0.0
--memory-pressure-policy pressure-only
--memory-pressure-protected-app-active
--no-memory-pressure-dry-run
```

可用参数：

```text
--memory-pressure-unload / --no-memory-pressure-unload
--memory-pressure-path PATH
--memory-pressure-interval-ms N
--memory-pressure-cooldown-ms N
--memory-pressure-step-mib N
--memory-pressure-max-fraction F
--memory-pressure-some-avg10-thold F
--memory-pressure-some-avg60-thold F
--memory-pressure-full-avg10-thold F
--memory-pressure-policy pressure-only|external-hint
--memory-pressure-protected-app-active / --no-memory-pressure-protected-app-active
--memory-pressure-dry-run / --no-memory-pressure-dry-run
--memory-pressure-log PATH
--memory-pressure-keep-regex REGEX
--drop-weight-file-cache-after-upload / --no-drop-weight-file-cache-after-upload
```

`pressure-only` 只根据 PSI 决策。`external-hint` 需要上层或测试脚本通过 `--memory-pressure-protected-app-active` 表示有应用需要保护；当 hint 为 inactive 时，策略仅观察和记录，不释放权重。

默认 keep regex 为：

```text
(^token_embd\.weight$|^blk\.[0-3]\.)
```

如果设置了环境变量 `LLAMA_WEIGHT_UNLOAD_KEEP_REGEX`，策略层会与 CLI keep regex 合并；底层 unload API 也会继续尊重该环境变量。

如果 Android 普通 uid 不能读取 `/proc/pressure/memory`，可用任务目录中的 root helper 镜像 PSI：

```bash
REMOTE_PATH=/data/local/tmp/pressure_memory_mirror \
/home/wuwang/workspace/llama_dev/task/分时分量释放内存/scripts/mirror_pressure_root.sh
```

然后让 llama 使用：

```text
--memory-pressure-path /data/local/tmp/pressure_memory_mirror
```

## 释放策略

策略事件按以下顺序处理：

1. 读取并解析 PSI 的 `some` / `full` 行。
2. 统计当前 resident/unloaded 权重字节数。
3. 判断压力等级：
   - `low`：低于阈值，仅观察。
   - `medium`：`some.avg10` 达阈值或 `some.total` 窗口增量明显。
   - `high`：`some.avg10` 明显升高或 `some.avg60` 达阈值。
   - `critical`：`full.avg10` 高于阈值。
4. 检查 cooldown 和最大释放比例。
5. 在安全点按预算选择 tensor：优先 `output`，再从高层 `blk.N` 向低层释放，并跳过 keep regex、已卸载或不可见的 tensor。
6. 调用 `llama_model_unload_tensor()`，下一次推理通过现有 restore/prefetch 路径恢复。

## JSONL 日志

使用 `--memory-pressure-log policy_events.jsonl` 后，每行是一个 JSON 事件：

```json
{"ts_ms":123,"event":"psi_sample","some_avg10":2.3,"some_avg60":0.8,"full_avg10":0.0,"resident_mib":820.5,"unloaded_mib":128.0,"mem_available_kib":512000}
{"ts_ms":124,"event":"unload_decision","action":"unload","state":"relieving","level":"medium","target_bytes":134217728,"reason":"some_avg10","busy":false,"dry_run":false}
{"ts_ms":125,"event":"tensor_unloaded","name":"blk.27.ffn_down.weight","layer":27,"bytes":58720256,"dry_run":false}
{"ts_ms":126,"event":"unload_summary","bytes_freed":146800640,"tensors":4,"resident_mib":820.5,"unloaded_mib":268.0,"dry_run":false}
```

`dry-run` 会记录相同决策和 tensor 选择，但不会调用实际 unload。

## Android Vulkan 示例

推荐环境：

```bash
export GGML_VK_PREFER_HOST_MEMORY=1
export LLAMA_VK_HOST_VISIBLE_DIRECT_SET=1
export LLAMA_ASYNC_IO_QUEUES=2
export LLAMA_WEIGHT_PREFIX_BARRIER=1
export LLAMA_WEIGHT_UNLOAD_KEEP_REGEX='(^token_embd\.weight$|^blk\.[0-3]\.)'
```

server 示例：

```bash
./llama-server \
  -m /data/local/tmp/gguf/Qwen3-1.7B-Q8_0.gguf \
  --host 127.0.0.1 --port 8080 \
  --no-mmap --parallel-load --async-io-load --load-micro-stats \
  --memory-pressure-unload \
  --memory-pressure-policy pressure-only \
  --memory-pressure-log /data/local/tmp/llama-policy-events.jsonl
```

先校准阈值时建议加：

```bash
--memory-pressure-dry-run
```

## 文件页 cache 提示

`--drop-weight-file-cache-after-upload` 会在非 direct-io 的权重恢复读取后，best-effort 调用 `posix_fadvise(..., POSIX_FADV_DONTNEED)`。这只提示内核回收模型文件 page cache，不等同于释放 Vulkan backend 权重 buffer，也不保证立即回收。报告中应单独记录该开关。

## 建议验证

本地回归：

```bash
cmake --build build --target test-memory-pressure test-arg-parser llama-cli llama-server -j$(nproc)
ctest --test-dir build -R 'test-memory-pressure|test-arg-parser|test-weight-unload|test-parallel-load' --output-on-failure
```

Android 环境检查：

```bash
scripts/build_vulkan_pixel_verify.sh --check-only
```

`scripts/build_vulkan_pixel_verify.sh` 会按顺序使用 `ANDROID_NDK_HOME`、`ANDROID_NDK_ROOT`、`ANDROID_NDK`，也会从 `ANDROID_HOME/ndk`、`ANDROID_SDK_ROOT/ndk` 和 `~/Android/Sdk/ndk` 自动发现 side-by-side NDK。手机验证脚本会推送 `llama-server` 和相关 `.so` 到 `/data/local/tmp/llama-vulkan-verify`。

任务目录的 `run_llama_server.sh` 默认使用 `-ngl 99 -c 4096 --parallel 1`，避免 Android server 直接采用模型训练上下文长度导致设备内存压力过大。需要放大压力时可用 `CTX_SIZE` 和 `N_PARALLEL` 覆盖。

任务目录 `task/分时分量释放内存/scripts/` 提供了设备信息采集、server 启动、保活 case 执行和 JSONL/CSV 汇总脚本。
