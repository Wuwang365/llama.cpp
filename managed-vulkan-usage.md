# Managed Vulkan Weights 使用说明

本文说明新增 managed Vulkan weights 功能的使用方法和可观察行为。

不展开内部技术实现。

## 快速开始

```bash
./build-vulkan/bin/llama-completion \
  -m /path/to/model.gguf \
  --device Vulkan0 -ngl 99 \
  --vk-managed --vk-managed-cache 8192 \
  --fit off --mmap --no-warmup \
  -c 128 -b 32 -ub 32 \
  -n 1 --no-display-prompt -no-cnv \
  -p "hello hello hello"
```

开启 `--vk-managed` 后：

- 请求开始时会启动权重预取。
- 计算会同时开始。
- 如果权重已经可用，计算直接继续。
- 如果计算追上尚未完成的权重加载，会等待对应权重。

## 模式

| 用法 | 行为 | 适合用途 |
| --- | --- | --- |
| `--vk-managed` | 默认 managed 模式，请求开始后流式预取权重，并与计算重叠 | 验证异步预取、评估 managed 权重 |
| `--vk-managed --vk-managed-eager` | eager 模式，首次执行时同步加载权重，加载完成后再计算 | correctness baseline、性能对照 |

## 常用参数

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--vk-managed-cache N` | 无固定默认建议 | 设置 managed weight cache 容量，单位 MiB |
| `GGML_VK_MANAGED_PREFETCH_INFLIGHT_MB` | `512` | 控制同时推进的预取规模 |
| `GGML_VK_MANAGED_PREFETCH_BATCH_MB` | `256` | 控制单次预取 batch 的规模 |
| `GGML_VK_MANAGED_PREFETCH_THREAD` | `1` | 控制是否使用后台线程推进预取；设为 `0` 可关闭 |
| `GGML_VK_MANAGED_STATS=1` | 关闭 | 打印 managed 权重统计 |
| `GGML_VK_MANAGED_DEBUG=1` | 关闭 | 打印更详细调试日志 |

当前 4B Q8_0 测试建议：

```bash
--vk-managed-cache 8192
GGML_VK_MANAGED_PREFETCH_INFLIGHT_MB=512
GGML_VK_MANAGED_PREFETCH_BATCH_MB=256
```

## Batch 参数行为

`GGML_VK_MANAGED_PREFETCH_BATCH_MB` 会影响预取节奏：

- 太小：提交次数变多，管理开销变大。
- 太大：早期计算可能等待过大的 batch 完成。
- 当前测试中，`256 MiB` 比 `128 MiB` 和 `512 MiB` 更稳定。

## 统计日志

开启：

```bash
GGML_VK_MANAGED_STATS=1
```

示例：

```text
managed stats ... hits=397 misses=2 prefetch_queued=396 prefetch_batches=28 prefetch_late=0 sync_fallback=2 ...
```

重点看这些字段：

| 字段 | 含义 |
| --- | --- |
| `hits` | 使用时权重已经可用 |
| `misses` | 使用时权重不在 cache 中 |
| `prefetch_queued` | 被预取覆盖的 tensor 数 |
| `prefetch_batches` | 预取 batch 数 |
| `prefetch_late` | 权重已在预取中，但使用时仍未完成 |
| `sync_fallback` | 预取没覆盖到，使用时同步加载 |
| `async` | 异步加载的数据量 |
| `sync` | 同步加载的数据量 |
| `resident` | 当前 cache 中的权重规模 |
| `prefetch_thread` | 本轮是否使用后台预取线程 |

一般判断：

- `prefetch_late` 越低，预取越及时。
- `sync_fallback` 越低，预取覆盖越完整。
- `prefetch_batches` 过高，batch 可能太碎。
- `prefetch_late` 过高，batch 可能太大或预取不够提前。

## 推荐测试

构造 32-token prompt：

```bash
PROMPT=$(printf 'hello %.0s' {1..32}); PROMPT=${PROMPT% }
```

确认 token 数：

```bash
./build-vulkan/bin/llama-tokenize \
  -m /path/to/model.gguf \
  -p "$PROMPT"
```

测试默认 managed 模式：

```bash
GGML_VK_MANAGED_STATS=1 \
./build-vulkan/bin/llama-completion \
  -m /path/to/model.gguf \
  --device Vulkan0 -ngl 99 \
  --vk-managed --vk-managed-cache 8192 \
  --fit off --mmap --no-warmup \
  -c 128 -b 32 -ub 32 \
  -n 1 --no-display-prompt -no-cnv \
  -p "$PROMPT" --verbose
```

测试 eager baseline：

```bash
GGML_VK_MANAGED_STATS=1 \
./build-vulkan/bin/llama-completion \
  -m /path/to/model.gguf \
  --device Vulkan0 -ngl 99 \
  --vk-managed --vk-managed-eager --vk-managed-cache 8192 \
  --fit off --mmap --no-warmup \
  -c 128 -b 32 -ub 32 \
  -n 1 --no-display-prompt -no-cnv \
  -p "$PROMPT" --verbose
```

关闭后台预取线程，对比主线程推进预取：

```bash
GGML_VK_MANAGED_STATS=1 \
GGML_VK_MANAGED_PREFETCH_THREAD=0 \
./build-vulkan/bin/llama-completion \
  -m /path/to/model.gguf \
  --device Vulkan0 -ngl 99 \
  --vk-managed --vk-managed-cache 8192 \
  --fit off --mmap --no-warmup \
  -c 128 -b 32 -ub 32 \
  -n 1 --no-display-prompt -no-cnv \
  -p "$PROMPT" --verbose
```

## 调试开关

```bash
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
```

行为：

- 用于验收更严格的 Vulkan 权重加载路径。
- 平时运行不一定需要设置。

## 注意事项

- 当前建议配合 `--mmap` 使用。
- 当前默认预取仍是 layer-level。
- 当前未启用 cache window。
- 首次运行的耗时通常包含权重加载影响，不应直接视为纯 prefill compute 时间。
- eager 适合作为对照，但不代表异步预取的目标行为。
