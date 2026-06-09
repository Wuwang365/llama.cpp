# Vulkan managed weights 4B Q4 keep-embedding TTFT 验证

本文记录 Android Pixel Vulkan 环境下，基于 Vulkan managed weights 保持 `llama_model` / `llama_context` 框架不变，但在请求前只保留 embedding resident、释放其它模型权重 resident 内存后的 TTFT 测试流程和结果。

## 测试目标

验证以下场景中 eager 与 non-eager 的耗时差异：

```text
模型: Qwen3-4B-Q4_K_M.gguf
上下文: -c 1024
prompt: The future of AI is
生成: -n 1
卸载策略: 保留 token_embd，其它 resident 权重全部 flush
计时口径: 从模型加载开始到首个 token decoding 完成
```

当前临时验证入口由环境变量打开：

```bash
GGML_VK_MANAGED_FLUSH_NON_PERMANENT=1
```

该模式在 `ggml_vk_managed_prepare_graph()` 前执行 flush。保留规则为 tensor 名称包含 `token_embd` 的 managed tensor，其它 resident managed tensor 释放。

## 测试环境

```text
设备: Pixel 8
ADB serial: 38051FDJH002BF
Vulkan device: Mali-G715
分支: elastic_memory_layer
模型路径: /data/local/tmp/gguf/Qwen3-4B-Q4_K_M.gguf
本次日志目录: /tmp/llama_4b_keepembd_c1024_20260608_231736
```

关键模型与 buffer 信息：

| 项目 | 数值 |
| --- | ---: |
| GPU offload | 37/37 layers |
| model buffer | 2375.91 MiB |
| KV buffer | 144.00 MiB |
| Vulkan compute buffer | 76.01 MiB |
| prompt tokens | 13 |

每轮测试前后 `dumpsys thermalservice` 均显示：

```text
Thermal Status: 0
```

## 构建与推送

```bash
cd /home/wuwang/workspace/llama_dev/llama.cpp
cmake --build build-android-vulkan-elastic-pixel --target llama-cli -j$(nproc)

adb -s 38051FDJH002BF shell 'mkdir -p /data/local/tmp/llama-vulkan-elastic'
adb -s 38051FDJH002BF push \
  build-android-vulkan-elastic-pixel/bin/llama-cli \
  build-android-vulkan-elastic-pixel/bin/*.so \
  /data/local/tmp/llama-vulkan-elastic/
```

## 测试流程

每次独立运行前先 drop cache：

```bash
adb -s 38051FDJH002BF shell 'su -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
```

non-eager 命令：

```bash
adb -s 38051FDJH002BF shell \
  "cd /data/local/tmp/llama-vulkan-elastic && \
   LD_LIBRARY_PATH=. \
   GGML_VK_MANAGED_STATS=1 \
   GGML_VK_MANAGED_FLUSH_NON_PERMANENT=1 \
   ./llama-cli \
     -m /data/local/tmp/gguf/Qwen3-4B-Q4_K_M.gguf \
     -p 'The future of AI is' \
     -n 1 -c 1024 -ngl 99 \
     --fit off \
     --single-turn \
     --no-display-prompt \
     --temp 0 --seed 42 \
     --perf --show-timings --no-warmup --verbose \
     --vk-managed"
```

eager 命令只额外增加：

```bash
--vk-managed-eager
```

本次采用交错顺序，每轮前 drop cache：

```text
non-eager 1
eager 1
non-eager 2
eager 2
non-eager 3
eager 3
```

## 时间口径

日志时间戳格式为 `M.SS.mmm.uuu`。本文使用以下字段：

| 字段 | 说明 |
| --- | --- |
| load-to-token | `load_model: loading model` 到首 token decoding 完成 |
| load-to-request | `load_model: loading model` 到请求开始 |
| request TTFT | 请求开始到首 token decoding 完成 |
| prompt eval time | llama.cpp 内部 prefill 统计，`-n 1` 时与 request TTFT 基本一致 |

首 token decoding 完成优先使用日志中的 `next token:` 时间戳。部分 eager 日志没有打印 `next token:` 调试行，此时使用紧随其后的 `prompt eval time` 行时间戳。已在 non-eager 日志中对比，`next token:` 与 `prompt eval time` 行只相差约 0.004 ms。

如果日志缺少 `new prompt` 行，则用：

```text
request_start = token_done - prompt_eval_ms
```

反推请求开始时间。

## Flush 行为确认

请求前真实 flush 行：

```text
managed flush non-permanent flushed=397 tensors 2071.63 MiB kept=1 tensors 304.28 MiB skipped=0
```

含义：

| 项目 | 数值 |
| --- | ---: |
| flushed tensors | 397 |
| flushed resident weights | 2071.63 MiB |
| kept tensors | 1 |
| kept resident weights | 304.28 MiB |

保留的 1 个 tensor 为 embedding 相关 managed tensor。初始化阶段第一次 prepare graph 可能出现 `flushed=0`，那次发生在权重尚未 resident 的阶段，不作为本测试的卸载效果判断。

## 测试结果

3 轮聚合结果：

| 模式 | load-to-token avg | load-to-token P50 | load-to-request avg | load-to-request P50 | request TTFT avg | request TTFT P50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| non-eager | 8802.36 ms | 8664.83 ms | 5761.58 ms | 5543.78 ms | 3040.78 ms | 3122.41 ms |
| eager | 12918.32 ms | 13030.66 ms | 7160.71 ms | 7094.34 ms | 5757.61 ms | 5735.00 ms |

逐轮数据：

| 模式 | run | load-to-token | load-to-request | request TTFT | prompt eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| non-eager | 1 | 9711.55 ms | 6198.54 ms | 3513.01 ms | 3513.01 ms |
| non-eager | 2 | 8664.83 ms | 5542.42 ms | 3122.41 ms | 3116.69 ms |
| non-eager | 3 | 8030.70 ms | 5543.78 ms | 2486.92 ms | 2483.73 ms |
| eager | 1 | 13030.66 ms | 7295.66 ms | 5735.00 ms | 5735.00 ms |
| eager | 2 | 13308.19 ms | 7094.34 ms | 6213.85 ms | 6213.85 ms |
| eager | 3 | 12416.11 ms | 7092.14 ms | 5323.97 ms | 5323.97 ms |

权重恢复统计为累计值，包含初始模型加载阶段与 flush 后首请求恢复阶段：

| 模式 | async upload | sync upload | upload_time avg | prefetch_thread |
| --- | ---: | ---: | ---: | ---: |
| non-eager | 4143.22 MiB | 304.31 MiB | 4949.21 ms | 1 |
| eager | 0.00 MiB | 4447.54 MiB | 4802.23 ms | 0 |

## 结论

在只保留 embedding、flush 其它 resident 权重，并把模型加载到首 token decoding 的时间都计入的口径下：

```text
load-to-token P50:
non-eager = 8664.83 ms
eager     = 13030.66 ms
non-eager 快 4365.83 ms

request TTFT P50:
non-eager = 3122.41 ms
eager     = 5735.00 ms
non-eager 快 2612.59 ms
```

主要原因是 non-eager 可以把 2071.63 MiB 非 embedding 权重恢复放到 prefetch 线程异步执行，最终累计同步上传量约 304.31 MiB。eager 没有 prefetch 线程，初始模型加载与 flush 后恢复累计 4447.54 MiB 都走同步路径，因此请求级 TTFT 和 load-to-token 都更慢。

本次原始日志与解析文件：

```text
/tmp/llama_4b_keepembd_c1024_20260608_231736/summary.csv
/tmp/llama_4b_keepembd_c1024_20260608_231736/aggregate.csv
/tmp/llama_4b_keepembd_c1024_20260608_231736/non_eager_*.log
/tmp/llama_4b_keepembd_c1024_20260608_231736/eager_*.log
```
