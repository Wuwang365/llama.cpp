# Vulkan managed weight 控制接口

本文记录 Vulkan managed weights 模式下，运行时查询和释放模型权重 resident 内存的第一版控制接口。

## 目标

在保持 `llama_model` / `llama_context` 存活的前提下，允许外部调试或服务控制逻辑按 weight entry 名称释放对应的 Vulkan resident 内存。

第一版并发语义：

- 当前没有正在处理的 slot/request 时，允许 unload。
- 当前有生成请求进行中，或者 task queue 中有 deferred request 时，server 返回 busy/HTTP 409。
- CLI 交互命令只在用户输入阶段执行，此时不处于生成中的 request 内。

释放粒度是 Vulkan managed weight entry。释放后不删除模型 metadata、GGUF mmap backing 或 tensor 名称；后续推理如果再次访问该权重，由 managed weights 机制重新加载。

## C API

新增接口位于 `include/llama.h`：

```c
int32_t llama_model_weight_count(const struct llama_model * model);
int32_t llama_model_weight_name(const struct llama_model * model, int32_t index, char * buf, size_t buf_size);
enum llama_weight_unload_result llama_model_unload_weight(struct llama_model * model, const char * name);
```

`llama_model_unload_weight()` 返回：

| 返回值 | 含义 |
| --- | --- |
| `LLAMA_WEIGHT_UNLOAD_SUCCESS` | 已释放 resident 内存 |
| `LLAMA_WEIGHT_UNLOAD_NOT_FOUND` | 找不到该权重名 |
| `LLAMA_WEIGHT_UNLOAD_NOT_MANAGED` | 该 tensor 不属于 Vulkan managed buffer，或当前 backend 不支持 |
| `LLAMA_WEIGHT_UNLOAD_NOT_RESIDENT` | 权重已经不在 resident 状态 |
| `LLAMA_WEIGHT_UNLOAD_BUSY` | 权重当前正在使用、准备或上传 |
| `LLAMA_WEIGHT_UNLOAD_ERROR` | 其它释放失败 |

## llama-server 使用方式

列出当前模型全部权重名：

```bash
curl -s http://127.0.0.1:8080/list_weights
```

返回格式：

```json
{
  "count": 399,
  "weights": [
    "token_embd.weight",
    "blk.0.attn_q.weight"
  ]
}
```

释放单个权重：

```bash
curl -s -X POST http://127.0.0.1:8080/unload_weight \
  -H 'Content-Type: application/json' \
  -d '{"name":"blk.0.attn_q.weight"}' | jq
```

`/unload_weight` 也接受以下输入形式，便于 shell 调试：

```bash
curl -s -X POST 'http://127.0.0.1:8080/unload_weight?name=blk.0.attn_q.weight'
curl -s -X POST http://127.0.0.1:8080/unload_weight -d '"blk.0.attn_q.weight"'
curl -s -X POST http://127.0.0.1:8080/unload_weight -d 'blk.0.attn_q.weight'
```

成功或已经不 resident 时返回 HTTP 200：

```json
{
  "ok": true,
  "name": "blk.0.attn_q.weight",
  "status": "unloaded"
}
```

如果 server 正在处理请求，返回 HTTP 409：

```json
{
  "error": {
    "code": 409,
    "message": "server is busy, retry when no slot/request is processing",
    "type": "unavailable_error"
  }
}
```

## llama-cli 使用方式

进入交互模式后可直接输入：

```text
/list_weights
/unload_weight blk.0.attn_q.weight
```

`/list_weights` 一行输出一个权重名，最后输出总数。`/unload_weight` 会打印 `unloaded`、`not resident`、`not found`、`not managed`、`busy` 或 `error` 对应状态。

## 兼容性与限制

- CPU backend 或未启用 Vulkan managed weights 时，释放接口会返回 `NOT_MANAGED`。
- server 端的并发保护粒度是整个 request/slot idle 窗口，不是 op 粒度。
- 第一版不在 HTTP handler 内持有 Vulkan managed context 级互斥锁，依赖 server idle 检查和 backend entry 状态检查共同保护。
- `NOT_RESIDENT` 在 server 返回中按幂等成功处理，便于重复执行卸载脚本。
- router server 会把 `/list_weights` 和 `/unload_weight` 代理给具体模型实例。

## 验证命令

host 构建与回归测试：

```bash
cd /home/wuwang/workspace/llama_dev/llama.cpp
cmake --build build --target test-weight-control llama-cli llama-server -j$(nproc)
ctest --test-dir build -R test-weight-control --output-on-failure
```

Android Vulkan 构建：

```bash
cd /home/wuwang/workspace/llama_dev/llama.cpp
cmake --build build-android-vulkan-elastic-pixel --target llama-cli llama-server -j$(nproc)
```

Pixel server 冒烟命令：

```bash
adb -s 38051FDJH002BF shell 'mkdir -p /data/local/tmp/llama-vulkan-elastic'
adb -s 38051FDJH002BF push \
  build-android-vulkan-elastic-pixel/bin/llama-cli \
  build-android-vulkan-elastic-pixel/bin/llama-server \
  build-android-vulkan-elastic-pixel/bin/*.so \
  /data/local/tmp/llama-vulkan-elastic/

adb -s 38051FDJH002BF forward tcp:8080 tcp:8080
adb -s 38051FDJH002BF shell \
  "cd /data/local/tmp/llama-vulkan-elastic && \
   LD_LIBRARY_PATH=. GGML_VK_MANAGED_STATS=1 \
   ./llama-server \
     -m /data/local/tmp/gguf/Qwen3-4B-Q4_K_M.gguf \
     -c 1024 -ngl 99 --host 127.0.0.1 --port 8080 --vk-managed"
```

另开终端执行：

```bash
curl -s http://127.0.0.1:8080/list_weights
curl -s -X POST http://127.0.0.1:8080/unload_weight \
  -H 'Content-Type: application/json' \
  -d '{"name":"blk.0.attn_q.weight"}'
```

如果本机安装了 `jq`，可以追加 `| jq` 或 `| jq '.weights[:5]'` 方便查看。

## 本次验证结果

验证日期：2026-06-09。

host 侧：

- `cmake --build build --target test-weight-control llama-cli llama-server -j$(nproc)` 通过。
- `ctest --test-dir build -R test-weight-control --output-on-failure` 通过，`1/1` passed。

Android Pixel 侧：

- `cmake --build build-android-vulkan-elastic-pixel --target llama-cli llama-server -j$(nproc)` 通过。
- 推送 `llama-cli`、`llama-server` 和 `bin/*.so` 到 `/data/local/tmp/llama-vulkan-elastic/` 通过。
- 使用 `/data/local/tmp/gguf/Qwen3-4B-Q4_K_M.gguf`、`-c 1024 -ngl 99 --vk-managed` 启动 server，日志显示 Vulkan device 为 `Mali-G715`。
- `GET /list_weights` 返回 `count=398`，前几个权重为 `token_embd.weight`、`output_norm.weight`、`blk.0.attn_norm.weight`、`blk.0.attn_q.weight`。
- `POST /unload_weight {"name":"blk.0.attn_q.weight"}` 返回 `status="not_resident"`。
- `POST /unload_weight {"name":"token_embd.weight"}` 返回 `status="unloaded"`。
- 在一个 `n_predict=512` 的 `/completion` 请求处理中调用 `/unload_weight`，返回 HTTP 409，body 中 `error.code=409`。
