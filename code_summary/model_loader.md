# llama-model-loader.cpp：UMA + Vulkan 权重加载内存管理总结

## 结论先行
- `llama-model-loader.cpp` **没有 Vulkan 专用分支**，采用的是 `ggml` 后端抽象下的通用策略：按 tensor 选择 buffer type，然后按 `mmap/非 mmap` 和 `host/device` 能力决定加载路径。
- 在统一内存架构（UMA）下使用 Vulkan 时，这个文件里可见的策略仍是：
1. 优先把权重放到目标后端声明可用的默认设备 buffer；
2. 若目标 buffer 被判定为 host buffer，则直接文件读入；
3. 若是 device buffer 且支持 async + host_buffer + events，则通过 host staging buffer 异步上传；
4. `mmap` 模式下会尽量避免使用后端 host buffer（会回退到 CPU buffer），并走映射地址+后端分配/拷贝路径。

## 关键流程（按代码路径）

### 1) 权重 buffer 类型选择（与 Vulkan 解耦）
- `weight_buft_supported()` + `select_weight_buft()`：通过 `ggml_backend_dev_supports_op()` 测试某个 tensor/op 是否支持指定 `buft`。
- `create_tensor()` 中 `buft_for_tensor()`：按输入层/输出层/重复层的 `buft_list` 选最终 buffer type。
- 若 `use_mmap == true` 且选中了某设备的 `host buffer type`，会改为 CPU 默认 buffer（“avoid using a host buffer when using mmap”）。

这意味着：是否“UMA 直接共享”不由这里硬编码决定，而由后端上报的 `buffer type` 能力决定。

### 2) mmap 模式的权重加载
- `load_all_data()` 中 `use_mmap` 分支：
1. 从映射文件拿到 `data = mapping->addr() + offset`；
2. 若有后端 buffer 且 `cur->data == nullptr`，调用 `ggml_backend_tensor_alloc(buf_mmap, cur, data)`；
3. 否则调用 `ggml_backend_tensor_set(cur, data, 0, n_size)`。

这里可看作“映射地址驱动加载”，是否零拷贝/别名依赖后端 `tensor_alloc` 实现；本文件不做 Vulkan 专门判断。

### 3) 非 mmap 模式：host 直读 vs staging 异步上传
- 若 `ggml_backend_buffer_is_host(cur->buffer)` 为真：直接 `read_raw(cur->data, n_size)`。
- 否则尝试创建 `upload_backend`（要求设备能力同时具备 `async + host_buffer + events`）：
1. 从 `ggml_backend_dev_host_buffer_type(dev)` 分配 4 个 host staging buffer；
2. 分块从文件读到 staging；
3. `ggml_backend_tensor_set_async()` 上传到目标 tensor；
4. 用 event 同步复用缓冲区。
- 若 async 条件不满足：走同步 `read_raw -> ggml_backend_tensor_set()`。

## 对“UMA + Vulkan”的具体含义（在本文件可确认范围内）
- 本文件并不直接判断 UMA，也不检查 Vulkan 内存类型（如 HOST_VISIBLE/DEVICE_LOCAL）；它只看抽象能力和 buffer 分类。
- 因此在 UMA 机器上，若 Vulkan 后端把目标权重 buffer 视作 host buffer，则会走“直接读入 host buffer”路径；若仍视作 device buffer，则走“host staging + 上传”路径（若支持 async）或同步上传。
- 是否真正零拷贝、是否使用统一物理内存的同一分配、以及 Vulkan memory type 选择逻辑，需要到 Vulkan 后端实现文件中确认（不在 `llama-model-loader.cpp` 内）。

## 可定位的关键代码段
- `weight_buft_supported` / `select_weight_buft`：约 `L892-L1041`
- `create_tensor` 中 `buft_for_tensor` 与 `use_mmap` + host buffer 规避：约 `L1078-L1200`
- `load_all_data` 中 async upload 初始化：约 `L1416-L1513`
- `load_all_data` 中 `mmap` 与 `non-mmap` 两条主路径：约 `L1530-L1640`
