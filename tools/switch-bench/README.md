# llama.cpp/tools/switch-bench

`llama-switch-bench` measures first-token latency when the model context is reused.
It is intended for weight-switching experiments where `llama_context`, scheduler,
KV cache, and backend resources stay alive across rounds.

Example:

```sh
./llama-switch-bench \
  -m model.gguf -ngl 99 -npp 512 -r 7 \
  --unload-layers 4 \
  --async-io-load --no-mmap \
  --output-format md
```

Useful modes:

- Context-reuse baseline: omit `--unload-layers`, `--unload-global`, and `--unload-output`.
- Differential reload: use `--unload-layers N` to mark the first `N` layer groups changed before each round.
- Later-layer reload: add `--unload-layer-start N` to move the changed layer window deeper into the graph.
- Differential + async IO: add `--async-io-load --no-mmap`; on Vulkan this also exercises backend upload batching when available.
- Cold pagecache check: add `--drop-caches-before-switch` to run `sync; echo 3 > /proc/sys/vm/drop_caches` after unloading selected groups and before switch timing. On Android this uses `su` if direct writes are not permitted.
- Eager base behavior: add `--preload-before-switch` to synchronously ready all selected changed weight groups before decode; the preload is included in `switch_to_first_token_ms`.
- Layer prefetch: `LLAMA_WEIGHT_PREFETCH_AHEAD=N` controls how many future layer groups are prefetched in the background while the current layer computes. The default is `0`.
- Ordered small-diff prefetch: `LLAMA_WEIGHT_PREFETCH_MAX_GROUPS=N` enables decode-start prefetch for at most `N` unloaded groups. The default is `16`; set it to `0` to disable. `LLAMA_WEIGHT_PREFETCH_MIN_ORDERED_LAYER=N` defaults to `4` to avoid prefetching layer-0 changes that have no useful overlap window.
- Upload prefetch: `LLAMA_WEIGHT_PREFETCH_UPLOAD=1` makes ordered prefetch load and upload the selected groups as one combined batch so they can become ready before decode reaches those layers. This is the default. Set it to `0` only for host-staging experiments.

Important columns:

- `context_init_ms`: one-time context creation cost, excluded from per-round switch metrics.
- `weight_prepare_ms`: time spent marking selected weight groups unloaded before the round.
- `weight_io_ms`: wall time spent reading weight bytes while preparing groups.
- `weight_upload_ms`: backend tensor upload submission and sync time.
- `weight_ready_wall_ms`: total wall time spent making weight groups ready, including background prefetch work.
- `weight_wait_during_decode_ms`: wall time spent in weight group readiness checks during decode.
- `switch_to_first_token_ms`: main metric, measured from the start of prompt decode after weight preparation until first-token logits are synchronized.

Markdown output reports median and p90 for `switch_to_first_token_ms`, discarding the first sample by default.
Use `--discard-first N` to change that behavior, or `--output-format csv|jsonl` for post-processing.
