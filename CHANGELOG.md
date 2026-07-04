## 0.6.8

- Cache loaded engines process-wide (`LlamaEngineCache`), keyed by model path
  plus load-time params. Previously every `LlamadartChatModel` owned its own
  `LlamaEngine`, so fresh-model-per-call orchestrators paid a full model load
  (weights + graph compile + context allocation) on every request and leaked
  the previous engine's native handles. Sessions remain per-model.
- `sendStream` calls `cancelGeneration()` before generating: a caller that
  timed out on a previous call cannot cancel the native generation via
  `.timeout`, and with a shared engine the zombie must be interrupted instead
  of raced. Safe when idle (the cancel token is per-generation).
- `LlamadartChatModel.dispose()` no longer disposes the (shared) engine; call
  `LlamaEngineCache.instance.disposeAll()` at app shutdown. Corruption
  recovery (`_resetEngine`) evicts from the cache so the next call reloads.

## 0.6.7

- Reserve llama.cpp MTP rollback snapshots at model load. GGUF draft-mtp
  generation failed with "MTP speculative decoding is not available for this
  model/context" because the context reserved no `n_rs_seq` rollback snapshots.
  `_ensureInitialized` now sets `ModelParams.speculativeRollbackTokenMax` to the
  draft token max when a GGUF drafter is configured, and the generation request
  passes a matching `draftTokenMax`. New `LlamadartChatOptions.mtpDraftTokenMax`
  (default 1) controls both.

## 0.6.6

- Route GGUF/llama.cpp multi-token-prediction speculative decoding. Added
  `LlamadartChatOptions.mtpDraftModelPath`; when set on a GGUF model,
  `buildGenerationParams` emits `SpeculativeDecodingConfig.mtp(draftModelPath:)`
  so llama.cpp runs its `draft-mtp` path with a separate drafter. The LiteRT-LM
  backend continues to use the `speculativeDecoding` flag (its MTP heads ship in
  the `.litertlm` bundle) and ignores the GGUF draft path.
- Requires `llamadart >=0.8.4` for the `SpeculativeDecodingConfig` API.

## 0.6.2

- Rebuild the engine after a failed generation instead of reusing it. The
  native LiteRT-LM runtime can leave the engine in a corrupted state when a
  generation fails internally (e.g. `send_message` returns null on the
  function-calling path); reusing it then crashes the process with a SIGSEGV
  (use-after-free in `Conversation::SendMessage`'s async cleanup). Tearing the
  engine down on error converts that fatal native crash into a recoverable
  per-call error.

## 0.6.1

- Fixed `UnsupportedError` from the LiteRT-LM backend by not sending the
  llama.cpp-only generation knobs (`minP`, `penalty`) for `.litertlm` models.
  These now stay at `GenerationParams` defaults for LiteRT-LM (which rejects
  non-default values) while GGUF/llama.cpp keeps the full sampler controls.
  Extracted the logic into `buildGenerationParams` (`@visibleForTesting`).

## 0.6.0

- Added `speculativeDecoding` (`bool?`) to `LlamadartChatOptions` and forwarded it
  to `GenerationParams`, enabling LiteRT-LM multi-token-prediction (MTP) /
  speculative decoding. Honored only by the native LiteRT-LM backend with a
  `.litertlm` bundle that ships MTP draft heads (e.g. the post-MTP Gemma 4 E2B
  revision); a no-op on GGUF/llama.cpp, WebGPU, and LiteRT-LM web. Default off.

## 0.5.0

- Upgraded `llamadart` dependency to `^0.7.0` for LiteRT-LM support
- Added `liteRtLmBackend` (`LiteRtLmBackendPreference`) to `LlamadartChatOptions` for
  selecting CPU/GPU/NPU on `.litertlm` models (default: `auto`)
- Added `chatTemplate` (`String?`) to `LlamadartChatOptions` to override detected
  chat template for models without embedded metadata
- Both new fields are forwarded to `ModelParams` when loading a model

## 0.4.0

- Added tool support with automatic format detection based on model GGUF metadata
- Added `preferredBackend` and `minP` options to `LlamadartChatOptions`
- Changed default context size to 8192 (was 2048)
- Changed default GPU layers to maxGpuLayers (was 0)
- Added tool schema examples extraction and injection into tool descriptions
- Added multi-format tool call pattern matching:
  - Gemma 4: `<|tool_call>call:name{...}<|tool_call|>`
  - FunctionGemma: `<start_function_call>...<end_function_call>`
  - Hermes/DeepSeek: `<tool_call>...</tool_call>`
- Added automatic value casting (bool/int/double) for tool arguments

## 0.3.0

- Upgraded to `dartantic_interface 4.0.0`
- Compatibility with `dartantic_ai 3.3.0`
- Added automatic FunctionGemma activation trigger prepending for tool use
- Added explicit unit tests for message conversion logic

## 0.2.0

- Added support for `<tool_call>` parsing in LlamadartChatModel
- Improved token buffering and tool part generation

## 0.1.0

- Initial release: Local GGUF model support for Dartantic AI using llamadart
- LlamadartChatOptions with nCtx, nGpuLayers, temp, topK, topP, repeatPenalty
- LlamadartProvider with modelPath support
- LlamadartChatModel with streaming and tool call parsing
- XML-tag wrapper (`<tool_call>`) for structured tool calls
- LlamadartEmbeddingsModel stub implementation
