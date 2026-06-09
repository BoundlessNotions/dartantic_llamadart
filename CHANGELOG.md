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
