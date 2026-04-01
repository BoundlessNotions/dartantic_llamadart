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
