## 0.1.0

- Initial release: Local GGUF model support for Dartantic AI using llamadart
- LlamadartChatOptions with nCtx, nGpuLayers, temp, topK, topP, repeatPenalty
- LlamadartProvider with modelPath support
- LlamadartChatModel with streaming and tool call parsing
- XML-tag wrapper (`<tool_call>`) for structured tool calls
- LlamadartEmbeddingsModel stub implementation
