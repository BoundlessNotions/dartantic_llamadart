# dartantic_llamadart

Local model support for [Dartantic AI](https://pub.dev/packages/dartantic_ai)
through the [llamadart](https://pub.dev/packages/llamadart) engine: GGUF models
on llama.cpp, and `.litertlm` bundles on LiteRT-LM. Inference runs on-device,
with no network and no API key.

## Getting started

```yaml
dependencies:
  dartantic_ai: ^3.4.2
  dartantic_llamadart: ^0.7.0
```

Point the provider at a model file and hand it to an `Agent`:

```dart
import 'package:dartantic_ai/dartantic_ai.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';

final provider = LlamadartProvider(
  name: 'llamadart',
  displayName: 'Local Llama',
  modelPath: 'models/SmolLM2-135M-Instruct-Q4_K_M.gguf',
);

final agent = Agent.forProvider(
  provider,
  chatModelOptions: const LlamadartChatOptions(nCtx: 4096, temp: 0.7),
);

await for (final chunk in agent.sendStream(
  'Hello! Who are you?',
  history: [ChatMessage.system('You are a pirate. Answer in one sentence.')],
)) {
  stdout.write(chunk.output);
}

// Engines are shared process-wide; release native memory on shutdown.
await LlamaEngineCache.instance.disposeAll();
```

`example/dartantic_llamadart_example.dart` is this program, ready to run:
`dart run example/dartantic_llamadart_example.dart path/to/model.gguf`.

## Options

`LlamadartChatOptions` covers both the load-time shape of the engine and the
sampler:

| Load time | Per request |
| --- | --- |
| `nCtx`, `nGpuLayers`, `preferredBackend`, `liteRtLmBackend`, `chatTemplate`, `mtpDraftModelPath`, `mtpDraftTokenMax` | `temp`, `topK`, `topP`, `minP`, `repeatPenalty`, `maxTokens`, `speculativeDecoding`, `reusePromptPrefix`, `streamBatchTokenThreshold`, `streamBatchByteThreshold` |

Options passed to a single call are merged over the model's defaults, so
`LlamadartChatOptions(temp: 0.2)` keeps the default `topK` and the rest. A
load-time field set per call to something other than the model's default
throws `ArgumentError`: it would need a second engine to take effect.

Thinking is off unless asked for: `provider.createChatModel(enableThinking:
true)` (or `Agent.forProvider(..., enableThinking: true)`). Prior-turn
reasoning is sent back as thinking content, and each chat template decides
whether to render or strip it.

## Tools

Pass tools to the model and llamadart renders them with the model's own
tool-call template, then parses the calls back out:

```dart
final agent = Agent.forProvider(provider, tools: [myTool]);
```

Tool calls arrive as `ToolPart`s and dartantic executes them. When no tools
are passed, a text fallback still recognises tool-call envelopes a
prompt-instructed model writes into its reply (`<tool_call>{"name": ...,
"arguments": {...}}</tool_call>` and Gemma's
`<|tool_call>call:name{...}<tool_call|>`); an envelope that doesn't parse stays
text.

## Structured output

An `outputSchema` becomes a JSON-schema response format, which llama.cpp
enforces with a grammar during decoding:

```dart
final result = await agent.send(
  'Is the sky blue?',
  outputSchema: Schema.fromMap({
    'type': 'object',
    'properties': {'ok': {'type': 'boolean'}},
    'required': ['ok'],
  }),
);
```

llamadart throws for schema keywords it can't convert. LiteRT-LM has no
grammar constraints, so there the schema is dropped and output is best effort.

## Embeddings

Give the provider an embedding GGUF:

```dart
final provider = LlamadartProvider(
  name: 'llamadart',
  displayName: 'Local Llama',
  modelPath: 'models/chat.gguf',
  embeddingsModelPath: 'models/embeddinggemma-300m-Q4_0.gguf',
);

final embeddings = provider.createEmbeddingsModel(
  options: const EmbeddingsModelOptions(batchSize: 8),
);
final vectors = await embeddings.embedDocuments(['one', 'two']);
```

`batchSize` chunks `embedDocuments` and sets how many texts the engine embeds
as parallel sequences.

## Engines are shared

`LlamaEngineCache` keeps one engine per model path and set of load parameters,
so building a fresh `LlamadartChatModel` per request doesn't reload the model.
Generations on one engine are serialized: a second call waits rather than
cutting the first one off. Cancelling a stream (what `.timeout` does) stops
native generation and releases the engine. Call
`LlamaEngineCache.instance.disposeAll()` at shutdown; `dispose()` on a model
doesn't tear down an engine other models may be using.

There is no history trimming. A prompt that overflows the context fails with
`LlamaInferenceException`, and compacting the conversation is the caller's job,
as with hosted providers.

## Speculative decoding

On GGUF, point `mtpDraftModelPath` at a draft model to run llama.cpp's
`draft-mtp` speculative decoding, with `mtpDraftTokenMax` for the draft budget.
On LiteRT-LM, `speculativeDecoding: true` uses the MTP heads inside a
`.litertlm` bundle.

## License

BSD-3-Clause - See [LICENSE](LICENSE) for details.
