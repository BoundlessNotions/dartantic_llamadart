import 'dart:io';

import 'package:dartantic_ai/dartantic_ai.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';

/// Streams a reply from a local GGUF model through a dartantic [Agent].
///
/// Run with `dart run example/dartantic_llamadart_example.dart path/to.gguf`.
Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    stderr.writeln('Usage: dartantic_llamadart_example <model.gguf>');
    exit(64);
  }

  final provider = LlamadartProvider(
    name: 'llamadart',
    displayName: 'Local Llama',
    modelPath: args.first,
  );
  final agent = Agent.forProvider(
    provider,
    chatModelOptions: const LlamadartChatOptions(nCtx: 4096, temp: 0.7),
  );

  try {
    await for (final chunk in agent.sendStream(
      'Hello! Who are you?',
      history: [
        ChatMessage.system('You are a pirate. Answer in one sentence.'),
      ],
    )) {
      stdout.write(chunk.output);
    }
    stdout.writeln();
  } finally {
    // Engines are shared process-wide; release native memory on shutdown.
    await LlamaEngineCache.instance.disposeAll();
  }
}
