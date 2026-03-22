import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:dartantic_llamadart/dartantic_llamadart.dart';

void main() async {
  // 1. Setup the Llamadart provider with a path to a GGUF model.
  final provider = LlamadartProvider(
    name: 'llamadart',
    displayName: 'Local Llama',
    modelPath: 'models/llama3.2-1b.gguf', // Replace with your actual model path
  );

  // 2. Create a chat model.
  final chatModel = provider.createChatModel(
    name: 'llama3.2-1b',
    options: const LlamadartChatOptions(
      temp: 0.7,
      nCtx: 2048,
    ),
  );

  // 3. Define messages.
  final messages = [
    const ChatMessage(role: ChatMessageRole.system, parts: [TextPart('You are a helpful assistant.')]),
    const ChatMessage(role: ChatMessageRole.user, parts: [TextPart('Hello! Who are you?')]),
  ];

  // 4. Generate a response.
  print('Generating response (offline)...');
  // In Dartantic v1.2.0, generate might be available on ChatModel extension or as a method.
  // If it's missing, we use sendStream().last.
  final result = await chatModel.sendStream(messages).last;

  print('Response:');
  for (final part in result.output.parts) {
    if (part is TextPart) {
      print(part.text);
    }
  }

  // 5. Cleanup
  chatModel.dispose();
}
