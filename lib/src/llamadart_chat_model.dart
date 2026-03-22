import 'dart:async';
import 'package:dartantic_interface/dartantic_interface.dart';
import 'package:llamadart/llamadart.dart';
import 'llamadart_chat_options.dart';
import 'llamadart_provider.dart';

/// A chat model implementation using the Llamadart engine.
class LlamadartChatModel extends ChatModel<LlamadartChatOptions> {
  @override
  final LlamadartProvider provider;

  @override
  final String modelName;

  @override
  final LlamadartChatOptions defaultOptions;

  LlamaModel? _model;
  LlamaContext? _context;

  LlamadartChatModel({
    required this.provider,
    required this.modelName,
    required this.defaultOptions,
  });

  Future<void> _ensureInitialized() async {
    if (_model != null) return;
    _model = LlamaModel(modelPath: provider.modelPath);
    _context = _model!.createContext();
  }

  @override
  Stream<ChatResult<ChatMessage>> stream(
    Iterable<ChatMessage> messages, {
    LlamadartChatOptions? options,
    Iterable<Tool>? tools,
  }) async* {
    await _ensureInitialized();
    final effectiveOptions = options ?? defaultOptions;

    // TODO: Implement more sophisticated prompt engineering for tools.
    // For now, we append a system instruction if tools are present.
    final prompt = _buildPrompt(messages, tools);

    final session = ChatSession(
      context: _context!,
      // grammar: _buildGrammar(tools), // TODO: Implement GBNF grammar
    );

    String fullContent = '';
    await for (final token in session.chat(prompt)) {
      fullContent += token;
      yield ChatResult(
        message: ChatMessage(
          role: ChatRole.assistant,
          parts: [TextPart(token)],
        ),
      );
    }

    // TODO: Parse tool calls from fullContent if they were generated.
  }

  @override
  Future<ChatResult<ChatMessage>> generate(
    Iterable<ChatMessage> messages, {
    LlamadartChatOptions? options,
    Iterable<Tool>? tools,
  }) async {
    final resultStream = stream(messages, options: options, tools: tools);
    ChatMessage? lastMessage;
    String fullContent = '';

    await for (final result in resultStream) {
      lastMessage = result.message;
      for (final part in result.message.parts) {
        if (part is TextPart) {
          fullContent += part.text;
        }
      }
    }

    return ChatResult(
      message: ChatMessage(
        role: ChatRole.assistant,
        parts: [TextPart(fullContent)],
      ),
    );
  }

  String _buildPrompt(Iterable<ChatMessage> messages, Iterable<Tool>? tools) {
    final buffer = StringBuffer();

    if (tools != null && tools.isNotEmpty) {
      buffer.writeln('SYSTEM: You have access to the following tools:');
      for (final tool in tools) {
        buffer.writeln('- ${tool.name}: ${tool.description}');
        buffer.writeln('  Arguments: ${tool.inputSchema}');
      }
      buffer.writeln(
          'To call a tool, use the format: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>');
    }

    for (final message in messages) {
      buffer.writeln('${message.role.name.toUpperCase()}: ${_messageContent(message)}');
    }
    buffer.writeln('ASSISTANT:');

    return buffer.toString();
  }

  String _messageContent(ChatMessage message) {
    return message.parts.whereType<TextPart>().map((e) => e.text).join('\n');
  }

  @override
  void dispose() {
    _context?.dispose();
    _model?.dispose();
  }
}
