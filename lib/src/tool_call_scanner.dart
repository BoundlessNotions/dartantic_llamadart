/// The opening and closing tags around a tool call a model writes into plain
/// content.
class ToolCallEnvelope {
  const ToolCallEnvelope(this.open, this.close);

  final String open;
  final String close;
}

/// A piece of streamed content: plain text, or the body of a complete
/// tool-call envelope.
sealed class ScanSegment {}

class TextSegment extends ScanSegment {
  TextSegment(this.text);

  final String text;
}

class EnvelopeSegment extends ScanSegment {
  EnvelopeSegment(this.envelope, this.body);

  final ToolCallEnvelope envelope;
  final String body;
}

/// Splits streamed content into text and tool-call envelopes.
///
/// Every character passed to [add] comes back exactly once, in order, either
/// as text or inside an envelope. Only text that could still become an
/// envelope is held back: an opening tag waiting for its close, or a tail
/// that is a proper prefix of an opening tag (e.g. a trailing `<tool_`).
class ToolCallScanner {
  ToolCallScanner(this.envelopes);

  final List<ToolCallEnvelope> envelopes;
  String _pending = '';

  List<ScanSegment> add(String content) {
    if (envelopes.isEmpty) return [TextSegment(content)];

    _pending += content;
    final segments = <ScanSegment>[];
    while (true) {
      final (start, envelope) = _earliestOpen();
      if (envelope == null) {
        final held = _heldPrefixLength();
        _emitText(segments, _pending.length - held);
        break;
      }
      _emitText(segments, start);
      final closeAt = _pending.indexOf(envelope.close, envelope.open.length);
      if (closeAt < 0) break;
      segments.add(
        EnvelopeSegment(
          envelope,
          _pending.substring(envelope.open.length, closeAt),
        ),
      );
      _pending = _pending.substring(closeAt + envelope.close.length);
    }
    return segments;
  }

  /// Returns whatever is still held back as text. An unterminated envelope
  /// is text.
  List<ScanSegment> close() {
    final rest = _pending;
    _pending = '';
    return rest.isEmpty ? [] : [TextSegment(rest)];
  }

  (int, ToolCallEnvelope?) _earliestOpen() {
    var best = -1;
    ToolCallEnvelope? found;
    for (final envelope in envelopes) {
      final at = _pending.indexOf(envelope.open);
      if (at >= 0 && (best < 0 || at < best)) {
        best = at;
        found = envelope;
      }
    }
    return (best, found);
  }

  int _heldPrefixLength() {
    var held = 0;
    for (final envelope in envelopes) {
      final open = envelope.open;
      final maxLen = open.length - 1 < _pending.length
          ? open.length - 1
          : _pending.length;
      for (var len = maxLen; len > held; len--) {
        if (_pending.endsWith(open.substring(0, len))) {
          held = len;
          break;
        }
      }
    }
    return held;
  }

  void _emitText(List<ScanSegment> segments, int end) {
    if (end <= 0) return;
    segments.add(TextSegment(_pending.substring(0, end)));
    _pending = _pending.substring(end);
  }
}
