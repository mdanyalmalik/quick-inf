// Mono 16-bit PCM WAV files from float audio, for saving and downloading generated speech.

/** A WAV file of the pieces of audio in order (samples in [-1, 1], clipped outside it). */
export function encodeWav(pieces: Float32Array[], sampleRate: number): Blob {
  const samples = pieces.reduce((n, p) => n + p.length, 0);
  const buffer = new ArrayBuffer(44 + samples * 2);
  const view = new DataView(buffer);
  const text = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  text(0, 'RIFF');
  view.setUint32(4, 36 + samples * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true); // format chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // bytes per second
  view.setUint16(32, 2, true); // bytes per frame
  view.setUint16(34, 16, true); // bits per sample
  text(36, 'data');
  view.setUint32(40, samples * 2, true);
  let offset = 44;
  for (const piece of pieces) {
    for (let i = 0; i < piece.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, piece[i]!));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
  }
  return new Blob([buffer], { type: 'audio/wav' });
}
