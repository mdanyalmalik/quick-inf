"""A Chatterbox voice as one small file: what the speech encoder (speech_encoder.onnx, from a reference clip) gives the
language model and the decoder, computed once, so the app needs neither the encoder (230-590 MB) nor a clip.

The file is little-endian float32: N, P, F, then audio_features [N, 1024] (the language model's conditioning
prefix), audio_tokens [P] (the clip's speech tokens, the decoder's prompt; whole numbers), speaker_embeddings [192]
and speaker_features [F, 80].

    python tools/export/chatterbox_voice.py SPEECH_ENCODER.onnx CLIP.wav OUT.bin
"""

import sys

import librosa
import numpy as np
import onnxruntime as ort


def main():
    encoder, clip, out = sys.argv[1:4]
    audio, _ = librosa.load(clip, sr=24000)  # as the model cards' ONNX example loads a reference
    s = ort.InferenceSession(encoder, providers=['CPUExecutionProvider'])
    features, tokens, embedding, speaker = s.run(None, {'audio_values': audio[np.newaxis, :].astype(np.float32)})
    n, p, f = features.shape[1], tokens.shape[1], speaker.shape[1]
    assert embedding.shape == (1, 192) and speaker.shape[2] == 80 and tokens.max() < 2 ** 24
    data = np.concatenate([np.array([n, p, f], np.float32), features.ravel(), tokens.ravel().astype(np.float32),
                           embedding.ravel(), speaker.ravel()]).astype('<f4')
    data.tofile(out)
    print(f'{out}: {n} conditioning positions, {p} prompt tokens, {f} speaker frames, {data.nbytes} bytes')


if __name__ == '__main__':
    main()
