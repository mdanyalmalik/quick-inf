// Text to speech in the browser: each model's ONNX graphs as ONNX Runtime sessions (WebGPU, or the CPU engine
// without a usable GPU), run as its pipeline says (src/catalogue/speech-models.ts). Runs in the worker. Audio is
// made a chunk of sentences at a time and each chunk is handed out as soon as it's ready, so playback can start
// before the whole text is spoken.
//
// - 'supertonic' (Supertonic 3): text → Unicode ids → duration predictor (how long the speech lasts) and text
//   encoder → flow matching from noise over the vector estimator's steps → vocoder → 44.1 kHz audio.
// - 'soprano' (Soprano 1.1): text spelled out → a small language model writes audio tokens per sentence → a
//   decoder turns its hidden states into 32 kHz audio.
// - 'phonon' (Phonon 31M; Pocket TTS, with its own text chunking and end rule): text normalised and tokenized → a flow LM conditioned on a voice makes one latent per
//   80 ms frame → Mimi's decoder streams 24 kHz audio frame by frame.
// - 'chatterbox' (Chatterbox, Chatterbox Turbo): per sentence, a voice's conditioning and the text's tokens → a language model
//   writes speech tokens → the conditional decoder makes 24 kHz audio in that voice.
// - 'cosyvoice' (Fun-CosyVoice 3): text normalised and chunked → a language model samples speech tokens after a
//   voice prompt → flow matching makes the mel → HiFT makes 24 kHz audio.
// - 'kokoro' (Kokoro): Misaki phonemes → one graph with a voice's style row → 24 kHz audio.
// - 'moss' (MOSS-TTS Nano): text normalized and chunked → a global transformer over the voice's codes and the text,
//   a local one sampling 16 codes per frame → the MOSS audio tokenizer streams 48 kHz audio as frames come.

import type { InferenceSession, Tensor } from 'onnxruntime-web/webgpu';
import type { ArtifactFile } from '../../catalogue/models';
import { speechGraphs, speechModelFiles, type SpeechModelSpec, type SpeechPipeline, type SpeechSettings } from '../../catalogue/speech-models';
import { getGpuDevice } from '../../device/gpu';
import { RuntimeError } from '../contract';
import type { ByteSource } from '../onnx/external-weights';
import { createGraphSession } from '../onnx/onnx-adapter';
import { getOrt } from '../onnx/ort-env';
import { fromHalf, gaussian, toHalf } from '../image/schedulers';
import { BpeTokenizer } from '../../tokenizer/bpe-tokenizer';
import { UnigramTokenizer } from '../../tokenizer/unigram-tokenizer';
import { CharBpeTokenizer } from './char-bpe';
import { encodeWithTags, prepareLanguage, puncNorm } from './chatterbox-text';
import { normalizeEnglish as normalizeCosyEnglish, splitParagraph } from './cosyvoice-text';
import { chunkPause, normalizeTtsText, sentencePieceText, splitText } from './moss-text';
import { qwenInstruct, qwenPrompt } from './qwen3-tts-text';
import { EnglishG2P, loadLexicon } from './g2p';
import { neuttsAddedTokens, neuttsPhonemes, neuttsPrompt } from './neutts-text';
import { mayaPrompt, mayaTags, orpheusPrompt, orpheusSnacCodes } from './orpheus-text';
import { voxcpmIds, voxcpmSplits } from './voxcpm-text';
import { phononChunks } from './phonon-text';
import { pocketChunks } from './pocket-text';
import { cleanText, splitSentences } from './soprano-text';
import { chunkText, prepareText, textIds } from './supertonic-text';

export interface SpeechGenerationRequest {
  text: string;
  settings: SpeechSettings;
}

export interface SpeechGenerationProgress {
  /** The next stretch of audio (mono, at the model's sample rate), in order. */
  audio: Float32Array;
  /** Chunks of text done, of `chunks`. */
  chunk: number;
  chunks: number;
}

export interface SpeechGenerationResult {
  sampleRate: number;
  /** Total samples handed out in progress events. */
  samples: number;
  timings: { firstAudioMs: number; totalMs: number };
}

export interface SpeechFileSource {
  source(file: ArtifactFile): ByteSource;
  bytes(file: ArtifactFile): Uint8Array<ArrayBuffer>;
}

interface Loaded {
  model: SpeechModelSpec;
  pipeline: SpeechPipeline;
  sessions: Map<string, InferenceSession>;
  files: SpeechFileSource;
  /** Parsed JSON files (configs, indexes, voice styles), by path. */
  json: Map<string, unknown>;
  /** Things built once per load (tokenizers). */
  cache: Map<string, unknown>;
}

/** Silence between chunks, as Supertonic's own pipeline leaves (seconds). */
const CHUNK_SILENCE_S = 0.3;

export class SpeechEngine {
  private loaded: Loaded | null = null;
  private cancelRequested = false;

  get modelId(): string | null {
    return this.loaded?.model.id ?? null;
  }

  async load(model: SpeechModelSpec, files: SpeechFileSource, processor: 'gpu' | 'cpu' = 'gpu'): Promise<{ sessionMs: number }> {
    await this.unload();
    const start = performance.now();
    const loaded: Loaded = { model, pipeline: model.pipeline, sessions: new Map(), files, json: new Map(), cache: new Map() };
    let device: GPUDevice | null = null;
    if (processor === 'gpu') {
      try {
        device = await getGpuDevice();
      } catch (err) {
        throw new RuntimeError('load-failed', `The GPU could not be opened: ${message(err)}`, true);
      }
    }
    try {
      const graphs = Object.entries(speechGraphs(model.pipeline)).sort(([, a], [, b]) => a.graph.sizeBytes - b.graph.sizeBytes);
      for (const [role, g] of graphs) {
        const data = Object.fromEntries((g.data ?? []).map((f) => [f.path.slice(f.path.lastIndexOf('/') + 1), files.source(f)]));
        // Caches and streaming state stay on the GPU between steps.
        const pipeline = model.pipeline;
        let gpuOutputs: string[] = [];
        if (device && pipeline.kind === 'soprano' && role === 'backbone') gpuOutputs = kvNames(pipeline.layers, 'present');
        if (device && pipeline.kind === 'chatterbox' && role === 'languageModel') gpuOutputs = kvNames(pipeline.layers, 'present');
        if (device && pipeline.kind === 'cosyvoice') {
          if (role === 'llm') gpuOutputs = kvNames(pipeline.layers, 'present');
          // The flow's inputs are fed to all ten steps, and each step's x to the next.
          if (role === 'flowEncoder') gpuOutputs = ['mu', 'spks', 'cond'];
          if (role === 'flowStep') gpuOutputs = ['x_next'];
        }
        if (device && pipeline.kind === 'csm') {
          if (role === 'backbone') gpuOutputs = kvNames(16, 'present');
          if (role === 'depthDecoder') gpuOutputs = kvNames(4, 'present');
        }
        // NeuTTS's logits (217,652 per position) stay on the GPU too: only the last position's are read.
        if (device && pipeline.kind === 'neutts' && role === 'lm') gpuOutputs = [...kvNames(NEUTTS_LAYERS, 'present'), 'logits'];
        if (device && pipeline.kind === 'orpheus' && role === 'lm') gpuOutputs = [...kvNames(ORPHEUS_LAYERS, 'present'), 'logits'];
        if (device && pipeline.kind === 'voxcpm') {
          if (role === 'baseLm') gpuOutputs = kvNames(VOX_BASE_LAYERS, 'present');
          if (role === 'residualLm') gpuOutputs = kvNames(VOX_RESIDUAL_LAYERS, 'present');
        }
        if (device && pipeline.kind === 'qwen3-tts') {
          if (role === 'talker') gpuOutputs = kvNames(QWEN_TALKER_LAYERS, 'present');
          if (role === 'codePredictor') gpuOutputs = kvNames(QWEN_CP_LAYERS, 'present');
        }
        if (device && pipeline.kind === 'vibevoice') {
          if (role === 'textLm') gpuOutputs = kvNames(VIBE_TEXT_LAYERS, 'present');
          if (role === 'ttsLm') gpuOutputs = kvNames(VIBE_TTS_LAYERS, 'present');
          if (role === 'decoder') gpuOutputs = (json(loaded, pipeline.states) as { states: number[][] }).states.map((_, i) => `new_state_${i}`);
        }
        if (device && pipeline.kind === 'moss') {
          const config = json(loaded, pipeline.config) as MossConfig;
          if (role === 'prefill' || role === 'decodeStep') gpuOutputs = mossKv(config, 'present');
        }
        if (device && pipeline.kind === 'phonon') {
          const config = json(loaded, pipeline.config) as PhononConfig;
          if (role === 'flowLm') gpuOutputs = [...phononKv(config, 'present'), 'next_emb'];
          if (role === 'mimiDecoder') gpuOutputs = config.mimi_decoder.state.map((s) => `${s.name}_out`);
        }
        // A batch of one: ONNX Runtime can then work out more of a graph's shapes ahead of time.
        const freeDimensions = pipeline.kind === 'chatterbox' ? { batch_size: 1 } : undefined;
        // MOSS's frame sampler and codec step are mostly small sequential and integer work that WebGPU would hand back
        // and forth to the CPU: they run faster on the CPU engine outright.
        const onCpu = pipeline.kind === 'moss' && (role === 'sampleFrame' || role === 'codec');
        if (onCpu) gpuOutputs = [];
        loaded.sessions.set(role, await createGraphSession(files.source(g.graph), onCpu ? null : device, data, gpuOutputs, { freeDimensions }));
      }
      // Binary data files (voices, embeddings) are read now: the files are only open while the model loads.
      for (const file of speechModelFiles(model).filter((f) => f.path.endsWith('.bin'))) {
        const bytes = files.source(file).read(0, file.sizeBytes);
        loaded.cache.set(file.path, new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));
      }
      if ('lexicon' in model.pipeline) {
        const lexicon = model.pipeline.lexicon;
        loaded.cache.set('g2p', new EnglishG2P(await loadLexicon(files.source(lexicon).read(0, lexicon.sizeBytes))));
      }
      this.loaded = loaded;
    } catch (err) {
      await release(loaded);
      if (OUT_OF_MEMORY.test(message(err))) {
        throw new RuntimeError('out-of-memory', `There isn’t enough free GPU memory to run ${model.name} on this device (${message(err)})`, true);
      }
      throw new RuntimeError('load-failed', `The model could not be loaded on the ${processor === 'cpu' ? 'CPU' : 'GPU'}: ${message(err)}`, true);
    }
    return { sessionMs: performance.now() - start };
  }

  cancel(): void {
    this.cancelRequested = true;
  }

  async unload(): Promise<void> {
    const loaded = this.loaded;
    this.loaded = null;
    if (loaded) await release(loaded);
  }

  async generate(request: SpeechGenerationRequest, onProgress: (p: SpeechGenerationProgress) => void): Promise<SpeechGenerationResult> {
    const loaded = this.loaded;
    if (!loaded) throw new RuntimeError('not-loaded', 'Load the model first.');
    this.cancelRequested = false;
    const start = performance.now();
    let firstAudioMs = 0;
    let samples = 0;
    const emit = (audio: Float32Array, chunk: number, chunks: number) => {
      if (!firstAudioMs) firstAudioMs = performance.now() - start;
      samples += audio.length;
      onProgress({ audio, chunk, chunks });
    };
    const pipeline = loaded.pipeline;
    switch (pipeline.kind) {
      case 'supertonic':
        await this.supertonic(loaded, pipeline, request, emit);
        break;
      case 'soprano':
        await this.soprano(loaded, pipeline, request, emit);
        break;
      case 'phonon':
        await this.phonon(loaded, pipeline, request, emit);
        break;
      case 'kitten':
        await this.kitten(loaded, pipeline, request, emit);
        break;
      case 'inflect':
        await this.inflect(loaded, request, emit);
        break;
      case 'paradee':
        await this.paradee(loaded, pipeline, request, emit);
        break;
      case 'chatterbox':
        await this.chatterbox(loaded, pipeline, request, emit);
        break;
      case 'moss':
        await this.moss(loaded, pipeline, request, emit);
        break;
      case 'kokoro':
        await this.kokoro(loaded, pipeline, request, emit);
        break;
      case 'cosyvoice':
        await this.cosyvoice(loaded, pipeline, request, emit);
        break;
      case 'vibevoice':
        await this.vibevoice(loaded, pipeline, request, emit);
        break;
      case 'qwen3-tts':
        await this.qwen3tts(loaded, pipeline, request, emit);
        break;
      case 'csm':
        await this.csm(loaded, pipeline, request, emit);
        break;
      case 'neutts':
        await this.neutts(loaded, pipeline, request, emit);
        break;
      case 'orpheus':
        await this.orpheus(loaded, pipeline, request, emit);
        break;
      case 'voxcpm':
        await this.voxcpm(loaded, pipeline, request, emit);
        break;
    }
    return { sampleRate: loaded.model.sampleRate, samples, timings: { firstAudioMs, totalMs: performance.now() - start } };
  }

  private throwIfCancelled(): void {
    if (this.cancelRequested) throw new RuntimeError('cancelled', 'Stopped.');
  }

  private async supertonic(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'supertonic' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    const config = json(loaded, pipeline.config) as { ae: { sample_rate: number; base_chunk_size: number }; ttl: { chunk_compress_factor: number; latent_dim: number } };
    const indexer = json(loaded, pipeline.unicodeIndexer) as number[];
    const voice = loaded.model.voices.find((v) => v.id === request.settings.voice) ?? loaded.model.voices[0]!;
    const style = json(loaded, voice.file!) as Record<'style_ttl' | 'style_dp', { dims: number[]; data: unknown }>;
    const styleTensor = (s: { dims: number[]; data: unknown }) => new ort.Tensor('float32', Float32Array.from((s.data as unknown[]).flat(Infinity) as number[]), s.dims);
    const styleTtl = styleTensor(style.style_ttl);
    const styleDp = styleTensor(style.style_dp);
    const sampleRate = config.ae.sample_rate;
    const chunkSamples = config.ae.base_chunk_size * config.ttl.chunk_compress_factor;
    const latentChannels = config.ttl.latent_dim * config.ttl.chunk_compress_factor;
    const steps = Math.max(1, Math.min(32, Math.round(request.settings.steps ?? 8)));
    const speed = Math.max(0.7, Math.min(2, request.settings.speed ?? 1.05));
    const run = (role: string, feeds: Record<string, Tensor>) => loaded.sessions.get(role)!.run(feeds);
    const chunks = chunkText(request.text);
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const silence = new Float32Array(Math.round(CHUNK_SILENCE_S * sampleRate));
    for (let c = 0; c < chunks.length; c++) {
      this.throwIfCancelled();
      const ids = textIds(prepareText(chunks[c]!, pipeline.language), indexer);
      const textIdsTensor = new ort.Tensor('int64', ids, [1, ids.length]);
      const textMask = new ort.Tensor('float32', new Float32Array(ids.length).fill(1), [1, 1, ids.length]);
      const duration = (await run('durationPredictor', { text_ids: textIdsTensor, style_dp: styleDp, text_mask: textMask })).duration!;
      const seconds = (duration.data as Float32Array)[0]! / speed;
      duration.dispose();
      const textEmb = (await run('textEncoder', { text_ids: textIdsTensor, style_ttl: styleTtl, text_mask: textMask })).text_emb!;
      const latentLength = Math.max(1, Math.ceil((seconds * sampleRate) / chunkSamples));
      const latentMask = new ort.Tensor('float32', new Float32Array(latentLength).fill(1), [1, 1, latentLength]);
      const totalStep = new ort.Tensor('float32', new Float32Array([steps]), [1]);
      let latent: Tensor = new ort.Tensor('float32', gaussian(latentChannels * latentLength, (Math.random() * 2 ** 31) | 0), [1, latentChannels, latentLength]);
      for (let step = 0; step < steps; step++) {
        this.throwIfCancelled();
        const next = (
          await run('vectorEstimator', {
            noisy_latent: latent,
            text_emb: textEmb,
            style_ttl: styleTtl,
            text_mask: textMask,
            latent_mask: latentMask,
            current_step: new ort.Tensor('float32', new Float32Array([step]), [1]),
            total_step: totalStep,
          })
        ).denoised_latent!;
        latent.dispose();
        latent = next;
      }
      textEmb.dispose();
      const wav = (await run('vocoder', { latent })).wav_tts!;
      latent.dispose();
      const audio = new Float32Array(wav.data as Float32Array);
      wav.dispose();
      emit(audio, c + 1, chunks.length);
      if (c < chunks.length - 1) emit(silence, c + 1, chunks.length);
    }
  }

  /**
   * Soprano, as its Python package runs it at its defaults: per sentence, `[STOP][TEXT]sentence[START]` →
   * greedy audio tokens (repetition penalty 1.2 over every token so far) until [STOP] → the decoder over the last
   * hidden state of each generated token → the audio of all but the first token.
   */
  private async soprano(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'soprano' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    let tokenizer = loaded.cache.get('tokenizer') as CharBpeTokenizer | undefined;
    if (!tokenizer) {
      tokenizer = new CharBpeTokenizer(json(loaded, pipeline.tokenizer));
      loaded.cache.set('tokenizer', tokenizer);
    }
    const backbone = loaded.sessions.get('backbone')!;
    const decoder = loaded.sessions.get('decoder')!;
    const sentences = splitSentences(cleanText(request.text));
    if (sentences.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const { hiddenSize, tokenSamples } = pipeline;
    for (const [s, sentence] of sentences.entries()) {
      const prompt = tokenizer.encode(`[STOP][TEXT]${sentence}[START]`);
      const seen = new Set(prompt);
      const hidden: Float32Array[] = [];
      let past: Record<string, Tensor> = Object.fromEntries(
        kvNames(pipeline.layers, 'past_key_values').map((n) => [n, new ort.Tensor('float16', new Uint16Array(0), [1, pipeline.kvHeads, 0, pipeline.headDim])]),
      );
      let step = prompt;
      let total = 0;
      try {
        for (let i = 0; i < MAX_SOPRANO_TOKENS; i++) {
          this.throwIfCancelled();
          total += step.length;
          const out = await backbone.run({
            input_ids: new ort.Tensor('int64', BigInt64Array.from(step, BigInt), [1, step.length]),
            attention_mask: new ort.Tensor('int64', new BigInt64Array(total).fill(1n), [1, total]),
            ...past,
          });
          for (const t of Object.values(past)) t.dispose();
          past = Object.fromEntries(kvNames(pipeline.layers, 'present').map((n, k) => [kvNames(pipeline.layers, 'past_key_values')[k]!, out[n]!]));
          const logits = floats(out.logits!);
          const states = floats(out.hidden_states!);
          out.logits!.dispose();
          out.hidden_states!.dispose();
          const vocab = logits.length / step.length;
          const best = argmaxWithPenalty(logits.subarray(logits.length - vocab), seen);
          if (best === pipeline.eosToken) break;
          hidden.push(states.slice(states.length - hiddenSize));
          seen.add(best);
          step = [best];
        }
      } finally {
        for (const t of Object.values(past)) t.dispose();
      }
      if (hidden.length < 2) continue;
      // The decoder takes [1, hidden, tokens].
      const input = new Float32Array(hiddenSize * hidden.length);
      hidden.forEach((h, t) => h.forEach((v, d) => (input[d * hidden.length + t] = v)));
      const audio = (await decoder.run({ hidden_states: new ort.Tensor('float32', input, [1, hiddenSize, hidden.length]) })).audio!;
      const samples = audio.data as Float32Array;
      emit(samples.slice(Math.max(0, samples.length - (hidden.length - 1) * tokenSamples)), s + 1, sentences.length);
      audio.dispose();
    }
  }

  /**
   * Phonon, as Gradium's runtime runs it (exports/…/README.md, from tools/export/export_phonon.py): the voice
   * prompt once into a KV cache; per chunk of text, its embeddings on a copy of that cache; then one frame at a
   * time from the BOS embedding, with Gaussian noise, until the EOS logit passes its threshold (plus a few frames)
   * or the chunk's frame budget runs out. Each frame's latent is decoded by Mimi straight away.
   */
  private async phonon(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'phonon' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    const config = json(loaded, pipeline.config) as PhononConfig;
    let tokenizer = loaded.cache.get('tokenizer') as UnigramTokenizer | undefined;
    if (!tokenizer) {
      tokenizer = UnigramTokenizer.fromJSON(json(loaded, pipeline.tokenizer));
      loaded.cache.set('tokenizer', tokenizer);
    }
    const pocket = pipeline.variant === 'pocket-tts';
    const chunks = (pocket ? pocketChunks : phononChunks)(request.text, tokenizer);
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const flow = loaded.sessions.get('flowLm')!;
    const mimi = loaded.sessions.get('mimiDecoder')!;
    const textEmbed = loaded.sessions.get('textEmbed')!;
    const voice = loaded.model.voices.find((v) => v.id === request.settings.voice) ?? loaded.model.voices[0]!;
    const voiceEmb = binary(loaded, voice.file!);
    const bos = binary(loaded, pipeline.bos);
    const pastNames = phononKv(config, 'past');
    const presentNames = phononKv(config, 'present');
    const { d_model: dim, ldim, num_heads: heads, head_dim: headDim } = config;
    const zeros = (n: number, dims: number[]) => new ort.Tensor('float32', new Float32Array(n), dims);
    const noStep = zeros(ldim, [1, ldim]);
    const asPast = (out: Record<string, Tensor>) => Object.fromEntries(presentNames.map((n, i) => [pastNames[i]!, out[n]!]));
    const dispose = (tensors: Record<string, Tensor>, keep: Set<Tensor> = new Set()) => {
      for (const t of Object.values(tensors)) if (!keep.has(t)) t.dispose();
    };

    // The voice prompt, once: its cache is the start of every chunk.
    const voiceOut = await flow.run({
      emb: new ort.Tensor('float32', voiceEmb, [1, voiceEmb.length / dim, dim]),
      noise: noStep,
      ...Object.fromEntries(pastNames.map((n) => [n, zeros(0, [1, heads, 0, headDim])])),
    });
    const voiceCache = asPast(voiceOut);
    const voiceTensors = new Set(Object.values(voiceCache));
    dispose(voiceOut, voiceTensors);
    const std = Math.sqrt(config.temperature);
    try {
      for (const [c, chunk] of chunks.entries()) {
        this.throwIfCancelled();
        const ids = new ort.Tensor('int64', BigInt64Array.from(chunk.ids, BigInt), [1, chunk.ids.length]);
        const textEmb = (await textEmbed.run({ token_ids: ids })).text_emb!;
        const promptOut = await flow.run({ emb: textEmb, noise: noStep, ...voiceCache });
        textEmb.dispose();
        let past = asPast(promptOut);
        dispose(promptOut, new Set(Object.values(past)));
        let emb: Tensor = new ort.Tensor('float32', bos, [1, 1, dim]);
        let state: Record<string, Tensor> = Object.fromEntries(config.mimi_decoder.state.map((s) => [s.name, zeros(s.shape.reduce((a, b) => a * b, 1), s.shape)]));
        let left = -1;
        let end = -1;
        try {
          for (let frame = 0; frame < chunk.frameBudget && left !== 0; frame++) {
            this.throwIfCancelled();
            const noise = gaussian(ldim, (Math.random() * 2 ** 31) | 0).map((v) => v * std);
            const out = await flow.run({ emb, noise: new ort.Tensor('float32', noise, [1, ldim]), ...past });
            dispose(past);
            emb.dispose();
            past = asPast(out);
            emb = out.next_emb!;
            const eos = (out.eos_logit!.data as Float32Array)[0]!;
            out.eos_logit!.dispose();
            if (pocket) {
              // Pocket TTS: the end counts from the sixth frame; the end frame and framesAfterEos - 1 more are kept.
              if (end < 0 && frame >= (config.min_frames_before_eos ?? 0) && eos > config.eos_threshold) end = frame;
              if (end >= 0 && frame >= end + chunk.framesAfterEos) {
                out.latent!.dispose();
                break;
              }
            } else if (left > 0) left--;
            else if (left < 0 && eos > config.eos_threshold) left = chunk.framesAfterEos;
            const decoded = await mimi.run({ latent: out.latent!, ...state });
            out.latent!.dispose();
            dispose(state);
            state = Object.fromEntries(config.mimi_decoder.state.map((s) => [s.name, decoded[`${s.name}_out`]!]));
            const pcm = new Float32Array(decoded.pcm!.data as Float32Array);
            decoded.pcm!.dispose();
            // Pocket TTS fades each chunk's first 5 ms in, against the decoder's start-up click.
            if (frame === 0 && config.fade_in_samples) {
              const n = Math.min(config.fade_in_samples, pcm.length);
              for (let i = 0; i < n; i++) pcm[i]! *= n > 1 ? i / (n - 1) : 0;
            }
            emit(pcm, c + 1, chunks.length);
          }
        } finally {
          dispose(past);
          dispose(state);
          emb.dispose();
        }
      }
    } finally {
      dispose(voiceCache);
    }
  }

  /**
   * Kitten TTS, as its Python package runs it: per chunk, the ids of its IPA, the voice's style row for the chunk's
   * text length and the speed (times the voice's prior) → audio, less the last 5000 samples.
   */
  private async kitten(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'kitten' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    const chunks = g2p(loaded).kitten(request.text);
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const voice = loaded.model.voices.find((v) => v.id === request.settings.voice) ?? loaded.model.voices[0]!;
    const styles = binary(loaded, voice.file!);
    const styleSize = 256;
    const rows = styles.length / styleSize;
    const speed = (request.settings.speed ?? 1) * (pipeline.speedPriors?.[voice.id] ?? 1);
    const session = loaded.sessions.get('model')!;
    for (const [c, chunk] of chunks.entries()) {
      this.throwIfCancelled();
      const row = Math.min(chunk.text.length, rows - 1);
      const out = await session.run({
        input_ids: new ort.Tensor('int64', BigInt64Array.from(chunk.ids, BigInt), [1, chunk.ids.length]),
        style: new ort.Tensor('float32', styles.slice(row * styleSize, (row + 1) * styleSize), [1, styleSize]),
        speed: new ort.Tensor('float32', new Float32Array([speed]), [1]),
      });
      const waveform = out[session.outputNames[0]!]!;
      const samples = waveform.data as Float32Array;
      emit(samples.slice(0, Math.max(0, samples.length - KITTEN_TRIM)), c + 1, chunks.length);
      for (const t of Object.values(out)) t.dispose();
    }
  }

  /**
   * Inflect v2, as its inference_onnx.py runs it: per chunk, durations → the decoder with Gaussian latent noise
   * (scale 0.667) → audio faded in and out over 5 ms; a pause between chunks by the punctuation that ends each.
   */
  private async inflect(loaded: Loaded, request: SpeechGenerationRequest, emit: (audio: Float32Array, chunk: number, chunks: number) => void): Promise<void> {
    const ort = getOrt();
    const chunks = g2p(loaded).inflect(request.text);
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const duration = loaded.sessions.get('duration')!;
    const decode = loaded.sessions.get('decode')!;
    const speed = Math.max(0.5, Math.min(2, request.settings.speed ?? 1));
    for (const [c, chunk] of chunks.entries()) {
      this.throwIfCancelled();
      if (c > 0) emit(new Float32Array(Math.round(loaded.model.sampleRate * inflectPause(chunks[c - 1]!.text))), c, chunks.length);
      const d = await duration.run({
        tokens: new ort.Tensor('int64', BigInt64Array.from(chunk.ids, BigInt), [1, chunk.ids.length]),
        lengths: new ort.Tensor('int64', new BigInt64Array([BigInt(chunk.ids.length)]), [1]),
        length_scale: new ort.Tensor('float32', new Float32Array([1 / speed]), []),
      });
      const mean = d.m_p_exp!;
      const out = await decode.run({
        m_p_exp: mean,
        logs_p_exp: d.logs_p_exp!,
        y_mask: d.y_mask!,
        zp_noise: new ort.Tensor('float32', gaussian(mean.size, (Math.random() * 2 ** 31) | 0), mean.dims),
        noise_scale: new ort.Tensor('float32', new Float32Array([INFLECT_VARIATION]), []),
      });
      for (const t of Object.values(d)) t.dispose();
      const audio = new Float32Array(out.waveform!.data as Float32Array);
      out.waveform!.dispose();
      const fade = Math.min(INFLECT_FADE, audio.length >> 1);
      for (let i = 0; i < fade; i++) {
        const ramp = i / Math.max(1, fade - 1);
        audio[i]! *= ramp;
        audio[audio.length - 1 - i]! *= ramp;
      }
      for (let i = 0; i < audio.length; i++) audio[i] = Math.max(-1, Math.min(1, audio[i]!));
      emit(audio, c + 1, chunks.length);
    }
  }

  /** Paradee, as its Python package runs it: per sentence, Misaki phonemes' ids and the speed → audio. */
  private async paradee(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'paradee' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    const vocab = (json(loaded, pipeline.config) as { vocab: Record<string, number> }).vocab;
    const chunks = g2p(loaded).paradee(request.text, vocab);
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const session = loaded.sessions.get('model')!;
    for (const [c, chunk] of chunks.entries()) {
      this.throwIfCancelled();
      const out = await session.run({
        input_ids: new ort.Tensor('int64', BigInt64Array.from(chunk.ids, BigInt), [1, chunk.ids.length]),
        speed: new ort.Tensor('float32', new Float32Array([request.settings.speed ?? 1]), [1]),
      });
      const waveform = out[session.outputNames[0]!]!;
      emit(new Float32Array(waveform.data as Float32Array), c + 1, chunks.length);
      for (const t of Object.values(out)) t.dispose();
    }
  }

  /**
   * Chatterbox, as the ONNX examples on Resemble AI's and onnx-community's model cards run it: per sentence
   * (punc_norm'd), the voice's conditioning then the text's embeddings into the language model, then greedy speech
   * tokens (repetition penalty 1.2 over those so far) until the stop token; the decoder takes the voice's prompt
   * tokens and the new ones (Turbo: and three of silence). The original's embeddings also take each token's position
   * and the exaggeration, 0.5 (its default).
   */
  private async chatterbox(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'chatterbox' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    const turbo = pipeline.variant === 'turbo';
    let textIds = loaded.cache.get('textIds') as ((sentence: string, language: string) => number[]) | undefined;
    if (!textIds) {
      const tokenizerJson = json(loaded, pipeline.tokenizer) as { added_tokens: { content: string }[] };
      if (turbo) {
        const tags = tokenizerJson.added_tokens.map((t) => t.content).filter((t) => t.startsWith('['));
        const tokenizer = BpeTokenizer.fromJSON(tokenizerJson);
        const eos = tokenizer.specialTokenId('<|endoftext|>');
        // Turbo's tokenizer ends every text with <|endoftext|> twice.
        textIds = (sentence) => [...encodeWithTags(tokenizer, puncNorm(sentence, 'turbo'), tags), eos, eos];
      } else {
        const tokenizer = new CharBpeTokenizer(tokenizerJson);
        const variant = pipeline.variant;
        // The template: [EXAGGERATION] [START] text [STOP] [START_SPEECH] [START_SPEECH]; the multilingual model's text
        // starts with its language's tag.
        textIds = (sentence, language) => {
          const text = puncNorm(sentence, variant);
          return [6563, 255, ...tokenizer.encode(variant === 'multilingual' ? prepareLanguage(text, language) : text), 0, CHATTERBOX_START, CHATTERBOX_START];
        };
      }
      loaded.cache.set('textIds', textIds);
    }
    const limits = turbo ? CHATTERBOX_LIMITS.turbo : CHATTERBOX_LIMITS.other;
    const sentences = splitSentences(request.text, 30, limits.characters);
    if (sentences.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const voice = chatterboxVoice(binary(loaded, (loaded.model.voices.find((v) => v.id === request.settings.voice) ?? loaded.model.voices[0]!).file!));
    const embed = loaded.sessions.get('embedTokens')!;
    const lm = loaded.sessions.get('languageModel')!;
    const decoder = loaded.sessions.get('decoder')!;
    const pastNames = kvNames(pipeline.layers, 'past_key_values');
    const presentNames = kvNames(pipeline.layers, 'present');
    const embedIds = async (ids: number[], positions: number[]) => {
      const out = await embed.run({
        input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]),
        ...(!turbo && {
          position_ids: new ort.Tensor('int64', BigInt64Array.from(positions, BigInt), [1, ids.length]),
          exaggeration: new ort.Tensor('float32', new Float32Array([CHATTERBOX_EXAGGERATION]), [1]),
        }),
      });
      return out.inputs_embeds!;
    };
    for (const [s, sentence] of sentences.entries()) {
      this.throwIfCancelled();
      const ids = textIds(sentence, request.settings.language ?? loaded.model.defaults.language ?? 'en');
      // The original's text positions: the speech markers at 0, the rest counting from the [START] token.
      const textEmbeds = await embedIds(ids, ids.map((id, i) => (id >= CHATTERBOX_START ? 0 : i - 1)));
      const prefix = new Float32Array(voice.conditioning.length + textEmbeds.size);
      prefix.set(voice.conditioning);
      prefix.set(textEmbeds.data as Float32Array, voice.conditioning.length);
      textEmbeds.dispose();
      let embeds: Tensor = new ort.Tensor('float32', prefix, [1, prefix.length / CHATTERBOX_WIDTH, CHATTERBOX_WIDTH]);
      let past: Record<string, Tensor> = Object.fromEntries(
        pastNames.map((n) => [n, new ort.Tensor('float32', new Float32Array(0), [1, pipeline.kvHeads, 0, pipeline.headDim])]),
      );
      const generated = [CHATTERBOX_START];
      let total = 0;
      try {
        for (let i = 0; i < limits.tokens; i++) {
          this.throwIfCancelled();
          const step = embeds.dims[1]!;
          const out = await lm.run({
            inputs_embeds: embeds,
            attention_mask: new ort.Tensor('int64', new BigInt64Array(total + step).fill(1n), [1, total + step]),
            ...(turbo && { position_ids: new ort.Tensor('int64', BigInt64Array.from({ length: step }, (_, k) => BigInt(total + k)), [1, step]) }),
            ...past,
          });
          total += step;
          embeds.dispose();
          for (const t of Object.values(past)) t.dispose();
          past = Object.fromEntries(presentNames.map((n, k) => [pastNames[k]!, out[n]!]));
          const logits = out.logits!.data as Float32Array;
          const vocab = logits.length / step;
          const next = argmaxWithPenalty(logits.subarray(logits.length - vocab), new Set(generated));
          out.logits!.dispose();
          generated.push(next);
          if (next === CHATTERBOX_STOP) break;
          embeds = await embedIds([next], [i + 1]);
        }
      } finally {
        for (const t of Object.values(past)) t.dispose();
      }
      const speech = generated.slice(1).filter((t) => t < CHATTERBOX_START);
      const tokens = [...voice.promptTokens, ...speech, ...Array<number>(turbo ? 3 : 0).fill(CHATTERBOX_SILENCE)];
      const wav = (
        await decoder.run({
          speech_tokens: new ort.Tensor('int64', BigInt64Array.from(tokens, BigInt), [1, tokens.length]),
          speaker_embeddings: new ort.Tensor('float32', voice.speakerEmbedding, [1, voice.speakerEmbedding.length]),
          speaker_features: new ort.Tensor('float32', voice.speakerFeatures, [1, voice.speakerFeatures.length / 80, 80]),
        })
      ).waveform!;
      emit(new Float32Array(wav.data as Float32Array), s + 1, sentences.length);
      wav.dispose();
    }
  }

  /**
   * Qwen3-TTS, as qwen_tts's generate_custom_voice (or generate_voice_design) runs it with the checkpoint's defaults:
   * the prompt (optional voice instruction, role, a codec prefix with the language and speaker, the whole text, then
   * codec BOS) through the talker; per frame, code 0 sampled from the talker (temperature 0.9, top-k 50, repetition
   * penalty 1.05, control codes masked, no stop before the third frame), codes 1-15 from the code predictor (fresh
   * cache, one codebook per step), and the sum of the 16 codes' embeddings plus the pad embedding fed back; finally
   * the codec decoder over all frames.
   */
  private async qwen3tts(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'qwen3-tts' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    let tokenizer = loaded.cache.get('tokenizer') as BpeTokenizer | undefined;
    if (!tokenizer) {
      tokenizer = BpeTokenizer.fromJSON(json(loaded, pipeline.tokenizer));
      loaded.cache.set('tokenizer', tokenizer);
    }
    const text = request.text.trim();
    if (!text) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const cfg = json(loaded, pipeline.config) as QwenConfig;
    const tc = cfg.talker_config;
    const [vocab, cpVocab, groups, eos] = [tc.vocab_size, tc.code_predictor_config.vocab_size, tc.num_code_groups, tc.codec_eos_token_id];
    const session = (role: string) => loaded.sessions.get(role)!;
    const [talker, cp, textEmbed, codecEmbed, decoder] = [session('talker'), session('codePredictor'), session('textEmbed'), session('codecEmbed'), session('decoder')];
    // The talker's width (1024 for 0.6B, 2048 for 1.7B); the code predictor's is 1024 in both, its inputs projected
    // for 1.7B (cp_project, and codec_embed's cp_embeds).
    const width = tc.hidden_size;
    const cpProject = pipeline.cpProject ? session('cpProject') : null;
    /** Rows of a lookup graph's output (`embeds`, or `cp_embeds` for the code predictor when there is one). */
    const lookup = async (graph: InferenceSession, ids: number[], output = 'embeds') => {
      const out = await graph.run({ ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]) });
      const t = out[output] ?? out.embeds!;
      const n = t.size / ids.length;
      const rows = Array.from({ length: ids.length }, (_, i) => (t.data as Float32Array).slice(i * n, (i + 1) * n));
      for (const v of Object.values(out)) v.dispose();
      return rows;
    };
    const embed = (graph: InferenceSession, ids: number[]) => lookup(graph, ids);
    const cpEmbed = (ids: number[]) => lookup(codecEmbed, ids, 'cp_embeds');

    // The prompt.
    const { role, body } = qwenPrompt(tokenizer, text);
    const [bosE, eosE, padE] = await embed(textEmbed, [cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]);
    const language = request.settings.language ?? loaded.model.defaults.language ?? 'english';
    const speaker = request.settings.voice;
    const dialect = tc.spk_is_dialect?.[speaker];
    const langId = language === 'chinese' && dialect ? tc.codec_language_id[dialect]! : tc.codec_language_id[language] ?? tc.codec_language_id.english!;
    const codecIds = [tc.codec_think_id, tc.codec_think_bos_id, langId, tc.codec_think_eos_id];
    if (tc.spk_id[speaker] !== undefined) codecIds.push(tc.spk_id[speaker]!);
    codecIds.push(tc.codec_pad_id, tc.codec_bos_id);
    const codec = await embed(codecEmbed, codecIds);
    const rows: Float32Array[] = [];
    if (request.settings.instruct?.trim()) rows.push(...(await embed(textEmbed, qwenInstruct(tokenizer, request.settings.instruct.trim()))));
    rows.push(...(await embed(textEmbed, role)));
    codec.slice(0, -1).forEach((c, i) => rows.push(add(i < codec.length - 2 ? padE! : bosE!, c)));
    const bodyE = await embed(textEmbed, body);
    const padCodes = await embed(codecEmbed, Array<number>(body.length + 1).fill(tc.codec_pad_id));
    [...bodyE, eosE!].forEach((t, i) => rows.push(add(t, padCodes[i]!)));
    rows.push(add(padE!, (await embed(codecEmbed, [tc.codec_bos_id]))[0]!));

    const talkerLm = builderDecoder(talker, { layers: QWEN_TALKER_LAYERS, kvHeads: QWEN_KV_HEADS, headDim: QWEN_HEAD_DIM, width, float32: pipeline.talkerFloat32 });
    const frames: number[][] = [];
    try {
      let { logits, hidden } = await talkerLm.run(rows);
      // Sampling occasionally never picks the stop code: frames are capped by the text's length at about twice normal
      // speech (4.5 frames per token on average, up to 8 on short sentences), within the checkpoints' max_new_tokens.
      const maxFrames = Math.min(QWEN_MAX_FRAMES, 25 + 9 * body.length);
      for (let step = 0; step < maxFrames; step++) {
        this.throwIfCancelled();
        const seen = new Set(frames.map((f) => f[0]!));
        for (const t of seen) logits[t] = logits[t]! < 0 ? logits[t]! * 1.05 : logits[t]! / 1.05;
        for (let t = vocab - 1024; t < vocab; t++) if (t !== eos) logits[t] = -Infinity;
        if (step < 2) logits[eos] = -Infinity;
        const c0 = sampleTopK(logits);
        if (c0 === eos) break;
        const frame = [c0];
        // The code predictor computes in float32 (its float16 build overflows).
        const cpLm = builderDecoder(cp, { layers: QWEN_CP_LAYERS, kvHeads: QWEN_KV_HEADS, headDim: QWEN_HEAD_DIM, width: QWEN_CP_WIDTH, float32: true });
        try {
          let h = hidden;
          if (cpProject) {
            const out = await cpProject.run({ hidden: new ort.Tensor('float32', hidden, [1, 1, width]) });
            h = (out.cp_hidden!.data as Float32Array).slice();
            out.cp_hidden!.dispose();
          }
          let x = [h, (await cpEmbed([c0]))[0]!];
          for (let j = 0; j < groups - 1; j++) {
            const { logits: cpLogits } = await cpLm.run(x);
            const c = sampleTopK(cpLogits.subarray(j * cpVocab, (j + 1) * cpVocab));
            frame.push(c);
            if (j < groups - 2) x = await cpEmbed([vocab + j * cpVocab + c]);
          }
        } finally {
          cpLm.dispose();
        }
        frames.push(frame);
        const sum = add(...(await embed(codecEmbed, frame.map((c, g) => (g === 0 ? c : vocab + (g - 1) * cpVocab + c)))), padE!);
        ({ logits, hidden } = await talkerLm.run([sum]));
      }
    } finally {
      talkerLm.dispose();
    }
    if (frames.length === 0) return;
    emit(await this.qwenDecode(decoder, frames), 1, 1);
  }

  /**
   * CSM-1B, as transformers' CsmForConditionalGeneration generates with the checkpoint's sampling (temperature 0.9,
   * top-k 50): the prompt "[0]{text}" through the backbone; per frame, codebook 0 from the backbone, codebooks 1-31
   * from the depth decoder (fresh cache, fed the projected backbone state and then one codebook's embedding per step),
   * the sum of the frame's 32 code embeddings fed back; a frame whose codebooks 0-30 are all zero ends the audio.
   * Unlike the original (125 frames, 10 s), frames are capped by the text's length, so longer text is spoken in one
   * voice; Mimi's decoder then makes the audio.
   */
  private async csm(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'csm' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    let tokenizer = loaded.cache.get('tokenizer') as BpeTokenizer | undefined;
    if (!tokenizer) {
      tokenizer = BpeTokenizer.fromJSON(json(loaded, pipeline.tokenizer));
      loaded.cache.set('tokenizer', tokenizer);
    }
    const text = request.text.trim();
    if (!text) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const ids = [tokenizer.specialTokenId('<|begin_of_text|>'), ...tokenizer.encode(`[0]${text}`), tokenizer.specialTokenId('<|end_of_text|>')];
    const session = (role: string) => loaded.sessions.get(role)!;
    const [textEmbed, audioEmbed, depthProject, mimi] = [session('textEmbed'), session('audioEmbed'), session('depthProject'), session('mimi')];
    const rowsOf = (t: Tensor, width: number) => Array.from({ length: t.size / width }, (_, i) => (t.data as Float32Array).slice(i * width, (i + 1) * width));
    const audioRows = async (rows: number[]) => {
      const out = await audioEmbed.run({ ids: new ort.Tensor('int64', BigInt64Array.from(rows, BigInt), [1, rows.length]) });
      const result = { embeds: rowsOf(out.embeds!, CSM_WIDTH), depth: rowsOf(out.depth_embeds!, CSM_DEPTH_WIDTH) };
      for (const t of Object.values(out)) t.dispose();
      return result;
    };
    const backbone = builderDecoder(session('backbone'), { layers: 16, kvHeads: 8, headDim: 64, width: CSM_WIDTH });
    const frames: number[][] = [];
    try {
      const prompt = (await textEmbed.run({ ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]) })).embeds!;
      let { logits, hidden } = await backbone.run(rowsOf(prompt, CSM_WIDTH));
      prompt.dispose();
      const maxFrames = Math.min(CSM_MAX_POSITIONS - ids.length, 50 + 15 * ids.length);
      for (let step = 0; step < maxFrames; step++) {
        this.throwIfCancelled();
        const c0 = sampleTopK(logits.subarray(0, CSM_VOCAB));
        const projected = await depthProject.run({ hidden: new ort.Tensor('float32', hidden, [1, 1, CSM_WIDTH]) });
        let x = [(projected.depth_hidden!.data as Float32Array).slice(), (await audioRows([c0])).depth[0]!];
        projected.depth_hidden!.dispose();
        const frame = [c0];
        const depth = builderDecoder(session('depthDecoder'), { layers: 4, kvHeads: 2, headDim: 128, width: CSM_DEPTH_WIDTH });
        try {
          for (let j = 0; j < CSM_CODEBOOKS - 1; j++) {
            const { logits: depthLogits } = await depth.run(x);
            const c = sampleTopK(depthLogits.subarray(j * CSM_VOCAB, (j + 1) * CSM_VOCAB));
            frame.push(c);
            if (j < CSM_CODEBOOKS - 2) x = (await audioRows([(j + 1) * CSM_VOCAB + c])).depth;
          }
        } finally {
          depth.dispose();
        }
        if (frame.slice(0, -1).every((c) => c === 0)) break;
        frames.push(frame);
        const { embeds } = await audioRows(frame.map((c, k) => k * CSM_VOCAB + c));
        ({ logits, hidden } = await backbone.run([add(...embeds)]));
      }
    } finally {
      backbone.dispose();
    }
    if (frames.length === 0) return;
    const codes = new BigInt64Array(CSM_CODEBOOKS * frames.length);
    frames.forEach((f, t) => f.forEach((c, k) => (codes[k * frames.length + t] = BigInt(c))));
    const audio = (await mimi.run({ codes: new ort.Tensor('int64', codes, [1, CSM_CODEBOOKS, frames.length]) })).audio!;
    emit(new Float32Array(audio.data as Float32Array), 1, 1);
    audio.dispose();
  }

  /**
   * The Qwen3-TTS codec decoder over all frames ([1, 16, frames] → 1920 samples a frame), in chunks of 300 frames with
   * 25 frames of left context beyond the first, as the original's chunked_decode.
   */
  /**
   * NeuTTS Air, as its NeuTTS.infer runs a phoneme prompt: per chunk of sentences, the voice's transcript phonemes and
   * the text's after the instruction, then the voice's codes; new codes sampled (temperature 1, top-k 50, no end in
   * the first 50) until <|SPEECH_GENERATION_END|> or 2048 positions; NeuCodec's decoder over the new codes.
   */
  private async neutts(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'neutts' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    let tokenizer = loaded.cache.get('tokenizer') as { tokenizer: BpeTokenizer; added: string[] } | undefined;
    if (!tokenizer) {
      const tokenizerJson = json(loaded, pipeline.tokenizer) as Parameters<typeof neuttsAddedTokens>[0];
      tokenizer = { tokenizer: BpeTokenizer.fromJSON(tokenizerJson), added: neuttsAddedTokens(tokenizerJson) };
      loaded.cache.set('tokenizer', tokenizer);
    }
    const voiceSpec = loaded.model.voices.find((v) => v.id === request.settings.voice) ?? loaded.model.voices[0]!;
    const voice = json(loaded, voiceSpec.file!) as { codes: number[]; ref_phonemes: string };
    const speech0 = tokenizer.tokenizer.specialTokenId('<|speech_0|>');
    const end = tokenizer.tokenizer.specialTokenId('<|SPEECH_GENERATION_END|>');
    const lm = loaded.sessions.get('lm')!;
    const codec = loaded.sessions.get('codec')!;
    const chunks = splitSentences(request.text, 30, NEUTTS_CHUNK_CHARACTERS);
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const pastNames = kvNames(NEUTTS_LAYERS, 'past_key_values');
    const presentNames = kvNames(NEUTTS_LAYERS, 'present');
    for (const [c, chunk] of chunks.entries()) {
      this.throwIfCancelled();
      const prompt = [
        ...neuttsPrompt(tokenizer.tokenizer, tokenizer.added, voice.ref_phonemes, neuttsPhonemes(g2p(loaded), chunk)),
        ...voice.codes.map((code) => speech0 + code),
      ];
      let past: Record<string, Tensor> = Object.fromEntries(
        pastNames.map((n) => [n, new ort.Tensor('float32', new Float32Array(0), [1, NEUTTS_KV_HEADS, 0, NEUTTS_HEAD_DIM])]),
      );
      let length = 0;
      // Runs ids through the model; returns the last position's logits if asked (the rest stay on the GPU, unread).
      const run = async (ids: number[], read: boolean) => {
        const out = await lm.run({
          input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]),
          attention_mask: new ort.Tensor('int64', new BigInt64Array(length + ids.length).fill(1n), [1, length + ids.length]),
          ...past,
        });
        for (const t of Object.values(past)) t.dispose();
        past = Object.fromEntries(presentNames.map((n, i) => [pastNames[i]!, out[n]!]));
        length += ids.length;
        const logits = read ? ((await out.logits!.getData()) as Float32Array) : null;
        out.logits!.dispose();
        return logits ? logits.subarray(logits.length - NEUTTS_VOCAB) : null;
      };
      const codes: number[] = [];
      try {
        // The prompt in slices (all positions' logits at once would be hundreds of MB), its last token on its own.
        for (let i = 0; i < prompt.length - 1; i += NEUTTS_PREFILL) await run(prompt.slice(i, Math.min(i + NEUTTS_PREFILL, prompt.length - 1)), false);
        let logits = (await run(prompt.slice(-1), true))!;
        for (let n = 0; length < NEUTTS_MAX_POSITIONS; n++) {
          this.throwIfCancelled();
          if (n < NEUTTS_MIN_TOKENS) logits[end] = -Infinity;
          const next = sampleTopK(logits, 1);
          if (next === end) break;
          if (next >= speech0 && next < speech0 + NEUTTS_CODES) codes.push(next - speech0);
          logits = (await run([next], true))!;
        }
      } finally {
        for (const t of Object.values(past)) t.dispose();
      }
      if (codes.length < 2) continue;
      const audio = (await codec.run({ codes: new ort.Tensor('int32', Int32Array.from(codes), [1, 1, codes.length]) })).audio!;
      emit(new Float32Array(audio.data as Float32Array), c + 1, chunks.length);
      audio.dispose();
    }
  }

  /**
   * Orpheus 3B, as canopyai's engine runs it: per chunk of sentences, the prompt "{voice}: {text}"; tokens sampled
   * (repetition penalty 1.3 over the prompt and what came so far, temperature 0.6, top-p 0.8) until end of speech or
   * 1200 new tokens; the audio tokens' frames decoded by SNAC at once, less its first 2048 samples (as the engine's
   * sliding-window decoder starts). Maya1: its description prompt and sampling (ORPHEUS_SAMPLING), otherwise the same.
   */
  private async orpheus(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'orpheus' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    let tokenizer = loaded.cache.get('tokenizer') as { tokenizer: BpeTokenizer; tags: string[] } | undefined;
    if (!tokenizer) {
      const tokenizerJson = json(loaded, pipeline.tokenizer) as Parameters<typeof mayaTags>[0];
      tokenizer = { tokenizer: BpeTokenizer.fromJSON(tokenizerJson), tags: mayaTags(tokenizerJson) };
      loaded.cache.set('tokenizer', tokenizer);
    }
    const maya = pipeline.variant === 'maya1';
    const sampling = ORPHEUS_SAMPLING[maya ? 'maya1' : 'orpheus'];
    const voice = (loaded.model.voices.find((v) => v.id === request.settings.voice) ?? loaded.model.voices[0]!).id;
    const description = request.settings.instruct?.trim() || loaded.model.defaults.instruct || '';
    const promptFor = (chunk: string) =>
      maya ? mayaPrompt(tokenizer.tokenizer, tokenizer.tags, description, chunk) : orpheusPrompt(tokenizer.tokenizer, voice, chunk);
    const lm = loaded.sessions.get('lm')!;
    const snac = loaded.sessions.get('snac')!;
    const chunks = splitSentences(request.text, 30, ORPHEUS_CHUNK_CHARACTERS);
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const pastNames = kvNames(ORPHEUS_LAYERS, 'past_key_values');
    const presentNames = kvNames(ORPHEUS_LAYERS, 'present');
    for (const [c, chunk] of chunks.entries()) {
      this.throwIfCancelled();
      const prompt = promptFor(chunk);
      let past: Record<string, Tensor> = Object.fromEntries(
        pastNames.map((n) => [n, new ort.Tensor('float16', new Uint16Array(0), [1, ORPHEUS_KV_HEADS, 0, ORPHEUS_HEAD_DIM])]),
      );
      let length = 0;
      // Runs ids through the model; returns the last position's logits if asked (the rest stay on the GPU, unread).
      const run = async (ids: number[], read: boolean) => {
        const out = await lm.run({
          input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]),
          attention_mask: new ort.Tensor('int64', new BigInt64Array(length + ids.length).fill(1n), [1, length + ids.length]),
          ...past,
        });
        for (const t of Object.values(past)) t.dispose();
        past = Object.fromEntries(presentNames.map((n, i) => [pastNames[i]!, out[n]!]));
        length += ids.length;
        const data = read ? ((await out.logits!.getData()) as Float32Array | Uint16Array) : null;
        const vocab = out.logits!.dims[2]!;
        out.logits!.dispose();
        if (!data) return null;
        const last = data.subarray(data.length - vocab);
        return last instanceof Float32Array ? last.slice() : fromHalf(last);
      };
      const seen = new Set(prompt);
      const generated: number[] = [];
      try {
        for (let i = 0; i < prompt.length - 1; i += ORPHEUS_PREFILL) await run(prompt.slice(i, Math.min(i + ORPHEUS_PREFILL, prompt.length - 1)), false);
        let logits = (await run(prompt.slice(-1), true))!;
        while (generated.length < ORPHEUS_MAX_NEW) {
          this.throwIfCancelled();
          for (const t of seen) logits[t] = logits[t]! > 0 ? logits[t]! / sampling.repetition : logits[t]! * sampling.repetition;
          if (generated.length < sampling.minTokens) logits[ORPHEUS_END_OF_SPEECH] = -Infinity;
          const next = sampleTopP(logits, sampling.temperature, sampling.topP);
          if (next === ORPHEUS_END_OF_SPEECH) break;
          generated.push(next);
          seen.add(next);
          logits = (await run([next], true))!;
        }
      } finally {
        for (const t of Object.values(past)) t.dispose();
      }
      const layers = orpheusSnacCodes(generated);
      if (layers[0].length === 0) continue;
      const audio = (
        await snac.run(Object.fromEntries(layers.map((l, i) => [`audio_codes.${i}`, new ort.Tensor('int64', BigInt64Array.from(l, BigInt), [1, l.length])])))
      ).audio_values!;
      emit(new Float32Array((audio.data as Float32Array).subarray(ORPHEUS_SKIP_SAMPLES)), c + 1, chunks.length);
      audio.dispose();
    }
  }

  /**
   * VoxCPM2, as VoxCPM2Model._inference runs without a reference voice (cfg 2, 10 timesteps): the text ids + audio
   * start through text_embed, the base LM, the bridge and the residual LM; per patch, noise denoised by the DiT's
   * Euler steps (none on the first, as CFG-zero*) on the sway schedule, its encoding fed back through the LMs; stop
   * after a patch whose state's stop head says so (from the fourth patch on), or at 6 patches per text token + 10.
   * The whole text is one generation, so it keeps one voice.
   */
  private async voxcpm(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'voxcpm' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    let text = loaded.cache.get('textIds') as ((t: string) => number[]) | undefined;
    if (!text) {
      const tokenizerJson = json(loaded, pipeline.tokenizer) as Parameters<typeof voxcpmSplits>[0];
      const tokenizer = BpeTokenizer.fromJSON(tokenizerJson);
      const splits = voxcpmSplits(tokenizerJson);
      text = (t) => voxcpmIds(tokenizer, splits, t);
      loaded.cache.set('textIds', text);
    }
    const body = text(request.text.replace(/\s+/g, ' ').trim());
    if (body.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const ids = [...body, VOX_AUDIO_START];
    const maxPatches = Math.min(Math.trunc(body.length * 6 + 10), VOX_MAX_PATCHES);
    const s = (role: string) => loaded.sessions.get(role)!;
    const f32 = (data: Float32Array, dims: number[]) => new ort.Tensor('float32', data, dims);
    const lm = (session: InferenceSession, layers: number) => {
      const pastNames = kvNames(layers, 'past_key_values');
      const presentNames = kvNames(layers, 'present');
      let past: Record<string, Tensor> = Object.fromEntries(pastNames.map((n) => [n, f32(new Float32Array(0), [1, VOX_KV_HEADS, 0, VOX_HEAD_DIM])]));
      return {
        async run(embeds: Tensor): Promise<Float32Array> {
          const out = await session.run({ inputs_embeds: embeds, ...past });
          for (const t of Object.values(past)) t.dispose();
          past = Object.fromEntries(presentNames.map((n, i) => [pastNames[i]!, out[n]!]));
          const hidden = out.hidden!.data as Float32Array;
          out.hidden!.dispose();
          return hidden;
        },
        dispose: () => Object.values(past).forEach((t) => t.dispose()),
      };
    };
    const base = lm(s('baseLm'), VOX_BASE_LAYERS);
    const residual = lm(s('residualLm'), VOX_RESIDUAL_LAYERS);
    const last = (rows: Float32Array) => rows.slice(rows.length - VOX_WIDTH);
    // Base LM rows → bridge (audio_mask 1 on audio positions) → residual LM: the last row of each, and the stop logits.
    const step = async (embeds: Float32Array, rows: number, audio: boolean) => {
      const raw = await base.run(f32(embeds, [1, rows, VOX_WIDTH]));
      const bridged = await s('bridge').run({
        hidden: f32(raw, [1, rows, VOX_WIDTH]),
        feat_embed: f32(audio ? embeds : new Float32Array(raw.length), [1, rows, VOX_WIDTH]),
        audio_mask: f32(new Float32Array(rows).fill(audio ? 1 : 0), [1, rows]),
      });
      const lmHidden = last(bridged.lm_hidden!.data as Float32Array);
      const stop = (bridged.stop_logits!.data as Float32Array).slice(-2);
      const res = await residual.run(bridged.residual_input!);
      for (const t of Object.values(bridged)) t.dispose();
      return { lmHidden, resHidden: last(res), stop: stop[1]! > stop[0]! };
    };
    const patches: Float32Array[] = [];
    try {
      const embedded = await s('textEmbed').run({ ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]) });
      const prompt = new Float32Array(embedded.embeds!.data as Float32Array);
      embedded.embeds!.dispose();
      let state = await step(prompt, ids.length, false);
      let previous = new Float32Array(VOX_PATCH * VOX_LATENT);
      const cfg = f32(new Float32Array([VOX_CFG]), []);
      for (let i = 0; i < maxPatches; i++) {
        this.throwIfCancelled();
        // x [1, 64, 4]: noise, then 9 Euler steps (step 0's velocity is zero under CFG-zero*).
        let x = gaussian(VOX_LATENT * VOX_PATCH, (Math.random() * 2 ** 31) | 0);
        for (let k = 1; k < VOX_STEPS; k++) {
          const out = await s('ditStep').run({
            x: f32(x, [1, VOX_LATENT, VOX_PATCH]),
            lm_hidden: f32(state.lmHidden, [1, VOX_WIDTH]),
            residual_hidden: f32(state.resHidden, [1, VOX_WIDTH]),
            cond: f32(previous, [1, VOX_PATCH, VOX_LATENT]),
            t: f32(new Float32Array([VOX_T[k]!]), [1]),
            cfg_value: cfg,
          });
          const v = out.v!.data as Float32Array;
          const dt = VOX_T[k]! - VOX_T[k + 1]!;
          x = x.map((value, j) => value - dt * v[j]!);
          out.v!.dispose();
        }
        // The patch as [4, 64] (frames × latent channels).
        const patch = new Float32Array(VOX_PATCH * VOX_LATENT);
        for (let c = 0; c < VOX_LATENT; c++) for (let f = 0; f < VOX_PATCH; f++) patch[f * VOX_LATENT + c] = x[c * VOX_PATCH + f]!;
        patches.push(patch);
        previous = patch;
        if (i > 2 && state.stop) break;
        const encoded = await s('featEncoder').run({ feats: f32(patch, [1, 1, VOX_PATCH, VOX_LATENT]) });
        const embeds = new Float32Array(encoded.embeds!.data as Float32Array);
        encoded.embeds!.dispose();
        state = await step(embeds, 1, true);
      }
    } finally {
      base.dispose();
      residual.dispose();
    }
    // Latents [1, 64, 4 × patches].
    const frames = patches.length * VOX_PATCH;
    const latents = new Float32Array(VOX_LATENT * frames);
    patches.forEach((p, i) => {
      for (let f = 0; f < VOX_PATCH; f++) for (let c = 0; c < VOX_LATENT; c++) latents[c * frames + i * VOX_PATCH + f] = p[f * VOX_LATENT + c]!;
    });
    const audio = (await s('vae').run({ latents: f32(latents, [1, VOX_LATENT, frames]) })).audio!;
    emit(new Float32Array(audio.data as Float32Array), 1, 1);
    audio.dispose();
  }

  private async qwenDecode(decoder: InferenceSession, frames: number[][]): Promise<Float32Array> {
    const ort = getOrt();
    const pieces: Float32Array[] = [];
    for (let start = 0; start < frames.length; start += QWEN_DECODE_CHUNK) {
      this.throwIfCancelled();
      const from = Math.max(0, start - QWEN_DECODE_CONTEXT);
      const chunk = frames.slice(from, start + QWEN_DECODE_CHUNK);
      const codes = new BigInt64Array(16 * chunk.length);
      chunk.forEach((f, t) => f.forEach((c, g) => (codes[g * chunk.length + t] = BigInt(c))));
      const audio = (await decoder.run({ codes: new ort.Tensor('int64', codes, [1, 16, chunk.length]) })).audio!;
      pieces.push((audio.data as Float32Array).slice((start - from) * QWEN_FRAME_SAMPLES));
      audio.dispose();
    }
    const out = new Float32Array(pieces.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of pieces) {
      out.set(p, at);
      at += p.length;
    }
    return out;
  }

  /**
   * VibeVoice Realtime, as its streaming generate runs it (the repository's README): the voice's prompt caches for
   * the text LM and the positive and negative TTS LM; then, while text is left, its next 5 tokens through the text LM
   * and the TTS LM, followed by 6 latents, each from the diffusion head (guidance 1.5 against the negative cache),
   * decoded to audio and fed back through the connector into both TTS-LM caches, until the EOS head passes 0.5.
   */
  private async vibevoice(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'vibevoice' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    let tokenizer = loaded.cache.get('tokenizer') as BpeTokenizer | undefined;
    if (!tokenizer) {
      tokenizer = BpeTokenizer.fromJSON(json(loaded, pipeline.tokenizer));
      loaded.cache.set('tokenizer', tokenizer);
    }
    if (!request.text.trim()) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const ids = tokenizer.encode(`${request.text.trim()}\n`);
    const voice = loaded.model.voices.find((v) => v.id === request.settings.voice) ?? loaded.model.voices[0]!;
    const data = binary(loaded, voice.file!);
    const index = json(loaded, pipeline.voiceIndexes[voice.id]!) as Record<string, { offset: number; shape: number[] }>;
    const array = (name: string) => {
      const { offset, shape } = index[name]!;
      return { values: data.subarray(offset, offset + shape.reduce((a, b) => a * b, 1)), shape };
    };
    const cache = (prefix: string, layers: number, half: boolean): Record<string, Tensor> =>
      Object.fromEntries(
        kvNames(layers, 'past_key_values').map((name, i) => {
          const { values, shape } = array(`${prefix}.${i >> 1}.${i % 2 ? 'value' : 'key'}`);
          return [name, half ? new ort.Tensor('float16', toHalf(values), shape) : new ort.Tensor('float32', values.slice(), shape)];
        }),
      );
    const session = (role: string) => loaded.sessions.get(role)!;
    const [textLm, ttsLm, diffusion, connector, eos, decoder] = [session('textLm'), session('ttsLm'), session('diffusion'), session('connector'), session('eos'), session('decoder')];
    const presentText = kvNames(VIBE_TEXT_LAYERS, 'present');
    const presentTts = kvNames(VIBE_TTS_LAYERS, 'present');
    const pastText = kvNames(VIBE_TEXT_LAYERS, 'past_key_values');
    const pastTts = kvNames(VIBE_TTS_LAYERS, 'past_key_values');
    const dispose = (tensors: Record<string, Tensor>) => Object.values(tensors).forEach((t) => t.dispose());

    let textPast = cache('lm', VIBE_TEXT_LAYERS, false);
    // Two caches on the TTS LM: the positive one (text and speech) and the negative one (speech only, for guidance).
    const tts = { past: cache('tts_lm', VIBE_TTS_LAYERS, true), length: index['tts_lm.0.key']!.shape[2]! };
    const neg = { past: cache('neg_tts_lm', VIBE_TTS_LAYERS, true), length: index['neg_tts_lm.0.key']!.shape[2]! };
    const stateShapes = (json(loaded, pipeline.states) as { states: number[][] }).states;
    let states: Record<string, Tensor> = Object.fromEntries(
      stateShapes.map((shape, i) => [`state_${i}`, new ort.Tensor('float32', new Float32Array(shape.reduce((a, b) => a * b, 1)), shape)]),
    );
    /** Runs float16 embeddings [1, S, 896] through the TTS LM on one cache; returns the last hidden state (float32). */
    const step = async (lm: { past: Record<string, Tensor>; length: number }, embeds: Uint16Array<ArrayBuffer>) => {
      const s = embeds.length / VIBE_WIDTH;
      const out = await ttsLm.run({
        inputs_embeds: new ort.Tensor('float16', embeds, [1, s, VIBE_WIDTH]),
        attention_mask: new ort.Tensor('int64', new BigInt64Array(lm.length + s).fill(1n), [1, lm.length + s]),
        ...lm.past,
      });
      dispose(lm.past);
      lm.past = Object.fromEntries(presentTts.map((n, i) => [pastTts[i]!, out[n]!]));
      lm.length += s;
      const hidden = out.hidden_states!.data as Uint16Array;
      const last = fromHalf(hidden.subarray(hidden.length - VIBE_WIDTH));
      out.hidden_states!.dispose();
      return last;
    };
    let negHidden = array('neg_tts_lm.hidden').values.slice();
    let hidden: Float32Array | null = null;
    let window = 0;
    let done = false;
    const chunks = Math.ceil(ids.length / VIBE_TEXT_WINDOW) || 1;
    try {
      while (!done && tts.length < VIBE_MAX_POSITIONS) {
        this.throwIfCancelled();
        const w = ids.slice(window * VIBE_TEXT_WINDOW, (window + 1) * VIBE_TEXT_WINDOW);
        window++;
        if (w.length) {
          const out = await textLm.run({ input_ids: new ort.Tensor('int64', BigInt64Array.from(w, BigInt), [1, w.length]), ...textPast });
          dispose(textPast);
          textPast = Object.fromEntries(presentText.map((n, i) => [pastText[i]!, out[n]!]));
          const textHidden = out.hidden!.data as Float32Array;
          hidden = await step(tts, toHalf(textHidden));
          out.hidden!.dispose();
        }
        for (let k = 0; k < VIBE_SPEECH_WINDOW; k++) {
          this.throwIfCancelled();
          const latent = (
            await diffusion.run({
              condition: new ort.Tensor('float32', hidden!, [1, VIBE_WIDTH]),
              neg_condition: new ort.Tensor('float32', negHidden, [1, VIBE_WIDTH]),
              noise: new ort.Tensor('float32', gaussian(64, (Math.random() * 2 ** 31) | 0), [1, 64]),
              cfg_scale: new ort.Tensor('float32', new Float32Array([VIBE_CFG]), [1]),
            })
          ).latent!;
          const latentData = new Float32Array(latent.data as Float32Array);
          latent.dispose();
          const decoded = await decoder.run({ latent: new ort.Tensor('float32', latentData, [1, 1, 64]), ...states });
          dispose(states);
          states = Object.fromEntries(stateShapes.map((_, i) => [`state_${i}`, decoded[`new_state_${i}`]!]));
          emit(new Float32Array(decoded.audio!.data as Float32Array), Math.min(window, chunks), chunks);
          decoded.audio!.dispose();
          const embeds = (await connector.run({ latent: new ort.Tensor('float32', latentData, [1, 1, 64]) })).embeds!;
          const half = toHalf(embeds.data as Float32Array);
          embeds.dispose();
          hidden = await step(tts, half);
          negHidden = await step(neg, half);
          const stop = (await eos.run({ hidden: new ort.Tensor('float32', hidden, [1, VIBE_WIDTH]) })).probability!;
          const p = (stop.data as Float32Array)[0]!;
          stop.dispose();
          if (p > 0.5) {
            done = true;
            break;
          }
        }
      }
    } finally {
      dispose(textPast);
      dispose(tts.past);
      dispose(neg.past);
      dispose(states);
    }
  }

  /**
   * Fun-CosyVoice 3, as upstream's zero-shot inference runs it (the repository's README): per chunk of up to 80 text
   * tokens, the language model reads [SOS, prompt text, text, TASK, prompt speech tokens] and samples speech tokens
   * (repetition-aware sampling) until a stop token; the flow encoder and ten flow steps from the fixed noise make the
   * mel of prompt and speech, the prompt part is dropped, and HiFT voices the rest with uniform noise.
   */
  private async cosyvoice(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'cosyvoice' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    let tokenizer = loaded.cache.get('tokenizer') as BpeTokenizer | undefined;
    if (!tokenizer) {
      tokenizer = BpeTokenizer.fromJSON(json(loaded, pipeline.tokenizer));
      loaded.cache.set('tokenizer', tokenizer);
    }
    const text = normalizeCosyEnglish(request.text);
    const chunks = text ? splitParagraph(text, (t) => tokenizer.encode(t).length) : [];
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const voice = json(loaded, pipeline.voice) as {
      prompt_text_ids: number[];
      prompt_speech_tokens: number[];
      embedding: number[];
      prompt_feat: number[];
      prompt_feat_shape: [number, number];
    };
    // The noise file is float16, read (as every .bin) into a Float32Array's bytes.
    const noiseBytes = binary(loaded, pipeline.flowNoise);
    const flowNoise = fromHalf(new Uint16Array(noiseBytes.buffer, noiseBytes.byteOffset, noiseBytes.byteLength / 2));
    const session = (role: string) => loaded.sessions.get(role)!;
    const [embed, llm, flowEncoder, flowStep, hift] = [session('embed'), session('llm'), session('flowEncoder'), session('flowStep'), session('hift')];
    const pastNames = kvNames(pipeline.layers, 'past_key_values');
    const presentNames = kvNames(pipeline.layers, 'present');
    const embedIds = async (ids: number[]) => (await embed.run({ ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]) })).embeds!;
    const promptFrames = voice.prompt_feat_shape[0];

    for (const [c, chunk] of chunks.entries()) {
      this.throwIfCancelled();
      // 1. Speech tokens.
      const textIds = tokenizer.encode(chunk);
      const ids = [COSY_TEXT_VOCAB + COSY_SOS, ...voice.prompt_text_ids, ...textIds, COSY_TEXT_VOCAB + COSY_TASK, ...voice.prompt_speech_tokens.map((s) => COSY_TEXT_VOCAB + s)];
      let embeds = await embedIds(ids);
      let past: Record<string, Tensor> = Object.fromEntries(pastNames.map((n) => [n, new ort.Tensor('float32', new Float32Array(0), [1, 2, 0, 64])]));
      const decoded: number[] = [];
      const kept: number[] = [];
      let silent = 0;
      let total = 0;
      try {
        for (let i = 0; i < textIds.length * 20; i++) {
          this.throwIfCancelled();
          total += embeds.dims[1]!;
          const out = await llm.run({ inputs_embeds: embeds, attention_mask: new ort.Tensor('int64', new BigInt64Array(total).fill(1n), [1, total]), ...past });
          embeds.dispose();
          for (const t of Object.values(past)) t.dispose();
          past = Object.fromEntries(presentNames.map((n, k) => [pastNames[k]!, out[n]!]));
          const logits = out.logits!.data as Float32Array;
          const logp = logSoftmax(logits.subarray(logits.length - COSY_LOGITS));
          out.logits!.dispose();
          // Upstream bans index 6561 (not the stop token 6562) before the minimum length.
          if (i < textIds.length * 2) logp[COSY_SOS] = -Infinity;
          const token = rasSample(logp, decoded);
          if (token >= COSY_SOS) break;
          decoded.push(token);
          silent = COSY_SILENT.has(token) ? silent + 1 : 0;
          // Upstream drops a silent token after five in a row (the model still sees it).
          if (silent <= 5) kept.push(token);
          embeds = await embedIds([COSY_TEXT_VOCAB + token]);
        }
      } finally {
        embeds.dispose();
        for (const t of Object.values(past)) t.dispose();
      }
      if (kept.length === 0) continue;

      // 2. Mel: flow matching over prompt + speech, then the prompt's frames dropped.
      const tokens = [...voice.prompt_speech_tokens, ...kept];
      const enc = await flowEncoder.run({
        tokens: new ort.Tensor('int64', BigInt64Array.from(tokens, BigInt), [1, tokens.length]),
        embedding: new ort.Tensor('float32', Float32Array.from(voice.embedding), [1, voice.embedding.length]),
        prompt_feat: new ort.Tensor('float32', Float32Array.from(voice.prompt_feat), [1, promptFrames, 80]),
      });
      const frames = enc.mu!.dims[2]!;
      const start = new Float32Array(80 * frames);
      const random = frames > COSY_NOISE_FRAMES ? gaussian(start.length, (Math.random() * 2 ** 31) | 0) : null;
      for (let ch = 0; ch < 80; ch++) for (let f = 0; f < frames; f++) start[ch * frames + f] = random ? random[ch * frames + f]! : flowNoise[ch * COSY_NOISE_FRAMES + f]!;
      let x: Tensor = new ort.Tensor('float32', start, [1, 80, frames]);
      const span = Array.from({ length: COSY_FLOW_STEPS + 1 }, (_, i) => Math.fround(1 - Math.cos(Math.fround(Math.fround(i / COSY_FLOW_STEPS) * Math.fround(0.5 * Math.PI)))));
      let t = span[0]!;
      let dt = Math.fround(span[1]! - t);
      for (let step = 1; step <= COSY_FLOW_STEPS; step++) {
        this.throwIfCancelled();
        const next = (
          await flowStep.run({
            x,
            mu: enc.mu!,
            spks: enc.spks!,
            cond: enc.cond!,
            t: new ort.Tensor('float32', new Float32Array([t]), [1]),
            dt: new ort.Tensor('float32', new Float32Array([dt]), [1]),
          })
        ).x_next!;
        x.dispose();
        x = next;
        t = Math.fround(t + dt);
        if (step < COSY_FLOW_STEPS) dt = Math.fround(span[step + 1]! - t);
      }
      for (const v of Object.values(enc)) v.dispose();
      const full = (await x.getData()) as Float32Array;
      x.dispose();
      const melFrames = frames - promptFrames;
      const mel = new Float32Array(80 * melFrames);
      for (let ch = 0; ch < 80; ch++) mel.set(full.subarray(ch * frames + promptFrames, (ch + 1) * frames), ch * melFrames);

      // 3. Audio.
      const noise = Float32Array.from({ length: 480 * melFrames * 9 }, Math.random);
      const audio = (
        await hift.run({ mel: new ort.Tensor('float32', mel, [1, 80, melFrames]), noise: new ort.Tensor('float32', noise, [1, 480 * melFrames, 9]) })
      ).audio!;
      emit(new Float32Array(audio.data as Float32Array), c + 1, chunks.length);
      audio.dispose();
    }
  }

  /**
   * Kokoro, as its KPipeline runs it: per sentence, Misaki phonemes' ids between two 0s (Paradee's front end, which
   * Paradee took from Kokoro), the voice's style row for len(phonemes) - 1 and the speed → audio.
   */
  private async kokoro(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'kokoro' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    const vocab = (json(loaded, pipeline.tokenizer) as { model: { vocab: Record<string, number> } }).model.vocab;
    const chunks = g2p(loaded).paradee(request.text, vocab);
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const voice = loaded.model.voices.find((v) => v.id === request.settings.voice) ?? loaded.model.voices[0]!;
    const styles = binary(loaded, voice.file!);
    const styleSize = 256;
    const rows = styles.length / styleSize;
    const speed = Math.max(0.5, Math.min(2, request.settings.speed ?? 1));
    const session = loaded.sessions.get('model')!;
    for (const [c, chunk] of chunks.entries()) {
      this.throwIfCancelled();
      const row = Math.min(Math.max([...chunk.phonemes].length - 1, 0), rows - 1);
      const out = await session.run({
        input_ids: new ort.Tensor('int64', BigInt64Array.from(chunk.ids, BigInt), [1, chunk.ids.length]),
        style: new ort.Tensor('float32', styles.slice(row * styleSize, (row + 1) * styleSize), [1, styleSize]),
        speed: new ort.Tensor('float32', new Float32Array([speed]), [1]),
      });
      const waveform = out[session.outputNames[0]!]!;
      emit(new Float32Array(waveform.data as Float32Array), c + 1, chunks.length);
      for (const t of Object.values(out)) t.dispose();
    }
  }

  /**
   * MOSS-TTS-Nano, as its onnx_tts_runtime.py runs it by default ("fixed" sampling, streaming decode): the text
   * normalized and split into chunks of 75 tokens; per chunk, the prompt rows (template, the voice's codes, the text)
   * through the prefill, then per frame the sampled-frame graph (with uniform random numbers and the codes seen so
   * far, for its repetition penalty) until it says stop, each frame fed back through the decode step. Frames go to
   * the codec's streaming decoder a few at a time (more once playback is ahead), and a pause follows each chunk.
   */
  private async moss(
    loaded: Loaded,
    pipeline: Extract<SpeechPipeline, { kind: 'moss' }>,
    request: SpeechGenerationRequest,
    emit: (audio: Float32Array, chunk: number, chunks: number) => void,
  ): Promise<void> {
    const ort = getOrt();
    const config = json(loaded, pipeline.config) as MossConfig;
    let tokenizer = loaded.cache.get('tokenizer') as BpeTokenizer | undefined;
    if (!tokenizer) {
      tokenizer = BpeTokenizer.fromJSON(json(loaded, pipeline.tokenizer));
      loaded.cache.set('tokenizer', tokenizer);
    }
    const encode = (text: string) => tokenizer.encode(sentencePieceText(text));
    const text = normalizeTtsText(request.text);
    const chunks = text ? splitText(text, (t) => encode(t).length) : [];
    if (chunks.length === 0) throw new RuntimeError('generation-failed', 'There is no text to speak.');
    const voice = config.voices[request.settings.voice] ?? config.voices[loaded.model.defaults.voice]!;
    const { tts, templates } = config;
    const width = tts.n_vq + 1;
    const vocab = tts.audio_codebook_sizes[0]!;
    const prefill = loaded.sessions.get('prefill')!;
    const decodeStep = loaded.sessions.get('decodeStep')!;
    const sampleFrame = loaded.sessions.get('sampleFrame')!;
    const codec = loaded.sessions.get('codec')!;
    const present = mossKv(config, 'present');
    const past = mossKv(config, 'past');
    const codecState = mossCodecState(config);
    const rate = loaded.model.sampleRate;
    const dispose = (tensors: Record<string, Tensor>) => Object.values(tensors).forEach((t) => t.dispose());
    // A row: a text token and 16 audio codes (padding where there are none).
    const row = (token: number, codes: readonly number[] = []) => [token, ...Array.from({ length: tts.n_vq }, (_, i) => codes[i] ?? tts.audio_pad_token_id)];
    const lastHidden = (hidden: Tensor) => {
      const data = hidden.data as Float32Array;
      const out = new ort.Tensor('float32', data.slice(data.length - 768), [1, 768]);
      hidden.dispose();
      return out;
    };

    for (const [c, chunk] of chunks.entries()) {
      this.throwIfCancelled();
      const rows = [
        ...[...templates.user_prompt_prefix_token_ids, tts.audio_start_token_id].map((t) => row(t)),
        ...voice.codes.map((codes) => row(tts.audio_user_slot_token_id, codes)),
        ...[
          tts.audio_end_token_id,
          ...templates.user_prompt_after_reference_token_ids,
          ...encode(chunk),
          ...templates.assistant_prompt_prefix_token_ids,
          tts.audio_start_token_id,
        ].map((t) => row(t)),
      ];
      const prefilled = await prefill.run({
        input_ids: new ort.Tensor('int32', Int32Array.from(rows.flat()), [1, rows.length, width]),
        attention_mask: new ort.Tensor('int32', new Int32Array(rows.length).fill(1), [1, rows.length]),
      });
      let hidden = lastHidden(prefilled.global_hidden!);
      let cache: Record<string, Tensor> = Object.fromEntries(present.map((n, i) => [past[i]!, prefilled[n]!]));
      let state: Record<string, Tensor> = mossCodecStart(config);
      let length = rows.length;
      const seen = new Int32Array(tts.n_vq * vocab);
      const pending: number[][] = [];
      let emitted = 0;
      let firstAt = 0;
      const decodePending = async (force: boolean) => {
        if (pending.length === 0) return;
        // The first frame alone (the first audio soonest), then 8 at a time: each call costs about the same however
        // many frames it takes, and the decoder's state makes the audio the same either way.
        const budget = firstAt ? 8 : 1;
        if (!force && pending.length < budget) return;
        const frames = pending.splice(0, force ? pending.length : budget);
        const out = await codec.run({
          audio_codes: new ort.Tensor('int32', Int32Array.from(frames.flat()), [1, frames.length, tts.n_vq]),
          audio_code_lengths: new ort.Tensor('int32', new Int32Array([frames.length]), [1]),
          ...state,
        });
        dispose(state);
        state = Object.fromEntries(codecState.map(([input, output]) => [input, out[output]!]));
        const audio = out.audio!.data as Float32Array;
        const samples = (out.audio_lengths!.data as Int32Array)[0]!;
        const channels = out.audio!.dims[1]!;
        const stride = out.audio!.dims[2]!;
        const mono = new Float32Array(samples);
        for (let ch = 0; ch < channels; ch++) for (let i = 0; i < samples; i++) mono[i]! += audio[ch * stride + i]! / channels;
        out.audio!.dispose();
        out.audio_lengths!.dispose();
        if (samples > 0) {
          if (!firstAt) firstAt = performance.now();
          emitted += samples;
          emit(mono, c + 1, chunks.length);
        }
      };
      try {
        for (let step = 0; step < config.generation.max_new_frames; step++) {
          this.throwIfCancelled();
          const sampled = await sampleFrame.run({
            global_hidden: hidden,
            repetition_seen_mask: new ort.Tensor('int32', seen, [1, tts.n_vq, vocab]),
            assistant_random_u: new ort.Tensor('float32', new Float32Array([uniform()]), [1]),
            audio_random_u: new ort.Tensor('float32', Float32Array.from({ length: tts.n_vq }, uniform), [1, tts.n_vq]),
          });
          const go = (sampled.should_continue!.data as Int32Array)[0]!;
          const frame = Array.from(sampled.frame_token_ids!.data as Int32Array);
          dispose(sampled);
          if (!go) break;
          frame.forEach((t, ch) => {
            if (t >= 0 && t < vocab) seen[ch * vocab + t] = 1;
          });
          pending.push(frame);
          await decodePending(false);
          const next = await decodeStep.run({
            input_ids: new ort.Tensor('int32', Int32Array.from(row(tts.audio_assistant_slot_token_id, frame)), [1, 1, width]),
            past_valid_lengths: new ort.Tensor('int32', new Int32Array([length]), [1]),
            ...cache,
          });
          length++;
          hidden.dispose();
          dispose(cache);
          hidden = lastHidden(next.global_hidden!);
          cache = Object.fromEntries(present.map((n, i) => [past[i]!, next[n]!]));
        }
        await decodePending(true);
      } finally {
        hidden.dispose();
        dispose(cache);
        dispose(state);
      }
      if (c < chunks.length - 1) emit(new Float32Array(Math.round(rate * chunkPause(chunk))), c + 1, chunks.length);
    }
  }
}

interface MossConfig {
  tts: {
    n_vq: number;
    audio_pad_token_id: number;
    audio_start_token_id: number;
    audio_end_token_id: number;
    audio_user_slot_token_id: number;
    audio_assistant_slot_token_id: number;
    audio_codebook_sizes: number[];
  };
  templates: { user_prompt_prefix_token_ids: number[]; user_prompt_after_reference_token_ids: number[]; assistant_prompt_prefix_token_ids: number[] };
  generation: { max_new_frames: number };
  global_layers: number;
  codec: {
    transformer_offsets: { input_name: string; output_name: string; shape: number[] }[];
    attention_caches: {
      offset_input_name: string;
      offset_output_name: string;
      cached_keys_input_name: string;
      cached_keys_output_name: string;
      cached_values_input_name: string;
      cached_values_output_name: string;
      cached_positions_input_name: string;
      cached_positions_output_name: string;
      offset_shape: number[];
      cache_shape: number[];
      positions_shape: number[];
    }[];
  };
  voices: Record<string, { codes: number[][] }>;
}

/** MOSS-TTS-Nano's global transformer cache names: past_key_0, past_value_0, … (or present_…). */
function mossKv(config: MossConfig, prefix: 'past' | 'present'): string[] {
  return Array.from({ length: config.global_layers }, (_, i) => [`${prefix}_key_${i}`, `${prefix}_value_${i}`]).flat();
}

/** The codec decoder's streaming state, as [input, output] name pairs. */
function mossCodecState(config: MossConfig): [string, string][] {
  return [
    ...config.codec.transformer_offsets.map((s): [string, string] => [s.input_name, s.output_name]),
    ...config.codec.attention_caches.flatMap((s): [string, string][] => [
      [s.offset_input_name, s.offset_output_name],
      [s.cached_keys_input_name, s.cached_keys_output_name],
      [s.cached_values_input_name, s.cached_values_output_name],
      [s.cached_positions_input_name, s.cached_positions_output_name],
    ]),
  ];
}

/** The codec decoder's state before the first frame: zero offsets and caches, no cached positions (-1). */
function mossCodecStart(config: MossConfig): Record<string, Tensor> {
  const ort = getOrt();
  const size = (shape: number[]) => shape.reduce((a, b) => a * b, 1);
  const state: Record<string, Tensor> = {};
  for (const s of config.codec.transformer_offsets) state[s.input_name] = new ort.Tensor('int32', new Int32Array(size(s.shape)), s.shape);
  for (const s of config.codec.attention_caches) {
    state[s.offset_input_name] = new ort.Tensor('int32', new Int32Array(size(s.offset_shape)), s.offset_shape);
    state[s.cached_keys_input_name] = new ort.Tensor('float32', new Float32Array(size(s.cache_shape)), s.cache_shape);
    state[s.cached_values_input_name] = new ort.Tensor('float32', new Float32Array(size(s.cache_shape)), s.cache_shape);
    state[s.cached_positions_input_name] = new ort.Tensor('int32', new Int32Array(size(s.positions_shape)).fill(-1), s.positions_shape);
  }
  return state;
}

/** A uniform random number in [0, 1) that float32 keeps below 1 (as the original clamps its draws). */
function uniform(): number {
  return Math.min(0.99999994, Math.random());
}

/** Chatterbox's speech-token ids: start and stop of speech, and silence (what the decoder ends each sentence with). */
const CHATTERBOX_START = 6561;
const CHATTERBOX_STOP = 6562;
const CHATTERBOX_SILENCE = 4299;
const CHATTERBOX_WIDTH = 1024;
/**
 * Speech tokens per sentence at most, as the ONNX examples' max_new_tokens: Turbo's 1024 (about 40 s), the others'
 * 256 (about 10 s), which also bounds a sentence whose greedy tokens never reach the stop token; their sentences are
 * split shorter (characters) to fit.
 */
const CHATTERBOX_LIMITS = { turbo: { tokens: 1024, characters: 300 }, other: { tokens: 256, characters: 150 } };
/** The original Chatterbox's emotion exaggeration, at its default. */
const CHATTERBOX_EXAGGERATION = 0.5;

/** A voice file of tools/export/chatterbox_voice.py: N, P, F, then the four arrays it describes. */
function chatterboxVoice(data: Float32Array) {
  const [n, p, f] = [data[0]!, data[1]!, data[2]!];
  let at = 3;
  const take = (count: number) => data.subarray(at, (at += count));
  return {
    conditioning: take(n * CHATTERBOX_WIDTH),
    promptTokens: Array.from(take(p)),
    speakerEmbedding: take(192),
    speakerFeatures: take(f * 80),
  };
}

/** CSM-1B: backbone and depth-decoder widths, codebooks, codes per codebook (with control ids), position limit. */
const CSM_WIDTH = 2048;
const CSM_DEPTH_WIDTH = 1024;
const CSM_CODEBOOKS = 32;
const CSM_VOCAB = 2051;
const CSM_MAX_POSITIONS = 2048;

/** NeuTTS Air: layers, KV heads and head size of its Qwen2 0.5B; its vocabulary (logits per position) and codes. */
const NEUTTS_LAYERS = 24;
const NEUTTS_KV_HEADS = 2;
const NEUTTS_HEAD_DIM = 64;
const NEUTTS_VOCAB = 217652;
const NEUTTS_CODES = 65536;
/** Its limits (NeuTTS's): positions in all, and new tokens before the end may come. */
const NEUTTS_MAX_POSITIONS = 2048;
const NEUTTS_MIN_TOKENS = 50;
/** Prompt tokens per prefill run, and characters of text per prompt (about 20 s of speech, within the positions). */
const NEUTTS_PREFILL = 128;
const NEUTTS_CHUNK_CHARACTERS = 250;

/** Orpheus 3B: layers, KV heads and head size of its Llama 3.2 3B; sampling (canopyai's engine) and limits. */
const ORPHEUS_LAYERS = 28;
const ORPHEUS_KV_HEADS = 8;
const ORPHEUS_HEAD_DIM = 128;
const ORPHEUS_SAMPLING = {
  orpheus: { temperature: 0.6, topP: 0.8, repetition: 1.3, minTokens: 0 },
  maya1: { temperature: 0.4, topP: 0.9, repetition: 1.1, minTokens: 28 },
};
const ORPHEUS_END_OF_SPEECH = 128258;
const ORPHEUS_MAX_NEW = 1200;
/** SNAC's start-up samples, dropped; prompt tokens per prefill run; characters of text per prompt (1200 tokens ≈ 14 s). */
const ORPHEUS_SKIP_SAMPLES = 2048;
const ORPHEUS_PREFILL = 128;
const ORPHEUS_CHUNK_CHARACTERS = 180;

/** VoxCPM2: LM layers, KV heads and head size, hidden width, latent channels and frames per patch, the audio-start id. */
const VOX_BASE_LAYERS = 28;
const VOX_RESIDUAL_LAYERS = 8;
const VOX_KV_HEADS = 2;
const VOX_HEAD_DIM = 128;
const VOX_WIDTH = 2048;
const VOX_LATENT = 64;
const VOX_PATCH = 4;
const VOX_AUDIO_START = 101;
const VOX_MAX_PATCHES = 2000;
/** Its sampler: classifier-free guidance, and 10 timesteps on the sway schedule t = s + cos(πs/2) − 1 + s, s from 1 to 0. */
const VOX_CFG = 2;
const VOX_STEPS = 10;
const VOX_T = Array.from({ length: VOX_STEPS + 1 }, (_, i) => {
  const t = Math.fround(1 - i / VOX_STEPS);
  return Math.fround(t + (Math.cos((Math.PI / 2) * t) - 1 + t));
});

/** Rows of embeddings summed element by element. */
function add(...rows: Float32Array[]): Float32Array {
  return rows.reduce((sum, r) => sum.map((v, i) => v + r[i]!));
}

/**
 * A GenAI-builder decoder graph (inputs_embeds in, logits and hidden_states out) with its own cache, float16 or
 * (`float32`) float32, starting empty: each run takes rows of embeddings and returns the last row's logits and hidden
 * state as float32.
 */
function builderDecoder(graph: InferenceSession, shape: { layers: number; kvHeads: number; headDim: number; width: number; float32?: boolean }) {
  const ort = getOrt();
  const { layers, kvHeads, headDim, width, float32 = false } = shape;
  const pastNames = kvNames(layers, 'past_key_values');
  const presentNames = kvNames(layers, 'present');
  const empty = () => (float32 ? new ort.Tensor('float32', new Float32Array(0), [1, kvHeads, 0, headDim]) : new ort.Tensor('float16', new Uint16Array(0), [1, kvHeads, 0, headDim]));
  const floats = (t: Tensor, from: number) => (float32 ? (t.data as Float32Array).slice(from) : fromHalf((t.data as Uint16Array).subarray(from)));
  let past: Record<string, Tensor> = Object.fromEntries(pastNames.map((n) => [n, empty()]));
  let length = 0;
  return {
    get length() {
      return length;
    },
    async run(rows: Float32Array[]) {
      const embeds = new Float32Array(rows.length * width);
      rows.forEach((r, i) => embeds.set(r, i * width));
      const dims = [1, rows.length, width];
      const out = await graph.run({
        inputs_embeds: float32 ? new ort.Tensor('float32', embeds, dims) : new ort.Tensor('float16', toHalf(embeds), dims),
        attention_mask: new ort.Tensor('int64', new BigInt64Array(length + rows.length).fill(1n), [1, length + rows.length]),
        ...past,
      });
      for (const t of Object.values(past)) t.dispose();
      past = Object.fromEntries(presentNames.map((n, i) => [pastNames[i]!, out[n]!]));
      length += rows.length;
      const n = out.logits!.size / rows.length;
      const h = out.hidden_states!.size / rows.length;
      const result = { logits: floats(out.logits!, out.logits!.size - n), hidden: floats(out.hidden_states!, out.hidden_states!.size - h) };
      out.logits!.dispose();
      out.hidden_states!.dispose();
      return result;
    },
    dispose: () => Object.values(past).forEach((t) => t.dispose()),
  };
}

/** Qwen3-TTS 12 Hz: the code predictor's width, the talker's and code predictor's layers, their KV heads and head size. */
const QWEN_CP_WIDTH = 1024;
const QWEN_TALKER_LAYERS = 28;
const QWEN_CP_LAYERS = 5;
const QWEN_KV_HEADS = 8;
const QWEN_HEAD_DIM = 128;
/** Frames at most per run (the checkpoints' max_new_tokens), samples per frame, and the decoder's chunking. */
const QWEN_MAX_FRAMES = 8192;
const QWEN_FRAME_SAMPLES = 1920;
const QWEN_DECODE_CHUNK = 300;
const QWEN_DECODE_CONTEXT = 25;

interface QwenConfig {
  tts_bos_token_id: number;
  tts_eos_token_id: number;
  tts_pad_token_id: number;
  talker_config: {
    hidden_size: number;
    vocab_size: number;
    num_code_groups: number;
    code_predictor_config: { vocab_size: number };
    codec_eos_token_id: number;
    codec_think_id: number;
    codec_think_bos_id: number;
    codec_think_eos_id: number;
    codec_pad_id: number;
    codec_bos_id: number;
    codec_language_id: Record<string, number>;
    spk_id: Record<string, number>;
    spk_is_dialect?: Record<string, string | false>;
  };
}

/** Indices of the `k` largest finite values, largest first (one pass, a small sorted buffer: vocabularies are large). */
function topIndices(values: Float32Array, k: number): number[] {
  const order: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (v === -Infinity || (order.length === k && v <= values[order[k - 1]!]!)) continue;
    let at = order.length === k ? k - 1 : order.length;
    while (at > 0 && values[order[at - 1]!]! < v) at--;
    order.splice(at, 0, i);
    if (order.length > k) order.pop();
  }
  return order;
}

/** Sampling at temperature 0.9 from the 50 most likely (the checkpoints' generation_config). */
function sampleTopK(logits: Float32Array, temperature = 0.9, k = 50): number {
  const order = topIndices(logits, k);
  const max = logits[order[0]!]! / temperature;
  const weights = order.map((i) => Math.exp(logits[i]! / temperature - max));
  let u = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < order.length; i++) if ((u -= weights[i]!) < 0) return order[i]!;
  return order[order.length - 1]!;
}

/**
 * Nucleus sampling: the most likely tokens (at `temperature`) whose probabilities, largest first, start below `topP`
 * of the whole, drawn in proportion.
 */
function sampleTopP(logits: Float32Array, temperature: number, topP: number): number {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let total = 0;
  for (const v of logits) total += Math.exp((v - max) / temperature);
  // The nucleus is almost always a few tokens: look among the top 256 first, and sort everything only if needed.
  let order = topIndices(logits, 256);
  const mass = (i: number) => Math.exp((logits[i]! - max) / temperature) / total;
  const nucleus = (candidates: number[]) => {
    const picked: number[] = [];
    let cum = 0;
    for (const i of candidates) {
      if (cum >= topP) return picked;
      cum += mass(i);
      picked.push(i);
    }
    return null;
  };
  let picked = nucleus(order);
  if (!picked) {
    order = Array.from(logits.keys()).sort((a, b) => logits[b]! - logits[a]!);
    picked = nucleus(order) ?? order;
  }
  const weights = picked.map(mass);
  let u = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let k = 0; k < picked.length; k++) if ((u -= weights[k]!) < 0) return picked[k]!;
  return picked[picked.length - 1]!;
}

/** VibeVoice Realtime: layers of its text LM and TTS LM, hidden width, text tokens and latents per window, guidance. */
const VIBE_TEXT_LAYERS = 4;
const VIBE_TTS_LAYERS = 20;
const VIBE_WIDTH = 896;
const VIBE_TEXT_WINDOW = 5;
const VIBE_SPEECH_WINDOW = 6;
const VIBE_CFG = 1.5;
/** The TTS LM's position cap (the original's). */
const VIBE_MAX_POSITIONS = 8192;

/** CosyVoice 3's ids: the text vocabulary's size (speech token s is embedded as it + s), start, task and logits size. */
const COSY_TEXT_VOCAB = 151936;
const COSY_SOS = 6561;
const COSY_TASK = 6563;
const COSY_LOGITS = 6761;
/** Speech tokens upstream treats as silence (dropped after five in a row). */
const COSY_SILENT = new Set([1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]);
const COSY_FLOW_STEPS = 10;
/** Frames of the fixed flow noise (longer mels start from fresh Gaussian noise). */
const COSY_NOISE_FRAMES = 3000;

function logSoftmax(logits: Float32Array): Float64Array {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  let sum = 0;
  for (const v of logits) sum += Math.exp(v - max);
  const log = Math.log(sum);
  return Float64Array.from(logits, (v) => v - max - log);
}

/** One draw from log-probabilities, by inverse CDF. */
function drawFrom(logp: Float64Array, indices: number[]): number {
  const max = Math.max(...indices.map((i) => logp[i]!));
  const weights = indices.map((i) => Math.exp(logp[i]! - max));
  let u = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let k = 0; k < indices.length; k++) if ((u -= weights[k]!) < 0) return indices[k]!;
  return indices[indices.length - 1]!;
}

/**
 * CosyVoice's repetition-aware sampling (cosyvoice.utils.common.ras_sampling): nucleus sampling (top-p 0.8 within the
 * top 25); a token already in the last 10 decoded is banned and one is drawn from the whole distribution instead.
 */
function rasSample(logp: Float64Array, decoded: number[]): number {
  const order = Array.from(logp.keys()).sort((a, b) => logp[b]! - logp[a]! || a - b);
  const max = logp[order[0]!]!;
  let total = 0;
  for (const v of logp) total += Math.exp(v - max);
  const picked: number[] = [];
  let cum = 0;
  for (const i of order) {
    if (cum >= 0.8 || picked.length >= 25) break;
    cum += Math.exp(logp[i]! - max) / total;
    picked.push(i);
  }
  const top = drawFrom(logp, picked);
  if (decoded.slice(-10).filter((t) => t === top).length >= 1) {
    logp[top] = -Infinity;
    return drawFrom(logp, Array.from(logp.keys()).filter((i) => logp[i]! > -Infinity));
  }
  return top;
}

/** The best token after a repetition penalty (Hugging Face's: positive logits divided, negative ones multiplied). */
function argmaxWithPenalty(logits: Float32Array, seen: Set<number>): number {
  let best = 0;
  let bestScore = -Infinity;
  for (let t = 0; t < logits.length; t++) {
    let score = logits[t]!;
    if (seen.has(t)) score = score > 0 ? score / REPETITION_PENALTY : score * REPETITION_PENALTY;
    if (score > bestScore) {
      bestScore = score;
      best = t;
    }
  }
  return best;
}

/** The model's phonemiser (built from its lexicon when it loaded). */
function g2p(loaded: Loaded): EnglishG2P {
  return loaded.cache.get('g2p') as EnglishG2P;
}

/** Kitten TTS trims this much from the end of each chunk's audio (its Python package does). */
const KITTEN_TRIM = 5000;
/** Inflect's latent noise scale ("variation") at its default, and the fade at each chunk's edges (5 ms at 24 kHz). */
const INFLECT_VARIATION = 0.667;
const INFLECT_FADE = 120;

/** Inflect's pause after a chunk, by how it ends (seconds; its inference_onnx.py boundary_pause_seconds). */
function inflectPause(chunk: string): number {
  return ({ '?': 0.28, '!': 0.24, '.': 0.22, ';': 0.16, ':': 0.13, ',': 0.09 } as Record<string, number>)[chunk.trimEnd().slice(-1)] ?? 0.08;
}

interface PhononConfig {
  eos_threshold: number;
  /** Pocket TTS: frames before an end can count, and samples faded in at each chunk's start. */
  min_frames_before_eos?: number;
  fade_in_samples?: number;
  temperature: number;
  ldim: number;
  d_model: number;
  num_layers: number;
  num_heads: number;
  head_dim: number;
  mimi_decoder: { state: { name: string; shape: number[] }[] };
}

/** The flow LM's cache names: past_key_0, past_value_0, … (or present_…). */
function phononKv(config: PhononConfig, prefix: 'past' | 'present'): string[] {
  return Array.from({ length: config.num_layers }, (_, i) => [`${prefix}_key_${i}`, `${prefix}_value_${i}`]).flat();
}

/** A little-endian float32 file, as read while loading. */
function binary(loaded: Loaded, file: ArtifactFile): Float32Array {
  return loaded.cache.get(file.path) as Float32Array;
}

/** Soprano's limit per sentence (its Python package's max_new_tokens) and repetition penalty. */
const MAX_SOPRANO_TOKENS = 512;
const REPETITION_PENALTY = 1.2;

function floats(tensor: Tensor): Float32Array {
  return tensor.type === 'float16' ? fromHalf(tensor.data as Uint16Array) : (tensor.data as Float32Array);
}

/** `prefix`.N.key and `prefix`.N.value for each layer (the builder's KV cache names). */
function kvNames(layers: number, prefix: 'present' | 'past_key_values'): string[] {
  return Array.from({ length: layers }, (_, i) => [`${prefix}.${i}.key`, `${prefix}.${i}.value`]).flat();
}

function json(loaded: Loaded, file: ArtifactFile): unknown {
  let value = loaded.json.get(file.path);
  if (value === undefined) {
    value = JSON.parse(new TextDecoder().decode(loaded.files.bytes(file)));
    loaded.json.set(file.path, value);
  }
  return value;
}

async function release(loaded: Loaded): Promise<void> {
  await Promise.all([...loaded.sessions.values()].map((s) => s.release().catch(() => {})));
  loaded.sessions.clear();
}

const OUT_OF_MEMORY = /out of memory|\bOOM\b|memory copy|allocat|GPUOutOfMemoryError/i;

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
