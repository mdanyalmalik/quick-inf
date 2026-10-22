// Text-to-speech models. Each names its pipeline (how src/runtime/speech/speech-engine.ts turns text into audio),
// its files and its voices. Our exports are made with tools/export/quantize_int4.py: int4 weights (int8 where 4 bits
// was audible), float32 activations; tools/export/.cache/tts-eval scores them against the originals.

import type { ModelLicense } from './image-models';
import { OWN_EXPORTS, type ArtifactFile, type Precision } from './models';

/** One ONNX graph and the weight files beside it (named in the graph by their file names). */
export interface SpeechGraph {
  graph: ArtifactFile;
  data?: ArtifactFile[];
}

export interface SpeechVoice {
  id: string;
  label: string;
  /** The voice's data (a style vector file), for models whose voices are files. */
  file?: ArtifactFile;
}

/** How a model makes speech (src/runtime/speech/speech-engine.ts runs each kind). */
export type SpeechPipeline =
  | {
      /**
       * Supertonic: characters (through a Unicode index) → duration predictor and text encoder → flow-matching steps
       * of the vector estimator over latent noise → vocoder. Voices are style vectors; 44.1 kHz.
       */
      kind: 'supertonic';
      config: ArtifactFile;
      unicodeIndexer: ArtifactFile;
      durationPredictor: SpeechGraph;
      textEncoder: SpeechGraph;
      vectorEstimator: SpeechGraph;
      vocoder: SpeechGraph;
      /** The language tag wrapped around the text (`<en>…</en>`). */
      language: string;
    }
  | {
      /**
       * Soprano: a small Qwen3 language model writes audio tokens for each sentence (greedy, with a repetition
       * penalty), and a Vocos decoder turns its last hidden states into 32 kHz audio. One voice.
       */
      kind: 'soprano';
      tokenizer: ArtifactFile;
      /** The language model (ONNX Runtime GenAI's builder: GroupQueryAttention, float16 cache), with hidden states. */
      backbone: SpeechGraph;
      decoder: SpeechGraph;
      layers: number;
      kvHeads: number;
      headDim: number;
      hiddenSize: number;
      eosToken: number;
      /** Audio samples per token. */
      tokenSamples: number;
    }
  | {
      /**
       * Phonon (after Kyutai's Pocket TTS): a 6-layer transformer, conditioned on a voice and the text, makes one Mimi
       * latent per 80 ms frame through a flow head, and Mimi's decoder turns each into 24 kHz audio as it comes.
       * Exported by tools/export/export_phonon.py; its README describes the graphs and this loop.
       */
      kind: 'phonon';
      /**
       * Pocket TTS, Kyutai's model Phonon was built on: the same graphs, with its own text chunking
       * (src/runtime/speech/pocket-text.ts), an end only from the sixth frame on, and a 5 ms fade-in per chunk.
       */
      variant?: 'pocket-tts';
      tokenizer: ArtifactFile;
      /** Constants and Mimi's state names and shapes (phonon.json). */
      config: ArtifactFile;
      /** The first frame's input embedding, float32 [1, 1, 512]. */
      bos: ArtifactFile;
      textEmbed: SpeechGraph;
      flowLm: SpeechGraph;
      mimiDecoder: SpeechGraph;
    }
  | {
      /**
       * Kitten TTS (StyleTTS 2 style): eSpeak-style IPA (our dictionary phonemiser, src/runtime/speech/g2p) → one
       * graph with a voice's style vector, picked by the chunk's text length, → 24 kHz audio.
       */
      kind: 'kitten';
      lexicon: ArtifactFile;
      model: SpeechGraph;
      /** Speed multipliers per voice, as the model's config.json gives them. */
      speedPriors?: Record<string, number>;
    }
  | {
      /** Inflect v2 (VITS): IPA → duration graph → decoder with latent noise → 24 kHz audio. One voice. */
      kind: 'inflect';
      lexicon: ArtifactFile;
      duration: SpeechGraph;
      decode: SpeechGraph;
    }
  | {
      /** Paradee (distilled from Kokoro): Misaki phonemes → one graph → 24 kHz audio. One voice. */
      kind: 'paradee';
      lexicon: ArtifactFile;
      /** config.json, whose `vocab` maps phonemes to ids. */
      config: ArtifactFile;
      model: SpeechGraph;
    }
  | {
      /**
       * Chatterbox (Resemble AI): a GPT-2-style language model, prefixed with a voice's conditioning, writes speech
       * tokens greedily (repetition penalty 1.2), and the conditional decoder turns them into 24 kHz audio in that
       * voice. Voices are the speech encoder's outputs for a reference clip, computed once
       * (tools/export/chatterbox_voice.py). Turbo reads GPT-2 tokens and decodes in one step; the original reads
       * characters and takes an exaggeration setting; the multilingual one also a language tag before the text.
       */
      kind: 'chatterbox';
      variant: 'turbo' | 'original' | 'multilingual';
      tokenizer: ArtifactFile;
      embedTokens: SpeechGraph;
      languageModel: SpeechGraph;
      decoder: SpeechGraph;
      layers: number;
      kvHeads: number;
      headDim: number;
    }
  | {
      /**
       * MOSS-TTS-Nano: a 12-layer global transformer over rows of a text token and 16 audio codes reads the voice's
       * prompt codes and the text; per 80 ms frame a small local transformer samples the frame's 16 codes (in its
       * graph, from random numbers given to it); the MOSS audio tokenizer's streaming decoder turns frames into
       * 48 kHz stereo (mixed to mono). Text: normalize_tts_text, chunks of 75 tokens.
       */
      kind: 'moss';
      /** tokenizer.json, converted from its SentencePiece model (tools/export/.cache/tts-eval/moss_tokenizer.py). */
      tokenizer: ArtifactFile;
      /** moss.json: prompt templates and ids, generation defaults, the codec's streaming state, the voices' codes. */
      config: ArtifactFile;
      prefill: SpeechGraph;
      decodeStep: SpeechGraph;
      sampleFrame: SpeechGraph;
      codec: SpeechGraph;
    }
  | {
      /**
       * Kokoro (StyleTTS 2 with an iSTFTNet decoder): Misaki phonemes (our dictionary phonemiser, as Paradee's) →
       * one graph with the voice's style row for the phoneme count → 24 kHz audio.
       */
      kind: 'kokoro';
      lexicon: ArtifactFile;
      /** tokenizer.json, whose model.vocab maps phonemes to ids. */
      tokenizer: ArtifactFile;
      model: SpeechGraph;
    }
  | {
      /**
       * Fun-CosyVoice 3: a Qwen2 0.5B language model (GenAI builder graph, inputs_embeds) samples speech tokens for
       * the text after a voice prompt's transcript and tokens; a DiT flow matcher (10 cosine-schedule steps, guidance
       * in the graph) makes an 80-band mel from them and the prompt's mel and speaker embedding; HiFT makes 24 kHz
       * audio. Graphs and loop: tools/export/export_cosyvoice3.py and the repository's README.
       */
      kind: 'cosyvoice';
      tokenizer: ArtifactFile;
      /** voice_default.json: the prompt's text ids, speech tokens, speaker embedding and mel. */
      voice: ArtifactFile;
      /** The upstream's fixed flow start noise, float16 [80, 3000]. */
      flowNoise: ArtifactFile;
      embed: SpeechGraph;
      llm: SpeechGraph;
      flowEncoder: SpeechGraph;
      flowStep: SpeechGraph;
      hift: SpeechGraph;
      layers: number;
    }
  | {
      /**
       * VibeVoice Realtime: the text, 5 tokens at a time, through a small text LM, then a 20-layer TTS LM conditioned on
       * a voice preset's prompt caches; per hidden state a diffusion head (guided by a second, negative cache) makes a
       * 64-d latent, a streaming VAE decoder turns each into 133 ms of 24 kHz audio, and an EOS head says when to stop.
       * Graphs and loop: tools/export/export_vibevoice.py and the repository's README.
       */
      kind: 'vibevoice';
      tokenizer: ArtifactFile;
      /** decoder_step_states.json: the decoder's state shapes. */
      states: ArtifactFile;
      /** Each voice's index (vibevoice_voice.py): array name → offset and shape in its .bin. */
      voiceIndexes: Record<string, ArtifactFile>;
      textLm: SpeechGraph;
      ttsLm: SpeechGraph;
      diffusion: SpeechGraph;
      connector: SpeechGraph;
      eos: SpeechGraph;
      decoder: SpeechGraph;
    }
  | {
      /**
       * Qwen3-TTS (12 Hz): a 28-layer talker (GenAI builder graph, inputs_embeds, float16 cache) samples each 80 ms
       * frame's first code from the text, a voice and the frames so far; a 5-layer code predictor samples the other 15
       * codebooks of the frame; the Qwen3-TTS tokenizer's decoder turns all frames into 24 kHz audio. Graphs and loop:
       * tools/export/export_qwen3_tts.py and the repositories' READMEs. CustomVoice: the voices are the model's
       * speakers; VoiceDesign: the voice is described in words (the voice description setting).
       */
      kind: 'qwen3-tts';
      tokenizer: ArtifactFile;
      /** The checkpoint's config.json: special token ids, codec ids, speakers and languages. */
      config: ArtifactFile;
      talker: SpeechGraph;
      codePredictor: SpeechGraph;
      textEmbed: SpeechGraph;
      /** Codec embeddings: `embeds` for the talker, and (1.7B) `cp_embeds`, the same rows projected for the code predictor. */
      codecEmbed: SpeechGraph;
      /** 1.7B: the talker's hidden state projected to the code predictor's width. */
      cpProject?: SpeechGraph;
      /** The talker computes in float32 (float32 inputs and cache; float16 overflowed into babble now and then). */
      talkerFloat32?: boolean;
      decoder: SpeechGraph;
    }
  | {
      /**
       * Sesame CSM-1B: a 16-layer Llama backbone (GenAI builder graph, inputs_embeds, float16 cache) samples each
       * 80 ms frame's first Mimi code from the prompt "[0]{text}" and the frames so far; a 4-layer depth decoder samples
       * the other 31; Mimi's decoder turns the frames into 24 kHz audio. The voice is the model's own for speaker 0.
       * Graphs and loop: tools/export/export_csm.py and the repository's README.
       */
      kind: 'csm';
      tokenizer: ArtifactFile;
      backbone: SpeechGraph;
      depthDecoder: SpeechGraph;
      textEmbed: SpeechGraph;
      audioEmbed: SpeechGraph;
      depthProject: SpeechGraph;
      mimi: SpeechGraph;
    }
  | {
      /**
       * NeuTTS Air: the text as eSpeak-style IPA (our G2P) after the voice's transcript phonemes; a Qwen2 0.5B model
       * (GenAI builder graph, input_ids, float32 cache) continues the voice's NeuCodec codes with the text's (50 a
       * second, temperature 1, top-k 50); NeuCodec's decoder turns them into 24 kHz audio. Voices are JSON files with
       * `codes` and `ref_phonemes`.
       */
      kind: 'neutts';
      tokenizer: ArtifactFile;
      lexicon: ArtifactFile;
      lm: SpeechGraph;
      codec: SpeechGraph;
    }
  | {
      /**
       * Orpheus 3B: a Llama 3.2 3B finetune (GenAI builder graph, input_ids, float16 cache) writes SNAC 24 kHz codes,
       * 7 tokens per frame, from "{voice}: {text}" (temperature 0.6, top-p 0.8, repetition penalty 1.3, as canopyai's
       * engine); SNAC's decoder turns them into audio.
       */
      kind: 'orpheus';
      /**
       * Maya1: the same layout with a voice described in words (`<description="…"> text`, the instruct setting),
       * temperature 0.4, top-p 0.9, repetition penalty 1.1, no end in the first 28 tokens (its model card's).
       */
      variant?: 'maya1';
      tokenizer: ArtifactFile;
      lm: SpeechGraph;
      snac: SpeechGraph;
    }
  | {
      /**
       * VoxCPM2: the text's embeddings through a 28-layer MiniCPM-4 LM and an 8-layer residual LM (float32 caches);
       * per patch of 4 latent frames, a local DiT run as 9 Euler steps (classifier-free guidance 2, CFG-zero*) from
       * noise, conditioned on both LMs' last states and the previous patch; the patch's encoding fed back, until the
       * stop head says so; the AudioVAE decoder turns the latents into 48 kHz audio. No reference voice.
       * Graphs and loop: tools/export/export_voxcpm.py and the repository's README.
       */
      kind: 'voxcpm';
      tokenizer: ArtifactFile;
      textEmbed: SpeechGraph;
      baseLm: SpeechGraph;
      residualLm: SpeechGraph;
      bridge: SpeechGraph;
      featEncoder: SpeechGraph;
      ditStep: SpeechGraph;
      vae: SpeechGraph;
    }
  | {
      /**
       * Zonos v0.1: the text as eSpeak-style IPA (our G2P in place of eSpeak) embedded, followed by the default
       * settings' conditioning rows (unconditional speaker), projected into a prefix; a 26-layer transformer with 9
       * codebook heads writes DAC codes in a delay pattern (classifier-free guidance 2, repetition penalty 3 over 2
       * frames, min-p 0.1) until codebook 0 ends; DAC's decoder turns them into 44.1 kHz audio. Graphs and loop:
       * tools/export/export_zonos.py and the repository's README.
       */
      kind: 'zonos';
      lexicon: ArtifactFile;
      /** The phoneme symbols and sampling constants (zonos.json). */
      config: ArtifactFile;
      /** The default settings' rows after the phonemes, float32 [2 (conditional, unconditional), 6, 2048]. */
      prefixRows: ArtifactFile;
      phonemeEmbed: SpeechGraph;
      prefix: SpeechGraph;
      embedCodes: SpeechGraph;
      backbone: SpeechGraph;
      dac: SpeechGraph;
    };

export interface SpeechModelSpec {
  id: string;
  name: string;
  publisher: string;
  parameters: string;
  /** Total parameters, for ordering by size. */
  parameterCount: number;
  summary: string;
  license: ModelLicense;
  source: { host: 'huggingface'; repo: string; revision: string };
  precision: Precision;
  sampleRate: number;
  voices: SpeechVoice[];
  /** The languages it reads, for multilingual models (the language setting's choices). */
  languages?: { id: string; label: string }[];
  pipeline: SpeechPipeline;
  /** Defaults of the settings the workspace shows. */
  defaults: SpeechSettings;
  /** Only in development builds (not yet uploaded). */
  devOnly?: boolean;
}

/** What a speech run is asked for: `language`, `speed` and `steps` (flow-matching steps) for models that take them. */
export interface SpeechSettings {
  voice: string;
  language?: string;
  /** A voice described in words, for models that design voices from text (Qwen3-TTS VoiceDesign). */
  instruct?: string;
  speed?: number;
  steps?: number;
}

const APACHE_2: ModelLicense = { name: 'Apache 2.0', url: 'https://www.apache.org/licenses/LICENSE-2.0' };

const CHATTERBOX_MIT: ModelLicense = { name: 'MIT', url: 'https://github.com/resemble-ai/chatterbox/blob/master/LICENSE' };

const OPENRAIL: ModelLicense = {
  name: 'BigScience OpenRAIL-M',
  url: 'https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE',
};

const file = (path: string, sizeBytes: number, sha256: string): ArtifactFile => ({ path, sizeBytes, sha256 });

/**
 * Our English pronunciation dictionary (Misaki's lexicons, compacted by tools/export/build_g2p_lexicon.py), shared
 * by the models that need phonemes: downloaded from its own repository and saved with each model.
 */
const G2P_LEXICON: ArtifactFile = {
  path: 'en-us-lexicon.txt.gz',
  sizeBytes: 1001546,
  sha256: '03acb4c31155ae08338174ae70db189dbb5790a11244b13e0f35846d5ee3c7e9',
  from: { repo: `${OWN_EXPORTS}/English-G2P-Lexicon`, revision: '3343f955b31ecdc8ea3cbc01105115b5aa52fd01', path: 'en-us-lexicon.txt.gz' },
};

/** A file of Resemble AI's own ONNX export of Chatterbox Turbo (its official int4 graphs), at a fixed revision. */
const turbo = (path: string, sizeBytes: number, sha256: string): ArtifactFile => ({
  path,
  sizeBytes,
  sha256,
  from: { repo: 'ResembleAI/chatterbox-turbo-ONNX', revision: 'd21799bd0354adb85e348b8a0442a8405110a2cf', path },
});

/** A file of onnx-community's ONNX export of Chatterbox (its int4 language model, float embeddings), at a fixed revision. */
const chatterboxOnnx = (path: string, sizeBytes: number, sha256: string): ArtifactFile => ({
  path,
  sizeBytes,
  sha256,
  from: { repo: 'onnx-community/chatterbox-ONNX', revision: '3cab09af388d3f02bba43443fce88c1f4525ac43', path },
});

/** A file of onnx-community's ONNX export of Chatterbox Multilingual, at a fixed revision. */
const chatterboxMultilingual = (path: string, sizeBytes: number, sha256: string): ArtifactFile => ({
  path,
  sizeBytes,
  sha256,
  from: { repo: 'onnx-community/chatterbox-multilingual-ONNX', revision: '452d3f434aa592098f1eedac9099f33642ab2da5', path },
});

/** Our int4 Chatterbox decoder (int8 convolutions), as Chatterbox Multilingual takes it from our Chatterbox repository. */
const CHATTERBOX_DECODER: ArtifactFile = {
  path: 'onnx/conditional_decoder.onnx',
  sizeBytes: 145358459,
  sha256: '4bc107a9996b313f5d1a094cdbccd9afaa20e6be596170eac74d8b0d903f1663',
  from: { repo: `${OWN_EXPORTS}/Chatterbox-ONNX-WebGPU`, revision: '54eb255c95ea64ea502f5efddb8a9869de30687f', path: 'onnx/conditional_decoder.onnx' },
};

/** A file of our int4 Qwen3-TTS 12 Hz tokenizer (the speech codec every Qwen3-TTS checkpoint shares). */
const qwenCodec = (path: string, sizeBytes: number, sha256: string): ArtifactFile => ({
  path: `codec/${path}`,
  sizeBytes,
  sha256,
  from: { repo: `${OWN_EXPORTS}/Qwen3-TTS-Tokenizer-12Hz-ONNX-WebGPU`, revision: 'b2aa169358f3ba36110a9c9ce7d5a75209e39b7f', path },
});

/** Qwen3-TTS CustomVoice's speakers (config.json spk_id), with the language each speaks natively. */
const QWEN_SPEAKERS: [string, string][] = [
  ['ryan', 'Ryan (English, male)'],
  ['aiden', 'Aiden (English, male)'],
  ['vivian', 'Vivian (Chinese, female)'],
  ['serena', 'Serena (Chinese, female)'],
  ['uncle_fu', 'Uncle Fu (Chinese, male)'],
  ['dylan', 'Dylan (Beijing dialect, male)'],
  ['eric', 'Eric (Sichuan dialect, male)'],
  ['ono_anna', 'Ono Anna (Japanese, female)'],
  ['sohee', 'Sohee (Korean, female)'],
];

/** Qwen3-TTS's languages (config.json codec_language_id). */
const QWEN_LANGUAGES = (
  [
    ['english', 'English'], ['chinese', 'Chinese'], ['japanese', 'Japanese'], ['korean', 'Korean'], ['german', 'German'],
    ['french', 'French'], ['spanish', 'Spanish'], ['italian', 'Italian'], ['portuguese', 'Portuguese'], ['russian', 'Russian'],
  ] as const
).map(([id, label]) => ({ id, label }));

/** MOSS-TTS-Nano's global transformer weights, shared by its prefill and decode-step graphs. */
const MOSS_GLOBAL = file('onnx/moss_tts_global_shared_q4.data', 69312512, 'df7c13ffcbcbb979d455bc59ca4448352ec4bfd60c7ad2d1e6159f3e2c5d1f05');

/** Kokoro's American voices: id, label (kokoro-js's names), the voice file's sha256. */
const KOKORO_VOICES: [string, string, string][] = [
  ['af_heart', 'Heart (female)', 'd583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b'],
  ['af_alloy', 'Alloy (female)', 'c4a6b876047fd7fb472edf4ebd63cfac7c3b958a7cae7c106e8f038ca6308c45'],
  ['af_aoede', 'Aoede (female)', '4a004c33430762e2461eedb2013fad808ef4ab3121f5300f554476caf58d8361'],
  ['af_bella', 'Bella (female)', 'f69d836209b78eb8c66e75e3cda491e26ea838a3674257e9d4e5703cbaf55c8b'],
  ['af_jessica', 'Jessica (female)', 'a240a5e3c15b43563d6e923bdca8ef5613a23471d9b77653694012435df23bd8'],
  ['af_kore', 'Kore (female)', '9be5221b6a941c04b561959b8ff0b06e809444dcc4ab7e75a7b23606f691819e'],
  ['af_nicole', 'Nicole (female)', 'cd2191ab31b914ed7b318416b0e4440fdf392ddad9106a060819aa600a64f59a'],
  ['af_nova', 'Nova (female)', '18778272caa0d0eebaea251c35fd635f038434f9eee5e691d02a174bd328414f'],
  ['af_river', 'River (female)', '00a2bcf82b1d86e8f19902ede58c65ccf6c0e43b44b7d74fad54e5d8933c9c30'],
  ['af_sarah', 'Sarah (female)', '4409fbc125afabacc615d94db5398d847006a737b0247d6892b7a9a0007a2f0a'],
  ['af_sky', 'Sky (female)', '4435255c9744f3f31659e0d714ab7689bf65d9e77ec1cce060f083912614f0b9'],
  ['am_adam', 'Adam (male)', '162b035ed91cfc48b6046982184c645f72edcdd1b82843347f605d7bf7b15716'],
  ['am_echo', 'Echo (male)', '3968b92c3c4cd1c4416dbded36c13eaa388a90d5788d02a13e4d781f5f8cf3c3'],
  ['am_eric', 'Eric (male)', 'e8b5be17edd1e3636901ce7598baafe2dc8dd8ff707a0c23bf9e461add7e2832'],
  ['am_fenrir', 'Fenrir (male)', 'c27989f741f7ee34d273a39d8a595cc0837d35f5ced9a29b7cc162614616df43'],
  ['am_liam', 'Liam (male)', '52403be32fd047c6a44517cb0bcd6b134f2a18baa73e70ef41651e0eab921ade'],
  ['am_michael', 'Michael (male)', '1d1f21dd8da39c30705cd4c75d039d265e9bc4a2a93ed09bc9e1b1225eb95ba1'],
  ['am_onyx', 'Onyx (male)', 'da5d135b424164916d75a68ffb4c2abce3d7d5ccc82dd1ee6cf447ce286145e6'],
  ['am_puck', 'Puck (male)', 'fcf73c989033e9233e0b98713eca600c8c74dcc1614b37009d5450ff4a2274a0'],
  ['am_santa', 'Santa (male)', '61150cf726ab6c5ed7a99f90a304f91f5a72c00c592e89ec94e5df11c319227a'],
];

/** Kitten TTS's eight voices, by the names its config gives them. */
const KITTEN_VOICES: [string, string][] = [
  ['expr-voice-2-f', 'Bella'],
  ['expr-voice-2-m', 'Jasper'],
  ['expr-voice-3-f', 'Luna'],
  ['expr-voice-3-m', 'Bruno'],
  ['expr-voice-4-f', 'Rosie'],
  ['expr-voice-4-m', 'Hugo'],
  ['expr-voice-5-f', 'Kiki'],
  ['expr-voice-5-m', 'Leo'],
];

/** A Kitten TTS model: its graph and its voices' style tables (float32 [400, 256] each), by voice id. */
function kitten(
  size: 'Nano' | 'Micro' | 'Mini',
  revision: string,
  parameters: string,
  parameterCount: number,
  summary: string,
  model: ArtifactFile,
  voices: Record<string, string>,
  speedPriors?: Record<string, number>,
): SpeechModelSpec {
  return {
    id: `kitten-tts-${size.toLowerCase()}-0.8`,
    name: `Kitten TTS ${size}`,
    publisher: 'KittenML',
    parameters,
    parameterCount,
    summary,
    license: APACHE_2,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Kitten-TTS-${size}-0.8-ONNX-WebGPU`, revision },
    precision: 'q4',
    sampleRate: 24000,
    voices: KITTEN_VOICES.map(([id, label]) => ({ id, label, file: file(`voices/${id}.bin`, 409600, voices[id]!) })),
    pipeline: { kind: 'kitten', lexicon: G2P_LEXICON, model: { graph: model }, speedPriors },
    defaults: { voice: 'expr-voice-2-f', speed: 1 },
  };
}

const SUPERTONIC_VOICES: [string, string, number, string][] = [
  ['F1', 'Female 1', 292046, 'bbdec6ee00231c2c742ad05483df5334cab3b52fda3ba38e6a07059c4563dbc2'],
  ['F2', 'Female 2', 292423, '7c722c6a72707b1a77f035d67f0d1351ba187738e06f7683e8c72b1df3477fc6'],
  ['F3', 'Female 3', 290794, '12f6ef2573baa2defa1128069cb59f203e3ab67c92af77b42df8a0e3a2f7c6ab'],
  ['F4', 'Female 4', 291808, 'c2fa764c1225a76dfc3e2c73e8aa4f70d9ee48793860eb34c295fff01c2e032b'],
  ['F5', 'Female 5', 291479, '45966e73316415626cf41a7d1c6f3b4c70dbc1ba2bee5c1978ef0ce33244fc8d'],
  ['M1', 'Male 1', 291748, 'e35604687f5d23694b8e91593a93eec0e4eca6c0b02bb8ed69139ab2ea6b0a5b'],
  ['M2', 'Male 2', 292055, 'b76cbf62bac707c710cf0ae5aba5e31eea1a6339a9734bfae33ab98499534a50'],
  ['M3', 'Male 3', 290198, 'ea1ac35ccb91b0d7ecad533a2fbd0eec10c91513d8951e3b25fbba99954e159b'],
  ['M4', 'Male 4', 291522, 'ca8eefad4fcd989c9379032ff3e50738adc547eeb5e221b82593a6d7b3bac303'],
  ['M5', 'Male 5', 291469, 'dd22b92740314321f8ae11c5e87f8dd60d060f15dd3a632b5adf77f471f77af2'],
];

const ALL_SPEECH_MODELS: readonly SpeechModelSpec[] = [
  {
    id: 'supertonic-3',
    name: 'Supertonic 3',
    publisher: 'Supertone',
    parameters: '99M',
    parameterCount: 99_000_000,
    summary: 'Clear, natural speech in 31 languages from a small flow-matching model, with ten voices. English here.',
    license: OPENRAIL,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Supertonic-3-ONNX-WebGPU`, revision: 'aed039b76a7472220e9d89f3ef537f7e2ff077a7' },
    precision: 'q4',
    sampleRate: 44100,
    voices: SUPERTONIC_VOICES.map(([id, label, size, sha]) => ({ id, label, file: file(`voice_styles/${id}.json`, size, sha) })),
    pipeline: {
      kind: 'supertonic',
      config: file('onnx/tts.json', 8253, '42078d3aef1cd43ab43021f3c54f47d2d75ceb4e75f627f118890128b06a0d09'),
      unicodeIndexer: file('onnx/unicode_indexer.json', 277676, '9bf7346e43883a81f8645c81224f786d43c5b57f3641f6e7671a7d6c493cb24f'),
      durationPredictor: { graph: file('onnx/duration_predictor.onnx', 1058573, 'a250979cab19427b65a29b488f639d1d1c53b66ab078015debb35e40e852e39e') },
      textEncoder: { graph: file('onnx/text_encoder.onnx', 9518425, 'd453ff479f38307d98a63da0853e0881983423366181da4438a5c4531d556467') },
      vectorEstimator: { graph: file('onnx/vector_estimator.onnx', 70634099, '26d0e5d82073f2cd2d3db6a335f238a4f7b8cf07c46d351e47398acc23208b50') },
      vocoder: { graph: file('onnx/vocoder.onnx', 28762717, '0f744ea78edd6e0b3fea4ba387f6a5e293ff89efeff63a0881291fe4a7f079a3') },
      language: 'en',
    },
    defaults: { voice: 'F1', speed: 1.05, steps: 8 },
  },
  {
    id: 'soprano-1.1-80m',
    name: 'Soprano 1.1',
    publisher: 'ekwek',
    parameters: '80M',
    parameterCount: 80_000_000,
    summary: 'Expressive English speech from a tiny language model that writes audio, sentence by sentence. One voice.',
    license: APACHE_2,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Soprano-1.1-80M-ONNX-WebGPU`, revision: '7d277623ab90126d95c2a032af6a262ce937bad5' },
    precision: 'q4',
    sampleRate: 32000,
    voices: [{ id: 'default', label: 'Default' }],
    pipeline: {
      kind: 'soprano',
      tokenizer: file('tokenizer.json', 1631056, 'a7df8e4e6082c7e01f0af6dea194dc19fa67c565fba6dd334930f4b28454e6d9'),
      backbone: {
        graph: file('onnx/model.onnx', 168140, 'a7cd39af1c2633b99f745a0560e21aa4ec135a0e475d61917a8d22b4d1083d97'),
        data: [file('onnx/model.onnx.data', 58327040, '2726b9acc8ebecf5c4a5448641419d319759c99ac40276d1c8f8732a5f53e1b4')],
      },
      decoder: { graph: file('onnx/decoder.onnx', 19219725, '22532e7ceefbfcf91d9f5fa86c175e378ff3c7990916176143895f37becc18c4') },
      layers: 17,
      kvHeads: 1,
      headDim: 128,
      hiddenSize: 512,
      eosToken: 3,
      tokenSamples: 2048,
    },
    defaults: { voice: 'default' },
  },
  {
    id: 'phonon-31m-en',
    name: 'Phonon 31M',
    publisher: 'Gradium',
    parameters: '31M',
    parameterCount: 31_000_000,
    summary: 'Natural English speech from a very small model, quick to make on most devices.',
    license: APACHE_2,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Phonon-31M-EN-ONNX-WebGPU`, revision: '5cc11081d9aaaea9fc4e71da79dcfefa7ecf1c03' },
    precision: 'q4',
    sampleRate: 24000,
    voices: [{ id: 'default', label: 'Default', file: file('voice_default.bin', 256000, '4302e4e490b76f9496e29b16df0367dc44e872ad868c22cfe780d81edd83a6a6') }],
    pipeline: {
      kind: 'phonon',
      tokenizer: file('tokenizer.json', 277967, '691a00068fa7b989f2f3e5ae5daa2d1a5bb319f2d5360deafcfb19a8bb6e7936'),
      config: file('phonon.json', 2145, 'c40cf1e5bbf0b747c50f58abcefff48d19b35fa39bcae305749b16d7e74d73a2'),
      bos: file('bos_emb.bin', 2048, '1a0c99983f06d9b07310c8348887193e55b3845f664ae7282fac162eedb79b51'),
      textEmbed: { graph: file('onnx/text_embed.onnx', 1280667, '0ad1bcc1be3c3a31693f8b791870d5acdd44f516697611f6866e44e37d72238b') },
      flowLm: { graph: file('onnx/flow_lm.onnx', 17500312, 'b8d59a947636c28263fda455b88148baf2ec489deaf722e086f7fc2cf54dbf21') },
      mimiDecoder: { graph: file('onnx/mimi_decoder.onnx', 8578311, '87c245ec800ab263c1cb7b8cd8bd8a9cb9ec84874a48f0c4b9812da7914a6486') },
    },
    defaults: { voice: 'default' },
  },
  kitten(
    'Nano',
    'b3fab281cdbd0275170471827645ce2a077a753d',
    '15M',
    15_000_000,
    'A tiny English voice model with eight voices: quick on any device.',
    file('onnx/model.onnx', 10329583, '9792b4d8d21abeb9fd2a06f3b32e9e90a45523dd970c8daa8a303b08572a2f7f'),
    {
      'expr-voice-2-f': '0f56274d033abbb4103140ff346e4f343c31dd0d2d3a45269a0988245291330e',
      'expr-voice-2-m': '5e0b649453ac74ca91dfc6730fb318567fc34a24bca9e910cdb39f07e611a83f',
      'expr-voice-3-f': '33c236c8ae79ba95c257c7b47196f22ec9d81a392e96aa0fb3a6397ac4294f29',
      'expr-voice-3-m': '388af21b8452d723738e1cc625838182cd306e36cbffc3ab2e9f310644783a18',
      'expr-voice-4-f': 'ae9a2bc901bfd2ca906e3be4996cbb5618b5a3724bef574f828cb1c1d94f7da9',
      'expr-voice-4-m': '1e036f1166acf869ba42d20a6f77ee13433f3959b44600af210e67b332cb53ca',
      'expr-voice-5-f': '4038768964607180b8e11e510056b9922b30b765a0f151517047143c56ff700b',
      'expr-voice-5-m': '106a8a5a8a02b82220cca9f467d638c8558e6e3d8f54b59ea54b326cbc91502d',
    },
    {
      'expr-voice-2-f': 0.8,
      'expr-voice-2-m': 0.8,
      'expr-voice-3-m': 0.8,
      'expr-voice-3-f': 0.8,
      'expr-voice-4-m': 0.9,
      'expr-voice-4-f': 0.8,
      'expr-voice-5-m': 0.8,
      'expr-voice-5-f': 0.8,
    },
  ),
  kitten(
    'Micro',
    'ddded952c8b2feeeebf4b0beb95bf8a5e7bc2356',
    '40M',
    40_000_000,
    'Kitten TTS with more capacity: clearer English speech, eight voices.',
    file('onnx/model.onnx', 26259434, 'ae54410aa05c12929c5b5ad96846df911e6db964dbbe59ac435acc85df3dd2f8'),
    {
      'expr-voice-2-f': '390b2ff5171712499f80423e10c625acbd7f112c349501ab03ab15718b89b80f',
      'expr-voice-2-m': '36f085b4a70855602749524990dd9793c6b69598f53284990bc7aa51289de12b',
      'expr-voice-3-f': '4d7402465c6a4280559a862ee618ee122f33d102dec8e1f80d68b01179f7a98a',
      'expr-voice-3-m': 'ae1e1a1e5d7e7cdaef70d510cb1847343d245b3b5822753eb0514b7ebd4abeef',
      'expr-voice-4-f': '8fad989b33bac33d42993fd86edb6c3f5c65025664580b0fa64b47cccee63e85',
      'expr-voice-4-m': 'c1f5ed7dcd05cc884835d30acf34edbbbf8dd6d241c926ccfbcca18270aa6d67',
      'expr-voice-5-f': 'feba460c07ac507ee4d632d26e5d8a5e398e9213e388a69303097e13467da0e0',
      'expr-voice-5-m': '077314eae6dba26121affd23a2958d5640ff457befc3a2feff1cd1bcb054eb6a',
    },
  ),
  kitten(
    'Mini',
    '8040b51f59734a0a45795a6ef2f3d04cc5b519d7',
    '80M',
    80_000_000,
    'The largest Kitten TTS: the most natural of the three, eight voices.',
    file('onnx/model.onnx', 51260062, '8bd2cec9533ce0f3c1ead36a12aeca2e192dd56aaf5eee9a8292e7e4e87cb5b2'),
    {
      'expr-voice-2-f': '04f743bea131051439bcbf4211aaa8834878aacb5e10fdf1858bd85fe41185be',
      'expr-voice-2-m': '4fc6fdcbaac820aad0d2f3040709bc0e23e3674d9c7f2eaa67ddc9573f128de4',
      'expr-voice-3-f': 'fea2513a70fb2c6b02aedf6291ea7faa2253893d2316dc2e3f3f79300f0d2a22',
      'expr-voice-3-m': '0c0b4b327eea6c8ea6926dba88a0ff3a12357db96cee1fb4426081a5c3ca2312',
      'expr-voice-4-f': '05cbcece857d3c2e0b188977933b67ebecbc589d45e3c63c31e27fc71e381b88',
      'expr-voice-4-m': '95ff42523e49ed1aec45c015846517396cd4a7abd1fb98c1eb67cfb4068eaa58',
      'expr-voice-5-f': '9b56657235dcff9d3e0112cac4c4fe8e571993ab102a0e7155bd6882129b1fd3',
      'expr-voice-5-m': '00d87de54370ac80e8a73d472dfd70c974b63cce83981e1165bcfd5f65037de7',
    },
  ),
  ...(
    [
      ['Micro', 'fb912b549304d6fa67332bef2869e327f6d8feae', '9M', 9_000_000, 5041033, '9b031d9ed7e229f6432397dcdce43adc8b7fcde0806fd46d055417eecd5ab40f', 1474013, 'ebb51dc345f86309ec7365717a173e00c34cd55a9e45a2c353c71f28bb132cfc'],
      ['Nano', '591a5cf8c7b107047a4424f1459649348452f036', '4M', 4_000_000, 2280959, 'd9d8ea6e7e1109ae92b2453894ab1c217273f777c4d2c27b5afb264265bb2bab', 905909, '99d87d1ecd0d3d9fd37de9f676f12ee2bb96434f210858c36d9d28760bba30bd'],
    ] as const
  ).map(
    ([size, revision, parameters, parameterCount, decodeBytes, decodeSha, durationBytes, durationSha]): SpeechModelSpec => ({
      id: `inflect-${size.toLowerCase()}-v2`,
      name: `Inflect ${size}`,
      publisher: 'owensong',
      parameters,
      parameterCount,
      summary: `A ${size === 'Nano' ? 'very small' : 'small'} English voice model (VITS) with one voice: a few megabytes, fast anywhere.`,
      license: APACHE_2,
      source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Inflect-${size}-v2-ONNX-WebGPU`, revision },
      precision: 'q4',
      sampleRate: 24000,
      voices: [{ id: 'default', label: 'Default' }],
      pipeline: {
        kind: 'inflect',
        lexicon: G2P_LEXICON,
        duration: { graph: file('onnx/duration.onnx', durationBytes, durationSha) },
        decode: { graph: file('onnx/decode.onnx', decodeBytes, decodeSha) },
      },
      defaults: { voice: 'default', speed: 1 },
    }),
  ),
  {
    id: 'paradee-8m',
    name: 'Paradee',
    publisher: 'sahilmahendrakar',
    parameters: '8M',
    parameterCount: 8_000_000,
    summary: 'Kokoro’s voice distilled into an 8M model: natural English speech at a fraction of the size.',
    license: APACHE_2,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Paradee-8M-ONNX-WebGPU`, revision: '075b5050e09adb91449883a6147b94548dd84989' },
    precision: 'q8',
    sampleRate: 24000,
    voices: [{ id: 'default', label: 'Default' }],
    pipeline: {
      kind: 'paradee',
      lexicon: G2P_LEXICON,
      config: file('config.json', 1774, 'c3f420e8a5aea41b97f698404a1b1e28e403c662f65927345335fc5493f2e075'),
      model: { graph: file('onnx/model.onnx', 14554226, 'bd0b243e3941dcebf98fa95edd04e380751949371563f4f8eb8978fc9a740c8b') },
    },
    defaults: { voice: 'default', speed: 1 },
  },
  {
    id: 'chatterbox-turbo',
    name: 'Chatterbox Turbo',
    publisher: 'Resemble AI',
    parameters: '350M',
    parameterCount: 350_000_000,
    summary: 'Natural, lively English speech from Resemble AI, with tags like [laugh], [sigh] or [cough] for sounds between words.',
    license: CHATTERBOX_MIT,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Chatterbox-Turbo-ONNX-WebGPU`, revision: '6b2b5fe0fbef30eb5a4fa8a7ef52ba95e4f062f8' },
    precision: 'q4',
    sampleRate: 24000,
    voices: [{ id: 'default', label: 'Default', file: file('voices/default.bin', 739060, 'eda1073fc6a7cd3d9d6ae24e3c6baa495abb112e5349fab6d0eb523dbc0e6a84') }],
    pipeline: {
      kind: 'chatterbox',
      variant: 'turbo',
      tokenizer: turbo('tokenizer.json', 3562272, '3f04e34bea22f9144d1a19151154095bc9ce0430bf421304f5797e716288a906'),
      embedTokens: {
        graph: turbo('onnx/embed_tokens_q4.onnx', 2844, 'fd6ba1d22902e8f539d3dd6d7c1c44b98ebb4c84ebbb5e47fcb826ddcf667561'),
        data: [turbo('onnx/embed_tokens_q4.onnx_data', 37286384, 'f54a51e234b509b64c3a03bb79e1149fba7e2eba6c2d9c222f18883379e1f5d8')],
      },
      languageModel: {
        graph: turbo('onnx/language_model_q4.onnx', 274572, 'b39d03d3f8b943b9e60c6fce3fb41191dbc1df4589f913291db1e214eef669b1'),
        data: [turbo('onnx/language_model_q4.onnx_data', 204456572, '2c029dc0acf48752473d8c74c72b5ceaaad76b9886fe106eaf2022142d5b5d5e')],
      },
      decoder: {
        graph: turbo('onnx/conditional_decoder_q4.onnx', 2179022, 'dccb7a6cea3472dc7f7d070eeb70ade18e6327fb4ec61a3d62cf211bfed90ea2'),
        data: [turbo('onnx/conditional_decoder_q4.onnx_data', 246397384, 'b5c5317e0b79a1a19dd3d5e2b2091ea06b15716716ab801a54eaeb906c6971ec')],
      },
      layers: 24,
      kvHeads: 16,
      headDim: 64,
    },
    defaults: { voice: 'default' },
  },
  {
    id: 'chatterbox',
    name: 'Chatterbox',
    publisher: 'Resemble AI',
    parameters: '500M',
    parameterCount: 500_000_000,
    summary: 'Resemble AI’s first Chatterbox: clear, expressive English speech, sentence by sentence.',
    license: CHATTERBOX_MIT,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Chatterbox-ONNX-WebGPU`, revision: '54eb255c95ea64ea502f5efddb8a9869de30687f' },
    precision: 'q4',
    sampleRate: 24000,
    voices: [{ id: 'default', label: 'Default', file: file('voices/default.bin', 255732, '1cb6ec75b9d8e58c4c5d7936c887307fe16c7b76bd91bbeda9e503fc9a9f8191') }],
    pipeline: {
      kind: 'chatterbox',
      variant: 'original',
      tokenizer: chatterboxOnnx('tokenizer.json', 28543, 'ecddf6aaae85e271610a2743409f7e447886957d0b0836687c1eaec5ccac2891'),
      embedTokens: {
        graph: chatterboxOnnx('onnx/embed_tokens.onnx', 13286, '160722ec14789f616abdb1e31916cbbf9223c03fde0ab546d64ca74fb72e430b'),
        data: [chatterboxOnnx('onnx/embed_tokens.onnx_data', 61640704, '898c563c3a5ca1b9ea10ce89b0cdcf252b0bb5ab460dfc4eadea003b56e5d2ee')],
      },
      languageModel: {
        graph: chatterboxOnnx('onnx/language_model_q4.onnx', 227911, '7f8cdca83b2493536cbf3acf421199808a3d68736f55f4eabd20ef8a99da4313'),
        data: [chatterboxOnnx('onnx/language_model_q4.onnx_data', 353621248, '5203d1e83c159316f9923c5c83759f6a34f87be1322ce4ad0facd9fc4aef4790')],
      },
      decoder: { graph: file('onnx/conditional_decoder.onnx', 145358459, '4bc107a9996b313f5d1a094cdbccd9afaa20e6be596170eac74d8b0d903f1663') },
      layers: 30,
      kvHeads: 16,
      headDim: 64,
    },
    defaults: { voice: 'default' },
  },
  {
    id: 'chatterbox-multilingual',
    name: 'Chatterbox Multilingual',
    publisher: 'Resemble AI',
    parameters: '500M',
    parameterCount: 500_000_000,
    summary: 'Chatterbox in 20 languages, from Arabic to Turkish: expressive speech in one voice across all of them.',
    license: CHATTERBOX_MIT,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Chatterbox-Multilingual-ONNX-WebGPU`, revision: 'b37cca40a5510b9525ca429a6f0af90f9557655e' },
    precision: 'q4',
    sampleRate: 24000,
    voices: [{ id: 'default', label: 'Default', file: file('voices/default.bin', 255732, 'd97aa83ba5286ccd87f5be2e939c02d5bdd10d7fe04631c3a75507277851308c') }],
    // Chinese, Japanese and Hebrew are left out: their text needs converters (Cangjie codes, kana, diacritics).
    languages: (
      [
        ['ar', 'Arabic'], ['da', 'Danish'], ['de', 'German'], ['el', 'Greek'], ['en', 'English'], ['es', 'Spanish'],
        ['fi', 'Finnish'], ['fr', 'French'], ['hi', 'Hindi'], ['it', 'Italian'], ['ko', 'Korean'], ['ms', 'Malay'],
        ['nl', 'Dutch'], ['no', 'Norwegian'], ['pl', 'Polish'], ['pt', 'Portuguese'], ['ru', 'Russian'], ['sv', 'Swedish'],
        ['sw', 'Swahili'], ['tr', 'Turkish'],
      ] as const
    ).map(([id, label]) => ({ id, label })),
    pipeline: {
      kind: 'chatterbox',
      variant: 'multilingual',
      tokenizer: chatterboxMultilingual('tokenizer.json', 71798, '29d48c4a178f6af3ad5130097c34744639e9294847b38a7b912c8c68027cb819'),
      embedTokens: {
        graph: chatterboxMultilingual('onnx/embed_tokens.onnx', 13286, 'f785819ca4f6271262d5bb8971d62796c3a909e3b031982c113dbe83a4c3b854'),
        data: [chatterboxMultilingual('onnx/embed_tokens.onnx_data', 68390912, '2a15f7dd73b2ee47f6edf87740324011594b5a528ed6471ae55e327ed6cad68c')],
      },
      languageModel: {
        graph: chatterboxMultilingual('onnx/language_model_q4.onnx', 227911, '7f8cdca83b2493536cbf3acf421199808a3d68736f55f4eabd20ef8a99da4313'),
        data: [chatterboxMultilingual('onnx/language_model_q4.onnx_data', 353621248, 'e79ab8784122a501718868b9631ff46e151c552d9b24e50f25d721f375e3526c')],
      },
      // The same decoder as Chatterbox's (onnx-community ships one file for both), from our Chatterbox repository.
      decoder: { graph: CHATTERBOX_DECODER },
      layers: 30,
      kvHeads: 16,
      headDim: 64,
    },
    defaults: { voice: 'default', language: 'en' },
  },
  {
    id: 'moss-tts-nano-100m',
    name: 'MOSS-TTS Nano',
    publisher: 'OpenMOSS',
    parameters: '100M',
    parameterCount: 100_000_000,
    summary: 'A small English and Chinese voice model with ten voices.',
    license: APACHE_2,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/MOSS-TTS-Nano-100M-ONNX-WebGPU`, revision: '95c5b2837c41178879a1794381a11b483bc7f518' },
    precision: 'q4',
    sampleRate: 48000,
    voices: (
      [
        ['Ava', 'Ava (English, female)'], ['Bella', 'Bella (English, female)'], ['Adam', 'Adam (English, male)'],
        ['Nathan', 'Nathan (English, male)'], ['Xiaoyu', 'Xiaoyu (Chinese, female)'], ['Yuewen', 'Yuewen (Chinese, female)'],
        ['Lingyu', 'Lingyu (Chinese, female)'], ['Junhao', 'Junhao (Chinese, male)'], ['Zhiming', 'Zhiming (Chinese, male)'],
        ['Weiguo', 'Weiguo (Chinese, male)'],
      ] as const
    ).map(([id, label]) => ({ id, label })),
    pipeline: {
      kind: 'moss',
      tokenizer: file('tokenizer.json', 505710, '6589916f2c3397dffd2f0b013ccd40657efdf36698a3001c95435b5d6c650d60'),
      config: file('moss.json', 88643, 'a0f26bae46ecaaf7c4e756c0326ca313975aae0a620eafb70e7bb4ea608e196c'),
      prefill: { graph: file('onnx/moss_tts_prefill_q4.onnx', 442349, 'f01ad811e1e05cc1bb1205b613faf2b7dd96083f5c72a6c26d1842a0fc031882'), data: [MOSS_GLOBAL] },
      decodeStep: { graph: file('onnx/moss_tts_decode_step_q4.onnx', 454013, '6b5f507f873d05c576f2cd1fbe02c036721fd1b1aabf698a555400e60b1aecc1'), data: [MOSS_GLOBAL] },
      // These two run on the CPU engine, which has no int4 lookups: float16 tables (tools/export/dequantize_gathers.py).
      sampleFrame: { graph: file('onnx/moss_tts_local_fixed_sampled_frame_cpu.onnx', 97983500, '9669d93d9d959042170229c97a6641aa25c56ac7623d0689aed63f66532f9b81') },
      codec: { graph: file('onnx/moss_audio_tokenizer_decode_step_cpu.onnx', 7719606, 'f14a7be3f7c0eac1100c5e750a73318146ca2b9cb8f0d1347957fc5175aaf1ff') },
    },
    defaults: { voice: 'Ava' },
  },
  {
    id: 'kokoro-82m',
    name: 'Kokoro',
    publisher: 'hexgrad',
    parameters: '82M',
    parameterCount: 82_000_000,
    summary: 'Natural, warm English speech from a small model, with twenty American voices.',
    license: APACHE_2,
    source: { host: 'huggingface', repo: 'onnx-community/Kokoro-82M-v1.0-ONNX', revision: '1939ad2a8e416c0acfeecc08a694d14ef25f2231' },
    precision: 'q4',
    sampleRate: 24000,
    // The American voices: the British ones need British phonemes, which our dictionary doesn't have.
    voices: KOKORO_VOICES.map(([id, label, sha]) => ({ id, label, file: file(`voices/${id}.bin`, 522240, sha) })),
    pipeline: {
      kind: 'kokoro',
      lexicon: G2P_LEXICON,
      tokenizer: file('tokenizer.json', 3497, '77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34'),
      model: { graph: file('onnx/model_q4.onnx', 305215966, '04cf570cf9c4153694f76347ed4b9a48c1b59ff1de0999e6605d123966b197c7') },
    },
    defaults: { voice: 'af_heart', speed: 1 },
  },
  {
    id: 'pocket-tts',
    name: 'Pocket TTS',
    publisher: 'Kyutai',
    parameters: '100M',
    parameterCount: 100_000_000,
    summary: 'Kyutai’s small English voice model: natural speech, quick to make on most devices.',
    license: {
      name: 'CC BY 4.0',
      url: 'https://creativecommons.org/licenses/by/4.0/',
      notice:
        'Pocket TTS by Kyutai (CC BY 4.0), converted to ONNX and quantized by Quick Inference; voice “Alba” from Alba MacKenna’s recording in kyutai/tts-voices (CC BY 4.0)',
    },
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Pocket-TTS-ONNX-WebGPU`, revision: '5daed3fdec494b7d7c2bb86b1de05e2d994a2481' },
    precision: 'q4',
    sampleRate: 24000,
    voices: [{ id: 'alba', label: 'Alba', file: file('voice_alba.bin', 520192, 'babf83491c56811b8d574849c6b25b95c329f7178401eca7aca7813c8b403357') }],
    pipeline: {
      kind: 'phonon',
      variant: 'pocket-tts',
      tokenizer: file('tokenizer.json', 245020, 'f498428e1eafee50492f7be13dc9bfafcfc12e508cd0eb1b01c92ecd5d8c6687'),
      config: file('pocket_tts.json', 2362, 'cd933b7de449d7a74ac5d3d7a100f012e9d170091ef9f47afdcd86c25885021f'),
      bos: file('bos_emb.bin', 4096, '5cfd884ea708e885fe8cf70fa7c3d9e19e4e6a5a6699e249a9d6a689a64aee97'),
      textEmbed: { graph: file('onnx/text_embed.onnx', 2560988, '6866f48280db7289707a726e4295e6575bdb0e12f1f429bc7b90fec9bd9704ec') },
      flowLm: { graph: file('onnx/flow_lm.onnx', 53124354, 'fb403eb14fc70ddb1675d1ab5d2a3a9f42bf999c511f5a148fa0b4c096a5958e') },
      mimiDecoder: { graph: file('onnx/mimi_decoder.onnx', 8579869, 'df0942a3ff388f4ad6184b06fb09cc633f4899bad8b535e5255d0bbb9829d073') },
    },
    defaults: { voice: 'alba' },
  },
  {
    id: 'fun-cosyvoice3-0.5b',
    name: 'Fun-CosyVoice 3',
    publisher: 'FunAudioLLM',
    parameters: '0.5B',
    parameterCount: 500_000_000,
    summary: 'Alibaba’s voice-cloning model: a small language model writes speech tokens, a flow model and vocoder voice them. English here.',
    license: {
      ...APACHE_2,
      notice: 'Default voice derived from LibriTTS-R (Koizumi et al., 2023), utterance 4446_2275_000002_000009, CC BY 4.0',
    },
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Fun-CosyVoice3-0.5B-ONNX-WebGPU`, revision: 'dcdb9df4bce4a47d318f5f8736de0b080efd944f' },
    precision: 'q8',
    sampleRate: 24000,
    voices: [{ id: 'default', label: 'Default' }],
    pipeline: {
      kind: 'cosyvoice',
      tokenizer: file('tokenizer.json', 10573953, 'caa0dadcb8c40c79a3244aaf290492e298a49c5c25e0db16cad12f2f8ff639c5'),
      voice: file('voice_default.json', 157335, '43f3ef22692d6dc21a45fbfa3330cc86dc1eb4105f152b3eb734b71ebe572fa3'),
      flowNoise: file('flow_noise_f16.bin', 480000, '7ab437a2da46f82936044a09ff25193db4044db8e028e67ad06915b887a0b869'),
      embed: { graph: file('onnx/embed_q4.onnx', 88870639, 'e723fac2773d3718023b4438b1e1c450d6a3ed6b502da112ef9e821865d98789') },
      llm: { graph: file('onnx/llm_q8.onnx', 418204025, '6221ef67f1d8b08f2537ef6bec01bb556af1792a5eb5aea1e98e9b6a196f1ac3') },
      flowEncoder: { graph: file('onnx/flow_encoder_q4.onnx', 740028, 'a830b1a28193cbf1cd79516cd367ddd0beaf66958a3c7ccddceb522eca6d50ec') },
      flowStep: { graph: file('onnx/flow_step_q8.onnx', 373851734, 'a56ab4d3654346a4da24d3b85211270a1fde5124fecd34a1d78d66cfe230058c') },
      hift: { graph: file('onnx/hift_q4c8.onnx', 24004242, '5b9f5bc83053523c19e446b64050582f43e729dbc182ea7b38258ca2b7c89ed3') },
      layers: 24,
    },
    defaults: { voice: 'default' },
  },
  {
    id: 'vibevoice-realtime-0.5b',
    name: 'VibeVoice Realtime',
    publisher: 'Microsoft',
    parameters: '0.5B',
    parameterCount: 500_000_000,
    summary: 'Microsoft’s English voice model: a language model and diffusion head make expressive, natural speech.',
    license: {
      name: 'MIT',
      url: 'https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B',
      notice:
        'Microsoft recommends VibeVoice for research and development, not commercial or real-world use without further testing, and prohibits voice impersonation without consent and deep-fakes',
    },
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/VibeVoice-Realtime-0.5B-ONNX-WebGPU`, revision: 'dd90d22e261c323a70cade8de3291a36ff990c9f' },
    precision: 'q4',
    sampleRate: 24000,
    voices: [{ id: 'en-Carter_man', label: 'Carter', file: file('voices/en-Carter_man.bin', 6938112, 'f97a7b018b610f2a3867c964aa3d6967c4004d7ea78bd5efc3a36c065327d5b5') }],
    pipeline: {
      kind: 'vibevoice',
      tokenizer: file('tokenizer.json', 7031645, 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'),
      states: file('decoder_step_states.json', 509, '5c61904069c0626ff97f52e301baf3882128e47b8ba8881b0fe50907e7b26bb8'),
      voiceIndexes: { 'en-Carter_man': file('voices/en-Carter_man.json', 5030, '5b97106cd31c5b44002e97ff1eb6ba93b32b068f95f76f83f3ccb98850d74890') },
      textLm: { graph: file('onnx/text_lm_q4.onnx', 122621508, '0bd1e061b7cddf764e40d1ace9a5f0bcaba154470890f1dddcb309b3a3b2bc51') },
      ttsLm: {
        graph: file('onnx/tts_lm/model.onnx', 182755, 'e1a7587f117280ebf12f5e4c86c22d710609f468a44b7047b8355af6f49715cb'),
        data: [file('onnx/tts_lm/model.onnx.data', 176504832, '30bb6e68dc964d44fa17bc05a44720e2542542ec104c280428670ba17345610b')],
      },
      diffusion: { graph: file('onnx/diffusion_q4.onnx', 26568116, '8813299c0b102d8efb884d41a2262d0e1041d7898e5002e0c8f80cc98bf6c14a') },
      connector: { graph: file('onnx/connector_q4.onnx', 554670, '12b08edefd57c82f7c61e7aec2488129c32ef9df55f79df2af9eb72db50d2178') },
      eos: { graph: file('onnx/eos_q4.onnx', 507037, 'c890c747ba65d381c0790772ce5952273cccdd57ead478b56c452d0e31c04642') },
      decoder: { graph: file('onnx/decoder_step_q4.onnx', 215867221, 'f066ed9e9db35db33a18dbfa4559e6feb180b331594598513734f5b38006ebfc') },
    },
    defaults: { voice: 'en-Carter_man' },
  },
  {
    id: 'qwen3-tts-0.6b-customvoice',
    name: 'Qwen3-TTS 0.6B',
    publisher: 'Qwen',
    parameters: '0.6B',
    parameterCount: 600_000_000,
    summary: 'Qwen’s speech model in ten languages, with nine built-in speakers.',
    license: APACHE_2,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX-WebGPU`, revision: 'ccdd14012062ee383dda616d7799cc955924af8a' },
    precision: 'q4',
    sampleRate: 24000,
    voices: QWEN_SPEAKERS.map(([id, label]) => ({ id, label })),
    languages: QWEN_LANGUAGES,
    pipeline: {
      kind: 'qwen3-tts',
      tokenizer: file('tokenizer.json', 11423986, '09267689b8362020b9763b65dd5be7e086b31e28d72e02837a9e781de9a91bc7'),
      config: file('config.json', 4908, '81aca2b6fac304944d8acf345272d8a9a727d5fc2e2e66b222ab4729340c7455'),
      talker: {
        graph: file('talker/model.onnx', 276512, '2edc5aed01da638848a17123ee0673782d6d5a406339969e4198e555a2ab8ff3'),
        data: [file('talker/model.onnx.data', 301203456, '026537338baf696232457b7e624a9c8794f16caa358a5a8cc88d92f61a091e3c')],
      },
      codePredictor: {
        graph: file('code_predictor/model.onnx', 56354, '6ac4c6c4cc831ff024015537d1ae16c0d3e72b17e4ef512c4c5924a26a7b7a4c'),
        data: [file('code_predictor/model.onnx.data', 157483008, '35284527836a5c52836f58d4fe0e140eb3a9a442cbeed44cd8d494c5eb056bf8')],
      },
      textEmbed: { graph: file('text_embed_q4.onnx', 97239399, '6a0d7fa5c29e1a60ccce507a92827ddb432ddf49dfea8c166a886c5c7d138df9') },
      codecEmbed: { graph: file('codec_embed_q4.onnx', 21627240, 'e5f424a00c8c565f1d660f6dffd69ff3be9bceeacdd55b5c0eb936bd72958cd5') },
      decoder: { graph: qwenCodec('tok_decoder_q4.onnx', 100611208, '75b53ce2ef29c390eeda17c6ffd76a34cf05cef9e0da0de05c42dfe563084a93') },
      talkerFloat32: true,
    },
    defaults: { voice: 'ryan', language: 'english' },
  },
  {
    id: 'qwen3-tts-1.7b-customvoice',
    name: 'Qwen3-TTS 1.7B',
    publisher: 'Qwen',
    parameters: '1.7B',
    parameterCount: 1_700_000_000,
    summary: 'The larger Qwen3-TTS: more natural speech in ten languages, with nine built-in speakers.',
    license: APACHE_2,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX-WebGPU`, revision: '098e415f848952a940a9ba685f65ec23e00fa582' },
    precision: 'q4',
    sampleRate: 24000,
    voices: QWEN_SPEAKERS.map(([id, label]) => ({ id, label })),
    languages: QWEN_LANGUAGES,
    pipeline: {
      kind: 'qwen3-tts',
      tokenizer: file('tokenizer.json', 11423986, '09267689b8362020b9763b65dd5be7e086b31e28d72e02837a9e781de9a91bc7'),
      config: file('config.json', 4908, '17a07f527a1c25ea30b4e023a184482a23d3e279d697b1dc81b1bde498d29cf9'),
      talker: {
        graph: file('talker/model.onnx', 277123, 'd2f997433acf81ed61034b21d8903cac9143f0b5b515358ce496a18321a391b3'),
        data: [file('talker/model.onnx.data', 924188672, '9f18ddd03c6e1b4a35804da668d63444973e835d3842ef684b2f124f0ff6a0cb')],
      },
      codePredictor: {
        graph: file('code_predictor/model.onnx', 56354, '6ac4c6c4cc831ff024015537d1ae16c0d3e72b17e4ef512c4c5924a26a7b7a4c'),
        data: [file('code_predictor/model.onnx.data', 157483008, '33fad8842b38c691750ab466379db80324fb685ada4b250bc96b975630d949bb')],
      },
      textEmbed: { graph: file('text_embed_q4.onnx', 194478439, 'd414d3c004a263adde99467f86bc1870a3f0c0c386152babdb3809c586cde296') },
      codecEmbed: { graph: file('codec_embed_q4.onnx', 64881305, '1f3bda3cfb2f4a09eb43734b2c3b991e5d9b8fb9811e8d7b8c3402ad2abaf551') },
      cpProject: { graph: file('cp_project_q4.onnx', 1315139, 'ed0a9a957eb5ebac177f37c044ad1227defd7595fc8946f63253a353123d837d') },
      decoder: { graph: qwenCodec('tok_decoder_q4.onnx', 100611208, '75b53ce2ef29c390eeda17c6ffd76a34cf05cef9e0da0de05c42dfe563084a93') },
      talkerFloat32: true,
    },
    defaults: { voice: 'ryan', language: 'english' },
  },
  {
    id: 'qwen3-tts-1.7b-voicedesign',
    name: 'Qwen3-TTS 1.7B VoiceDesign',
    publisher: 'Qwen',
    parameters: '1.7B',
    parameterCount: 1_700_000_000,
    summary: 'Qwen3-TTS with a voice you describe in words (age, accent, tone, pace), in ten languages.',
    license: APACHE_2,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Qwen3-TTS-12Hz-1.7B-VoiceDesign-ONNX-WebGPU`, revision: '66874037e93c4055295b1c13e90c02a21f68877f' },
    precision: 'q4',
    sampleRate: 24000,
    voices: [{ id: 'designed', label: 'Described voice' }],
    languages: QWEN_LANGUAGES,
    pipeline: {
      kind: 'qwen3-tts',
      tokenizer: file('tokenizer.json', 11423986, '09267689b8362020b9763b65dd5be7e086b31e28d72e02837a9e781de9a91bc7'),
      config: file('config.json', 4421, 'aecd2cc4c1fe9edef1cb7ca7c401685a43879ad43f3f9e883f1c6760b61731e0'),
      talker: {
        graph: file('talker/model.onnx', 277123, 'd2f997433acf81ed61034b21d8903cac9143f0b5b515358ce496a18321a391b3'),
        data: [file('talker/model.onnx.data', 924188672, '61a2d3862a3a666242d40e24e82d371a155e666454054fd7617cec6d1cbc9f71')],
      },
      codePredictor: {
        graph: file('code_predictor/model.onnx', 56354, '6ac4c6c4cc831ff024015537d1ae16c0d3e72b17e4ef512c4c5924a26a7b7a4c'),
        data: [file('code_predictor/model.onnx.data', 157483008, '78e4de197ec51ab9d5620ca580d6f3f39515041468a57da4a4f89c22b6b059ed')],
      },
      textEmbed: { graph: file('text_embed_q4.onnx', 194478439, '27c9c17ec13757cbc918578f508d1547b30da082de98c2e248fa8f4b2f935b34') },
      codecEmbed: { graph: file('codec_embed_q4.onnx', 64881305, 'f5e345ef5675f3644f8e885655b020df6505537736ffd5efb2bf7e019257dc62') },
      cpProject: { graph: file('cp_project_q4.onnx', 1315139, '80932ac192c1d300f3506a62db57e57afb8502f5980c0c55831da10795bcde78') },
      // The 8-bit-convolution decoder (the 4-bit one brightened this model's voices).
      decoder: { graph: qwenCodec('tok_decoder_q4c8.onnx', 129800340, 'a285e311d00e248ac3f2d1dfd80d6e181218e37e852545bce162624e1fd9214d') },
      talkerFloat32: true,
    },
    defaults: {
      voice: 'designed',
      language: 'english',
      instruct: 'A clear, calm adult male voice with a neutral American accent, speaking at a moderate pace.',
    },
  },
  {
    id: 'csm-1b',
    name: 'CSM 1B',
    publisher: 'Sesame',
    parameters: '1B',
    parameterCount: 1_000_000_000,
    summary: 'Sesame’s conversational speech model: natural, lively English. Its voice can change from one clip to the next.',
    license: {
      ...APACHE_2,
      notice: 'Includes Kyutai’s Mimi codec (CC BY 4.0), exported and quantized. Sesame prohibits impersonation without consent and misleading content',
    },
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/CSM-1B-ONNX-WebGPU`, revision: '088e80cbdc650ae8dae665040583a609c77b8990' },
    precision: 'q4',
    sampleRate: 24000,
    voices: [{ id: 'default', label: 'Default' }],
    pipeline: {
      kind: 'csm',
      tokenizer: file('tokenizer.json', 17209980, '80f511469b4aceda8c17ccafef2310eceeda9dd4589cf2c062bd0279eef5646d'),
      backbone: {
        graph: file('onnx/backbone/model.onnx', 140756, '38e20f29da306073d990da75a62b5b31d7537d939c399eca7e64c4ad23e0ecc4'),
        data: [file('onnx/backbone/model.onnx.data', 565510144, 'e0c8af2589da94a7e8073fd121582bda3a07df2c09322dee04164f6e59e66c3e')],
      },
      depthDecoder: {
        graph: file('onnx/depth_decoder/model.onnx', 37511, 'f651417339546a87f73d1122a43d5a609db0e85ab72743ee018278d433c2945a'),
        data: [file('onnx/depth_decoder/model.onnx.data', 102021632, 'e7fa802eeb3957cf1ca36d661682d6b937e9b552814bbde131ae0bd9d8251b3e')],
      },
      textEmbed: { graph: file('onnx/text_embed_q4.onnx', 164168039, '7aa9fb684a296d079bb7fbb49ea5d7d8bb31c89b0923a823f93eae91d5909e2b') },
      audioEmbed: { graph: file('onnx/audio_embed_q4.onnx', 126014126, '6c4300018627e56117eefa71c492a0f7216c396eb0b52763e234d0de5dce8fb1') },
      depthProject: { graph: file('onnx/depth_project_q4.onnx', 1311020, '21061a3a9690744cad6efdb3c3c04fe3149c972a0d8b995a7bfe88913f029816') },
      mimi: { graph: file('onnx/mimi_decoder_q4.onnx', 92753675, '901918d7e06fd22c060baf3ef11eb2858b7955072a9592d2766d342a6570ef93') },
    },
    defaults: { voice: 'default' },
  },
  {
    id: 'neutts-air',
    name: 'NeuTTS Air',
    publisher: 'Neuphonic',
    parameters: '0.7B',
    parameterCount: 748_000_000,
    summary: 'A small language model that speaks English in the voice of a short reference recording, from phonemes.',
    license: {
      ...APACHE_2,
      notice: 'The voice Dave is from Neuphonic’s neutts repository at an Apache 2.0 commit. No watermark is added to the audio',
    },
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/NeuTTS-Air-ONNX-WebGPU`, revision: '1d854fee3bac2ce7761905aa3fe8cf3aa68cf891' },
    precision: 'q4',
    sampleRate: 24000,
    voices: [{ id: 'dave', label: 'Dave', file: file('voice_dave.json', 3437, 'ecb53744a21b5ea507b7e61421b86693e6aa6be840305ff56189e85fcf55713a') }],
    pipeline: {
      kind: 'neutts',
      tokenizer: file('tokenizer.json', 24140235, 'a5cd7db9b07058f3742d541fbb4a053de43e4e6f8ab99a8edabe54daee3f0a5a'),
      lexicon: G2P_LEXICON,
      lm: {
        graph: file('onnx/model.onnx', 220917, 'b0bf6d61bb69cfc9b9fa64bcfd0ac064da035d6801912b106420ef7c828fe725'),
        data: [file('onnx/model.onnx.data', 366467840, '0162e666781098c58f65150ec10f546e89bd2105377ca6b6a77e95e2ce6fdef1')],
      },
      codec: { graph: file('onnx/codec_decoder.onnx', 134604036, 'bd5f867d37f738b4345898bb6ccf379139633828a7ef70e2b89b4e0c966ebb07') },
    },
    defaults: { voice: 'dave' },
  },
  {
    id: 'orpheus-3b',
    name: 'Orpheus 3B',
    publisher: 'Canopy Labs',
    parameters: '3.3B',
    parameterCount: 3_300_000_000,
    summary: 'Expressive English speech from a Llama-based model, with eight voices and tags such as <laugh> and <sigh>.',
    license: {
      name: 'Apache 2.0 and Llama 3.2 Community License',
      url: 'https://huggingface.co/danyalmalik/Orpheus-3B-ONNX-WebGPU/blob/34d7baff99b0b0ae8cc278cc61094537411b4da6/LICENSE-Llama-3.2.txt',
      notice:
        'Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. ' +
        'Includes the SNAC codec (MIT). Canopy Labs prohibits impersonation without consent, misinformation and deception',
    },
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Orpheus-3B-ONNX-WebGPU`, revision: '34d7baff99b0b0ae8cc278cc61094537411b4da6' },
    precision: 'q4',
    sampleRate: 24000,
    voices: ['tara', 'leah', 'jess', 'leo', 'dan', 'mia', 'zac', 'zoe'].map((id) => ({ id, label: id[0]!.toUpperCase() + id.slice(1) })),
    pipeline: {
      kind: 'orpheus',
      tokenizer: file('tokenizer.json', 22849547, 'fc3fecb199b4170636dbfab986d25f628157268d37b861f9cadaca60b1353bce'),
      lm: {
        graph: file('onnx/model.onnx', 246135, 'd43a92d7a2d15257f0284d6b5c3c12af49a40feb1e5a4962545eea606713b647'),
        data: [file('onnx/model.onnx.data', 3194030336, 'e38746bbaab94c87d0c34ce0a35b15618421a75e29f17a32f0f6dc70554c5c0d')],
      },
      snac: { graph: file('onnx/snac_decoder_q4.onnx', 8446438, 'e54e38693b7bd504ec76226e52f24dad894dbf2ec993d8799347a3ceb1ca1e2c') },
    },
    defaults: { voice: 'tara' },
  },
  {
    id: 'maya1',
    name: 'Maya1',
    publisher: 'Maya Research',
    parameters: '3.3B',
    parameterCount: 3_300_000_000,
    summary: 'Expressive English speech in a voice you describe in words, with tags such as <laugh>, <sigh> and <whisper>.',
    license: {
      name: 'Apache 2.0 and Llama 3.2 Community License',
      url: 'https://huggingface.co/danyalmalik/Maya1-ONNX-WebGPU/blob/751d3733144ee125541db50fbb6c57babdccf6ea/LICENSE-Llama-3.2.txt',
      notice:
        'Built with Llama. Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved. ' +
        'Includes the SNAC codec (MIT). Maya Research asks that it not be used for deceptive impersonation or fraud',
    },
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Maya1-ONNX-WebGPU`, revision: '751d3733144ee125541db50fbb6c57babdccf6ea' },
    precision: 'q8',
    sampleRate: 24000,
    voices: [{ id: 'designed', label: 'Described voice' }],
    pipeline: {
      kind: 'orpheus',
      variant: 'maya1',
      tokenizer: file('tokenizer.json', 22853258, '6c5e5b1d89b7e3738e5a5a4f93c326d8f3292ea83f9c560b8dbb6d66fb851973'),
      lm: {
        graph: file('onnx/model.onnx', 211227, '7f67e303df284487d067eae2f888a68fd04d23046d41113bc03814b4d426f929'),
        data: [file('onnx/model.onnx.data', 4505403392, '29d1693c5f4f97d1f267c3a1bcb79728e546ae89dccd2c1b64a6195166309366')],
      },
      snac: { graph: file('onnx/snac_decoder_q4.onnx', 8446438, 'e54e38693b7bd504ec76226e52f24dad894dbf2ec993d8799347a3ceb1ca1e2c') },
    },
    defaults: {
      voice: 'designed',
      instruct:
        'Realistic female voice in the 30s age with an american accent. Normal pitch, warm timbre, conversational pacing, neutral tone at medium intensity.',
    },
  },
  {
    id: 'voxcpm2',
    name: 'VoxCPM2',
    publisher: 'OpenBMB',
    parameters: '2B',
    parameterCount: 2_000_000_000,
    summary: 'Natural 48 kHz speech in 30 languages, written as continuous audio rather than tokens. Its voice can change from one clip to the next.',
    license: APACHE_2,
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/VoxCPM2-ONNX-WebGPU`, revision: '4b9c4acdf2e6caa718be098c7e1080a25554e294' },
    precision: 'q4',
    sampleRate: 48000,
    voices: [{ id: 'default', label: 'Default' }],
    pipeline: {
      kind: 'voxcpm',
      tokenizer: file('tokenizer.json', 3676772, 'f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20'),
      textEmbed: { graph: file('onnx/text_embed_q4.onnx', 94013799, '1d74eabef2f84b3ee55b1924a22048022cc32762301d2019efdade11a0828347') },
      baseLm: { graph: file('onnx/base_lm_q4.onnx', 860667961, '4f169f433bf9dbf2f35beaaba7b0c27fb666c5f32b657cb246376632b8024304') },
      residualLm: { graph: file('onnx/residual_lm_q4.onnx', 236262053, '1ad3b35969040f2e254b66dd1b016e10a5e2c2dbee2df63818f6e87edb95c032') },
      bridge: { graph: file('onnx/bridge_q4.onnx', 9209030, '5b9b494640a7811e38ccd582d467b1b851eb845e0838e4bbe984899e4e1df66b') },
      featEncoder: { graph: file('onnx/feat_encoder_q4.onnx', 165510260, 'a8b7f58d769169bab18a0864457d8f12ac1f269e9c23506a3ac0c04e356701f4') },
      ditStep: { graph: file('onnx/dit_step_q4.onnx', 136039200, '6c92d7dac948c30843e784082438fe270cbb4e442069f3ba4d8a5ef3bb0367bb') },
      vae: { graph: file('onnx/vae_decoder_q4c8.onnx', 51935118, '1b703b656ab68c8d7546531bc367cf5442e9e1fb52c6afba08e84bd069e20f3c') },
    },
    defaults: { voice: 'default' },
  },
  {
    id: 'zonos-v0.1',
    name: 'Zonos',
    publisher: 'Zyphra',
    parameters: '1.6B',
    parameterCount: 1_600_000_000,
    summary: 'Expressive 44.1 kHz English speech from phonemes, in a neutral voice of its own. Its voice can change from one clip to the next.',
    license: { ...APACHE_2, notice: 'Includes Descript’s DAC codec (MIT)' },
    source: { host: 'huggingface', repo: `${OWN_EXPORTS}/Zonos-v0.1-ONNX-WebGPU`, revision: 'feb0d9309265d61c047f6c607c87110ebe3a8ef4' },
    precision: 'q4',
    sampleRate: 44100,
    voices: [{ id: 'default', label: 'Default' }],
    pipeline: {
      kind: 'zonos',
      lexicon: G2P_LEXICON,
      config: file('zonos.json', 1995, 'c7e583e98d047e0580440ab5da85260e25ea10b5269169acf076153ab54ca146'),
      prefixRows: file('prefix_rows_default.bin', 98304, 'b1dbdbf0ad78c44612ebc5f15190bf953ee90080b7c033efd68d1793e4a3f50a'),
      phonemeEmbed: { graph: file('onnx/phoneme_embed_q4.onnx', 242261, 'c141cc4ed3136d1e8e01412e403f4dd30af83aa83db5dd7ef5d68872ac2e2f61') },
      prefix: { graph: file('onnx/prefix_q4.onnx', 2646841, '04c11397cc325798e7306a493582036094d00281535dc4a9a01bee00faca04e1') },
      embedCodes: { graph: file('onnx/embed_codes_q4.onnx', 11825438, 'fe3eed313cfe0c6679c0f9afa255507e2125af04180e38b954f77e9095fa822c') },
      backbone: { graph: file('onnx/backbone_q4.onnx', 1003190548, '16356f782c8c8715c4aedc67aaff5a2813bab9c7d7903218ac03511d6b839e88') },
      dac: { graph: file('onnx/dac_decoder_q4.onnx', 34211154, 'df7c99ddb4b7a9db992eec7d582db9bd31b93589b1585f02dec85edf12758848') },
    },
    defaults: { voice: 'default' },
  },
];

export const SPEECH_MODELS: readonly SpeechModelSpec[] = ALL_SPEECH_MODELS.filter((m) => !m.devOnly || import.meta.env.DEV);

export function findSpeechModel(id: string): SpeechModelSpec | undefined {
  return SPEECH_MODELS.find((m) => m.id === id);
}

/** Every file of a model, smallest first (voices and configs before graphs). */
export function speechModelFiles(model: SpeechModelSpec): ArtifactFile[] {
  const files: ArtifactFile[] = [];
  const walk = (value: unknown) => {
    if (!value || typeof value !== 'object') return;
    if ('sha256' in value && 'path' in value) files.push(value as ArtifactFile);
    else for (const v of Object.values(value)) walk(v);
  };
  walk(model.pipeline);
  walk(model.voices);
  return [...new Set(files)].sort((a, b) => a.sizeBytes - b.sizeBytes);
}

export function speechModelBytes(model: SpeechModelSpec): number {
  return speechModelFiles(model).reduce((n, f) => n + f.sizeBytes, 0);
}

/** Every graph of a model's pipeline, by role. */
export function speechGraphs(pipeline: SpeechPipeline): Record<string, SpeechGraph> {
  const out: Record<string, SpeechGraph> = {};
  for (const [role, value] of Object.entries(pipeline)) {
    if (typeof value === 'object' && value !== null && 'graph' in value) out[role] = value as SpeechGraph;
  }
  return out;
}
