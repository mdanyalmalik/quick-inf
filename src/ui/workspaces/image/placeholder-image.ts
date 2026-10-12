// Stand-in pictures until image models run: an abstract image drawn from the prompt and seed (the same
// prompt and seed always give the same picture), labelled as a placeholder. `drawFrame` imitates a
// diffusion model's steps: the picture starts as noise and sharpens as `progress` goes from 0 to 1.

/** Draws the finished picture onto `ctx`. */
function drawPicture(ctx: CanvasRenderingContext2D, prompt: string, seed: number, width: number, height: number): void {
  const random = rng(hash(prompt) ^ seed);
  const hue = Math.floor(random() * 360);
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, `hsl(${hue} 55% 32%)`);
  gradient.addColorStop(1, `hsl(${(hue + 60 + random() * 90) % 360} 60% 18%)`);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  for (let i = 0; i < 9; i++) {
    ctx.beginPath();
    ctx.fillStyle = `hsla(${(hue + random() * 120) % 360} 70% ${45 + random() * 25}% / ${0.25 + random() * 0.35})`;
    ctx.arc(random() * width, random() * height, (0.08 + random() * 0.25) * Math.min(width, height), 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
  ctx.fillRect(0, height - 44, width, 44);
  ctx.fillStyle = '#fff';
  ctx.font = `600 ${Math.round(width / 28)}px system-ui, sans-serif`;
  ctx.fillText('Placeholder', 14, height - 26);
  ctx.font = `${Math.round(width / 38)}px system-ui, sans-serif`;
  ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
  ctx.fillText(ellipsis(ctx, prompt, width - 28), 14, height - 10);
}

const pictures = new Map<string, HTMLCanvasElement>();

function picture(prompt: string, seed: number, width: number, height: number): HTMLCanvasElement {
  const cacheKey = `${seed}|${width}x${height}|${prompt}`;
  let canvas = pictures.get(cacheKey);
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (ctx) drawPicture(ctx, prompt, seed, width, height);
    pictures.set(cacheKey, canvas);
  }
  return canvas;
}

/** One step of the reveal: noise fading out and blur clearing as `progress` goes from 0 to 1. */
export function drawFrame(target: HTMLCanvasElement, prompt: string, seed: number, progress: number): void {
  const { width, height } = target;
  const ctx = target.getContext('2d');
  if (!ctx) return;
  const t = Math.min(1, Math.max(0, progress));
  ctx.filter = t < 1 ? `blur(${Math.round((1 - t) * 24)}px)` : 'none';
  ctx.drawImage(picture(prompt, seed, width, height), 0, 0);
  ctx.filter = 'none';
  if (t < 1) {
    const noise = ctx.createImageData(width, height);
    const random = rng(seed + Math.round(t * 1000));
    const alpha = Math.round((1 - t) ** 1.5 * 255);
    for (let i = 0; i < noise.data.length; i += 4) {
      const v = Math.floor(random() * 255);
      noise.data[i] = noise.data[i + 1] = noise.data[i + 2] = v;
      noise.data[i + 3] = alpha;
    }
    const layer = document.createElement('canvas');
    layer.width = width;
    layer.height = height;
    layer.getContext('2d')?.putImageData(noise, 0, 0);
    ctx.drawImage(layer, 0, 0);
  }
}

/** The finished picture as a PNG data URL, for downloading. */
export function placeholderImageUrl(prompt: string, seed: number, width: number, height: number): string {
  return picture(prompt, seed, width, height).toDataURL('image/png');
}

function ellipsis(ctx: CanvasRenderingContext2D, text: string, max: number): string {
  if (ctx.measureText(text).width <= max) return text;
  let out = text;
  while (out.length > 1 && ctx.measureText(`${out}…`).width > max) out = out.slice(0, -1);
  return `${out}…`;
}

function hash(text: string): number {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return h >>> 0;
}

function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}
