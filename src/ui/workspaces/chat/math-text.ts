// Models write maths as \( … \), \[ … \], $ … $ or a bare \boxed{…}; the Markdown maths plugin reads
// $$ … $$ (inline when within a line). Converts the former to the latter, leaving code untouched.
// A $ … $ counts as maths only when it looks like it (see INLINE), so prices ("$5 and $10") stay text.

const CODE = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g;
const DISPLAY = /(\$\$[\s\S]*?\$\$)/g;
// $ … $ within a line: no space just inside either dollar, and no digit (or word) just after the closing one, as in
// Pandoc; an escaped \$ is not a delimiter.
const INLINE = /(?<![\\$\w])\$(?![\s$])([^$\n]*?[^\s\\$])\$(?![\w$])/g;
// What makes a $ … $ maths: a TeX command or a maths symbol, or a short variable or call ($x$, $M_1$, $f(x)$).
const MATHS = /[\\^_=<>{}]|^[A-Za-z](?:'|_?\w)?(?:\([^()]*\))?$/;

export function normalizeMath(text: string): string {
  return text
    .split(CODE)
    .map((part, i) => (i % 2 === 1 ? part : convert(part)))
    .join('');
}

function convert(text: string): string {
  let out = text
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, m: string) => `$$${m}$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, m: string) => `$$${m}$$`);
  // A \boxed{…} outside any $$ … $$ (balanced braces, one level of nesting).
  out = out.replace(/(\$\$[\s\S]*?\$\$)|\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}/g, (match, math: string | undefined) =>
    math ? match : `$$${match}$$`,
  );
  // Single-dollar maths outside the $$ … $$ above.
  return out
    .split(DISPLAY)
    .map((part, i) =>
      i % 2 === 1 ? part : part.replace(INLINE, (match, math: string) => (MATHS.test(math) ? `$$${math}$$` : match)),
    )
    .join('');
}
