// Models write maths as \( … \), \[ … \] or a bare \boxed{…}; the Markdown maths plugin reads
// $$ … $$ (inline when within a line). Converts the former to the latter, leaving code untouched.
// Single dollars stay plain text, so prices ("$5 and $10") aren't taken for maths.

const CODE = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g;

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
  return out;
}
