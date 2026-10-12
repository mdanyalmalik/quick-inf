// Filtering the model list by what the user types: every word must match the model's name, publisher,
// size or kind (case-insensitive), e.g. "qwen 4b" or "thinking".

export interface Searchable {
  name: string;
  publisher: string;
  parameters: string;
  reasoning?: boolean;
}

export function matchesSearch(model: Searchable, query: string): boolean {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return true;
  const text = `${model.name} ${model.publisher} ${model.parameters} ${model.reasoning ? 'thinking reasoning thinks' : ''}`.toLowerCase();
  return words.every((w) => text.includes(w));
}
