export const MAX_COMPARE_CONCEPTS = 3;

export function updateCompareSelection(current: number[], conceptId: number) {
  if (current.includes(conceptId)) return { next: current.filter(id => id !== conceptId), limitReached: false };
  if (current.length >= MAX_COMPARE_CONCEPTS) return { next: current, limitReached: true };
  return { next: [...current, conceptId], limitReached: false };
}
