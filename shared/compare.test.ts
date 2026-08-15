import { describe, expect, it } from "vitest";
import { MAX_COMPARE_CONCEPTS, updateCompareSelection } from "./compare";

describe("compare selection", () => {
  it("caps the tray at three concepts and preserves its existing selection", () => {
    const result = updateCompareSelection([11, 12, 13], 14);
    expect(MAX_COMPARE_CONCEPTS).toBe(3);
    expect(result).toEqual({ next: [11, 12, 13], limitReached: true });
  });

  it("removes an already selected concept", () => {
    expect(updateCompareSelection([11, 12], 11)).toEqual({ next: [12], limitReached: false });
  });
});
