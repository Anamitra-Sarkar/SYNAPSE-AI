import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

describe("workspace accessibility markup", () => {
  it("includes keyboard-friendly selection controls and a live generation announcement", async () => {
    const source = await readFile(resolve(process.cwd(), "client/src/pages/SynapseWorkspace.tsx"), "utf8");
    expect(source).toContain("aria-live=\"polite\"");
    expect(source).toContain("aria-pressed");
    expect(source).toContain("onKeyDown");
  });
});
