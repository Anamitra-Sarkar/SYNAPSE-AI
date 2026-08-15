import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

describe("Groq credential boundary", () => {
  it("does not reference GROQ_API_KEY from browser source", async () => {
    const clientSource = await readFile(resolve(process.cwd(), "client/src/pages/SynapseWorkspace.tsx"), "utf8");
    const clientBootstrap = await readFile(resolve(process.cwd(), "client/src/main.tsx"), "utf8");
    expect(`${clientSource}\n${clientBootstrap}`).not.toContain("GROQ_API_KEY");
  });
});
