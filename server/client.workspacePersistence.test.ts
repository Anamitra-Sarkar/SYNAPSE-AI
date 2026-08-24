import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

describe("persisted browser workspace safeguards", () => {
  it("restores the route project identifier before enabling blueprint promotion", async () => {
    const source = await readFile(resolve(process.cwd(), "client/src/pages/SynapseWorkspace.tsx"), "utf8");
    expect(source).toContain("setProjectId(morrowProjectId);");
    expect(source).toContain("if (!promoteId || !projectId) return;");
  });

  it("merges client blueprint display artifacts without replacing server-owned export context", async () => {
    const source = await readFile(resolve(process.cwd(), "client/src/lib/projectRepository.ts"), "utf8");
    expect(source).toContain('"blueprints", "latest"), { ownerId, immutableOutput, userEdits, blueprintId, selectedConceptId, updatedAt: serverTimestamp() }, { merge: true }');
  });
});
