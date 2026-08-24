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

  it("stacks blueprint actions at the mobile breakpoint to keep every action inside the viewport", async () => {
    const source = await readFile(resolve(process.cwd(), "client/src/index.css"), "utf8");
    expect(source).toContain(".morrow-integrated-studio .blueprint-actions{width:100%;align-items:stretch;flex-direction:column}");
    expect(source).toContain(".morrow-integrated-studio .blueprint-actions>button{width:100%;min-height:44px}");
  });
});
