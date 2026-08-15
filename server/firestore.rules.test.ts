import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

describe("Firestore ownership rules", () => {
  it("requires authenticated project ownership and denies deletion", async () => {
    const rules = await readFile(resolve(process.cwd(), "firestore.rules"), "utf8");
    expect(rules).toContain("request.auth.uid");
    expect(rules).toContain("request.resource.data.ownerId == request.auth.uid");
    expect(rules).toContain("allow delete: if false");
  });

  it("protects nested Morrow artifacts with the owning project rule", async () => {
    const rules = await readFile(resolve(process.cwd(), "firestore.rules"), "utf8");
    expect(rules).toContain("match /projects/{projectId}");
    expect(rules).toContain("match /{document=**} { allow read, write: if ownsProject(projectId); }");
  });

  it("keeps the active Morrow API workflow on the Firestore repository", async () => {
    const router = await readFile(resolve(process.cwd(), "server/routers/synapse.ts"), "utf8");
    expect(router).toContain('from "../firestoreDb"');
    expect(router).not.toContain("synapseDb");
    expect(router).not.toContain("DATABASE_URL");
  });
});
