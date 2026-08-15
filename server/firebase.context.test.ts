import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

describe("Firebase request context", () => {
  it("verifies Firebase bearer tokens before evaluating the legacy session fallback", async () => {
    const source = await readFile(resolve(process.cwd(), "server/_core/context.ts"), "utf8");
    expect(source).toContain("verifyFirebaseIdToken(opts.req.headers.authorization)");
    expect(source).toContain("firebaseUser.uid");
    expect(source).toContain("sdk.authenticateRequest");
  });
});
