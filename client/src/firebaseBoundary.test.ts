import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

describe("Firebase credential boundary", () => {
  it("keeps Firebase Admin credentials out of browser code", async () => {
    const source = await readFile(resolve(process.cwd(), "client/src/lib/firebase.ts"), "utf8");
    expect(source).not.toContain("FIREBASE_PRIVATE_KEY");
    expect(source).not.toContain("FIREBASE_CLIENT_EMAIL");
  });
});
