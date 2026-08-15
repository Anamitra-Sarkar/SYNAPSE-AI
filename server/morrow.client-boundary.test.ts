import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

describe("Morrow client boundaries", () => {
  it("protects the direct project workspace, keeps Admin credentials out of browser configuration, and sends only Firebase bearer tokens", async () => {
    const [routes, firebaseClient, trpcClient] = await Promise.all([
      readFile(resolve(process.cwd(), "client/src/pages/MorrowApp.tsx"), "utf8"),
      readFile(resolve(process.cwd(), "client/src/lib/firebase.ts"), "utf8"),
      readFile(resolve(process.cwd(), "client/src/main.tsx"), "utf8"),
    ]);
    expect(routes).toContain('<Route path="/app/projects/:id"><Protected>');
    expect(routes).not.toContain('/app/projects/:id/studio');
    expect(routes).toContain('<Route path="/app"><Protected>');
    expect(firebaseClient).not.toContain("FIREBASE_PRIVATE_KEY");
    expect(firebaseClient).not.toContain("FIREBASE_CLIENT_EMAIL");
    expect(trpcClient).toContain("firebaseAuth?.currentUser?.getIdToken()");
    expect(trpcClient).not.toContain("manus-cookie");
    expect(trpcClient).not.toContain("startLogin");
  });
});
