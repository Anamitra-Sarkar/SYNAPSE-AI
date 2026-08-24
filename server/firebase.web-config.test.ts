import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

describe("Firebase web configuration", () => {
  it("targets the verified non-default Firestore database in the browser client", async () => {
    const clientSource = await readFile(resolve(process.cwd(), "client/src/lib/firebase.ts"), "utf8");
    expect(clientSource).toContain('getFirestore(firebaseApp, FIRESTORE_DATABASE_ID)');
    expect(clientSource).toContain('"database-1"');
  });

  it("accepts the configured web API key for a non-mutating Firebase token lookup", async () => {
    const apiKey = process.env.VITE_FIREBASE_API_KEY;
    expect(apiKey).toBeTruthy();

    const response = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:lookup?key=${apiKey}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ idToken: "intentionally-invalid-token" }),
    });
    const payload = await response.json() as { error?: { message?: string } };

    expect(response.status).toBe(400);
    expect(payload.error?.message).toMatch(/INVALID_ID_TOKEN|INVALID_IDP_RESPONSE|INVALID_REFRESH_TOKEN/);
  }, 20_000);
});
