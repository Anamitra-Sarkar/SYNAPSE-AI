import { describe, expect, it } from "vitest";

describe("Firebase web configuration", () => {
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
