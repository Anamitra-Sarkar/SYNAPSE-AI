import { describe, expect, it } from "vitest";

describe("Groq credential", () => {
  it("authenticates against the server-side models endpoint", async () => {
    const apiKey = process.env.GROQ_API_KEY;
    expect(apiKey, "GROQ_API_KEY must be configured for the server").toBeTruthy();

    const response = await fetch("https://api.groq.com/openai/v1/models", {
      headers: { Authorization: `Bearer ${apiKey}` },
    });

    expect(response.status, `Groq credential validation failed with ${response.status}`).toBe(200);
    const payload = await response.json() as { data?: unknown[] };
    expect(Array.isArray(payload.data)).toBe(true);
  }, 20_000);
});
