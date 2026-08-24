import { describe, expect, it } from "vitest";
import { extractJson, GroqPipelineError, PREFERRED_MODELS, retryTransientGroq } from "./groq";

describe("Groq response handling", () => {
  it("prioritizes the intended Llama 3.3 model before capacity-sensitive fallbacks", () => {
    expect(PREFERRED_MODELS[0]).toBe("llama-3.3-70b-versatile");
  });

  it("maps malformed provider output to a retryable invalid-response error", () => {
    try {
      extractJson("This is not valid JSON");
      throw new Error("Expected malformed JSON to throw");
    } catch (error) {
      expect(error).toBeInstanceOf(GroqPipelineError);
      expect((error as GroqPipelineError).code).toBe("INVALID_RESPONSE");
    }
  });

  it("retries transient provider failures with bounded attempts", async () => {
    let calls = 0;
    const pauses: number[] = [];
    const result = await retryTransientGroq(
      async () => {
        calls += 1;
        if (calls < 3) throw new GroqPipelineError("Groq is busy.", "RATE_LIMITED", 5);
        return "ready";
      },
      { wait: async (delayMs) => { pauses.push(delayMs); } },
    );

    expect(result).toBe("ready");
    expect(calls).toBe(3);
    expect(pauses).toEqual([5, 5]);
  });

  it("does not retry non-transient provider failures", async () => {
    let calls = 0;
    await expect(retryTransientGroq(async () => {
      calls += 1;
      throw new GroqPipelineError("Configuration missing.", "MISCONFIGURED");
    }, { wait: async () => undefined })).rejects.toMatchObject({ code: "MISCONFIGURED" });
    expect(calls).toBe(1);
  });
});
