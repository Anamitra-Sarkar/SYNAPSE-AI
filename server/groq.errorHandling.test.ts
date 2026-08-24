import { describe, expect, it } from "vitest";
import { extractJson, GroqPipelineError, GROQ_TOKEN_BUDGETS, modelGenerationOptions, normalizeBlueprintPayload, normalizeGroqListField, normalizeScorecardPayload, PREFERRED_MODELS, retryTransientGroq, supportsStrictStructuredOutput } from "./groq";

describe("Groq response handling", () => {
  it("prioritizes the available Qwen JSON-mode model before strict-output fallbacks", () => {
    expect(PREFERRED_MODELS[0]).toBe("qwen/qwen3.6-27b");
  });

  it("uses strict structured output only for the supported GPT-OSS models", () => {
    expect(supportsStrictStructuredOutput("openai/gpt-oss-20b")).toBe(true);
    expect(supportsStrictStructuredOutput("qwen/qwen3.6-27b")).toBe(false);
  });

  it("uses Qwen non-thinking mode to reserve the response budget for JSON", () => {
    expect(modelGenerationOptions("qwen/qwen3.6-27b")).toEqual({ reasoning_effort: "none", reasoning_format: "hidden" });
    expect(modelGenerationOptions("openai/gpt-oss-20b")).toEqual({});
  });

  it("keeps the immediate concept-to-blueprint token budget below the account TPM limit", () => {
    expect(Object.values(GROQ_TOKEN_BUDGETS).reduce((total, value) => total + value, 0)).toBeLessThan(8_000);
  });

  it("normalizes compact planning lists before validation", () => {
    expect(normalizeGroqListField("React, Firebase\n- Demo dashboard")).toEqual(["React", "Firebase", "Demo dashboard"]);
  });

  it("normalizes predictable Qwen blueprint text fields without changing core sections", () => {
    expect(normalizeBlueprintPayload({
      overview: { summary: "A compact overview" },
      fallbackPlan: { plan: "Use manual input" },
    })).toMatchObject({
      overview: "A compact overview",
      fallbackPlan: "Use manual input",
      extensions: ["Add a lightweight feedback loop", "Extend the MVP after the demo"],
    });
  });

  it("fills omitted Qwen scorecard fallback fields before validation", () => {
    expect(normalizeScorecardPayload({
      evaluations: [{ rank: 4, assumptions: undefined, risks: undefined, nextStep: undefined }],
    })).toMatchObject({
      evaluations: [{
        rank: 4,
        assumptions: ["Validate the core assumption with a representative user."],
        risks: ["Keep the first version scoped to the available build window."],
        nextStep: "Validate the narrowest viable demo flow.",
      }],
    });
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
