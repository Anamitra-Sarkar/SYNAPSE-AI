import { describe, expect, it } from "vitest";
import { extractJson, GroqPipelineError } from "./groq";

describe("Groq response handling", () => {
  it("maps malformed provider output to a retryable invalid-response error", () => {
    try {
      extractJson("This is not valid JSON");
      throw new Error("Expected malformed JSON to throw");
    } catch (error) {
      expect(error).toBeInstanceOf(GroqPipelineError);
      expect((error as GroqPipelineError).code).toBe("INVALID_RESPONSE");
    }
  });
});
