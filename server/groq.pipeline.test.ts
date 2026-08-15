import { describe, expect, it } from "vitest";
import { generateConcepts } from "./groq";

describe("Groq concept pipeline", () => {
  const liveTest = process.env.RUN_GROQ_INTEGRATION === "true" ? it : it.skip;

  liveTest("creates four to six distinct, structured, scored concept cards", async () => {
    const result = await generateConcepts("integration-test-user", {
      title: "Accessible transit demo",
      skills: ["React", "TypeScript", "Figma"],
      problemStatement: "Help visually impaired commuters make safer last-mile transit decisions during a 24-hour hackathon.",
      availableHours: 24,
      teamSize: 3,
      teamRoles: ["Frontend", "Product", "AI integration"],
      domain: "Accessibility",
      preferredTech: ["React", "Groq"],
      resources: ["Public transit feeds"],
      constraints: ["No hardware", "Mobile-friendly demo"],
      scoringWeights: { skillsFit: 25, feasibility: 30, novelty: 15, impact: 15, demoPotential: 15 },
    });

    expect(result.concepts.length).toBeGreaterThanOrEqual(4);
    expect(result.concepts.length).toBeLessThanOrEqual(6);
    expect(new Set(result.concepts.map(concept => concept.name.toLowerCase())).size).toBe(result.concepts.length);
    expect(result.concepts.every(concept => concept.scores.overall >= 1 && concept.scores.overall <= 10)).toBe(true);
    expect(result.concepts.every(concept => concept.assumptions.length > 0 && concept.risks.length > 0)).toBe(true);
  }, 60_000);
});
