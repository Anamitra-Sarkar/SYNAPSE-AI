import { describe, expect, it } from "vitest";
import { buildMarkdown } from "./routers/synapse";
import type { BlueprintArtifact, ConceptCard } from "../shared/synapse";

const concept: ConceptCard = { rank: 1, name: "SignalBridge", hook: "A clearer path to care", targetUser: "Clinic staff", painPoint: "No-shows waste scarce appointments", solution: "A rescheduling assistant", differentiator: "Uses simple, respectful nudges", difficulty: "Intermediate", buildTime: "24 hours", techStack: ["React"], scores: { skillsFit: 9, feasibility: 8, novelty: 7, impact: 9, demoPotential: 8, overall: 8.3 }, scoreRationale: { skillsFit: "Matches the team", feasibility: "Focused mock", novelty: "Intent-aware", impact: "Protects access", demoPotential: "Clear before-and-after" }, assumptions: ["A mock is acceptable"], risks: ["No live data"], nextStep: "Prototype the intake flow" };
const blueprint: BlueprintArtifact = { overview: "A scoped clinic rescheduling prototype.", mvpFeatures: [{ title: "Intake", detail: "Capture appointment intent", priority: "Must" }], architecture: [{ layer: "Client", purpose: "Collect choices", technologies: ["React"] }], dataAndApis: [{ name: "Mock data", need: "Demo appointments" }], buildPlan: [{ window: "Hour 1", goal: "Set the path", tasks: ["Sketch screens"] }], teamPlan: [{ role: "Frontend", responsibilities: ["Build flow"] }], demoFlow: ["Open a missed appointment"], judgePitch: { opening: "Missed care is avoidable.", problem: "Appointments disappear", solution: "A quick alternative", proof: "Show recovery", close: "Protect access" }, risks: [{ risk: "No live data", mitigation: "Use synthetic records" }], extensions: ["Add reminders"], fallbackPlan: "Use a static prototype." };

describe("portable Markdown export", () => {
  it("contains the essential self-contained blueprint sections", () => {
    const output = buildMarkdown("Care pilot", concept, blueprint);
    expect(output).toContain("# Care pilot");
    expect(output).toContain("## MVP");
    expect(output).toContain("## Architecture");
    expect(output).toContain("## Build plan");
    expect(output).toContain("## Fallback plan");
  });
});
