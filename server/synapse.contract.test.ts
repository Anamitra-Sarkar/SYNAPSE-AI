import { describe, expect, it } from "vitest";
import { briefSchema } from "./routers/synapse";

describe("brief contract", () => {
  it("rejects a generation request without skills or a viable problem statement", () => {
    const result = briefSchema.safeParse({
      skills: [], problemStatement: "Too short", availableHours: 1, teamSize: 0, teamRoles: [], preferredTech: [], resources: [], constraints: [],
      scoringWeights: { skillsFit: 25, feasibility: 30, novelty: 15, impact: 15, demoPotential: 15 },
    });
    expect(result.success).toBe(false);
  });
});
