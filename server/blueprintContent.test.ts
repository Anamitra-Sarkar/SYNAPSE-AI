import { describe, expect, it } from "vitest";
import { resolveBlueprintContent } from "./blueprintContent";
import type { BlueprintArtifact } from "../shared/synapse";

const raw: BlueprintArtifact = {
  overview: "Original model artifact", mvpFeatures: [], architecture: [], dataAndApis: [], buildPlan: [], teamPlan: [], demoFlow: [],
  judgePitch: { opening: "", problem: "", solution: "", proof: "", close: "" }, risks: [], extensions: [], fallbackPlan: "Original fallback",
};

describe("blueprint content resolution", () => {
  it("uses an edit without mutating or replacing the immutable raw model artifact", () => {
    const edited = { ...raw, overview: "User-authored revision" };
    const visible = resolveBlueprintContent(raw, edited);
    expect(visible.overview).toBe("User-authored revision");
    expect(raw.overview).toBe("Original model artifact");
  });
});
