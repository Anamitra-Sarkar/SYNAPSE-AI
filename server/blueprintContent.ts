import type { BlueprintArtifact } from "../shared/synapse";

export function resolveBlueprintContent(rawModelOutput: BlueprintArtifact, editedContent?: BlueprintArtifact) {
  return editedContent ?? rawModelOutput;
}
