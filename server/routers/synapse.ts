import { TRPCError } from "@trpc/server";
import { z } from "zod";
import type { BlueprintArtifact, BriefInput, ConceptCard, MarkdownExport, ScoreWeights } from "../../shared/synapse";
import { generateBlueprint, generateConcepts, GroqPipelineError, normalizeBrief } from "../groq";
import { resolveBlueprintContent } from "../blueprintContent";
import * as synapseDb from "../synapseDb";
import { protectedProcedure, router } from "../_core/trpc";

const scoreWeightsSchema = z.object({ skillsFit: z.number().min(0).max(100), feasibility: z.number().min(0).max(100), novelty: z.number().min(0).max(100), impact: z.number().min(0).max(100), demoPotential: z.number().min(0).max(100) });
export const briefSchema = z.object({
  title: z.string().max(160).optional(),
  skills: z.array(z.string().min(1).max(48)).min(1).max(12),
  problemStatement: z.string().min(12).max(4000),
  availableHours: z.number().int().min(2).max(240),
  teamSize: z.number().int().min(1).max(12),
  teamRoles: z.array(z.string().min(1).max(48)).max(12),
  domain: z.string().max(120).optional(),
  preferredTech: z.array(z.string().min(1).max(48)).max(12),
  resources: z.array(z.string().min(1).max(120)).max(12),
  constraints: z.array(z.string().min(1).max(180)).max(12),
  scoringWeights: scoreWeightsSchema,
});
const scoreDimensions = z.enum(["skillsFit", "feasibility", "novelty", "impact", "demoPotential"]);
const blueprintSchema = z.object({
  overview: z.string(),
  mvpFeatures: z.array(z.object({ title: z.string(), detail: z.string(), priority: z.enum(["Must", "Should", "Could"]) })),
  architecture: z.array(z.object({ layer: z.string(), purpose: z.string(), technologies: z.array(z.string()) })),
  dataAndApis: z.array(z.object({ name: z.string(), need: z.string(), alternative: z.string().optional() })),
  buildPlan: z.array(z.object({ window: z.string(), goal: z.string(), tasks: z.array(z.string()) })),
  teamPlan: z.array(z.object({ role: z.string(), responsibilities: z.array(z.string()) })),
  demoFlow: z.array(z.string()),
  judgePitch: z.object({ opening: z.string(), problem: z.string(), solution: z.string(), proof: z.string(), close: z.string() }),
  risks: z.array(z.object({ risk: z.string(), mitigation: z.string() })),
  extensions: z.array(z.string()),
  fallbackPlan: z.string(),
});

function providerError(error: unknown): never {
  if (error instanceof GroqPipelineError) {
    const code = error.code === "RATE_LIMITED" ? "TOO_MANY_REQUESTS" : error.code === "MISCONFIGURED" ? "PRECONDITION_FAILED" : "BAD_GATEWAY";
    throw new TRPCError({ code, message: error.message });
  }
  throw error;
}

function markdownList(items: string[]) { return items.map(item => `- ${item}`).join("\n"); }

export function buildMarkdown(title: string, concept: ConceptCard, blueprint: BlueprintArtifact) {
  return `# ${title}\n\n## ${concept.name}\n\n> ${concept.hook}\n\n## Problem and solution\n\n**Target user:** ${concept.targetUser}\n\n**Pain point:** ${concept.painPoint}\n\n**Solution:** ${concept.solution}\n\n**Differentiator:** ${concept.differentiator}\n\n## Scorecard\n\n| Dimension | Score |\n| --- | ---: |\n| Skills fit | ${concept.scores.skillsFit}/10 |\n| Feasibility | ${concept.scores.feasibility}/10 |\n| Novelty | ${concept.scores.novelty}/10 |\n| Impact | ${concept.scores.impact}/10 |\n| Demo potential | ${concept.scores.demoPotential}/10 |\n| Weighted overall | ${concept.scores.overall}/10 |\n\n## MVP\n\n${blueprint.mvpFeatures.map(item => `### ${item.title} — ${item.priority}\n${item.detail}`).join("\n\n")}\n\n## Architecture\n\n${blueprint.architecture.map(item => `### ${item.layer}\n${item.purpose}\n\nTech: ${item.technologies.join(", ")}`).join("\n\n")}\n\n## Data and APIs\n\n${blueprint.dataAndApis.map(item => `- **${item.name}:** ${item.need}${item.alternative ? ` Alternative: ${item.alternative}` : ""}`).join("\n")}\n\n## Build plan\n\n${blueprint.buildPlan.map(item => `### ${item.window}: ${item.goal}\n${markdownList(item.tasks)}`).join("\n\n")}\n\n## Team plan\n\n${blueprint.teamPlan.map(item => `### ${item.role}\n${markdownList(item.responsibilities)}`).join("\n\n")}\n\n## Demo flow\n\n${blueprint.demoFlow.map((step, index) => `${index + 1}. ${step}`).join("\n")}\n\n## Judge pitch\n\n${blueprint.judgePitch.opening}\n\n**Problem:** ${blueprint.judgePitch.problem}\n\n**Solution:** ${blueprint.judgePitch.solution}\n\n**Proof:** ${blueprint.judgePitch.proof}\n\n**Close:** ${blueprint.judgePitch.close}\n\n## Risks and mitigations\n\n${blueprint.risks.map(item => `- **${item.risk}:** ${item.mitigation}`).join("\n")}\n\n## Extensions\n\n${markdownList(blueprint.extensions)}\n\n## Fallback plan\n\n${blueprint.fallbackPlan}\n`;
}

export const synapseRouter = router({
  projects: protectedProcedure.query(({ ctx }) => synapseDb.listProjects(ctx.user.id)),
  workspace: protectedProcedure.input(z.object({ projectId: z.number().int().positive() })).query(async ({ ctx, input }) => {
    const workspace = await synapseDb.getProjectWorkspace(ctx.user.id, input.projectId);
    if (!workspace) throw new TRPCError({ code: "NOT_FOUND", message: "Workspace not found." });
    return workspace;
  }),
  generate: protectedProcedure.input(briefSchema).mutation(async ({ ctx, input }) => {
    const brief = normalizeBrief(input as BriefInput);
    const { projectId, briefId } = await synapseDb.createProjectWithBrief(ctx.user.id, brief.title || "Untitled hackathon workspace", brief);
    const recipe = { normalizedBrief: brief, model: "pending", promptVersion: "synapse-concepts-v1", schemaVersion: "concept-card-v1", createdAt: new Date().toISOString() };
    const runId = await synapseDb.createGenerationRun(ctx.user.id, projectId, briefId, recipe);
    try {
      const result = await generateConcepts(String(ctx.user.id), brief);
      const persistedConcepts = await synapseDb.completeGenerationRun(ctx.user.id, runId, result.raw, result.concepts, {
        ...recipe,
        normalizedBrief: result.brief,
        model: result.model,
        scoreSnapshot: result.concepts.map(concept => ({ rank: concept.rank, name: concept.name, scores: concept.scores })),
      });
      return { projectId, runId, concepts: persistedConcepts, normalizedBrief: result.brief };
    } catch (error) {
      await synapseDb.failGenerationRun(ctx.user.id, runId, error instanceof Error ? error.message : "Unknown generation error");
      return providerError(error);
    }
  }),
  saveComparison: protectedProcedure.input(z.object({ projectId: z.number().int().positive(), conceptIds: z.array(z.number().int().positive()).min(1).max(3) })).mutation(async ({ ctx, input }) => {
    await synapseDb.saveComparison(ctx.user.id, input.projectId, input.conceptIds);
    return { success: true } as const;
  }),
  promoteToBlueprint: protectedProcedure.input(z.object({ conceptId: z.number().int().positive() })).mutation(async ({ ctx, input }) => {
    const conceptRow = await synapseDb.getConcept(ctx.user.id, input.conceptId);
    if (!conceptRow) throw new TRPCError({ code: "NOT_FOUND", message: "Concept not found." });
    const workspace = await synapseDb.getProjectWorkspace(ctx.user.id, conceptRow.projectId);
    if (!workspace?.brief) throw new TRPCError({ code: "NOT_FOUND", message: "The source brief could not be found." });
    try {
      const result = await generateBlueprint(String(ctx.user.id), workspace.brief, conceptRow.content);
      const blueprintId = await synapseDb.createBlueprint(ctx.user.id, conceptRow.projectId, conceptRow.id, conceptRow.generationRunId, result.blueprint);
      return { blueprintId, blueprint: result.blueprint };
    } catch (error) {
      return providerError(error);
    }
  }),
  blueprint: protectedProcedure.input(z.object({ blueprintId: z.number().int().positive() })).query(async ({ ctx, input }) => {
    const blueprint = await synapseDb.getBlueprint(ctx.user.id, input.blueprintId);
    if (!blueprint) throw new TRPCError({ code: "NOT_FOUND", message: "Blueprint not found." });
    return blueprint;
  }),
  saveBlueprintEdits: protectedProcedure.input(z.object({ blueprintId: z.number().int().positive(), content: blueprintSchema })).mutation(async ({ ctx, input }) => {
    const existing = await synapseDb.getBlueprint(ctx.user.id, input.blueprintId);
    if (!existing) throw new TRPCError({ code: "NOT_FOUND", message: "Blueprint not found." });
    await synapseDb.saveBlueprintEdit(ctx.user.id, input.blueprintId, input.content as BlueprintArtifact);
    return { success: true } as const;
  }),
  exportMarkdown: protectedProcedure.input(z.object({ blueprintId: z.number().int().positive() })).mutation(async ({ ctx, input }) => {
    const blueprint = await synapseDb.getBlueprint(ctx.user.id, input.blueprintId);
    if (!blueprint) throw new TRPCError({ code: "NOT_FOUND", message: "Blueprint not found." });
    const concept = await synapseDb.getConcept(ctx.user.id, blueprint.conceptId);
    const workspace = await synapseDb.getProjectWorkspace(ctx.user.id, blueprint.projectId);
    if (!concept || !workspace) throw new TRPCError({ code: "NOT_FOUND", message: "The blueprint context could not be found." });
    const artifact = resolveBlueprintContent(blueprint.rawModelOutput, blueprint.editedContent);
    const exportValue: MarkdownExport = { filename: `${workspace.project.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "synapse-blueprint"}.md`, content: buildMarkdown(workspace.project.title, concept.content, artifact), exportedAt: new Date().toISOString(), projectId: blueprint.projectId, blueprintId: blueprint.id };
    await synapseDb.recordExport(ctx.user.id, blueprint.projectId, blueprint.id);
    return exportValue;
  }),
});
