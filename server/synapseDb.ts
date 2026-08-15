import { and, desc, eq } from "drizzle-orm";
import { blueprintEdits, blueprints, briefs, concepts, exports, generationRuns, projects, savedComparisons } from "../drizzle/schema";
import type { BlueprintArtifact, BriefInput, ConceptCard, GenerationRecipe } from "../shared/synapse";
import { getDb } from "./db";

const jsonRecord = (value: unknown) => JSON.parse(JSON.stringify(value)) as Record<string, unknown>;
const insertedId = (result: unknown) => ((result as Array<{ insertId?: number }>)[0]?.insertId ?? 0);

async function database() {
  const db = await getDb();
  if (!db) throw new Error("Database is unavailable. Please try again shortly.");
  return db;
}

export async function createProjectWithBrief(userId: number, title: string, content: BriefInput) {
  const db = await database();
  const projectResult = await db.insert(projects).values({ userId, title, status: "exploring" });
  const projectId = insertedId(projectResult);
  const briefResult = await db.insert(briefs).values({ projectId, userId, content: jsonRecord(content) });
  return { projectId, briefId: insertedId(briefResult) };
}

export async function createGenerationRun(userId: number, projectId: number, briefId: number, recipe: GenerationRecipe) {
  const db = await database();
  const result = await db.insert(generationRuns).values({ projectId, briefId, userId, recipe: jsonRecord(recipe), status: "pending" });
  return insertedId(result);
}

export async function completeGenerationRun(userId: number, runId: number, generated: unknown, items: ConceptCard[], recipe: GenerationRecipe) {
  const db = await database();
  const run = await db.select().from(generationRuns).where(and(eq(generationRuns.id, runId), eq(generationRuns.userId, userId))).limit(1);
  if (!run[0]) throw new Error("Generation run not found.");
  await db.update(generationRuns).set({ status: "complete", recipe: jsonRecord(recipe), rawModelOutput: jsonRecord(generated), completedAt: new Date() }).where(eq(generationRuns.id, runId));
  await db.insert(concepts).values(items.map(item => ({ projectId: run[0].projectId, generationRunId: runId, userId, rank: item.rank, content: jsonRecord(item), rawModelOutput: jsonRecord(item) })));
  await db.update(projects).set({ status: "exploring" }).where(and(eq(projects.id, run[0].projectId), eq(projects.userId, userId)));
  const persisted = await db.select().from(concepts).where(and(eq(concepts.generationRunId, runId), eq(concepts.userId, userId))).orderBy(concepts.rank);
  return persisted.map(row => ({ ...row.content as unknown as ConceptCard, id: row.id }));
}

export async function failGenerationRun(userId: number, runId: number, summary: string) {
  const db = await database();
  await db.update(generationRuns).set({ status: "failed", errorSummary: summary.slice(0, 500), completedAt: new Date() }).where(and(eq(generationRuns.id, runId), eq(generationRuns.userId, userId)));
}

export async function listProjects(userId: number) {
  const db = await database();
  return db.select().from(projects).where(eq(projects.userId, userId)).orderBy(desc(projects.updatedAt));
}

export async function getProjectWorkspace(userId: number, projectId: number) {
  const db = await database();
  const project = await db.select().from(projects).where(and(eq(projects.id, projectId), eq(projects.userId, userId))).limit(1);
  if (!project[0]) return null;
  const [brief] = await db.select().from(briefs).where(and(eq(briefs.projectId, projectId), eq(briefs.userId, userId))).orderBy(desc(briefs.updatedAt)).limit(1);
  const conceptRows = await db.select().from(concepts).where(and(eq(concepts.projectId, projectId), eq(concepts.userId, userId))).orderBy(desc(concepts.createdAt));
  return { project: project[0], brief: brief ? (brief.content as unknown as BriefInput) : null, concepts: conceptRows.map(row => ({ ...row, content: row.content as unknown as ConceptCard })) };
}

export async function getConcept(userId: number, conceptId: number) {
  const db = await database();
  const rows = await db.select().from(concepts).where(and(eq(concepts.id, conceptId), eq(concepts.userId, userId))).limit(1);
  return rows[0] ? { ...rows[0], content: rows[0].content as unknown as ConceptCard } : null;
}

export async function createBlueprint(userId: number, projectId: number, conceptId: number, generationRunId: number, artifact: BlueprintArtifact) {
  const db = await database();
  const result = await db.insert(blueprints).values({ projectId, conceptId, generationRunId, userId, rawModelOutput: jsonRecord(artifact) });
  const blueprintId = insertedId(result);
  await db.update(projects).set({ status: "blueprint", activeBlueprintId: blueprintId }).where(and(eq(projects.id, projectId), eq(projects.userId, userId)));
  return blueprintId;
}

export async function getBlueprint(userId: number, blueprintId: number) {
  const db = await database();
  const rows = await db.select().from(blueprints).where(and(eq(blueprints.id, blueprintId), eq(blueprints.userId, userId))).limit(1);
  const blueprint = rows[0];
  if (!blueprint) return null;
  const edits = await db.select().from(blueprintEdits).where(and(eq(blueprintEdits.blueprintId, blueprintId), eq(blueprintEdits.userId, userId))).orderBy(desc(blueprintEdits.updatedAt)).limit(1);
  return { ...blueprint, rawModelOutput: blueprint.rawModelOutput as unknown as BlueprintArtifact, editedContent: edits[0]?.content as unknown as BlueprintArtifact | undefined, editUpdatedAt: edits[0]?.updatedAt };
}

export async function saveBlueprintEdit(userId: number, blueprintId: number, artifact: BlueprintArtifact) {
  const db = await database();
  const existing = await db.select().from(blueprintEdits).where(and(eq(blueprintEdits.blueprintId, blueprintId), eq(blueprintEdits.userId, userId))).limit(1);
  if (existing[0]) {
    await db.update(blueprintEdits).set({ content: jsonRecord(artifact), updatedAt: new Date() }).where(eq(blueprintEdits.id, existing[0].id));
  } else {
    await db.insert(blueprintEdits).values({ blueprintId, userId, content: jsonRecord(artifact) });
  }
}

export async function saveComparison(userId: number, projectId: number, conceptIds: number[]) {
  const db = await database();
  await db.insert(savedComparisons).values({ projectId, userId, conceptIds });
}

export async function recordExport(userId: number, projectId: number, blueprintId: number) {
  const db = await database();
  await db.insert(exports).values({ userId, projectId, blueprintId, format: "markdown" });
}
