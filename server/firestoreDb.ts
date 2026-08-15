import { randomUUID } from "node:crypto";
import { FieldValue } from "firebase-admin/firestore";
import type { BlueprintArtifact, BriefInput, ConceptCard, GenerationRecipe } from "../shared/synapse";
import { getFirebaseAdminFirestore } from "./firebaseAdmin";

type ProjectRecord = { id: string; ownerId: string; title: string; stage: string; updatedAt?: unknown };
type BlueprintRecord = { id: string; projectId: string; conceptId: string; rawModelOutput: BlueprintArtifact; editedContent?: BlueprintArtifact };

function projects() { return getFirebaseAdminFirestore().collection("projects"); }
function project(projectId: string) { return projects().doc(projectId); }
function ownedProject(projectId: string, ownerId: string) { return project(projectId).get().then(snapshot => {
  if (!snapshot.exists || snapshot.data()?.ownerId !== ownerId) return null;
  return snapshot;
}); }

export async function listProjects(ownerId: string): Promise<ProjectRecord[]> {
  const snapshot = await projects().where("ownerId", "==", ownerId).get();
  return snapshot.docs.map(item => ({ id: item.id, ...item.data() } as ProjectRecord))
    .sort((left, right) => String(right.updatedAt ?? "").localeCompare(String(left.updatedAt ?? "")));
}

export async function createProjectWithBrief(ownerId: string, projectId: string, title: string, brief: BriefInput) {
  const projectRef = project(projectId);
  const existing = await projectRef.get();
  if (existing.exists && existing.data()?.ownerId !== ownerId) throw new Error("Project ownership mismatch.");
  const runId = randomUUID();
  const batch = getFirebaseAdminFirestore().batch();
  batch.set(projectRef, { ownerId, title, stage: "Exploring", updatedAt: FieldValue.serverTimestamp(), ...(existing.exists ? {} : { createdAt: FieldValue.serverTimestamp() }) }, { merge: true });
  batch.set(projectRef.collection("briefs").doc("current"), { ownerId, content: brief, challenge: brief.problemStatement, updatedAt: FieldValue.serverTimestamp() }, { merge: true });
  batch.set(projectRef.collection("generations").doc("latest"), { ownerId, runId, status: "pending", createdAt: FieldValue.serverTimestamp() }, { merge: true });
  await batch.commit();
  return { projectId, runId };
}

export async function completeGenerationRun(ownerId: string, projectId: string, runId: string, raw: unknown, concepts: ConceptCard[], recipe: GenerationRecipe) {
  const projectRef = project(projectId);
  if (!await ownedProject(projectId, ownerId)) throw new Error("Project not found.");
  const persisted = concepts.map(concept => ({ ...concept, id: `concept-${concept.rank}` }));
  const batch = getFirebaseAdminFirestore().batch();
  batch.set(projectRef.collection("generations").doc("latest"), { ownerId, runId, status: "complete", rawModelOutput: raw, recipe, completedAt: FieldValue.serverTimestamp() }, { merge: true });
  batch.set(projectRef.collection("concepts").doc("latest"), { ownerId, concepts: persisted, updatedAt: FieldValue.serverTimestamp() }, { merge: true });
  persisted.forEach(concept => batch.set(projectRef.collection("concepts").doc(concept.id!), { ownerId, projectId, generationRunId: runId, rank: concept.rank, content: concept, rawModelOutput: concept, updatedAt: FieldValue.serverTimestamp() }, { merge: true }));
  batch.set(projectRef, { stage: "Exploring", updatedAt: FieldValue.serverTimestamp() }, { merge: true });
  await batch.commit();
  return persisted;
}

export async function failGenerationRun(ownerId: string, projectId: string, runId: string, summary: string) {
  if (!await ownedProject(projectId, ownerId)) return;
  await project(projectId).collection("generations").doc("latest").set({ ownerId, runId, status: "failed", errorSummary: summary.slice(0, 500), completedAt: FieldValue.serverTimestamp() }, { merge: true });
}

export async function getProjectWorkspace(ownerId: string, projectId: string) {
  const projectSnapshot = await ownedProject(projectId, ownerId);
  if (!projectSnapshot) return null;
  const [briefSnapshot, conceptsSnapshot] = await Promise.all([
    project(projectId).collection("briefs").doc("current").get(),
    project(projectId).collection("concepts").doc("latest").get(),
  ]);
  return {
    project: { id: projectId, ...projectSnapshot.data() } as ProjectRecord,
    brief: briefSnapshot.exists ? briefSnapshot.data()?.content as BriefInput : null,
    concepts: conceptsSnapshot.exists ? conceptsSnapshot.data()?.concepts as ConceptCard[] : [],
  };
}

export async function getConcept(ownerId: string, projectId: string, conceptId: string) {
  if (!await ownedProject(projectId, ownerId)) return null;
  const snapshot = await project(projectId).collection("concepts").doc(conceptId).get();
  if (!snapshot.exists || snapshot.data()?.ownerId !== ownerId) return null;
  return { id: snapshot.id, ...snapshot.data(), content: snapshot.data()?.content as ConceptCard };
}

export async function saveComparison(ownerId: string, projectId: string, conceptIds: string[]) {
  if (!await ownedProject(projectId, ownerId)) throw new Error("Project not found.");
  await project(projectId).collection("comparisons").doc("latest").set({ ownerId, conceptIds, updatedAt: FieldValue.serverTimestamp() }, { merge: true });
  await project(projectId).set({ stage: "Choosing", updatedAt: FieldValue.serverTimestamp() }, { merge: true });
}

export async function createBlueprint(ownerId: string, projectId: string, conceptId: string, artifact: BlueprintArtifact) {
  if (!await ownedProject(projectId, ownerId)) throw new Error("Project not found.");
  const blueprintId = "latest";
  await project(projectId).collection("blueprints").doc(blueprintId).set({ ownerId, projectId, conceptId, rawModelOutput: artifact, updatedAt: FieldValue.serverTimestamp() }, { merge: true });
  await project(projectId).set({ stage: "Planning", updatedAt: FieldValue.serverTimestamp() }, { merge: true });
  return blueprintId;
}

export async function getBlueprint(ownerId: string, projectId: string, blueprintId: string): Promise<BlueprintRecord | null> {
  if (!await ownedProject(projectId, ownerId)) return null;
  const [blueprintSnapshot, editSnapshot] = await Promise.all([
    project(projectId).collection("blueprints").doc(blueprintId).get(),
    project(projectId).collection("blueprintEdits").doc("latest").get(),
  ]);
  if (!blueprintSnapshot.exists || blueprintSnapshot.data()?.ownerId !== ownerId) return null;
  const blueprint = blueprintSnapshot.data()!;
  return { id: blueprintId, projectId, conceptId: blueprint.conceptId, rawModelOutput: blueprint.rawModelOutput as BlueprintArtifact, editedContent: editSnapshot.data()?.revision as BlueprintArtifact | undefined };
}

export async function saveBlueprintEdit(ownerId: string, projectId: string, blueprintId: string, content: BlueprintArtifact) {
  const existing = await getBlueprint(ownerId, projectId, blueprintId);
  if (!existing) return false;
  await project(projectId).collection("blueprintEdits").doc("latest").set({ ownerId, blueprintId, revision: content, updatedAt: FieldValue.serverTimestamp() }, { merge: true });
  return true;
}

export async function recordExport(ownerId: string, projectId: string, blueprintId: string, content: string) {
  if (!await getBlueprint(ownerId, projectId, blueprintId)) return false;
  await project(projectId).collection("exports").doc("latest").set({ ownerId, blueprintId, markdown: content, generatedAt: FieldValue.serverTimestamp() }, { merge: true });
  return true;
}
