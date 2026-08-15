import { collection, doc, getDoc, getDocs, orderBy, query, serverTimestamp, setDoc, where } from "firebase/firestore";
import { firestore } from "./firebase";

export type MorrowProject = { id: string; ownerId: string; title: string; stage: "Framing" | "Exploring" | "Choosing" | "Planning"; };
function db() { if (!firestore) throw new Error("Firestore is not configured."); return firestore; }
export async function listProjects(ownerId: string) { const snapshot = await getDocs(query(collection(db(), "projects"), where("ownerId", "==", ownerId), orderBy("updatedAt", "desc"))); return snapshot.docs.map(item => ({ id: item.id, ...item.data() } as MorrowProject)); }
export async function saveBrief(projectId: string, ownerId: string, challenge: string) { await setDoc(doc(db(), "projects", projectId, "briefs", "current"), { ownerId, challenge, updatedAt: serverTimestamp() }, { merge: true }); }
export async function getBrief(projectId: string) { const snapshot = await getDoc(doc(db(), "projects", projectId, "briefs", "current")); return snapshot.exists() ? snapshot.data() : null; }
export async function loadWorkspaceArtifacts(projectId: string) {
  const paths = ["briefs", "generations", "concepts", "comparisons", "blueprints", "blueprintEdits", "exports"] as const;
  const snapshots = await Promise.all(paths.map(path => getDoc(doc(db(), "projects", projectId, path, "latest"))));
  const [brief, generation, concepts, comparison, blueprint, blueprintEdits, exported] = snapshots.map(snapshot => snapshot.exists() ? snapshot.data() : null);
  return { brief, generation, concepts, comparison, blueprint, blueprintEdits, exported };
}
export async function saveGenerationArtifacts(projectId: string, ownerId: string, concepts: unknown[]) { await setDoc(doc(db(), "projects", projectId, "generations", "latest"), { ownerId, concepts, createdAt: serverTimestamp() }); }
export async function saveConceptArtifacts(projectId: string, ownerId: string, concepts: unknown[]) { await setDoc(doc(db(), "projects", projectId, "concepts", "latest"), { ownerId, concepts, updatedAt: serverTimestamp() }); }
export async function saveComparisonArtifacts(projectId: string, ownerId: string, conceptIds: string[]) { await setDoc(doc(db(), "projects", projectId, "comparisons", "latest"), { ownerId, conceptIds, updatedAt: serverTimestamp() }); }
export async function saveBlueprintArtifacts(projectId: string, ownerId: string, immutableOutput: unknown, userEdits: unknown, legacyBlueprintId: number, selectedConceptId: number) { await setDoc(doc(db(), "projects", projectId, "blueprints", "latest"), { ownerId, immutableOutput, userEdits, legacyBlueprintId, selectedConceptId, updatedAt: serverTimestamp() }); }
export async function saveBlueprintRevision(projectId: string, ownerId: string, revision: unknown) { await setDoc(doc(db(), "projects", projectId, "blueprintEdits", "latest"), { ownerId, revision, updatedAt: serverTimestamp() }); }
export async function savePortableExport(projectId: string, ownerId: string, markdown: string) { await setDoc(doc(db(), "projects", projectId, "exports", "latest"), { ownerId, markdown, generatedAt: serverTimestamp() }); }
