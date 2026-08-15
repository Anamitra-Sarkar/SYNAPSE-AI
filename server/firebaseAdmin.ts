import { cert, getApps, initializeApp } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";
import { getFirestore } from "firebase-admin/firestore";

export function getFirebaseAdminApp() {
  const projectId = process.env.FIREBASE_PROJECT_ID;
  const clientEmail = process.env.FIREBASE_CLIENT_EMAIL;
  const privateKey = process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, "\n");
  if (!projectId || !clientEmail || !privateKey) throw new Error("Firebase Admin credentials are not configured.");
  return getApps()[0] ?? initializeApp({ credential: cert({ projectId, clientEmail, privateKey }) });
}

export async function verifyFirebaseIdToken(authorization?: string) {
  const token = authorization?.startsWith("Bearer ") ? authorization.slice(7) : undefined;
  if (!token) return null;
  return getAuth(getFirebaseAdminApp()).verifyIdToken(token);
}

export function getFirebaseAdminFirestore() {
  return getFirestore(getFirebaseAdminApp(), process.env.FIREBASE_DATABASE_ID ?? "(default)");
}
