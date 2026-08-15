import { describe, expect, it } from "vitest";
import { cert, getApps, initializeApp } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";

describe("Firebase Admin credentials", () => {
  it("authenticates to Firebase with the configured service account", async () => {
    const projectId = process.env.FIREBASE_PROJECT_ID;
    const clientEmail = process.env.FIREBASE_CLIENT_EMAIL;
    const privateKey = process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, "\n");
    expect(projectId).toBeTruthy();
    expect(clientEmail).toBeTruthy();
    expect(privateKey).toContain("BEGIN PRIVATE KEY");
    const app = getApps()[0] ?? initializeApp({ credential: cert({ projectId, clientEmail, privateKey }) });
    const result = await getAuth(app).listUsers(1);
    expect(Array.isArray(result.users)).toBe(true);
  }, 20_000);
});
