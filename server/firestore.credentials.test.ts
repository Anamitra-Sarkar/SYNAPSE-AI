import { describe, expect, it } from "vitest";
import { getFirebaseAdminFirestore } from "./firebaseAdmin";

describe("Firestore Admin credentials", () => {
  it("can read Firestore metadata with the configured Firebase Admin service account", async () => {
    const collections = await getFirebaseAdminFirestore().listCollections();
    expect(Array.isArray(collections)).toBe(true);
  }, 20_000);
});
