import { describe, expect, it } from "vitest";
import { getFirebaseAdminFirestore } from "./firebaseAdmin";

describe("Firestore database discovery", () => {
  it("uses the configured named database without writing any documents", async () => {
    expect(process.env.FIREBASE_DATABASE_ID).toBe("database-1");
    const collections = await getFirebaseAdminFirestore().listCollections();
    expect(Array.isArray(collections)).toBe(true);
  }, 20_000);
});
