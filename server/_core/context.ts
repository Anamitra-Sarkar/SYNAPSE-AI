import type { CreateExpressContextOptions } from "@trpc/server/adapters/express";
import { verifyFirebaseIdToken } from "../firebaseAdmin";

export type FirebaseContextUser = {
  id: string;
  openId: string;
  email: string | null;
  name: string | null;
  loginMethod: "firebase";
  role: "user" | "admin";
  createdAt: Date;
  updatedAt: Date;
  lastSignedIn: Date;
};

export type TrpcContext = {
  req: CreateExpressContextOptions["req"];
  res: CreateExpressContextOptions["res"];
  user: FirebaseContextUser | null;
};

export async function createContext(
  opts: CreateExpressContextOptions
): Promise<TrpcContext> {
  let user: FirebaseContextUser | null = null;

  try {
    const firebaseUser = await verifyFirebaseIdToken(opts.req.headers.authorization);
    if (firebaseUser) {
      const now = new Date();
      user = { id: firebaseUser.uid, openId: firebaseUser.uid, name: firebaseUser.name ?? null, email: firebaseUser.email ?? null, loginMethod: "firebase", role: "user", createdAt: now, updatedAt: now, lastSignedIn: now };
    }
  } catch (error) {
    // Authentication is optional for public procedures.
    user = null;
  }

  return {
    req: opts.req,
    res: opts.res,
    user,
  };
}
