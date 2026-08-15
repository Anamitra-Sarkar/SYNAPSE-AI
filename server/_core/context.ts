import type { CreateExpressContextOptions } from "@trpc/server/adapters/express";
import type { User } from "../../drizzle/schema";
import { sdk } from "./sdk";
import * as db from "../db";
import { verifyFirebaseIdToken } from "../firebaseAdmin";

export type TrpcContext = {
  req: CreateExpressContextOptions["req"];
  res: CreateExpressContextOptions["res"];
  user: User | null;
};

export async function createContext(
  opts: CreateExpressContextOptions
): Promise<TrpcContext> {
  let user: User | null = null;

  try {
    const firebaseUser = await verifyFirebaseIdToken(opts.req.headers.authorization);
    if (firebaseUser) {
      await db.upsertUser({ openId: firebaseUser.uid, name: firebaseUser.name ?? null, email: firebaseUser.email ?? null, loginMethod: "firebase", lastSignedIn: new Date() });
      user = await db.getUserByOpenId(firebaseUser.uid) ?? null;
    } else {
      user = await sdk.authenticateRequest(opts.req);
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
