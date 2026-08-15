import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  authenticateRequest: vi.fn(),
  getUserByOpenId: vi.fn(),
  listProjects: vi.fn(),
  upsertUser: vi.fn(),
  verifyFirebaseIdToken: vi.fn(),
}));

vi.mock("./firebaseAdmin", () => ({ verifyFirebaseIdToken: mocks.verifyFirebaseIdToken }));
vi.mock("./db", () => ({ getUserByOpenId: mocks.getUserByOpenId, upsertUser: mocks.upsertUser }));
vi.mock("./synapseDb", () => ({ listProjects: mocks.listProjects }));
vi.mock("./_core/sdk", () => ({ sdk: { authenticateRequest: mocks.authenticateRequest } }));

import { createContext, type TrpcContext } from "./_core/context";
import { appRouter } from "./routers";

const firebaseUser = {
  id: 42,
  openId: "firebase-user-42",
  email: "builder@morrow.test",
  name: "Morrow Builder",
  loginMethod: "firebase",
  role: "user" as const,
  createdAt: new Date(),
  updatedAt: new Date(),
  lastSignedIn: new Date(),
};

function requestWithBearer(authorization?: string) {
  return {
    req: { protocol: "https", headers: authorization ? { authorization } : {} } as TrpcContext["req"],
    res: {} as TrpcContext["res"],
  };
}

describe("Firebase bearer authorization for protected Morrow procedures", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.authenticateRequest.mockResolvedValue(null);
  });

  it("accepts a verified Firebase bearer token and permits the protected project procedure", async () => {
    mocks.verifyFirebaseIdToken.mockResolvedValue({ uid: firebaseUser.openId, email: firebaseUser.email, name: firebaseUser.name });
    mocks.getUserByOpenId.mockResolvedValue(firebaseUser);
    mocks.listProjects.mockResolvedValue([{ id: 7, title: "Campus night map" }]);

    const context = await createContext(requestWithBearer("Bearer verified-firebase-token"));
    const result = await appRouter.createCaller(context).synapse.projects();

    expect(mocks.verifyFirebaseIdToken).toHaveBeenCalledWith("Bearer verified-firebase-token");
    expect(mocks.upsertUser).toHaveBeenCalledWith(expect.objectContaining({ openId: firebaseUser.openId, loginMethod: "firebase" }));
    expect(mocks.authenticateRequest).not.toHaveBeenCalled();
    expect(mocks.listProjects).toHaveBeenCalledWith(firebaseUser.id);
    expect(result).toEqual([{ id: 7, title: "Campus night map" }]);
  });

  it.each([
    ["missing bearer token", undefined],
    ["invalid bearer token", "Bearer invalid-firebase-token"],
  ])("rejects the protected project procedure for a %s", async (_label, authorization) => {
    if (authorization) mocks.verifyFirebaseIdToken.mockRejectedValue(new Error("invalid token"));
    else mocks.verifyFirebaseIdToken.mockResolvedValue(null);

    const context = await createContext(requestWithBearer(authorization));

    await expect(appRouter.createCaller(context).synapse.projects()).rejects.toMatchObject({ code: "UNAUTHORIZED" });
    expect(mocks.listProjects).not.toHaveBeenCalled();
  });
});
