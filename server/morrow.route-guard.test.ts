import { describe, expect, it } from "vitest";
import { protectedRouteDestination } from "../client/src/lib/routeGuard";

describe("Morrow protected-route behavior", () => {
  it("redirects unauthenticated dashboard and studio navigation to login", () => {
    expect(protectedRouteDestination(undefined, false)).toBe("login");
  });
  it("holds navigation while authentication state is loading and allows signed-in members", () => {
    expect(protectedRouteDestination(undefined, true)).toBe("loading");
    expect(protectedRouteDestination("firebase-user-1", false)).toBe("allow");
  });
});
