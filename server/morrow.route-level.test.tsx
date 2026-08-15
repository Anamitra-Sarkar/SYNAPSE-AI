/** @vitest-environment jsdom */

import "@testing-library/jest-dom/vitest";
import { cleanup, render, waitFor } from "@testing-library/react";
import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

const authState = vi.hoisted(() => ({
  value: {
    user: null,
    loading: false,
    login: vi.fn(),
    signup: vi.fn(),
    loginWithGoogle: vi.fn(),
    logout: vi.fn(),
    resetPassword: vi.fn(),
  },
}));

vi.mock("@/contexts/FirebaseAuthContext", () => ({
  FirebaseAuthProvider: ({ children }: { children: React.ReactNode }) => children,
  useFirebaseAuth: () => authState.value,
}));
vi.mock("@/lib/firebase", () => ({ firestore: null }));

import MorrowApp from "../client/src/pages/MorrowApp";

function open(path: string) {
  window.history.replaceState({}, "", path);
  return render(<MorrowApp />);
}

describe("Morrow protected route navigation", () => {
  afterEach(() => cleanup());

  it("redirects an unauthenticated dashboard visitor to sign-in", async () => {
    open("/app");

    await waitFor(() => expect(window.location.pathname).toBe("/login"));
  });

  it("redirects an unauthenticated project-workspace visitor to sign-in", async () => {
    open("/app/projects/private-project");

    await waitFor(() => expect(window.location.pathname).toBe("/login"));
  });
});
