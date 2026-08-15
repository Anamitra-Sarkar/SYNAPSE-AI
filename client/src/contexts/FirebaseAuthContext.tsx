import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { GoogleAuthProvider, User, createUserWithEmailAndPassword, onAuthStateChanged, sendPasswordResetEmail, signInWithEmailAndPassword, signInWithPopup, signOut } from "firebase/auth";
import { firebaseAuth } from "@/lib/firebase";

type AuthValue = { user: User | null; loading: boolean; login: (email: string, password: string) => Promise<void>; signup: (email: string, password: string) => Promise<void>; loginWithGoogle: () => Promise<void>; resetPassword: (email: string) => Promise<void>; logout: () => Promise<void> };
const FirebaseAuthContext = createContext<AuthValue | null>(null);

export function FirebaseAuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null); const [loading, setLoading] = useState(true);
  useEffect(() => { if (!firebaseAuth) { setLoading(false); return; } return onAuthStateChanged(firebaseAuth, value => { setUser(value); setLoading(false); }); }, []);
  const value = useMemo<AuthValue>(() => ({ user, loading,
    login: async (email, password) => { if (!firebaseAuth) throw new Error("Firebase is not configured."); await signInWithEmailAndPassword(firebaseAuth, email, password); },
    signup: async (email, password) => { if (!firebaseAuth) throw new Error("Firebase is not configured."); await createUserWithEmailAndPassword(firebaseAuth, email, password); },
    loginWithGoogle: async () => { if (!firebaseAuth) throw new Error("Firebase is not configured."); await signInWithPopup(firebaseAuth, new GoogleAuthProvider()); },
    resetPassword: async email => { if (!firebaseAuth) throw new Error("Firebase is not configured."); await sendPasswordResetEmail(firebaseAuth, email); },
    logout: async () => { if (firebaseAuth) await signOut(firebaseAuth); },
  }), [user, loading]);
  return <FirebaseAuthContext.Provider value={value}>{children}</FirebaseAuthContext.Provider>;
}
export function useFirebaseAuth() { const value = useContext(FirebaseAuthContext); if (!value) throw new Error("useFirebaseAuth must be used within FirebaseAuthProvider"); return value; }
