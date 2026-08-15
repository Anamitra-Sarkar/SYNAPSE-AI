export function protectedRouteDestination(userId: string | undefined, loading: boolean) {
  if (loading) return "loading" as const;
  return userId ? "allow" as const : "login" as const;
}
