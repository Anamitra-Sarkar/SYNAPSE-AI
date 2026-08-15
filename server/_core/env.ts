export const ENV = {
  appId: process.env.VITE_APP_ID ?? "",
  cookieSecret: process.env.JWT_SECRET ?? "",
  databaseUrl: process.env.DATABASE_URL ?? "",
  oAuthServerUrl: process.env.OAUTH_SERVER_URL ?? "",
  ownerOpenId: process.env.OWNER_OPEN_ID ?? "",
  isProduction: process.env.NODE_ENV === "production",
  forgeApiUrl: process.env.BUILT_IN_FORGE_API_URL ?? "",
  forgeApiKey: process.env.BUILT_IN_FORGE_API_KEY ?? "",
  groqApiKey: process.env.GROQ_API_KEY ?? "",
  // Split-deployment settings (set these on Render)
  // FRONTEND_ORIGIN: the Vercel app URL, e.g. https://synapse-ai.vercel.app
  frontendOrigin: process.env.FRONTEND_ORIGIN ?? "",
  // POST_AUTH_REDIRECT: where to send the browser after OAuth completes.
  // Set to the Vercel app URL in production; defaults to "/" for same-origin dev.
  postAuthRedirect: process.env.POST_AUTH_REDIRECT ?? "/",
  // API_ONLY: when "true", the server skips static-file serving and serves only /api/* routes.
  apiOnly: process.env.API_ONLY === "true",
};
