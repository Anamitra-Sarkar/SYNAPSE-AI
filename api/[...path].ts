import "dotenv/config";
import { createApiApp } from "../server/_core/apiApp";

// Vercel sends all /api/* requests through this serverless Express handler.
// The client uses same-origin `/api/trpc`, while Vercel serves Vite assets.
export default createApiApp({ healthPath: "/api/health" });
