import express, { type Express } from "express";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import { registerOAuthRoutes } from "./oauth";
import { registerStorageProxy } from "./storageProxy";
import { appRouter } from "../routers";
import { createContext } from "./context";
import { ENV } from "./env";

type ApiAppOptions = { healthPath?: string };

/**
 * Creates the shared API layer used by the local Express server and Vercel's
 * catch-all serverless function. Static files are served by Vite/Vercel, not
 * this application factory.
 */
export function createApiApp({ healthPath = "/health" }: ApiAppOptions = {}): Express {
  const app = express();
  app.set("trust proxy", 1);

  app.use((_req, res, next) => {
    res.setHeader("X-Content-Type-Options", "nosniff");
    res.setHeader("X-Frame-Options", "DENY");
    res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
    res.setHeader("X-XSS-Protection", "0");
    next();
  });

  const allowedOrigins = new Set<string>([
    ENV.frontendOrigin,
    ...(ENV.isProduction ? [] : ["http://localhost:3000", "http://localhost:5173"]),
  ].filter(Boolean));
  app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (origin && allowedOrigins.has(origin)) {
      res.setHeader("Access-Control-Allow-Origin", origin);
      res.setHeader("Access-Control-Allow-Credentials", "true");
      res.setHeader("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
      res.setHeader("Vary", "Origin");
    }
    if (req.method === "OPTIONS") {
      res.sendStatus(204);
      return;
    }
    next();
  });

  app.use(express.json({ limit: ENV.apiOnly ? "2mb" : "50mb" }));
  app.use(express.urlencoded({ limit: ENV.apiOnly ? "2mb" : "50mb", extended: true }));
  app.get(healthPath, (_req, res) => res.json({ status: "ok", ts: Date.now() }));
  registerStorageProxy(app);
  registerOAuthRoutes(app);
  app.use("/api/trpc", createExpressMiddleware({ router: appRouter, createContext }));

  return app;
}
