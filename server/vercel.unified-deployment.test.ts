import { describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

describe("unified Vercel deployment", () => {
  it("keeps the Vite frontend and the catch-all API handler in one Vercel project", async () => {
    const [vercelConfig, apiHandler, apiApp] = await Promise.all([
      readFile(resolve(process.cwd(), "vercel.json"), "utf8"),
      readFile(resolve(process.cwd(), "api/[...path].ts"), "utf8"),
      readFile(resolve(process.cwd(), "server/_core/apiApp.ts"), "utf8"),
    ]);
    expect(vercelConfig).toContain('"framework": "vite"');
    expect(vercelConfig).toContain('"api/[...path].ts"');
    expect(apiHandler).toContain('healthPath: "/api/health"');
    expect(apiApp).toContain('app.use("/api/trpc"');
  });
});
