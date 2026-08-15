import "dotenv/config";
import { createApiApp } from "./_core/apiApp";

// This module is bundled into api/_morrowApi.cjs during Vercel builds.
export default createApiApp({ healthPath: "/api/health" });
