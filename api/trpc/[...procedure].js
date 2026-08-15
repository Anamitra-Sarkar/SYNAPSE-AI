import handler from "../_morrowApi.mjs";

// Vercel resolves this explicit procedure catch-all before the generic API
// handler, ensuring `/api/trpc/synapse.generate` reaches Express unchanged.
export default handler;
