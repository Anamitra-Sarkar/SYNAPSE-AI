# Morrow Release QA Notes

## 2026-08-24 — Production-hardening preview

- Fresh Vercel preview: `https://morrow-8yt3zi5xb-anamitra-sarkars-projects.vercel.app`
- `GET /api/health` returned `200` with `{"status":"ok"}`.
- The public landing page rendered Morrow navigation, editorial hero, planning process, product qualities, calls to action, and footer successfully in the default browser.
- The public navigation exposes the expected How it works, Pricing, FAQ, Sign in, and sign-up paths.

## 2026-08-24 — Final authenticated workflow preview

- Final Vercel preview: `https://morrow-np233a6cf-anamitra-sarkars-projects.vercel.app`
- `GET /api/health` returned `200` with `{"status":"ok"}`.
- The public Morrow landing page rendered successfully, including the editorial navigation, hero, planning process, calls to action, and footer.
- A temporary Firebase-authenticated test identity completed the deployed workflow: generated **4** concepts, created a Firestore-backed blueprint, edited it, and produced the portable Markdown export `release-smoke-test-workspace.md`.
- The smoke test automatically removed its temporary Firebase user and all created Firestore artifacts after completion.
- The provider pipeline uses bounded retries, Qwen non-thinking JSON mode, schema normalization for predictable compact artifact forms, and server-side Zod validation. No Groq credential is exposed to client code.
