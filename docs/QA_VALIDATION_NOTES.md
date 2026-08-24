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

## 2026-08-24 — SPA deep-link routing correction

- Final routing preview: `https://morrow-5ul9t4fsk-anamitra-sarkars-projects.vercel.app`
- The direct `/signup` browser request now renders Morrow’s email/Google sign-up screen instead of returning Vercel `404: NOT_FOUND`.
- The regression test requires Vercel’s filesystem-first routing and the `/index.html` SPA fallback, preserving the serverless `/api/*` routes.

## 2026-08-24 — Final release-candidate browser acceptance

- Final PR #8 preview: `https://morrow-o5kw7hjt4-anamitra-sarkars-projects.vercel.app` (commit `1b630a4`).
- Routing acceptance passed: `/api/health` returned `200`; unauthenticated `/api/trpc/synapse.projects` returned an API authorization response rather than SPA HTML; direct `/signup` rendered successfully.
- The Firestore browser client addressed `synapse-ai-98e8a` database `database-1`. After publishing the owner-only rules release, an authenticated browser created the `SignalBridge` workspace and navigated into its project route without a permission-denied error.
- Desktop authenticated acceptance passed: live Groq generation returned four persisted concept directions; the sticky comparison tray accepted all three allowed selections after its clearance repair; promotion produced an editable blueprint; a browser-authored edit displayed its saved indicator; and `synapse.exportMarkdown` returned `200`.
- Browser download history contained `signalbridge.md` and `signalbridge (1).md`, confirming the portable Markdown export was downloaded from the final preview.
- A reload-specific defect was corrected: hydrated workspaces now restore their project identifier before promotion. A persistence defect was also corrected: client display artifacts now merge into the blueprint document rather than replacing server-owned concept/export context.
- Mobile visual acceptance at `375×812` showed the public editorial landing and email-auth screens without horizontal overflow; controls remained visibly usable and legible. The final desktop workspace verification confirmed the accessible labeled inputs, focusable controls, live generation status, and compare-button states in the authenticated journey.
- The temporary Firebase quality-assurance user and its owner-scoped Firestore project were deleted with Firebase Admin immediately after verification. No temporary scripts remain in the project root.

## 2026-08-24 — True 375px authenticated mobile acceptance

- A second short-lived Firebase QA identity completed the final-preview mobile path: sign-up, onboarding, Firestore project creation, quick-start brief, live Groq generation (four directions), all-three compare selection, and blueprint promotion.
- Computed layout checks at `375×812` verified the off-canvas sidebar (`left: -250px`), main workspace width of `375px`, mobile inspector visibility, single-column MVP grid, and no document-level horizontal overflow in the Concept Studio (`scrollWidth: 360px`).
- Release-candidate commit `51eebef` corrected a measured `19px` blueprint-editor overflow: on the final preview `https://morrow-13zeerbjw-anamitra-sarkars-projects.vercel.app`, the three blueprint actions stack vertically at `316px` width with `44px` heights, and the editor reports `scrollWidth: 360px` within a `375px` viewport.
- The same true-mobile run edited the blueprint overview, showed the persisted `Edited` state, and completed `synapse.saveBlueprintEdits` plus `synapse.exportMarkdown` with `200` responses. The temporary mobile QA user and project `fcG86zN4A4I43uc88RUN` were then deleted via Firebase Admin.
- Browser download history confirmed the mobile portable export as `signalbridge (2).md` from the final preview, completing the true-mobile edit, save, export, and downloaded-file acceptance path.
