# Morrow Release Verification

This document records the final validation performed for the Firebase-authenticated Morrow experience after the native project-workspace migration.

| Area | Validation performed | Outcome |
|---|---|---|
| Public product pages | Captured the landing page, pricing page, and sign-in page at desktop and 375px mobile breakpoints. | The warm editorial paper/ink visual system, typography, CTAs, and responsive layouts rendered without visible overflow or overlap. |
| Protected navigation | Mounted the actual Morrow router in jsdom with an unauthenticated Firebase state. | Both `/app` and `/app/projects/:id` redirect to `/login`. |
| Firebase API authorization | Mocked verified, missing, and invalid Firebase bearer-token requests at the server request-context boundary. | A verified token reaches a protected project procedure; missing and invalid tokens are rejected before any project lookup. |
| Client secret boundary | Examined the browser Firebase configuration and active API-header path in automated regression coverage. | Firebase Admin credentials remain server-only; active API requests use Firebase ID tokens only and no legacy session-storage bearer fallback remains. |
| Artifact persistence | Reviewed the native workspace route and Firestore repository wiring. | The workspace stores and restores the brief, generation concepts, comparison choice, immutable blueprint output, independent edit revision, and portable Markdown export metadata under the owner-scoped project subtree. |
| Accessibility and motion | Reviewed interactive controls and the global responsive stylesheet. | The experience uses visible focus styles, semantic labels for interactive form controls, 44px-or-larger principal controls, and motion configured to respect the user’s reduced-motion preference. |
| Release build | Executed the full automated test suite, TypeScript check, and production build after the changes. | Passed; the only expected exception is the opt-in live Groq integration test, which remains skipped without an explicit live-provider test run. |

## Deployment readiness

The project remains prepared for a split deployment: the static client can be deployed to Vercel and the Express API to Render. Before publishing, enable Firebase Authentication providers in the Firebase console, deploy `firestore.rules` and `firestore.indexes.json`, populate the documented client and server environment variables in each host, and set the server’s trusted frontend origin to the deployed Vercel domain. Refer to [Firebase setup](./FIREBASE_SETUP.md) and [deployment configuration](./DEPLOYMENT.md) for the complete operational sequence.
