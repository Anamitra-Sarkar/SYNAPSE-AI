# Production Deployment: Morrow on Render + Vercel

Morrow is configured for a split deployment. Vercel serves the Vite/React client, while Render runs the Firebase-authorized Express/tRPC API, talks to Groq, and connects to the supporting MySQL/TiDB persistence used by the generation workflow. **`GROQ_API_KEY` and Firebase Admin credentials are never configured in, bundled with, or exposed by Vercel.**

## 1. Provision the API on Render

Create a new Render **Web Service** from this repository. Render will detect `render.yaml`; choose the generated `synapse-ai-api` service. Its health check is `GET /health`, and it runs with `API_ONLY=true`, so it exposes only API routes rather than trying to serve a second copy of the client.

| Render environment variable | Value | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Your Groq secret | Server-side structured concept and blueprint generation. |
| `DATABASE_URL` | MySQL/TiDB connection string | Persistent user-owned projects, recipes, concepts, and edits. |
| `FIREBASE_PROJECT_ID` | Firebase project ID | Identifies the Firebase Admin project used to verify ID tokens. |
| `FIREBASE_CLIENT_EMAIL` | Service-account client email | Firebase Admin service identity. |
| `FIREBASE_PRIVATE_KEY` | Service-account private key, preserving line breaks | Firebase Admin credential used only by Render. |
| `FRONTEND_ORIGIN` | `https://<your-vercel-domain>` | Exact CORS allow-list origin. |
| `API_ONLY` | `true` | Keeps Render in API-only mode. |

After first deploy, copy the Render API URL, such as `https://synapse-ai-api.onrender.com`.

## 2. Activate Firebase before deploying the frontend

In the Firebase console for the Morrow project, enable **Email/Password** and **Google** under **Authentication → Sign-in method**. Add both the final Vercel domain and the local development domain to **Authentication → Settings → Authorized domains**. In **Firestore Database**, create the database in production mode, deploy the repository’s `firestore.rules` and `firestore.indexes.json`, and wait for the required composite index to report as enabled.

> The browser uses only the Firebase web configuration values. Firebase Admin values belong exclusively in the Render service configuration.

## 3. Deploy the frontend on Vercel

Import the same repository into Vercel. The included `vercel.json` uses the Vite client build and publishes `dist/public`. Add this one Vercel environment variable before deploying:

| Vercel environment variable | Value | 
| --- | --- |
| `VITE_API_BASE_URL` | The full Render API origin, e.g. `https://synapse-ai-api.onrender.com` |
| `VITE_FIREBASE_API_KEY` | Firebase web API key |
| `VITE_FIREBASE_AUTH_DOMAIN` | Firebase web auth domain |
| `VITE_FIREBASE_PROJECT_ID` | Firebase project ID |
| `VITE_FIREBASE_STORAGE_BUCKET` | Firebase storage bucket |
| `VITE_FIREBASE_MESSAGING_SENDER_ID` | Firebase web sender ID |
| `VITE_FIREBASE_APP_ID` | Firebase web app ID |

Redeploy the frontend after setting the variables. The browser sends each protected tRPC call to `${VITE_API_BASE_URL}/api/trpc` with the signed Firebase ID token in its `Authorization` header; no legacy OAuth cookie fallback is required.

## 4. Configure domains and CORS

Set `FRONTEND_ORIGIN` on Render to the exact HTTPS Vercel domain, or to the custom `app.example.com` domain once it is connected. If a custom domain is used, add that same frontend domain to Firebase Authentication’s Authorized domains list and redeploy Render after changing the CORS allow-list.

## 5. Pre-launch checklist

Confirm that `https://<api-domain>/health` returns `{ "status": "ok" }`. From the Vercel app, complete a Firebase sign-in, create a project, frame the brief, generate concept cards, compare at least one card, promote a selection from the compare tray, save a blueprint edit, reload the project workspace, and download the Markdown export. Verify browser DevTools never shows `GROQ_API_KEY`, `FIREBASE_PRIVATE_KEY`, or `FIREBASE_CLIENT_EMAIL` in source, network payloads, or runtime configuration.

> This project is built within Manus for development, but the included manifests target the user-selected Render and Vercel deployment. Once a checkpoint is created, export the code to GitHub and connect that repository to both hosts. Do not copy live production secrets into the repository.
