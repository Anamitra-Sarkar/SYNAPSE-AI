# Production Deployment: Morrow on One Vercel Project

Morrow can run as one Vercel project. Vercel serves the Vite/React frontend and routes all `/api/*` requests to a lightweight Express serverless function. This keeps the browser and Firebase-authorized Groq API on the same origin while retaining `GROQ_API_KEY` and Firebase Admin credentials only on the server.

## 1. Activate Firebase

In the Firebase console for the Morrow project, enable **Email/Password** and **Google** under **Authentication → Sign-in method**. Add both the final Vercel domain and the local development domain to **Authentication → Settings → Authorized domains**. In **Firestore Database**, create the database in production mode, deploy the repository’s `firestore.rules` and `firestore.indexes.json`, and wait for the required composite index to report as enabled.

> The browser uses only the Firebase web configuration values. Firebase Admin values belong exclusively in Vercel’s server-side environment configuration.

## 2. Configure the unified Vercel project

Import the repository into Vercel. The included `vercel.json` builds the Vite client into `dist/public` and deploys `api/[...path].ts` as the Express serverless function. Set the following variables for the Production, Preview, and Development environments as appropriate:

| Vercel environment variable | Value | Scope |
| --- | --- |
| `VITE_FIREBASE_API_KEY` | Firebase web API key | Browser |
| `VITE_FIREBASE_AUTH_DOMAIN` | Firebase web auth domain | Browser |
| `VITE_FIREBASE_PROJECT_ID` | Firebase project ID | Browser |
| `VITE_FIREBASE_STORAGE_BUCKET` | Firebase storage bucket | Browser |
| `VITE_FIREBASE_MESSAGING_SENDER_ID` | Firebase web sender ID | Browser |
| `VITE_FIREBASE_APP_ID` | Firebase web app ID | Browser |
| `GROQ_API_KEY` | Groq secret | Server only |
| `FIREBASE_PROJECT_ID` | Firebase project ID | Server only |
| `FIREBASE_CLIENT_EMAIL` | Firebase Admin service-account email | Server only |
| `FIREBASE_PRIVATE_KEY` | Firebase Admin private key with preserved line breaks | Server only |
| `DATABASE_URL` | MySQL/TiDB connection string used by the generation workflow | Server only |
| `FRONTEND_ORIGIN` | The final Vercel HTTPS origin | Server only |

Leave `VITE_API_BASE_URL` unset for the unified deployment. The browser then calls same-origin `/api/trpc` with the signed Firebase ID token in its `Authorization` header; no cross-origin cookie or legacy OAuth fallback is required.

## 3. Configure domains and CORS

Set `FRONTEND_ORIGIN` to the exact HTTPS Vercel domain, or to the custom `app.example.com` domain once it is connected. If a custom domain is used, add that same frontend domain to Firebase Authentication’s Authorized domains list and redeploy after changing the CORS allow-list.

## 4. Pre-launch checklist

Confirm that `https://<your-vercel-domain>/api/health` returns `{ "status": "ok" }`. From the Vercel app, complete a Firebase sign-in, create a project, frame the brief, generate concept cards, compare at least one card, promote a selection from the compare tray, save a blueprint edit, reload the project workspace, and download the Markdown export. Verify browser DevTools never shows `GROQ_API_KEY`, `FIREBASE_PRIVATE_KEY`, or `FIREBASE_CLIENT_EMAIL` in source, network payloads, or runtime configuration.

> This project is built within Manus for development, but the included manifests target the user-selected Render and Vercel deployment. Once a checkpoint is created, export the code to GitHub and connect that repository to both hosts. Do not copy live production secrets into the repository.
