# Morrow Firebase Activation

The supplied Firebase project credentials have been configured in the Morrow application and verified with a read-only Admin SDK call. Before end users can sign in and create projects, open the Firebase console for `cabbage-guard` and complete the following production activation steps.

| Firebase area | Required action |
| --- | --- |
| Authentication | Enable **Email/Password** and **Google** providers. Add the Vercel domain and local development domain to Authorized domains. |
| Firestore Database | Create the Firestore database in production mode and deploy `firestore.rules` plus `firestore.indexes.json`. |
| Web App settings | Confirm the supplied web app remains associated with the Vercel deployment domain. |
| Service account | Keep the supplied service-account credential only in Render’s secret configuration. Rotate it immediately if it has ever been committed, shared publicly, or exposed outside the Firebase console and secure project configuration. |

The React client uses only Firebase’s public web configuration. Morrow project documents are written to the authenticated user’s `projects` collection, while the Firestore rules restrict access to the document owner. Groq remains a server-side concern and must continue to use a Render-only secret.

## Index requirement

`firestore.indexes.json` defines the single composite index required by the dashboard query: `projects` ordered by `ownerId` ascending and `updatedAt` descending. Deploy this file with the Firestore rules before using the project-list dashboard in production. The nested artifact lookups use direct document reads at predictable paths and do not require an additional composite index.

## Legacy-to-Firestore migration path

Migrate each existing workspace in ownership order. First create or update `projects/{morrowProjectId}` with the Firebase user ID as `ownerId`; then copy its current brief to `briefs/current`. Save the latest generated directions to both `generations/latest` and `concepts/latest`, the selected IDs to `comparisons/latest`, the immutable model blueprint plus current user state to `blueprints/latest`, any later revision to `blueprintEdits/latest`, and a portable Markdown export to `exports/latest`. Keep the legacy records read-only until the Firestore copy has been verified for the owner account; do not delete source data during the migration.
