# SaaS Redesign Plan — Morrow

## Goal

Transform the current SYNAPSE-AI workspace into **Morrow**, a polished SaaS product for hackathon teams to turn a rough challenge into a selected concept and a practical execution plan. The redesign will eliminate the current dark “AI/neural” visual language, replace the product name and identity, introduce a conversion-oriented public site, and place authenticated users in a calm, useful private dashboard.

> **Product position:** Morrow is a planning studio for ambitious short-form builds. It helps teams decide what to make, scope it intelligently, and execute with confidence.

## Assumptions and Decisions

| Area | Decision |
| --- | --- |
| Product name | Use **Morrow** as the working name, with a concise wordmark and no explicit “AI” suffix. The name can be changed before the brand pass if the user prefers another direction. |
| Product style | Use a warm, editorial productivity aesthetic: paper-white and soft oat surfaces, ink-blue typography, muted coral and moss accents, generous whitespace, soft borders, and subtle tactile textures. Avoid gradients, neon, or “AI dashboard” motifs. |
| Public experience | Build a responsive landing page with clear product value, visual workflow demonstration, social-proof placeholders, feature storytelling, FAQs, and primary sign-up calls to action. |
| Private experience | Build an authenticated SaaS dashboard with recent workspaces, a new-project entry point, usage/account panel, settings, and a project shell containing the existing brief, concept, comparison, and blueprint flows. |
| Authentication | Replace the current OAuth UI flow with Firebase Authentication, initially supporting email/password, Google sign-in, password reset, session persistence, protected routes, and account sign-out. |
| Persistence | Replace the application’s project/brief/concept/blueprint persistence layer with Firestore collections protected by per-user security rules. |
| Groq safety | Retain Groq only on the protected Render API. The browser will never receive `GROQ_API_KEY`; authenticated requests will pass a Firebase ID token to the server for verification. |
| Deployment | Keep the split deployment: Vercel hosts the React client and Render hosts the Groq API. Add Firebase web configuration only as public client configuration; keep Firebase Admin credentials only on Render. |

## Information Architecture

| Route | Audience | Purpose |
| --- | --- | --- |
| `/` | Public | Landing page that communicates value and drives sign-up. |
| `/pricing` | Public | Clear free/pro plan structure, designed so billing can be connected later without a UI rewrite. |
| `/login` and `/signup` | Public | Focused Firebase-authentication screens with social and email paths. |
| `/forgot-password` | Public | Password-reset request flow. |
| `/app` | Authenticated | Dashboard showing recent workspaces, quick start, account usage, and helpful empty states. |
| `/app/projects/new` | Authenticated | Guided project brief creation. |
| `/app/projects/:projectId` | Authenticated | Project overview and current decision status. |
| `/app/projects/:projectId/concepts` | Authenticated | Concept ranking, filters, inspector, and comparison tray. |
| `/app/projects/:projectId/blueprint` | Authenticated | Editable execution blueprint and Markdown export. |
| `/app/settings` | Authenticated | Profile, sign-in method, preferences, and account controls. |

## Landing Page Plan

The public landing page will be designed as a deliberate product narrative rather than a copy of the application dashboard. A minimal top navigation will use **Product**, **How it works**, **Pricing**, and a quiet sign-in link, with a prominent “Start planning” action. The hero will use editorial typography, a clear benefit statement, concise supporting copy, and a product screenshot-style composition made from real Morrow UI components rather than generic visual effects.

Below the hero, the page will present an outcome-led workflow: **Frame the problem**, **Choose the strongest direction**, and **Turn it into a build plan**. A focused feature grid, a “built for the weekend sprint” section, concise testimonials/placeholders, transparent pricing, FAQ, and a calm final conversion block will create a complete SaaS journey. The responsive version will use a mobile navigation sheet, stacked feature media, and consistently sized touch targets.

## Dashboard and Product UI Plan

The dashboard will use a light application shell with a slim sidebar on desktop and a compact top bar or drawer on mobile. The primary visual anchor will be a “Continue planning” section, followed by recently edited project cards, a blank-state guide for first-time users, and a small account/usage summary. The dashboard will not show implementation details, model names, or technical backend language.

Each project will use a clear stepper: **Brief → Directions → Decision → Plan**. The Brief Composer will preserve progressive disclosure but use larger natural-language prompts and a clear project context. The Concept Studio will become a card-and-detail experience with neutral score explanations and visual comparisons. The compare tray remains the only place from which a concept is promoted. The Blueprint will read as a well-structured project document, with editable sections, version clarity, and a portable Markdown export.

## Visual System and Interaction Design

| System element | Direction |
| --- | --- |
| Color | `#FAF7F2` paper background, `#18202C` ink, `#51606D` slate, `#D96C54` muted coral action accent, `#607D62` moss success accent, and warm neutral borders. |
| Typography | A characterful editorial display font for page headlines and a highly legible sans-serif for interface content. Use modest letter spacing and clear hierarchy, not futuristic or monospaced “AI” styling. |
| Components | Softly rounded but not bubble-like cards; slim 1px warm borders; sparse shadows; clear filled primary buttons; outlined secondary actions; readable forms and tables. |
| Illustration | Use abstract paper-cut shapes, hand-drawn planning marks, and product UI screenshots or diagrams. Do not use brains, constellations, chat bubbles, gradients, glows, network lines, or sci-fi visuals. |
| Motion | Use brief, natural opacity/translate transitions for sheets, dialogs, cards, and route sections. Add an optional subtle landing-page parallax only where it supports depth. Respect `prefers-reduced-motion` globally. |
| Accessibility | Preserve visible focus styles, semantic headings, screen-reader labels, keyboard-operable filters and compare controls, focus-managed dialogs, 44px touch targets, and readable contrast. |

## Firebase Architecture

### Client

Install and configure the Firebase Web SDK using public Vite variables. Create a single Firebase client module, an authentication provider, and a route guard. Firebase Authentication will manage sign-up, sign-in, Google login, password reset, sign-out, user profile state, and token refresh. Firestore reads will power the dashboard and project views with local cache-friendly real-time subscriptions only where they improve the experience.

### Server

Install the Firebase Admin SDK on Render only. The existing tRPC context will verify the `Authorization: Bearer <Firebase ID token>` header and derive the authenticated Firebase UID. Server-side Groq procedures will use that UID to authorize requests and will write immutable model artifacts, score snapshots, and generated blueprints to Firestore. This preserves the security boundary: **Firebase browser configuration is public by design; Groq and Firebase Admin credentials are not.**

### Firestore Data Model

| Collection | Key fields | Access rule |
| --- | --- | --- |
| `users/{uid}` | Profile, created timestamp, display name, onboarding status, plan metadata | User reads/writes own profile; server writes authoritative plan fields. |
| `projects/{projectId}` | `ownerId`, title, stage, created/updated timestamps, brief summary | Owner only; server enforces ownership on generation actions. |
| `projects/{projectId}/briefs/{briefId}` | Normalized brief, scoring weights, revision metadata | Owner only. |
| `projects/{projectId}/generations/{runId}` | Status, prompt/model/schema versions, raw model response, score snapshot | Owner read only; server write only for immutable model artifacts. |
| `projects/{projectId}/concepts/{conceptId}` | Rank, card content, scores, selection state | Owner only. |
| `projects/{projectId}/blueprints/{blueprintId}` | Raw model output, edited content, latest edit timestamp, export count | Owner only; raw output server write only; edits owner write under strict validated shape. |

Firestore rules will deny all cross-user reads and writes. The API will independently check the Firebase UID before any Groq operation or server-side write, providing defense in depth.

## Migration and Compatibility Approach

The current MySQL schema and Manus-specific authentication will be retired from the active product flow after Firestore is verified. The migration will be implemented as an application-level replacement rather than a partial dual-write system to avoid data divergence. Existing development records need not be migrated unless explicitly requested; if migration is desired, a one-time authenticated admin script will import the existing data with a dry-run report first.

The existing Groq prompt and structured-output pipeline will remain, but its copy will be adjusted to produce human, actionable project language rather than technical or “AI” themed phrasing. The client will call the Render API with Firebase ID tokens, not expose any Groq-related configuration, and persist project artifacts in Firestore.

## Implementation Phases

1. **Foundation and branding.** Replace the product name, metadata, fonts, color tokens, and current global visual language. Add the new navigation model and route structure without deleting the secure Groq server code prematurely.
2. **Firebase setup.** Add Firebase client and admin configuration, implement Firebase Authentication, protected routes, sign-out, password reset, and Firebase token verification on the Render API. Request Firebase project configuration and Admin credentials only at the secure configuration step.
3. **Firestore persistence.** Create typed Firestore repositories, validated security rules, indexes, ownership checks, and a development emulator/test strategy. Transition projects, briefs, generations, concepts, blueprints, edits, and export audit records from the current database layer.
4. **Public SaaS site.** Build the new landing page, pricing page, auth pages, responsive navigation, and editorial visual system with production-quality loading, error, and empty states.
5. **Private SaaS workspace.** Build the dashboard, new-project flow, project shell, concept experience, comparison, blueprint, settings, and account surfaces against Firestore data.
6. **Quality and deployment.** Test authentication, Firestore rules, cross-user protection, Groq server authorization, loading/error/retry behavior, keyboard and screen-reader access, mobile layouts, Vercel build, Render API health, and Firebase configuration. Update Render/Vercel/Firebase deployment documentation.

## Test Plan

Automated tests will validate Firebase token verification, unauthorized API rejection, Firestore repository ownership checks, Firestore rules using the emulator, data-contract validation, Groq provider error mapping, compare-limit behavior, raw-versus-edited blueprint separation, Markdown export, and environment boundaries preventing Groq or Admin credentials from entering the browser build.

Manual QA will cover the full user journey: landing-page CTA, sign-up, email verification if enabled, Google sign-in, password reset, onboarding, dashboard empty state, new project, concept generation, comparison selection, promotion from the compare tray, blueprint edit/save/export, sign-out, mobile navigation, keyboard-only navigation, and account settings.

## Firebase and Deployment Prerequisites

Implementation will need the user to create or select a Firebase project and enable **Email/Password** and **Google** sign-in. The Vercel client will need Firebase public web configuration values, while Render will need a Firebase Admin service-account credential or equivalent server credential. The existing `GROQ_API_KEY` remains on Render only. The user must authorize access to Firebase, Vercel, and Render only at the final configuration/deployment stage; secrets should be supplied through their respective host dashboards rather than committed or pasted into source control.

## Risks and Open Items

The name **Morrow** is a working brand choice and should be confirmed before implementation to avoid later visual rework. Firebase free-tier constraints, regional requirements, and Google sign-in authorized domains will be validated once a Firebase project is available. Billing UI will be designed as a launch-ready information architecture but will not charge users until a payment provider and pricing model are explicitly chosen.
