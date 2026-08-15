export const SCORE_DIMENSIONS = [
  "skillsFit",
  "feasibility",
  "novelty",
  "impact",
  "demoPotential",
] as const;

export type ScoreDimension = (typeof SCORE_DIMENSIONS)[number];

export type ScoreWeights = Record<ScoreDimension, number>;

export type BriefInput = {
  title?: string;
  skills: string[];
  problemStatement: string;
  availableHours: number;
  teamSize: number;
  teamRoles: string[];
  domain?: string;
  preferredTech: string[];
  resources: string[];
  constraints: string[];
  scoringWeights: ScoreWeights;
};

export type ConceptScores = Record<ScoreDimension, number> & {
  overall: number;
};

export type ConceptCard = {
  id?: string;
  rank: number;
  name: string;
  hook: string;
  targetUser: string;
  painPoint: string;
  solution: string;
  differentiator: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  buildTime: string;
  techStack: string[];
  scores: ConceptScores;
  scoreRationale: Record<ScoreDimension, string>;
  assumptions: string[];
  risks: string[];
  nextStep: string;
};

export type BlueprintArtifact = {
  overview: string;
  mvpFeatures: Array<{ title: string; detail: string; priority: "Must" | "Should" | "Could" }>;
  architecture: Array<{ layer: string; purpose: string; technologies: string[] }>;
  dataAndApis: Array<{ name: string; need: string; alternative?: string }>;
  buildPlan: Array<{ window: string; goal: string; tasks: string[] }>;
  teamPlan: Array<{ role: string; responsibilities: string[] }>;
  demoFlow: string[];
  judgePitch: { opening: string; problem: string; solution: string; proof: string; close: string };
  risks: Array<{ risk: string; mitigation: string }>;
  extensions: string[];
  fallbackPlan: string;
};

export type GenerationRecipe = {
  normalizedBrief: BriefInput;
  model: string;
  promptVersion: string;
  schemaVersion: string;
  createdAt: string;
  scoreSnapshot?: Array<{ rank: number; name: string; scores: ConceptScores }>;
};

export type MarkdownExport = {
  filename: string;
  content: string;
  exportedAt: string;
  projectId: string;
  blueprintId: string;
};
