import { useFirebaseAuth } from "@/contexts/FirebaseAuthContext";
import { loadWorkspaceArtifacts, saveBlueprintArtifacts, saveBlueprintRevision, saveBrief, saveComparisonArtifacts, saveConceptArtifacts, saveGenerationArtifacts, savePortableExport } from "@/lib/projectRepository";
import { trpc } from "@/lib/trpc";
import type { BlueprintArtifact, BriefInput, ConceptCard, ScoreDimension, ScoreWeights } from "@shared/synapse";
import { updateCompareSelection } from "@shared/compare";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Archive, ArrowLeft, ArrowRight, BrainCircuit, Check, ChevronDown, ChevronUp, CircleAlert, ClipboardCheck, Code2, Download, Gauge, GitCompareArrows, Lightbulb, Loader2, Plus, Rocket, Sparkles, Target, Users, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useLocation } from "wouter";

const DEFAULT_WEIGHTS: ScoreWeights = { skillsFit: 25, feasibility: 30, novelty: 15, impact: 15, demoPotential: 15 };
const STAGE_LABEL = "Shaping brief → Exploring concepts → Checking feasibility → Assembling blueprints";
const SCORE_LABELS: Record<ScoreDimension, string> = { skillsFit: "Skills fit", feasibility: "Feasibility", novelty: "Novelty", impact: "Impact", demoPotential: "Demo potential" };

const blankBrief = (): BriefInput => ({
  title: "",
  skills: [],
  problemStatement: "",
  availableHours: 24,
  teamSize: 3,
  teamRoles: [],
  domain: "",
  preferredTech: [],
  resources: [],
  constraints: [],
  scoringWeights: DEFAULT_WEIGHTS,
});

function tokenise(value: string) {
  return value.split(",").map(item => item.trim()).filter(Boolean);
}

function chipText(values: string[]) {
  return values.join(", ");
}

function SkillField({ label, values, onChange, placeholder, helper }: { label: string; values: string[]; onChange: (next: string[]) => void; placeholder: string; helper?: string }) {
  const [input, setInput] = useState("");
  const commit = () => {
    const next = Array.from(new Set([...values, ...tokenise(input)])).slice(0, 12);
    onChange(next);
    setInput("");
  };
  return <label className="field-group">
    <span className="field-label">{label}</span>
    <div className="tag-input" onClick={event => (event.currentTarget.querySelector("input") as HTMLInputElement | null)?.focus()}>
      {values.map(value => <span key={value} className="tag"><span>{value}</span><button type="button" onClick={event => { event.stopPropagation(); onChange(values.filter(item => item !== value)); }} aria-label={`Remove ${value}`}><X size={12} /></button></span>)}
      <input value={input} onChange={event => setInput(event.target.value)} onKeyDown={event => { if (["Enter", ","].includes(event.key)) { event.preventDefault(); commit(); } if (event.key === "Backspace" && !input && values.length) onChange(values.slice(0, -1)); }} onBlur={commit} placeholder={values.length ? "Add another" : placeholder} />
    </div>
    {helper && <span className="field-helper">{helper}</span>}
  </label>;
}

function WeightControl({ dimension, value, onChange }: { dimension: ScoreDimension; value: number; onChange: (value: number) => void }) {
  return <label className="weight-control">
    <span>{SCORE_LABELS[dimension]}</span><strong>{value}%</strong>
    <input aria-label={`${SCORE_LABELS[dimension]} weight`} type="range" min="0" max="50" step="5" value={value} onChange={event => onChange(Number(event.target.value))} />
  </label>;
}

function ScoreBar({ label, value, highlight = false }: { label: string; value: number; highlight?: boolean }) {
  return <div className="score-bar">
    <div><span>{label}</span><strong>{value.toFixed(1)}</strong></div>
    <div className="score-track"><motion.span initial={{ scaleX: 0 }} animate={{ scaleX: value / 10 }} transition={{ duration: 0.35, ease: [0.23, 1, 0.32, 1] }} className={highlight ? "score-fill score-fill-hot" : "score-fill"} /></div>
  </div>;
}

function CardScore({ concept }: { concept: ConceptCard }) {
  return <div className="overall-score"><span>Weighted score</span><strong>{concept.scores.overall.toFixed(1)}</strong><small>/10</small></div>;
}

function MiniScoreChart({ concept }: { concept: ConceptCard }) {
  return <div className="mini-score-chart" role="img" aria-label={`${concept.name} score bars: skills fit ${concept.scores.skillsFit}, feasibility ${concept.scores.feasibility}, novelty ${concept.scores.novelty}, impact ${concept.scores.impact}, demo potential ${concept.scores.demoPotential}`}>
    {(Object.keys(SCORE_LABELS) as ScoreDimension[]).map(key => <span key={key} style={{ height: `${Math.max(16, concept.scores[key] * 3.2)}px` }} />)}
  </div>;
}

function ConceptCardView({ concept, active, compared, onSelect, onCompare }: { concept: ConceptCard; active: boolean; compared: boolean; onSelect: () => void; onCompare: () => void }) {
  return <motion.article layout initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }} className={`concept-card ${active ? "is-active" : ""}`}>
    <button type="button" className="concept-main" onClick={onSelect} aria-pressed={active}>
      <div className="card-topline"><span className="rank-label">0{concept.rank}</span><CardScore concept={concept} /></div>
      <h3>{concept.name}</h3><p className="concept-hook">{concept.hook}</p>
      <div className="concept-facts"><span><Target size={13} />{concept.targetUser}</span><span><Gauge size={13} />{concept.difficulty}</span><span><Archive size={13} />{concept.buildTime}</span></div>
      <p className="concept-solution"><strong>Solution</strong>{concept.solution}</p>
    </button>
    <div className="card-actions"><span className="tech-line">{concept.techStack.slice(0, 3).join(" · ")}</span><button type="button" className={`compare-toggle ${compared ? "is-selected" : ""}`} onClick={onCompare} aria-pressed={compared}>{compared ? <Check size={15} /> : <Plus size={15} />}{compared ? "In compare" : "Compare"}</button></div>
  </motion.article>;
}

function Inspector({ concept, onClose }: { concept: ConceptCard; onClose?: () => void }) {
  return <motion.aside initial={{ opacity: 0, x: 18, scale: 0.98 }} animate={{ opacity: 1, x: 0, scale: 1 }} exit={{ opacity: 0, x: 18, scale: 0.98 }} transition={{ duration: 0.24, ease: [0.23, 1, 0.32, 1] }} className="inspector-panel">
    <div className="inspector-heading"><div><span className="eyebrow">Concept rationale</span><h2>{concept.name}</h2></div>{onClose && <button type="button" className="icon-button" onClick={onClose} aria-label="Close concept details"><X size={18} /></button>}</div>
    <p>{concept.differentiator}</p>
    <div className="score-stack">{(Object.keys(SCORE_LABELS) as ScoreDimension[]).map(key => <ScoreBar key={key} label={SCORE_LABELS[key]} value={concept.scores[key]} />)}</div>
    <section><h4>Why it fits</h4>{(Object.keys(SCORE_LABELS) as ScoreDimension[]).map(key => <p key={key} className="rationale"><span>{SCORE_LABELS[key]}</span>{concept.scoreRationale[key]}</p>)}</section>
    <section className="assumption-section"><h4>Assumptions</h4><ul>{concept.assumptions.map(item => <li key={item}>{item}</li>)}</ul></section>
    <section className="risk-section"><h4>Risks to contain</h4><ul>{concept.risks.map(item => <li key={item}>{item}</li>)}</ul></section>
    <div className="next-step"><Sparkles size={16} /><span><strong>Best next step</strong>{concept.nextStep}</span></div>
  </motion.aside>;
}

function CompareTray({ concepts, promoteId, promoting, onPromoteIdChange, onPromote, onRemove }: { concepts: ConceptCard[]; promoteId: string | undefined; promoting: boolean; onPromoteIdChange: (id: string) => void; onPromote: () => void; onRemove: (id: string) => void }) {
  if (!concepts.length) return null;
  return <motion.section initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 28 }} transition={{ duration: 0.24, ease: [0.23, 1, 0.32, 1] }} className="compare-tray" aria-label="Concept comparison tray">
    <div className="compare-title"><div className="compare-icon"><GitCompareArrows size={18} /></div><div><span>Compare tray</span><strong>{concepts.length} of 3 concepts</strong></div></div>
    <div className="compare-cards">{concepts.map(concept => <div key={concept.id ?? concept.rank} className={`compare-mini ${promoteId === concept.id ? "is-promoted" : ""}`}><button type="button" className="mini-select" onClick={() => concept.id && onPromoteIdChange(concept.id)} aria-pressed={promoteId === concept.id}><span>{concept.name}</span><MiniScoreChart concept={concept} /><strong>{concept.scores.overall.toFixed(1)}</strong></button><button type="button" className="mini-remove" aria-label={`Remove ${concept.name} from comparison`} onClick={() => concept.id && onRemove(concept.id)}><X size={14} /></button></div>)}</div>
    <Button className="promote-button" onClick={onPromote} disabled={!promoteId || promoting}>{promoting ? <Loader2 className="animate-spin" /> : <Rocket />}{promoting ? "Building blueprint" : "Promote to Blueprint"}</Button>
  </motion.section>;
}

function BlueprintView({ blueprint, concept, onBack, onSave, onExport, saving, exporting }: { blueprint: BlueprintArtifact; concept: ConceptCard; onBack: () => void; onSave: (next: BlueprintArtifact) => void; onExport: () => void; saving: boolean; exporting: boolean }) {
  const [draft, setDraft] = useState<BlueprintArtifact>(blueprint);
  const [editedAt, setEditedAt] = useState<Date>();
  const [editing, setEditing] = useState(false);
  const [jsonDraft, setJsonDraft] = useState(JSON.stringify(blueprint, null, 2));
  const update = (key: keyof BlueprintArtifact, value: BlueprintArtifact[keyof BlueprintArtifact]) => setDraft(current => ({ ...current, [key]: value }));
  const applyJson = () => { try { const parsed = JSON.parse(jsonDraft) as BlueprintArtifact; setDraft(parsed); setEditing(false); setEditedAt(new Date()); toast.success("Blueprint edits applied locally."); } catch { toast.error("The blueprint editor needs valid JSON before it can apply changes."); } };
  const handleSave = async (next: BlueprintArtifact) => { setEditedAt(new Date()); await onSave(next); };
  return <main className="blueprint-page">
    <header className="blueprint-header"><button type="button" className="back-link" onClick={onBack}><ArrowLeft size={16} />Concept Studio</button><div className="blueprint-actions">{editedAt && <span className="edited-badge"><Check size={12} />Edited · {editedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>}<button type="button" className="text-button" onClick={() => setEditing(value => !value)}>{editing ? "Close editor" : "Edit all sections"}</button><Button variant="outline" onClick={onExport} disabled={exporting}>{exporting ? <Loader2 className="animate-spin" /> : <Download /> }Export Markdown</Button><Button onClick={() => handleSave(draft)} disabled={saving}>{saving ? <Loader2 className="animate-spin" /> : <Check />}Save edits</Button></div></header>
    <section className="blueprint-hero"><div><span className="eyebrow">Execution blueprint</span><h1>{concept.name}</h1><p>{concept.hook}</p></div><CardScore concept={concept} /></section>
    <AnimatePresence>{editing && <motion.section initial={{ opacity: 0, y: -10, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -10, scale: 0.98 }} className="advanced-editor"><div><strong>Full blueprint editor</strong><span>Edits are saved separately from the original model artifact.</span></div><Textarea value={jsonDraft} onChange={event => setJsonDraft(event.target.value)} aria-label="Full blueprint JSON editor" /><div><Button variant="outline" onClick={() => setEditing(false)}>Cancel</Button><Button onClick={applyJson}>Apply edits</Button></div></motion.section>}</AnimatePresence>
    <section className="blueprint-grid">
      <article className="blueprint-intro"><span className="eyebrow">Build intent</span><Textarea className="overview-editor" value={draft.overview} onChange={event => update("overview", event.target.value)} aria-label="Blueprint overview" /></article>
      <article className="blueprint-section span-two"><div className="section-heading"><ClipboardCheck /><div><span className="eyebrow">MVP scope</span><h2>Build the signal, not the entire platform.</h2></div></div><div className="mvp-grid">{draft.mvpFeatures.map((feature, index) => <div className="mvp-card" key={`${feature.title}-${index}`}><select value={feature.priority} onChange={event => update("mvpFeatures", draft.mvpFeatures.map((item, itemIndex) => itemIndex === index ? { ...item, priority: event.target.value as "Must" | "Should" | "Could" } : item))}><option>Must</option><option>Should</option><option>Could</option></select><Input value={feature.title} onChange={event => update("mvpFeatures", draft.mvpFeatures.map((item, itemIndex) => itemIndex === index ? { ...item, title: event.target.value } : item))} /><Textarea value={feature.detail} onChange={event => update("mvpFeatures", draft.mvpFeatures.map((item, itemIndex) => itemIndex === index ? { ...item, detail: event.target.value } : item))} /></div>)}</div></article>
      <article className="blueprint-section"><div className="section-heading"><Code2 /><div><span className="eyebrow">System shape</span><h2>Architecture</h2></div></div><div className="architecture-list">{draft.architecture.map(item => <div key={item.layer}><strong>{item.layer}</strong><p>{item.purpose}</p><span>{item.technologies.join(" · ")}</span></div>)}</div></article>
      <article className="blueprint-section"><div className="section-heading"><BrainCircuit /><div><span className="eyebrow">Dependencies</span><h2>Data & APIs</h2></div></div><div className="dependency-list">{draft.dataAndApis.map(item => <div key={item.name}><strong>{item.name}</strong><p>{item.need}</p>{item.alternative && <small>Fallback: {item.alternative}</small>}</div>)}</div></article>
      <article className="blueprint-section span-two"><div className="section-heading"><Rocket /><div><span className="eyebrow">Momentum</span><h2>Build plan</h2></div></div><div className="timeline">{draft.buildPlan.map((item, index) => <div className="timeline-item" key={`${item.window}-${index}`}><span>{item.window}</span><div><strong>{item.goal}</strong><ul>{item.tasks.map(task => <li key={task}>{task}</li>)}</ul></div></div>)}</div></article>
      <article className="blueprint-section"><div className="section-heading"><Users /><div><span className="eyebrow">Team</span><h2>Role allocation</h2></div></div>{draft.teamPlan.map(item => <div className="team-row" key={item.role}><strong>{item.role}</strong><span>{item.responsibilities.join(" · ")}</span></div>)}</article>
      <article className="blueprint-section"><div className="section-heading"><Lightbulb /><div><span className="eyebrow">Story</span><h2>Demo & pitch</h2></div></div><ol className="demo-list">{draft.demoFlow.map(step => <li key={step}>{step}</li>)}</ol><div className="pitch-card"><strong>{draft.judgePitch.opening}</strong><p><b>Problem:</b> {draft.judgePitch.problem}</p><p><b>Solution:</b> {draft.judgePitch.solution}</p><p><b>Proof:</b> {draft.judgePitch.proof}</p><p><b>Close:</b> {draft.judgePitch.close}</p></div></article>
      <article className="blueprint-section risk-blueprint"><div className="section-heading"><CircleAlert /><div><span className="eyebrow">Safeguards</span><h2>Risks & fallback</h2></div></div>{draft.risks.map(item => <div className="risk-row" key={item.risk}><strong>{item.risk}</strong><span>{item.mitigation}</span></div>)}<Textarea value={draft.fallbackPlan} onChange={event => update("fallbackPlan", event.target.value)} aria-label="Fallback plan" /></article>
      <article className="blueprint-section"><div className="section-heading"><Sparkles /><div><span className="eyebrow">After the demo</span><h2>Extensions</h2></div></div><ul className="extension-list">{draft.extensions.map(item => <li key={item}>{item}</li>)}</ul></article>
    </section>
  </main>;
}

type SynapseWorkspaceProps = { projectId?: string; embedded?: boolean };

export default function SynapseWorkspace({ projectId: suppliedProjectId, embedded = false }: SynapseWorkspaceProps) {
  const { user, loading } = useFirebaseAuth();
  const [location] = useLocation();
  const morrowProjectId = suppliedProjectId ?? location.split("/")[3] ?? "";
  const isAuthenticated = Boolean(user);
  const reducedMotion = useReducedMotion();
  const [brief, setBrief] = useState<BriefInput>(blankBrief);
  const [expanded, setExpanded] = useState(false);
  const [skillText, setSkillText] = useState("");
  const [concepts, setConcepts] = useState<ConceptCard[]>([]);
  const [projectId, setProjectId] = useState<string>();
  const [activeId, setActiveId] = useState<string>();
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [promoteId, setPromoteId] = useState<string>();
  const [filter, setFilter] = useState("all");
  const [sortBy, setSortBy] = useState<"score" | "feasibility" | "novelty">("score");
  const [view, setView] = useState<"brief" | "studio" | "blueprint">("brief");
  const [blueprint, setBlueprint] = useState<BlueprintArtifact>();
  const [blueprintId, setBlueprintId] = useState<string>();
  const [generationError, setGenerationError] = useState<string>();
  const generate = trpc.synapse.generate.useMutation();
  const saveComparison = trpc.synapse.saveComparison.useMutation();
  const promote = trpc.synapse.promoteToBlueprint.useMutation();
  const saveBlueprint = trpc.synapse.saveBlueprintEdits.useMutation();
  const exportMarkdown = trpc.synapse.exportMarkdown.useMutation();
  const projects = trpc.synapse.projects.useQuery(undefined, { enabled: isAuthenticated });
  const [generationStage, setGenerationStage] = useState(0);

  useEffect(() => {
    if (!user || !morrowProjectId) return;
    let active = true;
    loadWorkspaceArtifacts(morrowProjectId).then(artifacts => {
      if (!active) return;
      const savedChallenge = typeof artifacts.brief?.challenge === "string" ? artifacts.brief.challenge : "";
      if (savedChallenge) setBrief(current => current.problemStatement ? current : { ...current, problemStatement: savedChallenge });
      const savedConcepts = Array.isArray(artifacts.concepts?.concepts) ? artifacts.concepts.concepts as ConceptCard[] : [];
      if (!savedConcepts.length) return;
      setConcepts(savedConcepts);
      const selectedConceptId = artifacts.blueprint?.selectedConceptId as string | undefined;
      const selectedConcept = savedConcepts.find(concept => concept.id === selectedConceptId) ?? savedConcepts[0];
      setActiveId(selectedConcept?.id);
      const savedBlueprintArtifact = artifacts.blueprint;
      const savedBlueprint = savedBlueprintArtifact?.userEdits ?? savedBlueprintArtifact?.immutableOutput;
      if (savedBlueprint && selectedConcept && savedBlueprintArtifact && typeof savedBlueprintArtifact.blueprintId === "string") {
        setBlueprint(savedBlueprint as BlueprintArtifact);
        setBlueprintId(savedBlueprintArtifact.blueprintId);
        setView("blueprint");
      } else {
        setView("studio");
      }
    }).catch(() => undefined);
    return () => { active = false; };
  }, [morrowProjectId, user]);

  useEffect(() => {
    if (!generate.isPending || reducedMotion) return;
    setGenerationStage(0);
    const timer = window.setInterval(() => setGenerationStage(stage => Math.min(stage + 1, 3)), 900);
    return () => window.clearInterval(timer);
  }, [generate.isPending, reducedMotion]);

  const activeConcept = concepts.find(concept => concept.id === activeId) ?? concepts[0];
  const compareConcepts = concepts.filter(concept => concept.id && compareIds.includes(concept.id));
  const visibleConcepts = useMemo(() => concepts.filter(concept => filter === "all" || concept.difficulty === filter).sort((left, right) => {
    const key = sortBy === "score" ? "overall" : sortBy;
    return right.scores[key] - left.scores[key];
  }), [concepts, filter, sortBy]);

  const surpriseMe = () => {
    setBrief({ title: "SignalBridge", skills: ["React", "Python", "UI/UX"], problemStatement: "Help small community clinics anticipate no-shows and offer patients a simple, respectful way to reschedule scarce appointments.", availableHours: 24, teamSize: 3, teamRoles: ["Frontend", "AI / data", "Design"], domain: "Health equity", preferredTech: ["React", "Groq", "Python"], resources: ["Synthetic appointment data", "SMS mock"], constraints: ["No real patient data", "Mobile-first demo"], scoringWeights: { skillsFit: 25, feasibility: 30, novelty: 15, impact: 20, demoPotential: 10 } });
    toast.success("A high-signal brief is ready to refine.");
  };

  const generateConcepts = async () => {
    const skills = brief.skills.length ? brief.skills : tokenise(skillText);
    if (!skills.length || brief.problemStatement.trim().length < 12) { toast.error("Add at least one skill and a concise problem statement first."); return; }
    if (!isAuthenticated) { toast.message("Sign in to generate and save a private workspace."); return; }
    try {
      setGenerationError(undefined);
      setView("studio");
      if (!morrowProjectId) throw new Error("A Morrow project is required before generation can begin.");
      const result = await generate.mutateAsync({ ...brief, projectId: morrowProjectId, skills, title: brief.title || "Untitled hackathon workspace" });
      setConcepts(result.concepts);
      setProjectId(result.projectId);
      setActiveId(result.concepts[0]?.id);
      setCompareIds([]);
      setPromoteId(undefined);
      setView("studio");
      if (user && morrowProjectId) {
        await Promise.all([
          saveBrief(morrowProjectId, user.uid, brief.problemStatement),
          saveGenerationArtifacts(morrowProjectId, user.uid, result.concepts),
          saveConceptArtifacts(morrowProjectId, user.uid, result.concepts),
        ]);
      }
      toast.success(`${result.concepts.length} distinct directions are ready to compare.`);
      projects.refetch();
    } catch (error) { const message = error instanceof Error ? error.message : "Generation could not be completed. Please retry."; setView("brief"); setGenerationError(message); toast.error(message); }
  };

  const toggleCompare = (id: string | undefined) => {
    if (!id) return;
    setCompareIds(current => {
      const result = updateCompareSelection(current, id);
      if (result.limitReached) { toast.error("The compare tray holds up to three concepts."); return current; }
      const next = result.next;
      setPromoteId(previous => previous ?? id);
      return next;
    });
  };

  const promoteBlueprint = async () => {
    if (!promoteId || !projectId) return;
    try {
      await saveComparison.mutateAsync({ projectId, conceptIds: compareIds });
      if (!morrowProjectId) return;
      if (user) await saveComparisonArtifacts(morrowProjectId, user.uid, compareIds);
      const result = await promote.mutateAsync({ projectId: morrowProjectId, conceptId: promoteId });
      if (user) await saveBlueprintArtifacts(morrowProjectId, user.uid, result.blueprint, result.blueprint, result.blueprintId, promoteId);
      setBlueprint(result.blueprint);
      setBlueprintId(result.blueprintId);
      setView("blueprint");
      toast.success("Your execution blueprint is ready to shape.");
    } catch (error) { toast.error(error instanceof Error ? error.message : "The blueprint could not be created."); }
  };

  const saveEdits = async (next: BlueprintArtifact) => {
    if (!blueprintId) return;
    try { if (!morrowProjectId) return; await saveBlueprint.mutateAsync({ projectId: morrowProjectId, blueprintId, content: next }); if (user) await saveBlueprintRevision(morrowProjectId, user.uid, next); setBlueprint(next); toast.success("Your edits are stored separately from the original AI blueprint."); } catch (error) { toast.error(error instanceof Error ? error.message : "Edits could not be saved."); }
  };

  const downloadMarkdown = async () => {
    if (!blueprintId) return;
    try {
      if (!morrowProjectId) return;
      const result = await exportMarkdown.mutateAsync({ projectId: morrowProjectId, blueprintId });
      if (user && morrowProjectId) await savePortableExport(morrowProjectId, user.uid, result.content);
      const url = URL.createObjectURL(new Blob([result.content], { type: "text/markdown;charset=utf-8" }));
      const anchor = document.createElement("a"); anchor.href = url; anchor.download = result.filename; anchor.click(); URL.revokeObjectURL(url);
      toast.success("Portable Markdown export downloaded.");
    } catch (error) { toast.error(error instanceof Error ? error.message : "Markdown export could not be created."); }
  };

  if (view === "blueprint" && blueprint && activeConcept) return <BlueprintView blueprint={blueprint} concept={activeConcept} onBack={() => setView("studio")} onSave={saveEdits} onExport={downloadMarkdown} saving={saveBlueprint.isPending} exporting={exportMarkdown.isPending} />;

  return <div className={embedded ? "synapse-shell morrow-embedded-workspace" : "synapse-shell"}>
    <div className="ambient-field" aria-hidden="true"><span /><span /><span /></div>
    {!embedded && <header className="topbar"><a className="brand" href="/"><span className="brand-mark"><BrainCircuit size={20} /></span><span>Morrow</span></a><nav><a href="#workspace">Workspace</a><a href="#how-it-works">Method</a>{isAuthenticated ? <span className="user-chip">{user?.displayName?.split(" ")[0] ?? "Builder"}</span> : <Button variant="outline" disabled={loading}>Sign in</Button>}</nav></header>}
    {view === "brief" ? <main className="brief-page" id="workspace">
      <section className="brief-hero"><motion.div initial={{ opacity: 0, y: reducedMotion ? 0 : 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, ease: [0.23, 1, 0.32, 1] }}><span className="eyebrow"><Sparkles size={14} />From blank brief to build-ready</span><h1>Find the idea<br /><em>worth building.</em></h1><p>Morrow turns your context into distinct project directions, helps you compare the trade-offs, and converts your choice into a focused execution blueprint.</p><div className="hero-proof"><span><Check />Private workspace</span><span><Check />Server-side generation</span><span><Check />Exportable plan</span></div></motion.div><div className="hero-orbit" aria-hidden="true"><span className="orbit-core"><BrainCircuit size={28} /></span><span className="orbit orbit-a" /><span className="orbit orbit-b" /><span className="signal signal-a" /><span className="signal signal-b" /></div></section>
      <section className="brief-card"><div className="brief-card-heading"><div><span className="eyebrow">01 — Frame the challenge</span><h2>Your build context</h2><p>Start with the non-negotiables. Expand the context when you want a sharper match.</p></div><button type="button" className="surprise-button" onClick={surpriseMe}><Sparkles size={15} />Surprise Me</button></div>
        <div className="brief-form"><div className="form-row"><label className="field-group"><span className="field-label">Workspace name <small>Optional</small></span><Input value={brief.title ?? ""} placeholder="e.g. SignalBridge" onChange={event => setBrief(current => ({ ...current, title: event.target.value }))} /></label><label className="field-group"><span className="field-label">Available build time</span><div className="segmented-control">{[12, 24, 36, 48].map(hours => <button type="button" key={hours} className={brief.availableHours === hours ? "is-selected" : ""} onClick={() => setBrief(current => ({ ...current, availableHours: hours }))}>{hours}h</button>)}</div></label></div>
          <SkillField label="Your skills" values={brief.skills} onChange={skills => setBrief(current => ({ ...current, skills }))} placeholder="Type a skill and press Enter" helper="Use skills your team can actually ship with." />
          <label className="field-group"><span className="field-label">Problem statement</span><Textarea value={brief.problemStatement} placeholder="What real friction, unmet need, or opportunity will your team solve?" onChange={event => setBrief(current => ({ ...current, problemStatement: event.target.value }))} /><span className="field-helper">Focus on an observable user problem, not a preferred solution.</span></label>
          <button type="button" className="expand-trigger" onClick={() => setExpanded(value => !value)} aria-expanded={expanded}>{expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}{expanded ? "Hide context controls" : "Add context for a better fit"}</button>
          <AnimatePresence>{expanded && <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.24 }} className="advanced-brief"><div className="form-row three"><label className="field-group"><span className="field-label">Team size</span><Input type="number" min="1" max="12" value={brief.teamSize} onChange={event => setBrief(current => ({ ...current, teamSize: Number(event.target.value) }))} /></label><label className="field-group"><span className="field-label">Domain or track</span><Input value={brief.domain ?? ""} placeholder="e.g. Climate, Open innovation" onChange={event => setBrief(current => ({ ...current, domain: event.target.value }))} /></label><label className="field-group"><span className="field-label">Team roles</span><Input value={chipText(brief.teamRoles)} placeholder="Design, frontend, ML" onChange={event => setBrief(current => ({ ...current, teamRoles: tokenise(event.target.value) }))} /></label></div><div className="form-row"><SkillField label="Preferred tech" values={brief.preferredTech} onChange={preferredTech => setBrief(current => ({ ...current, preferredTech }))} placeholder="React, Python, APIs" /><SkillField label="Available resources" values={brief.resources} onChange={resources => setBrief(current => ({ ...current, resources }))} placeholder="Public dataset, SMS mock" helper="Things you can use: datasets, APIs, hardware." /></div><SkillField label="Hard constraints" values={brief.constraints} onChange={constraints => setBrief(current => ({ ...current, constraints }))} placeholder="No real user data, mobile-first demo" helper="Non-negotiable limits the AI must respect." /><div className="weights-panel"><div><span className="eyebrow">Decision lens</span><h3>What should win?</h3><p>Weights personalize the final ranking.</p></div><div className="weight-grid">{(Object.keys(SCORE_LABELS) as ScoreDimension[]).map(key => <WeightControl key={key} dimension={key} value={brief.scoringWeights[key]} onChange={value => setBrief(current => ({ ...current, scoringWeights: { ...current.scoringWeights, [key]: value } }))} />)}</div></div></motion.div>}</AnimatePresence>
        </div>{generationError && <div className="generation-error" role="alert"><CircleAlert size={17} /><div><strong>Generation paused</strong><span>{generationError}</span></div><button type="button" onClick={generateConcepts}>Try again</button></div>}<div className="brief-footer"><p><span>Private by design.</span> Your brief is sent only through the protected generation service.</p><Button size="lg" className="generate-button" onClick={generateConcepts} disabled={generate.isPending}>{generate.isPending ? <Loader2 className="animate-spin" /> : <Sparkles />}{generate.isPending ? "Generating your directions" : "Generate concept directions"}<ArrowRight /></Button></div>
      </section><section id="how-it-works" className="method-strip"><span>Brief</span><i /><span>Explore</span><i /><span>Compare</span><i /><span>Commit</span><i /><span>Build</span></section>
    </main> : <main className="studio-page"><header className="studio-header"><div><button type="button" className="back-link" onClick={() => setView("brief")}><ArrowLeft size={16} />New brief</button><span className="eyebrow">02 — Explore directions</span><h1>Concept Studio</h1><p>{brief.problemStatement}</p></div><div className="studio-meta"><span>{concepts.length} directions</span><span>·</span><span>{brief.availableHours}h build window</span></div></header>
      <section className="studio-controls"><div className="filter-set" aria-label="Filter concept cards"><button type="button" className={filter === "all" ? "is-selected" : ""} onClick={() => setFilter("all")}>All</button>{["Beginner", "Intermediate", "Advanced"].map(level => <button type="button" key={level} className={filter === level ? "is-selected" : ""} onClick={() => setFilter(level)}>{level}</button>)}</div><label className="sort-control">Sort by<select value={sortBy} onChange={event => setSortBy(event.target.value as typeof sortBy)}><option value="score">Best fit</option><option value="feasibility">Feasibility</option><option value="novelty">Novelty</option></select></label></section>
      {generate.isPending ? <section className="generation-state" aria-live="polite"><div className="generation-orbit"><span /><span /><BrainCircuit /></div><div><span className="eyebrow">Working with your context</span><h2>Mapping the viable edge.</h2><p>{STAGE_LABEL}</p><div className="stage-dots">{[0, 1, 2, 3].map(index => <span className={index <= generationStage ? "is-active" : ""} key={index} />)}</div></div></section> : <section className="studio-layout"><div className="concept-deck">{visibleConcepts.length ? visibleConcepts.map(concept => <ConceptCardView key={concept.id ?? concept.rank} concept={concept} active={activeConcept?.id === concept.id} compared={Boolean(concept.id && compareIds.includes(concept.id))} onSelect={() => setActiveId(concept.id)} onCompare={() => toggleCompare(concept.id)} />) : <div className="empty-studio"><Sparkles size={24} /><h2>No directions match this filter.</h2><p>Clear the difficulty filter to see every generated concept.</p><Button variant="outline" onClick={() => setFilter("all")}>Show all directions</Button></div>}</div><div className="desktop-inspector"><AnimatePresence mode="wait">{activeConcept && <Inspector key={activeConcept.id ?? activeConcept.rank} concept={activeConcept} />}</AnimatePresence></div><div className="mobile-inspector"><AnimatePresence mode="wait">{activeConcept && <Inspector key={activeConcept.id ?? activeConcept.rank} concept={activeConcept} />}</AnimatePresence></div></section>}
      <AnimatePresence>{compareConcepts.length > 0 && <CompareTray concepts={compareConcepts} promoteId={promoteId} promoting={promote.isPending} onPromoteIdChange={setPromoteId} onPromote={promoteBlueprint} onRemove={id => { setCompareIds(current => current.filter(item => item !== id)); if (promoteId === id) setPromoteId(compareIds.find(item => item !== id)); }} />}</AnimatePresence>
    </main>}
  </div>;
}
