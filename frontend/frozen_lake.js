"use strict";

/* ============================================================
 *  Frozen Lake front-end (vanilla JS, no frameworks)
 *  Talks to the FastAPI backend (same origin, served at "/").
 *  Builds the grid MDP client-side (mirrors backend/utils/frozen_lake.py),
 *  sends P to /train for policy iteration, plays over /predict + /ws/predict.
 * ============================================================ */

// ---------- API ----------------------------------------------------

function apiBase() {
  return location.hostname === "" ? "http://127.0.0.1:8123" : "";
}

async function api(path, body) {
  const res = await fetch(apiBase() + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request to ${path} failed (HTTP ${res.status})`);
  }
  return data;
}

async function apiGet(path) {
  const res = await fetch(apiBase() + path);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request to ${path} failed (HTTP ${res.status})`);
  }
  return data;
}

function wsBase() {
  if (location.hostname === "") return "ws://127.0.0.1:8123";
  return (location.protocol === "https:" ? "wss://" : "ws://") + location.host;
}

// ---------- Actions -------------------------------------------------
// Codes follow the Gymnasium FrozenLake convention.

const ACTION_NAMES = { 0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP" };
const ACTION_ARROWS = { 0: "←", 1: "↓", 2: "→", 3: "↑" };
const DELTAS = { 0: [0, -1], 1: [1, 0], 2: [0, 1], 3: [-1, 0] }; // (drow, dcol)

// ---------- DOM -----------------------------------------------------

const $ = (id) => document.getElementById(id);

const els = {
  preset: $("preset"),
  intendedProb: $("intendedProb"),
  intendedProbLabel: $("intendedProbLabel"),
  goalReward: $("goalReward"),
  holeReward: $("holeReward"),
  seed: $("seed"),
  initBtn: $("initBtn"),
  gamma: $("gamma"),
  tol: $("tol"),
  initPolicy: $("initPolicy"),
  agentType: $("agentType"),
  piParams: $("piParams"),
  tdParams: $("tdParams"),
  tdGamma: $("tdGamma"),
  tdEpisodes: $("tdEpisodes"),
  tdMaxSteps: $("tdMaxSteps"),
  tdInitPolicy: $("tdInitPolicy"),
  initAlpha: $("initAlpha"),
  minAlpha: $("minAlpha"),
  alphaDecay: $("alphaDecay"),
  epsilonDecay: $("epsilonDecay"),
  initEpsilon: $("initEpsilon"),
  minEpsilon: $("minEpsilon"),
  tdLive: $("tdLive"),
  liveEpisode: $("liveEpisode"),
  liveStep: $("liveStep"),
  liveAlpha: $("liveAlpha"),
  liveEpsilon: $("liveEpsilon"),
  liveSuccess: $("liveSuccess"),
  successWrap: $("successWrap"),
  successChart: $("successChart"),
  chartTitle: $("chartTitle"),
  trainBtn: $("trainBtn"),
  statusLine: $("statusLine"),
  errorLine: $("errorLine"),
  grid: $("grid"),
  lastMove: $("lastMove"),
  valueChart: $("valueChart"),
  chartFallback: $("chartFallback"),
  modelCard: $("modelCard"),
  aiSpeed: $("aiSpeed"),
  aiSpeedLabel: $("aiSpeedLabel"),
  aiRunBtn: $("aiRunBtn"),
  aiControls: $("aiControls"),
  manualControls: $("manualControls"),
  upBtn: $("upBtn"),
  leftBtn: $("leftBtn"),
  downBtn: $("downBtn"),
  rightBtn: $("rightBtn"),
  resetEpisodeBtn: $("resetEpisodeBtn"),
  statState: $("statState"),
  statSteps: $("statSteps"),
  statReward: $("statReward"),
  statResult: $("statResult"),
  sessionSelect: $("sessionSelect"),
  sessionStatus: $("sessionStatus"),
  refreshSessionsBtn: $("refreshSessionsBtn"),
  loadModelBtn: $("loadModelBtn"),
  bulkCount: $("bulkCount"),
  bulkRunBtn: $("bulkRunBtn"),
  bulkSuccess: $("bulkSuccess"),
  bulkTotal: $("bulkTotal"),
  bulkRate: $("bulkRate"),
  bulkSteps: $("bulkSteps"),
  bulkProgress: $("bulkProgress"),
};

els.intendedProb.addEventListener("input", () => {
  els.intendedProbLabel.textContent = Number(els.intendedProb.value).toFixed(3);
});

// ---------- App state ----------------------------------------------

const S = {
  expId: null,
  envId: null,
  env: null,          // env descriptor (n, holes, start, goal, P, ...)
  models: [],
  agentTypes: [],   // agent type catalog fetched from the backend /agents
  state: null,
  steps: 0,
  episodeReward: 0,
  finished: false,
  busy: false,
  value: null,        // V(s) array returned by /train
  policy: null,       // learned action per state (for the grid arrows)
  trained: false,
  bulkRunning: false,
  training: false,       // policy iteration run in flight
  tdRunning: false,      // live TD (SARSA / Q-Learning) training loop active
  tdAbort: false,        // set by the Stop button to break the training loop
  tdEpisodeResults: [],  // 1/0 per completed episode of the current TD run
  chartMode: null,       // "bar" (policy iteration) | "line" (TD live)
  wsMsgId: 0,
  mode: "manual",
  aiTimer: null,
};

// ---------- Board presets ------------------------------------------

const PRESETS = {
  classic: { n: 4, start: 0, goal: 15, holes: [5, 7, 11, 12] },
  easy:    { n: 4, start: 0, goal: 15, holes: [7] },
  tricky:  { n: 4, start: 0, goal: 15, holes: [2, 5, 6, 9, 10, 11] },
  wide:    { n: 5, start: 0, goal: 24, holes: [4, 6, 9, 10, 14, 19, 20] },
  narrow:  { n: 3, start: 0, goal: 8,  holes: [1, 6] },
};

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function goalReachable(n, start, goal, holes) {
  const blocked = new Set(holes);
  const seen = new Set([start]);
  const queue = [start];
  while (queue.length) {
    const s = queue.shift();
    if (s === goal) return true;
    const r = Math.floor(s / n), c = s % n;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= n || nc < 0 || nc >= n) continue;
      const ns = nr * n + nc;
      if (blocked.has(ns) || seen.has(ns)) continue;
      seen.add(ns);
      queue.push(ns);
    }
  }
  return false;
}

function randomHoles(n, start, goal, seed, count) {
  const rng = mulberry32(seed || 1);
  const candidates = [];
  for (let s = 0; s < n * n; s++) if (s !== start && s !== goal) candidates.push(s);
  for (let i = candidates.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [candidates[i], candidates[j]] = [candidates[j], candidates[i]];
  }
  return candidates.slice(0, Math.min(count, candidates.length));
}

// Resolve the board layout for the currently selected preset. "loaded" keeps
// the layout of whatever environment is currently in S.env.
function currentLayout() {
  const key = els.preset.value;
  if (key === "loaded" && S.env) {
    return { n: S.env.n, start: S.env.start, goal: S.env.goal, holes: [...S.env.holes] };
  }
  if (key === "random") {
    const n = 4, start = 0, goal = n * n - 1;
    const seed = parseInt(els.seed.value, 10) || 1;
    let count = Math.max(1, Math.floor((n * n) * 0.2));
    let holes = [];
    do {
      holes = randomHoles(n, start, goal, seed, count);
      if (goalReachable(n, start, goal, holes)) break;
      count -= 1;
    } while (count >= 0);
    return { n, start, goal, holes };
  }
  const p = PRESETS[key] || PRESETS.classic;
  return { n: p.n, start: p.start, goal: p.goal, holes: [...p.holes] };
}

// ---------- MDP construction ----------------------------------------

// Mirrors FrozenLake._build_transitions: for a chosen action a the ACTUAL move
// is drawn from [(a-1)%4, a, (a+1)%4] with slip_probs [left, intended, right].
// Off-board moves clamp (agent stays put); branches landing on the same state
// are merged. Terminal states (goal + holes) are absorbing.
function buildTransitions(cfg) {
  const { n, n_states, goal, holes, goal_reward, hole_reward, step_reward, slip_probs } = cfg;
  const holeSet = new Set(holes);
  const rewardFor = (s) =>
    s === goal ? [goal_reward, true]
    : holeSet.has(s) ? [hole_reward, true]
    : [step_reward, false];

  const P = {};
  for (let s = 0; s < n_states; s++) {
    P[s] = {};
    for (let a = 0; a < 4; a++) {
      if (s === goal || holeSet.has(s)) {
        P[s][a] = [{ prob: 1, next_state: s, reward: step_reward, done: true }];
        continue;
      }
      const codes = [(a - 1 + 4) % 4, a, (a + 1) % 4];
      const branches = new Map();
      for (let i = 0; i < codes.length; i++) {
        const prob = slip_probs[i];
        if (prob <= 0) continue;
        const [dr, dc] = DELTAS[codes[i]];
        const r = Math.floor(s / n), c = s % n;
        const nr = Math.min(n - 1, Math.max(0, r + dr));
        const nc = Math.min(n - 1, Math.max(0, c + dc));
        const nxt = nr * n + nc;
        const [reward, done] = rewardFor(nxt);
        const key = `${nxt}|${reward}|${done}`;
        const existing = branches.get(key);
        branches.set(key, existing
          ? { ...existing, prob: existing.prob + prob }
          : { prob, next_state: nxt, reward, done });
      }
      P[s][a] = Array.from(branches.values());
    }
  }
  return P;
}

function sampleOutcome(outcomes) {
  const r = Math.random();
  let cumulative = 0;
  for (const o of outcomes) {
    cumulative += o.prob;
    if (r < cumulative) return o;
  }
  return outcomes[outcomes.length - 1];
}

function readEnvConfig() {
  const layout = currentLayout();
  const p = clampField(els.intendedProb, 0, 1, 1 / 3);
  const cfg = {
    name: "FrozenLake",
    env_type: "FrozenLake",
    n: layout.n,
    n_states: layout.n * layout.n,
    n_actions: 4,
    start: layout.start,
    goal: layout.goal,
    holes: layout.holes,
    goal_reward: parseFloat(els.goalReward.value),
    hole_reward: parseFloat(els.holeReward.value),
    step_reward: 0,
    intended_prob: p,
    slip_probs: [(1 - p) / 2, p, (1 - p) / 2],
    seed: parseInt(els.seed.value, 10),
    preset: els.preset.value,
  };
  cfg.actions = { LEFT: 0, DOWN: 1, RIGHT: 2, UP: 3 };
  cfg.skip_states = [...layout.holes, layout.goal];
  cfg.P = buildTransitions(cfg);
  return cfg;
}

function readTrainParams() {
  return {
    P: S.env.P,
    skip_states: S.env.skip_states,
    init_policy: els.initPolicy.value,
    gamma: parseFloat(els.gamma.value),
    tolerance: parseFloat(els.tol.value),
    max_iters: 1000,
    eval_max_iters: 10000,
  };
}

function readTdTrainParams() {
  return {
    initial_state: S.env.start,
    gamma: parseFloat(els.tdGamma.value),
    n_episodes: clampIntField(els.tdEpisodes, 1, 100000, 1000),
    max_steps: clampIntField(els.tdMaxSteps, 1, 5000, 100),
    init_alpha: parseFloat(els.initAlpha.value),
    min_alpha: parseFloat(els.minAlpha.value),
    alpha_decay_ratio: clampField(els.alphaDecay, 0, 1, 0.5),
    init_epsilon: clampField(els.initEpsilon, 0, 1, 1),
    min_epsilon: clampField(els.minEpsilon, 0, 1, 0.1),
    epsilon_decay_ratio: clampField(els.epsilonDecay, 0, 1, 0.9),
  };
}

// ---------- Chart ---------------------------------------------------

let chart = null;

function destroyChart() {
  if (chart) { chart.destroy(); chart = null; }
  S.chartMode = null;
}

function plotValue(V) {
  if (typeof Chart === "undefined") {
    const lines = V.map((v, s) => `V(${s}) = ${Number(v).toFixed(4)}`).join("<br>");
    els.chartFallback.innerHTML = lines;
    els.chartFallback.classList.remove("hidden");
    return;
  }
  if (S.chartMode !== "bar") destroyChart();   // leave TD live mode
  const holeSet = new Set(S.env.holes);
  const labels = V.map((_, s) => String(s));
  const color = (s) =>
    s === S.env.goal ? "rgba(251,191,36,0.85)"
    : holeSet.has(s) ? "rgba(148,163,184,0.85)"
    : "rgba(56,189,248,0.8)";
  if (!chart) {
    chart = new Chart(els.valueChart, {
      type: "bar",
      data: {
        labels,
        datasets: [{ label: "V(s) — State value", data: V, backgroundColor: V.map((_, s) => color(s)), borderRadius: 6 }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#e2e8f0" } },
          tooltip: { callbacks: { label: (c) => `V(${c.label}) = ${Number(c.raw).toFixed(4)}` } },
        },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" },
               title: { display: true, text: "Value", color: "#94a3b8" } },
        },
      },
    });
    S.chartMode = "bar";
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = V;
    chart.data.datasets[0].backgroundColor = V.map((_, s) => color(s));
    chart.update();
  }
}

// Live TD chart (SARSA / Q-Learning): one line per state, x-axis = episode, updated every
// episode boundary while the training loop runs.
function stateLineColor(s) {
  if (s === S.env.goal) return "rgb(251,191,36)";
  if (S.env.holes.includes(s)) return "rgb(148,163,184)";
  return `hsl(${Math.round((s * 360) / S.env.n_states)}, 70%, 62%)`;
}

function initLiveChart() {
  if (typeof Chart === "undefined") return;
  destroyChart();
  els.chartFallback.classList.add("hidden");
  const datasets = [];
  for (let s = 0; s < S.env.n_states; s++) {
    datasets.push({
      label: "s" + s,
      data: [],
      borderColor: stateLineColor(s),
      backgroundColor: stateLineColor(s),
      pointRadius: 0,
      borderWidth: (s === S.env.goal || S.env.holes.includes(s)) ? 2.5 : 1.5,
      tension: 0.25,
    });
  }
  chart = new Chart(els.valueChart, {
    type: "line",
    data: { labels: [], datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { labels: { color: "#e2e8f0", boxWidth: 10, font: { size: 10 } } },
      },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" },
             title: { display: true, text: "Episode", color: "#94a3b8" } },
        y: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" },
             title: { display: true, text: "Value V(s)", color: "#94a3b8" } },
      },
    },
  });
  S.chartMode = "line";
}

function pushLivePoint(episode, V) {
  if (!chart || S.chartMode !== "line" || !V) return;
  chart.data.labels.push(String(episode));
  chart.data.datasets.forEach((ds, s) => ds.data.push(Number(V[s]) || 0));
  chart.update("none");
}

function clearChart() {
  destroyChart();
  destroySuccessChart();
  els.chartFallback.classList.add("hidden");
}

// ---------- Small helpers -------------------------------------------

function clampField(input, min, max, fallback) {
  let v = parseFloat(input.value);
  if (Number.isNaN(v)) v = fallback;
  v = Math.min(max, Math.max(min, v));
  input.value = v;
  return v;
}

function clampIntField(input, min, max, fallback) {
  let v = parseInt(input.value, 10);
  if (Number.isNaN(v)) v = fallback;
  v = Math.min(max, Math.max(min, v));
  input.value = v;
  return v;
}

function setStatus(text, className) {
  els.statusLine.textContent = text;
  els.statusLine.className = "status" + (className ? " " + className : "");
}

function setError(text) {
  els.errorLine.textContent = text;
  if (text) console.error(text);
}

function setBusyGame(busy) {
  const locked = busy || S.tdRunning;
  const disabled = locked || !S.env || S.finished || S.mode !== "manual";
  els.upBtn.disabled = disabled;
  els.downBtn.disabled = disabled;
  els.leftBtn.disabled = disabled;
  els.rightBtn.disabled = disabled;
  els.resetEpisodeBtn.disabled = !S.env || S.busy;
}

function refreshSessionUI() {
  const active = !!S.env;
  els.trainBtn.disabled = !active;
  updateAIButton();
  els.refreshSessionsBtn.disabled = false;
  updateBulkButton();
  updateLoadButton();
  setBusyGame(false);
}

// ---------- Training agent selection --------------------------------

// The backend agent registry is the single source of truth: fetch /agents
// and build the dropdown from it, so registering a new agent in the backend
// needs no front-end change. `stepwise` marks interaction-based agents
// (TD control) that train live instead of solving the MDP in one shot.
async function loadAgentTypes() {
  try {
    const data = await apiGet("/agents");
    S.agentTypes = data.agents || [];
  } catch (err) {
    setError(err.message);   // keep whatever options are already rendered
    return;
  }
  if (!S.agentTypes.length) return;
  const current = els.agentType.value;
  els.agentType.innerHTML = "";
  for (const a of S.agentTypes) {
    const opt = document.createElement("option");
    opt.value = a.agent_type;
    opt.textContent = `${a.description} (${a.stepwise ? "learns by stepping" : "needs the MDP"})`;
    els.agentType.appendChild(opt);
  }
  if (S.agentTypes.some((a) => a.agent_type === current)) els.agentType.value = current;
  setAgentUI();
}

function agentInfo() {
  return S.agentTypes.find((a) => a.agent_type === els.agentType.value) || null;
}

function isTd() {
  const info = agentInfo();
  if (info) return !!info.stepwise;
  return els.agentType.value === "Sarsa" || els.agentType.value === "QLearning";
}

function tdAgentLabel() {
  const info = agentInfo();
  return (info && info.description) || els.agentType.value;
}

// Show the hyper-parameters of the selected agent, retitle the chart, and
// relabel the train button ("Train ..." normally, "Stop training" live).
function setAgentUI() {
  const td = isTd();
  els.piParams.classList.toggle("hidden", td);
  els.tdParams.classList.toggle("hidden", !td);
  els.tdLive.classList.toggle("hidden", !(td && S.tdRunning));
  els.chartTitle.textContent = td
    ? `Value function — live V(s) per episode (${tdAgentLabel()})`
    : "Value function (trained)";
  if (!S.tdRunning) {
    els.trainBtn.textContent = td ? `Train ${tdAgentLabel()} (live)` : "Train & plot V(s)";
  }
}

function updateBulkButton() {
  els.bulkRunBtn.disabled = !S.env || S.bulkRunning || S.tdRunning || S.training;
}

function updateLoadButton() {
  els.loadModelBtn.disabled = !els.sessionSelect.value || S.tdRunning || S.training;
}

function updateAIButton() {
  els.aiRunBtn.disabled = !S.env || S.mode !== "ai" || S.tdRunning || S.training;
}

function setSessionStatus(data) {
  if (!data) return;
  els.sessionStatus.textContent = data.trained
    ? `"${data.experiment_name}" — trained, ready for AI prediction`
    : `"${data.experiment_name}" — not trained yet`;
  els.sessionStatus.className = data.trained ? "ok" : "";
}

// ---------- Model card ----------------------------------------------

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmtNum(v) {
  if (v === undefined || v === null) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}

// Human-readable duration (cumulative training seconds) for the model card.
function fmtDuration(sec) {
  if (!sec) return "—";
  if (sec < 1) return `${Math.round(sec * 1000)} ms`;
  if (sec < 60) return `${sec.toFixed(1)} s`;
  const m = Math.floor(sec / 60);
  return `${m}m ${(sec - m * 60).toFixed(0)}s`;
}

function fmtTrainingParams(tp) {
  if (!tp) return "—";
  const parts = [];
  for (const k of ["gamma", "tolerance", "max_iters", "eval_max_iters", "init_policy",
                   "n_episodes", "max_steps", "init_alpha", "min_alpha",
                   "init_epsilon", "min_epsilon"]) {
    if (tp[k] !== undefined) parts.push(`${k}=${tp[k]}`);
  }
  return parts.length ? parts.join(" · ") : "—";
}

async function fetchEnvConfig(envId, agentParams) {
  if (envId) {
    try {
      const env = await apiGet(`/envs/${encodeURIComponent(envId)}`);
      return env.config;
    } catch (_) { /* fall back below */ }
  }
  return (agentParams && agentParams.env_config) || null;
}

async function fetchExperimentDetail(experimentId) {
  const detail = await apiGet(`/experiments/${encodeURIComponent(experimentId)}`);
  const envCfg = await fetchEnvConfig(detail.env_id, detail.agent_params);
  return { detail, envCfg };
}

function renderModelCard(detail, envCfg) {
  if (!detail) { els.modelCard.textContent = "No model loaded yet."; return; }
  const n = envCfg && envCfg.n;
  const rows = [
    ["Model", detail.experiment_name],
    ["Experiment", detail.experiment_id],
    ["Environment", `${detail.env_type} · ${n ? n + "×" + n : (envCfg && envCfg.n_states) || "—"} grid`],
    ["env_id", detail.env_id || "—"],
    ["Layout", envCfg
      ? `start ${envCfg.start} · goal ${envCfg.goal} · ${envCfg.holes ? envCfg.holes.length : 0} holes`
      : "—"],
    ["Holes", envCfg && envCfg.holes ? envCfg.holes.join(", ") : "—"],
    ["Intended prob", fmtNum(envCfg && envCfg.intended_prob)],
    ["Rewards", envCfg ? `goal ${fmtNum(envCfg.goal_reward)} · hole ${fmtNum(envCfg.hole_reward)}` : "—"],
    ["Agent", detail.agent_type],
    ["Trained", detail.trained ? "yes" : "no"],
    ["Training", fmtTrainingParams(detail.training_params)],
    ["Episodes trained", fmtNum(detail.episodes_trained)],
    ["Training time", fmtDuration(detail.training_time)],
    ["Value fn", detail.value && detail.value.length
      ? `min ${Math.min(...detail.value).toFixed(3)} · max ${Math.max(...detail.value).toFixed(3)}`
      : "—"],
  ];
  els.modelCard.innerHTML = rows
    .map(([label, value]) => {
      const empty = value === undefined || value === null || value === "—";
      const cls = label === "Trained" && !empty && String(value) === "yes"
        ? "ok" : empty ? "empty" : "";
      return `<div class="mc-label">${escapeHtml(label)}</div>` +
             `<div class="mc-value ${cls}">${escapeHtml(value)}</div>`;
    })
    .join("");
}

async function renderCurrentModelCard() {
  if (!S.expId) return;
  try {
    const { detail, envCfg } = await fetchExperimentDetail(S.expId);
    renderModelCard(detail, envCfg);
  } catch (_) { /* keep prior card */ }
}

// Reflect a loaded environment's parameters in the setup inputs so the UI
// matches the environment currently in use.
function applyEnvToConfigInputs(envCfg) {
  if (!envCfg) return;
  els.preset.value = "loaded";
  if (envCfg.intended_prob !== undefined) {
    els.intendedProb.value = envCfg.intended_prob;
    els.intendedProbLabel.textContent = Number(envCfg.intended_prob).toFixed(3);
  }
  if (envCfg.seed !== undefined) els.seed.value = envCfg.seed;
  if (envCfg.goal_reward !== undefined) els.goalReward.value = envCfg.goal_reward;
  if (envCfg.hole_reward !== undefined) els.holeReward.value = envCfg.hole_reward;
}

// ---------- Session library (experiment registry) -------------------

async function refreshSessions(selectId) {
  try {
    const data = await apiGet("/experiments");
    const list = data.experiments || [];
    S.models = list;
    els.sessionSelect.innerHTML = "";
    if (list.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "— no models yet —";
      els.sessionSelect.appendChild(opt);
      els.sessionStatus.textContent = "No models registered yet";
      els.sessionStatus.className = "";
    } else {
      list.forEach((model) => {
        const opt = document.createElement("option");
        opt.value = model.experiment_id;
        opt.textContent = `${model.experiment_name} · ${model.experiment_id.slice(0, 8)}${model.trained ? " ✓" : ""}`;
        els.sessionSelect.appendChild(opt);
      });
      if (selectId) els.sessionSelect.value = selectId;
      const trained = list.filter((m) => m.trained).length;
      els.sessionStatus.textContent = `${list.length} model(s) · ${trained} trained`;
      els.sessionStatus.className = "ok";
    }
    els.sessionSelect.disabled = false;
    els.refreshSessionsBtn.disabled = false;
    updateLoadButton();
    return data;
  } catch (err) {
    els.sessionStatus.textContent = "Could not load models";
    els.sessionStatus.className = "bad";
    els.sessionSelect.disabled = true;
    els.refreshSessionsBtn.disabled = false;
    setError(err.message);
    return null;
  }
}

async function selectSession(experimentId) {
  if (S.bulkRunning) { setError("Wait for bulk prediction to finish first."); return; }
  if (S.tdRunning) { setError(`Stop ${tdAgentLabel()} training first.`); return; }
  stopAI();
  setError("");
  if (!experimentId) return;
  try {
    const { detail, envCfg } = await fetchExperimentDetail(experimentId);
    if (!envCfg || !envCfg.P) {
      throw new Error("Selected model has no stored environment to restore.");
    }
    S.expId = detail.experiment_id;
    S.envId = detail.env_id;
    S.env = envCfg;
    S.value = detail.value || null;
    S.policy = null;
    S.trained = !!detail.trained;

    // Match the training-agent selector to the loaded model (no change event).
    if ([...els.agentType.options].some((o) => o.value === detail.agent_type)) {
      els.agentType.value = detail.agent_type;
    }
    setAgentUI();

    applyEnvToConfigInputs(envCfg);
    resetBulkStats();

    S.state = S.env.start;
    S.steps = 0;
    S.episodeReward = 0;
    S.finished = false;

    if (S.trained && S.value) {
      plotValue(S.value);
      S.policy = [];
      for (let s = 0; s < S.env.n_states; s++) {
        const p = await api("/predict", { experiment_id: S.expId, state: s });
        S.policy[s] = p.action;
      }
    } else {
      // Untrained or only partially trained: no policy / values to show.
      clearChart();
    }
    renderGrid();
    updateStats();
    setSessionStatus(detail);
    renderModelCard(detail, envCfg);
    setStatus(`Model loaded: ${detail.experiment_name} (${S.expId.slice(0, 8)}).`, detail.trained ? "ok" : "");
    refreshSessionUI();
  } catch (err) {
    setError(err.message);
    setStatus("Failed to load the selected model.", "bad");
  }
}

// ---------- Experiment lifecycle ------------------------------------

async function initExperiment() {
  if (S.bulkRunning) { setError("Wait for bulk prediction to finish first."); return; }
  if (S.tdRunning) { setError(`Stop ${tdAgentLabel()} training first.`); return; }
  stopAI();
  setError("");
  els.initBtn.disabled = true;
  els.initBtn.textContent = "Initializing...";
  try {
    const cfg = readEnvConfig();
    const td = isTd();
    const data = await api("/register_experiment", {
      agent_type: els.agentType.value,
      experiment_name: `frozenlake_${cfg.preset}_${cfg.n}x${cfg.n}${td ? "_" + els.agentType.value.toLowerCase() : ""}`,
      env_type: cfg.env_type,
      agent_params: {
        name: cfg.name.toLowerCase(),
        env_type: cfg.env_type,
        n_states: cfg.n_states,
        n_actions: cfg.n_actions,
        env_config: cfg,
        ...(td ? { initial_policy: els.tdInitPolicy.value } : {}),
      },
      env_config: cfg,
    });
    S.expId = data.experiment_id;
    S.envId = data.env_id;
    S.env = cfg;
    S.state = cfg.start;
    S.steps = 0;
    S.episodeReward = 0;
    S.finished = false;
    S.value = null;
    S.policy = null;
    S.trained = false;
    resetBulkStats();

    clearChart();
    renderGrid();
    updateStats();
    await refreshSessions(data.experiment_id);
    els.sessionStatus.textContent = "New session registered — train it to enable AI.";
    els.sessionStatus.className = "ok";
    refreshSessionUI();
    renderCurrentModelCard();
    setStatus(
      `Experiment ready (id ${S.expId.slice(0, 8)}). ` +
      `Grid ${cfg.n}×${cfg.n}, start ${cfg.start}, goal ${cfg.goal}, ${cfg.holes.length} hole(s).`,
      "ok"
    );
  } catch (err) {
    setError(err.message);
    setStatus("Registration failed.", "bad");
  } finally {
    els.initBtn.disabled = false;
    els.initBtn.textContent = "Initialize experiment";
    setBusyGame(false);
  }
}

async function trainAgent() {
  if (isTd()) return trainTd();
  if (!S.expId) { setError("Initialize the experiment first."); return; }
  setError("");
  els.trainBtn.disabled = true;
  els.trainBtn.textContent = "Training...";
  S.training = true;
  updateBulkButton();
  updateLoadButton();
  updateAIButton();
  setStatus("Training agent (policy iteration)...");
  try {
    const data = await api("/train", { experiment_id: S.expId, train_params: readTrainParams() });
    S.value = data.value;

    const policy = [];
    for (let s = 0; s < S.env.n_states; s++) {
      const p = await api("/predict", { experiment_id: S.expId, state: s });
      policy[s] = p.action;
    }
    S.policy = policy;
    S.trained = true;
    plotValue(S.value);
    renderGrid();
    updateBulkButton();
    await refreshSessions(S.expId);
    setSessionStatus({ experiment_name: "this session", trained: true });
    renderCurrentModelCard();
    setError("");
    setStatus(`Trained. V ranges ${Math.min(...S.value).toFixed(3)} .. ${Math.max(...S.value).toFixed(3)}.`, "ok");
  } catch (err) {
    setError(err.message);
    setStatus("Training failed.");
  } finally {
    S.training = false;
    els.trainBtn.disabled = false;
    els.trainBtn.textContent = "Train & plot V(s)";
    updateBulkButton();
    updateLoadButton();
    updateAIButton();
  }
}

// ---------- TD live training — SARSA / Q-Learning (WebSocket /ws/train_step) ----------
//
// The front end owns the rollouts: for every episode it starts an episode on
// the server, steps the local grid with the action the server returns, and
// sends (state, action, reward, next_state, done) back. Each reply gives the
// next action plus the live value function, plotted per episode as soon as it
// arrives. The frontend also enforces max_steps and the episode count.

function openTrainSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsBase() + "/ws/train_step");
    ws.onopen = () => resolve(ws);
    ws.onerror = () => reject(new Error("Training WebSocket could not be opened."));
  });
}

// One request/reply round-trip on the training socket (id-matched).
function wsTrain(ws, payload) {
  return new Promise((resolve, reject) => {
    const id = ++S.wsMsgId;
    const timer = setTimeout(() => {
      ws.removeEventListener("message", onMsg);
      reject(new Error("Training request timed out."));
    }, 15000);
    function onMsg(e) {
      let msg;
      try { msg = JSON.parse(e.data); } catch (_) { return; }
      if (msg.id === undefined || Number(msg.id) !== id) return;
      clearTimeout(timer);
      ws.removeEventListener("message", onMsg);
      if (msg.type === "error") reject(new Error(msg.detail || "Training failed."));
      else resolve(msg);
    }
    ws.addEventListener("message", onMsg);
    ws.send(JSON.stringify({ id, ...payload }));
  });
}

// Rolling window (in episodes) for the live success-rate readout.
const TD_SUCCESS_WINDOW = 100;

let successChart = null;

function destroySuccessChart() {
  if (successChart) { successChart.destroy(); successChart = null; }
  els.successWrap.classList.add("hidden");
}

// Rolling success-rate line: one point per episode boundary, y = % goals
// reached over the last TD_SUCCESS_WINDOW completed episodes.
function initSuccessChart() {
  if (typeof Chart === "undefined") return;   // text tile still works
  destroySuccessChart();
  els.successWrap.classList.remove("hidden");
  successChart = new Chart(els.successChart, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: `Success rate — last ${TD_SUCCESS_WINDOW} episodes`,
        data: [],
        borderColor: "rgb(74,222,128)",
        backgroundColor: "rgba(74,222,128,0.12)",
        fill: true,
        pointRadius: 0,
        borderWidth: 2,
        tension: 0.25,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { labels: { color: "#e2e8f0", boxWidth: 10, font: { size: 10 } } },
        tooltip: { callbacks: { label: (c) => `${Number(c.raw).toFixed(0)}%` } },
      },
      scales: {
        x: { ticks: { color: "#94a3b8" }, grid: { color: "#1e293b" },
             title: { display: true, text: "Episode", color: "#94a3b8" } },
        y: { min: 0, max: 100,
             ticks: { color: "#94a3b8", callback: (v) => v + "%" },
             grid: { color: "#1e293b" },
             title: { display: true, text: "Goal reached", color: "#94a3b8" } },
      },
    },
  });
}

// Goal-reaching rate over the last TD_SUCCESS_WINDOW completed episodes of
// the current training run; refreshed (and plotted) at every episode boundary.
function updateLiveSuccess(episode) {
  const results = S.tdEpisodeResults;
  if (!results.length) { els.liveSuccess.textContent = "—"; return; }
  const win = results.slice(-TD_SUCCESS_WINDOW);
  const rate = (100 * win.reduce((a, b) => a + b, 0)) / win.length;
  els.liveSuccess.textContent = `${Math.round(rate)}% (${win.length})`;
  if (successChart) {
    successChart.data.labels.push(String(episode));
    successChart.data.datasets[0].data.push(Number(rate.toFixed(1)));
    successChart.update("none");
  }
}

function updateLiveStats(reply) {
  const total = reply.episodes_trained;
  els.liveEpisode.textContent = reply.episode === undefined
    ? "—"
    : (total === undefined
        ? String(reply.episode + 1)
        : `${reply.episode + 1} / ${total} total`);
  els.liveStep.textContent = reply.step === undefined ? "—" : String(reply.step);
  els.liveAlpha.textContent = reply.alpha === undefined ? "—" : Number(reply.alpha).toFixed(3);
  els.liveEpsilon.textContent = reply.epsilon === undefined ? "—" : Number(reply.epsilon).toFixed(3);
}

async function trainTd() {
  if (!S.expId) { setError("Initialize the experiment first."); return; }
  if (S.bulkRunning) { setError("Wait for bulk prediction to finish first."); return; }
  if (S.tdRunning) {           // second click while running = stop
    S.tdAbort = true;
    els.trainBtn.disabled = true;
    els.trainBtn.textContent = "Stopping...";
    return;
  }

  stopAI();
  setError("");
  const tp = readTdTrainParams();
  S.tdRunning = true;
  S.tdAbort = false;
  S.tdEpisodeResults = [];
  S.busy = true;
  setBusyGame(true);
  updateBulkButton();
  updateLoadButton();
  updateAIButton();
  els.trainBtn.disabled = false;
  els.trainBtn.textContent = "Stop training";
  els.tdLive.classList.remove("hidden");
  initLiveChart();
  initSuccessChart();

  let ws = null;
  let lastEpisode = 0;
  try {
    ws = await openTrainSocket();
    let reply = await wsTrain(ws, { type: "start", experiment_id: S.expId, train_params: tp });
    pushLivePoint(0, reply.value);
    let action = reply.action;
    S.value = reply.value;

    for (let ep = 0; ep < tp.n_episodes && !S.tdAbort; ep++) {
      lastEpisode = ep;
      if (ep > 0) {                       // ask the backend to reset to the start state
        reply = await wsTrain(ws, { type: "episode" });
        action = reply.action;
        S.value = reply.value;
        pushLivePoint(ep, S.value);
      }
      let state = tp.initial_state;
      let reachedGoal = false;               // episode outcome for the rolling success rate

      // live view: replay this training episode on the board itself
      S.state = state;
      S.steps = 0;
      S.episodeReward = 0;
      S.finished = false;
      updateStats();

      for (let step = 0; step < tp.max_steps && !S.tdAbort; step++) {
        const outcome = sampleOutcome(S.env.P[state][action]);
        reply = await wsTrain(ws, {
          type: "step",
          state,
          action,
          reward: outcome.reward,
          next_state: outcome.next_state,
          done: !!outcome.done,
        });

        // animate the play view exactly like a manual/AI step
        showLastMove(action, state, outcome.next_state);
        state = outcome.next_state;
        S.state = state;
        S.steps = step + 1;
        S.episodeReward += outcome.reward;
        if (outcome.done) S.finished = true;   // goal/hole → next episode
        S.value = reply.value;
        action = reply.action;                 // next action from the backend
        renderGrid();
        updateStats();
        updateLiveStats(reply);
        if (outcome.done) { reachedGoal = outcome.next_state === S.env.goal; break; }
      }

      // episode boundary: feed the rolling success-rate window
      S.tdEpisodeResults.push(reachedGoal ? 1 : 0);
      updateLiveSuccess(ep + 1);

      const doneEp = ep + 1;
      if (doneEp === tp.n_episodes || doneEp % 10 === 0) {
        renderGrid();                     // refresh the live V= labels on the cells
        setStatus(`${tdAgentLabel()} training — episode ${doneEp} / ${tp.n_episodes}.`, "ok");
        await new Promise((r) => setTimeout(r, 0));  // let the UI repaint
      }
    }

    pushLivePoint(lastEpisode + 1, S.value);   // final value snapshot

    // park the marker back at the start; the final board shows the learned policy
    S.state = S.env.start;
    S.steps = 0;
    S.episodeReward = 0;
    S.finished = false;

    if (typeof Chart === "undefined" && S.value) {
      els.chartFallback.innerHTML = S.value
        .map((v, s) => `V(${s}) = ${Number(v).toFixed(4)}`)
        .join("<br>");
      els.chartFallback.classList.remove("hidden");
    }

    // Only a fully completed run is marked trained on the backend; an
    // interrupted one stays "not trained" everywhere (sessions, model card).
    const completed = !S.tdAbort;
    if (completed) {
      await wsTrain(ws, { type: "finish" });
    }
    S.trained = completed;

    const policy = [];
    if (completed) {
      for (let s = 0; s < S.env.n_states; s++) {
        const p = await api("/predict", { experiment_id: S.expId, state: s });
        policy[s] = p.action;
      }
    }
    S.policy = completed ? policy : null;
    renderGrid();
    await refreshSessions(S.expId);
    setSessionStatus({ experiment_name: "this session", trained: completed });
    renderCurrentModelCard();
    const range = `V ranges ${Math.min(...S.value).toFixed(3)} .. ${Math.max(...S.value).toFixed(3)}`;
    const win = S.tdEpisodeResults.slice(-TD_SUCCESS_WINDOW);
    const success = win.length
      ? ` Success rate (last ${win.length}): ${Math.round((100 * win.reduce((a, b) => a + b, 0)) / win.length)}%.`
      : "";
    setStatus(
      completed
        ? `${tdAgentLabel()} trained for ${tp.n_episodes} episodes. ${range}.${success}`
        : `${tdAgentLabel()} stopped at episode ${lastEpisode + 1} / ${tp.n_episodes} — partially trained, not marked as trained. ${range}.${success}`,
      "ok"
    );
  } catch (err) {
    setError(err.message);
    setStatus(`${tdAgentLabel()} training failed.`, "bad");
  } finally {
    if (ws) { try { ws.close(); } catch (_) { /* noop */ } }
    S.tdRunning = false;
    S.tdAbort = false;
    S.busy = false;
    els.tdLive.classList.add("hidden");
    els.trainBtn.disabled = !S.env;
    els.trainBtn.textContent = isTd() ? `Train ${tdAgentLabel()} (live)` : "Train & plot V(s)";
    updateBulkButton();
    updateLoadButton();
    updateAIButton();
    setBusyGame(false);
  }
}

// ---------- Grid rendering ------------------------------------------

function markerFor(s) {
  if (s === S.env.goal) return "G";
  if (S.env.holes.includes(s)) return "H";
  if (s === S.env.start) return "S";
  return "F";
}

function arrowFor(s) {
  if (!S.policy) return "";
  if (s === S.env.goal || S.env.holes.includes(s)) return "";
  const a = S.policy[s];
  return `${ACTION_ARROWS[a]} ${ACTION_NAMES[a]}`;
}

function renderGrid() {
  els.grid.innerHTML = "";
  if (!S.env) return;
  const n = S.env.n;
  els.grid.style.gridTemplateColumns = `repeat(${n}, 64px)`;
  for (let s = 0; s < S.env.n_states; s++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    if (s === S.env.goal) cell.classList.add("goal");
    if (S.env.holes.includes(s)) cell.classList.add("hole");
    if (s === S.state) cell.classList.add("active");

    const idx = document.createElement("div");
    idx.className = "sid";
    idx.textContent = "s" + s;

    const mid = document.createElement("div");
    mid.className = "mid";
    mid.textContent = s === S.state ? "A" : markerFor(s);

    const prow = document.createElement("div");
    prow.className = "prow";
    prow.textContent = arrowFor(s);

    const vval = document.createElement("div");
    vval.className = "vvalue";
    vval.textContent = S.value ? "V=" + Number(S.value[s]).toFixed(3) : "";

    cell.append(idx, mid, prow, vval);
    els.grid.appendChild(cell);
  }
}

// ---------- Stats / play --------------------------------------------

function updateStats() {
  els.statState.textContent = S.state === null ? "—" : String(S.state);
  els.statSteps.textContent = String(S.steps);
  els.statReward.textContent = S.episodeReward.toFixed(2);
  els.statResult.textContent = S.finished
    ? (S.state === S.env.goal ? "GOAL" : "HOLE")
    : "—";
}

function showLastMove(actionCode, from, to) {
  if (!S.env || from === null || to === null) return;
  const n = S.env.n;
  const dr = Math.floor(to / n) - Math.floor(from / n);
  const dc = (to % n) - (from % n);
  const actual = dr === -1 ? "UP" : dr === 1 ? "DOWN" : dc === -1 ? "LEFT" : dc === 1 ? "RIGHT" : null;
  const intent = ACTION_NAMES[actionCode];
  let text, cls;
  if (actual === null) {
    text = `Pressed ${intent} → hit the edge (stayed at s${to})`;
    cls = "bounce";
  } else if (actual === intent) {
    text = `Pressed ${intent} → moved ${actual} as intended (no slip)`;
    cls = "ok";
  } else {
    text = `Pressed ${intent} → SLIPPED → moved ${actual}`;
    cls = "slip";
  }
  els.lastMove.textContent = text;
  els.lastMove.className = "lastmove " + cls;
}

function takeAction(actionCode) {
  if (!S.env || S.finished) return;
  const from = S.state;
  const outcome = sampleOutcome(S.env.P[from][actionCode]);
  showLastMove(actionCode, from, outcome.next_state);
  S.state = outcome.next_state;
  S.steps += 1;
  S.episodeReward += outcome.reward;
  if (outcome.done) {
    S.finished = true;
    stopAI();
  }
  renderGrid();
  updateStats();
}

function resetEpisode() {
  if (!S.env) return;
  stopAI();
  S.state = S.env.start;
  S.steps = 0;
  S.episodeReward = 0;
  S.finished = false;
  els.lastMove.textContent = "";
  els.lastMove.className = "lastmove";
  renderGrid();
  updateStats();
  setError("");
}

// ---------- AI predict mode -----------------------------------------

function startAI() {
  stopAI();
  if (S.tdRunning || S.training) { setError("Stop training first."); return; }
  if (!S.expId) { setError("Initialize the experiment first."); return; }
  if (S.finished) resetEpisode();
  const ms = parseInt(els.aiSpeed.value, 10);
  setStatus(`AI running — asks the backend for a prediction every ${ms} ms.`, "ok");
  setError("");
  S.aiTimer = setInterval(async () => {
    if (S.busy || S.finished || !S.expId) return;
    try {
      const pred = await api("/predict", { experiment_id: S.expId, state: S.state });
      await takeAction(pred.action);
    } catch (err) {
      setError(err.message);
      stopAI();
    }
  }, ms);
}

function stopAI() {
  if (S.aiTimer) {
    clearInterval(S.aiTimer);
    S.aiTimer = null;
  }
  els.aiRunBtn.textContent = "Start AI";
}

// ---------- Bulk prediction ------------------------------------------

function resetBulkStats() {
  els.bulkSuccess.textContent = "—";
  els.bulkTotal.textContent = "—";
  els.bulkRate.textContent = "—";
  els.bulkSteps.textContent = "—";
  els.bulkProgress.textContent = "";
}

function openPredictSocket() {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsBase() + "/ws/predict");
    ws.onopen = () => resolve(ws);
    ws.onerror = () => reject(new Error("WebSocket prediction channel could not be opened."));
  });
}

function wsPredict(ws, experimentId, state) {
  return new Promise((resolve, reject) => {
    const id = ++S.wsMsgId;
    const timer = setTimeout(() => {
      ws.removeEventListener("message", onMsg);
      reject(new Error("Prediction timed out."));
    }, 15000);
    function onMsg(e) {
      let msg;
      try { msg = JSON.parse(e.data); } catch (_) { return; }
      if (msg.id === undefined || Number(msg.id) !== id) return;
      clearTimeout(timer);
      ws.removeEventListener("message", onMsg);
      if (msg.error) reject(new Error(msg.error));
      else resolve(msg.action);
    }
    ws.addEventListener("message", onMsg);
    ws.send(JSON.stringify({ id, experiment_id: experimentId, state }));
  });
}

async function runBulkPrediction() {
  if (S.bulkRunning) return;
  if (S.tdRunning) { setError(`Stop ${tdAgentLabel()} training first.`); return; }
  if (!S.expId || !S.env) { setError("Select a session first."); return; }
  stopAI();
  setError("");
  const total = clampIntField(els.bulkCount, 1, 10000, 100);
  const expId = S.expId;
  const env = S.env;
  S.bulkRunning = true;
  els.bulkRunBtn.disabled = true;
  els.bulkRunBtn.textContent = "Running...";
  setStatus("Bulk predicting — one persistent WebSocket, many lookups.", "ok");
  els.bulkTotal.textContent = String(total);
  els.bulkProgress.textContent = `Running 0 / ${total} episodes…`;

  let ws = null;
  let successes = 0;
  let totalSteps = 0;
  const maxSteps = Math.max(200, env.n_states * 50);
  try {
    ws = await openPredictSocket();
    for (let ep = 1; ep <= total; ep++) {
      let state = env.start;
      let finished = false;
      let reachedGoal = false;
      let steps = 0;
      while (!finished && steps < maxSteps) {
        const action = await wsPredict(ws, expId, state);
        const outcome = sampleOutcome(env.P[state][action]);
        state = outcome.next_state;
        steps += 1;
        if (outcome.done) {
          reachedGoal = state === env.goal;
          finished = true;
        }
      }
      if (reachedGoal) successes += 1;
      totalSteps += steps;
      if (ep === total || ep % 10 === 0) {
        els.bulkProgress.textContent = `Running ${ep} / ${total} episodes…`;
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    const rate = ((100 * successes) / total).toFixed(1);
    els.bulkSuccess.textContent = String(successes);
    els.bulkRate.textContent = rate + "%";
    els.bulkSteps.textContent = (totalSteps / total).toFixed(1);
    els.bulkProgress.textContent = `${successes} / ${total} episodes reached the goal (${rate}%)`;
    setStatus(`Bulk prediction: ${successes}/${total} successes (${rate}%).`, "ok");
  } catch (err) {
    setError(err.message);
    setStatus("Bulk prediction failed.", "bad");
  } finally {
    if (ws) { try { ws.close(); } catch (_) { /* noop */ } }
    S.bulkRunning = false;
    els.bulkRunBtn.textContent = "Run bulk prediction";
    updateBulkButton();
  }
}

// ---------- Mode / control wiring -----------------------------------

function setMode(mode) {
  S.mode = mode;
  stopAI();
  const ai = mode === "ai";
  els.aiControls.classList.toggle("hidden", !ai);
  els.manualControls.classList.toggle("hidden", ai);
  updateAIButton();
  els.aiRunBtn.textContent = "Start AI";
  if (!ai) setBusyGame(false);
}

els.initBtn.addEventListener("click", initExperiment);
els.trainBtn.addEventListener("click", trainAgent);
els.agentType.addEventListener("change", () => {
  setAgentUI();
  // The agent type is chosen at registration; re-register so the backend
  // instantiates the newly selected agent for the current environment.
  if (S.env && !S.tdRunning && !S.bulkRunning) initExperiment();
});
els.upBtn.addEventListener("click", () => takeAction(3));
els.downBtn.addEventListener("click", () => takeAction(1));
els.leftBtn.addEventListener("click", () => takeAction(0));
els.rightBtn.addEventListener("click", () => takeAction(2));
els.resetEpisodeBtn.addEventListener("click", resetEpisode);

els.sessionSelect.addEventListener("change", () => updateLoadButton());
els.refreshSessionsBtn.addEventListener("click", () => refreshSessions(S.expId));
els.loadModelBtn.addEventListener("click", () => selectSession(els.sessionSelect.value));
els.bulkRunBtn.addEventListener("click", runBulkPrediction);

els.aiSpeed.addEventListener("input", () => {
  els.aiSpeedLabel.textContent = els.aiSpeed.value;
  if (S.aiTimer) startAI();
});

els.aiRunBtn.addEventListener("click", () => {
  if (S.aiTimer) {
    stopAI();
    setStatus("AI paused.", "ok");
  } else {
    startAI();
    els.aiRunBtn.textContent = "Stop AI";
  }
});

document.querySelectorAll('input[name="mode"]').forEach((radio) => {
  radio.addEventListener("change", () => setMode(radio.value));
});

// Keyboard: arrows / WASD play the grid in Manual mode.
document.addEventListener("keydown", (e) => {
  if (S.mode !== "manual" || S.busy) return;
  if (!S.env || S.finished) return;
  const map = {
    ArrowLeft: 0, ArrowDown: 1, ArrowRight: 2, ArrowUp: 3,
    a: 0, s: 1, d: 2, w: 3,
  };
  const code = map[e.key];
  if (code !== undefined) {
    e.preventDefault();
    takeAction(code);
  }
});

// ---------- Init UI --------------------------------------------------

setAgentUI();
setBusyGame(true);
renderGrid();
updateStats();
refreshSessions();
loadAgentTypes();
