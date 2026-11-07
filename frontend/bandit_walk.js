"use strict";

/* ============================================================
 *  BanditSlipperyWalk front-end (vanilla JS, no frameworks)
 *  Talks to the FastAPI backend (same origin, served at "/").
 * ============================================================ */

// ---------- API ----------------------------------------------------

// When opened via file:// we must call the backend at its fixed URL;
// when served by the backend itself we can use relative paths.
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

// WebSocket base, mirroring apiBase(): same-host when served by the backend,
// a fixed local URL when opened via file://.
function wsBase() {
  if (location.hostname === "") return "ws://127.0.0.1:8123";
  return (location.protocol === "https:" ? "wss://" : "ws://") + location.host;
}

// ---------- DOM -----------------------------------------------------

const $ = (id) => document.getElementById(id);

const els = {
  nStates: $("nStates"),
  startPos: $("startPos"),
  holePos: $("holePos"),
  goalPos: $("goalPos"),
  slipProb: $("slipProb"),
  seed: $("seed"),
  goalReward: $("goalReward"),
  holeReward: $("holeReward"),
  gamma: $("gamma"),
  tol: $("tol"),
  initPolicy: $("initPolicy"),
  agentType: $("agentType"),
  piParams: $("piParams"),
  sarsaParams: $("sarsaParams"),
  sarsaGamma: $("sarsaGamma"),
  sarsaEpisodes: $("sarsaEpisodes"),
  sarsaMaxSteps: $("sarsaMaxSteps"),
  sarsaInitPolicy: $("sarsaInitPolicy"),
  initAlpha: $("initAlpha"),
  minAlpha: $("minAlpha"),
  alphaDecay: $("alphaDecay"),
  epsilonDecay: $("epsilonDecay"),
  initEpsilon: $("initEpsilon"),
  minEpsilon: $("minEpsilon"),
  sarsaLive: $("sarsaLive"),
  liveEpisode: $("liveEpisode"),
  liveStep: $("liveStep"),
  liveAlpha: $("liveAlpha"),
  liveEpsilon: $("liveEpsilon"),
  chartTitle: $("chartTitle"),
  initBtn: $("initBtn"),
  trainBtn: $("trainBtn"),
  statusLine: $("statusLine"),
  errorLine: $("errorLine"),
  corridor: $("corridor"),
  valueChart: $("valueChart"),
  chartFallback: $("chartFallback"),
  manualControls: $("manualControls"),
  aiControls: $("aiControls"),
  aiSpeed: $("aiSpeed"),
  aiSpeedLabel: $("aiSpeedLabel"),
  aiRunBtn: $("aiRunBtn"),
  leftBtn: $("leftBtn"),
  rightBtn: $("rightBtn"),
  resetEpisodeBtn: $("resetEpisodeBtn"),
  statState: $("statState"),
  statSteps: $("statSteps"),
  statReward: $("statReward"),
  statResult: $("statResult"),
  sessionSelect: $("sessionSelect"),
  refreshSessionsBtn: $("refreshSessionsBtn"),
  loadModelBtn: $("loadModelBtn"),
  sessionStatus: $("sessionStatus"),
  lastMove: $("lastMove"),
  modelCard: $("modelCard"),
  bulkCount: $("bulkCount"),
  bulkRunBtn: $("bulkRunBtn"),
  bulkSuccess: $("bulkSuccess"),
  bulkTotal: $("bulkTotal"),
  bulkRate: $("bulkRate"),
  bulkSteps: $("bulkSteps"),
  bulkProgress: $("bulkProgress"),
};

els.slipProb.addEventListener("blur", () => clampField(els.slipProb, 0, 1, 0.2));
els.nStates.addEventListener("change", clampEnvFromLength);

// ---------- App state ----------------------------------------------

const S = {
  expId: null,
  envId: null,         // env_id of the current environment (for model filtering)
  env: null,           // env descriptor returned by /register_experiment
  models: [],          // all models from the backend experiment registry
  state: null,         // current walk position
  steps: 0,
  episodeReward: 0,
  finished: false,     // episode ended (goal or hole)
  busy: false,         // guard against overlapping requests
  training: false,
  value: null,         // V(s) array returned by /train
  policy: null,        // learned action per state (for the corridor arrows)
  trained: false,
  bulkRunning: false,
  sarsaRunning: false,   // live SARSA training loop active
  sarsaAbort: false,     // set by the Stop button to break the training loop
  chartMode: null,       // "bar" (policy iteration) | "line" (SARSA live)
  wsMsgId: 0,
  mode: "manual",
  aiTimer: null,
};

// ---------- Chart ---------------------------------------------------

let chart = null;

function destroyChart() {
  if (chart) { chart.destroy(); chart = null; }
  S.chartMode = null;
}

function plotValue(V) {
  if (typeof Chart === "undefined") {
    const lines = V
      .map((v, s) => `V(${s}) = ${Number(v).toFixed(4)}`)
      .join("<br>");
    els.chartFallback.innerHTML = lines;
    els.chartFallback.classList.remove("hidden");
    return;
  }
  if (S.chartMode !== "bar") destroyChart();   // leave SARSA live mode
  const labels = V.map((_, s) => String(s));
  if (!chart) {
    chart = new Chart(els.valueChart, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "V(s) — State value",
          data: V,
          backgroundColor: V.map((v, s) =>
            s === S.env.goal ? "rgba(251,191,36,0.85)"
            : s === S.env.hole ? "rgba(148,163,184,0.85)"
            : "rgba(56,189,248,0.8)"
          ),
          borderRadius: 6,
        }],
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
    chart.update();
  }
}

// Live SARSA chart: one line per state, x-axis = episode, updated every
// episode boundary while the training loop runs.
function stateLineColor(s) {
  if (s === S.env.goal) return "rgb(251,191,36)";
  if (s === S.env.hole) return "rgb(148,163,184)";
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
      borderWidth: (s === S.env.goal || s === S.env.hole) ? 2.5 : 1.5,
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

function clampEnvFromLength() {
  const n = clampIntField(els.nStates, 3, 20, 6);
  const keep = (el, max) => {
    const v = parseInt(el.value, 10);
    if (Number.isNaN(v) || v > max || v < 0) el.value = Math.floor(max / 2);
  };
  keep(els.startPos, n - 1);
  keep(els.holePos, n - 1);
  keep(els.goalPos, n - 1);
  const max = n - 1;
  if (parseInt(els.holePos.value, 10) > max) els.holePos.value = 0;
  if (parseInt(els.goalPos.value, 10) > max) els.goalPos.value = max;
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
  const locked = busy || S.sarsaRunning;
  els.leftBtn.disabled = locked || !S.env || S.finished || S.mode !== "manual";
  els.rightBtn.disabled = els.leftBtn.disabled;
  els.resetEpisodeBtn.disabled = !S.env || S.busy;
}

function refreshSessionUI() {
  const active = !!S.env;
  els.trainBtn.disabled = !active;
  els.aiRunBtn.disabled = !active || S.mode !== "ai";
  els.refreshSessionsBtn.disabled = false;
  updateBulkButton();
  updateLoadButton();
  setBusyGame(false);
}

// ---------- Training agent selection --------------------------------

function isSarsa() {
  return els.agentType.value === "Sarsa";
}

// Show the hyper-parameters of the selected agent, retitle the chart, and
// relabel the train button ("Train ..." normally, "Stop training" live).
function setAgentUI() {
  const sarsa = isSarsa();
  els.piParams.classList.toggle("hidden", sarsa);
  els.sarsaParams.classList.toggle("hidden", !sarsa);
  els.sarsaLive.classList.toggle("hidden", !(sarsa && S.sarsaRunning));
  els.chartTitle.textContent = sarsa
    ? "Value function — live V(s) per episode (SARSA)"
    : "Value function (trained)";
  if (!S.sarsaRunning) {
    els.trainBtn.textContent = sarsa ? "Train SARSA (live)" : "Train & plot V(s)";
  }
}

function updateBulkButton() {
  els.bulkRunBtn.disabled = !S.env || S.bulkRunning;
}

function updateLoadButton() {
  els.loadModelBtn.disabled = !els.sessionSelect.value;
}

function clearChart() {
  destroyChart();
  els.chartFallback.classList.add("hidden");
}

function setSessionStatus(data) {
  if (!data) return;
  els.sessionStatus.textContent = data.trained
    ? `"${data.experiment_name}" — trained, ready for AI prediction`
    : `"${data.experiment_name}" — not trained yet`;
  els.sessionStatus.className = data.trained ? "ok" : "";
}

// ---------- Model card + env loading --------------------------------

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

// Fetch a model's environment from the env registry by env_id, falling back to
// the config stored on the model (agent_params.env_config) for older entries.
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
  const rows = [
    ["Model", detail.experiment_name],
    ["Experiment", detail.experiment_id],
    ["Environment", `${detail.env_type} · ${fmtNum(envCfg && envCfg.n_states)} states`],
    ["env_id", detail.env_id || "—"],
    ["Layout", envCfg
      ? `start ${envCfg.start} · hole ${envCfg.hole} · goal ${envCfg.goal}`
      : "—"],
    ["Slip prob", fmtNum(envCfg && envCfg.slip_prob)],
    ["Rewards", envCfg ? `goal ${fmtNum(envCfg.goal_reward)} · hole ${fmtNum(envCfg.hole_reward)}` : "—"],
    ["Agent", detail.agent_type],
    ["Trained", detail.trained ? "yes" : "no"],
    ["Training", fmtTrainingParams(detail.training_params)],
    ["Episodes trained", fmtNum(detail.episodes_trained)],
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

// Render the model card for whatever experiment is currently active (no side
// effects on the corridor/stats).
async function renderCurrentModelCard() {
  if (!S.expId) return;
  try {
    const { detail, envCfg } = await fetchExperimentDetail(S.expId);
    renderModelCard(detail, envCfg);
  } catch (_) { /* keep prior card */ }
}

// Reflect a loaded environment's parameters in the "Environment setup" inputs
// so the UI always matches the environment currently in use.
function applyEnvToConfigInputs(envCfg) {
  if (!envCfg) return;
  const set = (id, val) => {
    if (val !== undefined && val !== null) $(id).value = val;
  };
  set("nStates", envCfg.n_states);
  set("startPos", envCfg.start);
  set("holePos", envCfg.hole);
  set("goalPos", envCfg.goal);
  set("goalReward", envCfg.goal_reward);
  set("holeReward", envCfg.hole_reward);
  set("slipProb", envCfg.slip_prob);
  set("seed", envCfg.seed);
}

// ---------- Session library (experiment registry) -------------------

// List every model stored in the backend experiment registry. These persist in
// memory across page refreshes; picking one loads the model's own environment.
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
  if (S.sarsaRunning) { setError("Stop SARSA training first."); return; }
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
    renderCorridor();
    updateStats();
    setSessionStatus(detail);
    renderModelCard(detail, envCfg);
    setStatus(
      `Model loaded: ${detail.experiment_name} (${S.expId.slice(0, 8)}).`,
      detail.trained ? "ok" : ""
    );
    refreshSessionUI();
  } catch (err) {
    setError(err.message);
    setStatus("Failed to load the selected model.", "bad");
  }
}

// ---------- Experiment lifecycle ------------------------------------

function wallBounce(s, nStates) {
  if (s < 0) return 0;
  if (s >= nStates) return nStates - 1;
  return s;
}

function buildTransitions(cfg) {
  // Mirrors BanditSlipperyWalk P[s][a] = [{prob, next_state, reward, done} ...].
  const {
    n_states, hole, goal,
    goal_reward, hole_reward, step_reward, slip_prob,
  } = cfg;
  const rewardFor = (s) =>
    s === goal ? [goal_reward, true]
    : s === hole ? [hole_reward, true]
    : [step_reward, false];

  const P = {};
  for (let s = 0; s < n_states; s++) {
    P[s] = {};
    for (const [code, direction] of [[0, -1], [1, 1]]) { // LEFT, RIGHT
      if (s === hole || s === goal) {
        P[s][code] = [{ prob: 1, next_state: s, reward: step_reward, done: true }];
        continue;
      }
      const intended = wallBounce(s + direction, n_states);
      const slipped = wallBounce(s - direction, n_states);
      const branches = new Map();
      for (const [prob, target] of [[1 - slip_prob, intended], [slip_prob, slipped]]) {
        if (prob <= 0) continue;
        const [reward, done] = rewardFor(target);
        const key = `${target}|${reward}|${done}`;
        const existing = branches.get(key);
        branches.set(key, existing
          ? { ...existing, prob: existing.prob + prob }
          : { prob, next_state: target, reward, done });
      }
      P[s][code] = Array.from(branches.values());
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
  const cfg = {
    name: "BanditSlipperyWalk",
    env_type: "SlipperyBanditWalk",
    n_states: parseInt(els.nStates.value, 10),
    n_actions: 2,
    start: parseInt(els.startPos.value, 10),
    hole: parseInt(els.holePos.value, 10),
    goal: parseInt(els.goalPos.value, 10),
    goal_reward: parseFloat(els.goalReward.value),
    hole_reward: parseFloat(els.holeReward.value),
    step_reward: 0,
    slip_prob: clampField(els.slipProb, 0, 1, 0.2),
    seed: parseInt(els.seed.value, 10),
  };
  cfg.actions = { LEFT: 0, RIGHT: 1 };
  cfg.skip_states = [cfg.hole, cfg.goal];
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

function readSarsaTrainParams() {
  return {
    initial_state: S.env.start,
    gamma: parseFloat(els.sarsaGamma.value),
    n_episodes: clampIntField(els.sarsaEpisodes, 1, 100000, 1000),
    max_steps: clampIntField(els.sarsaMaxSteps, 1, 5000, 100),
    init_alpha: parseFloat(els.initAlpha.value),
    min_alpha: parseFloat(els.minAlpha.value),
    alpha_decay_ratio: clampField(els.alphaDecay, 0, 1, 0.5),
    init_epsilon: clampField(els.initEpsilon, 0, 1, 1),
    min_epsilon: clampField(els.minEpsilon, 0, 1, 0.1),
    epsilon_decay_ratio: clampField(els.epsilonDecay, 0, 1, 0.9),
  };
}

async function initExperiment() {
  if (S.bulkRunning) { setError("Wait for bulk prediction to finish first."); return; }
  if (S.sarsaRunning) { setError("Stop SARSA training first."); return; }
  stopAI();
  setError("");
  els.initBtn.disabled = true;
  els.initBtn.textContent = "Initializing...";
  try {
    clampEnvFromLength();
    const cfg = readEnvConfig();
    const sarsa = isSarsa();
    const data = await api("/register_experiment", {
      agent_type: els.agentType.value,
      experiment_name: `${cfg.name.toLowerCase()}_n${cfg.n_states}${sarsa ? "_sarsa" : ""}`,
      env_type: cfg.env_type,
      agent_params: {
        name: cfg.name.toLowerCase(),
        env_type: cfg.env_type,
        n_states: cfg.n_states,
        n_actions: cfg.n_actions,
        env_config: cfg, // round-tripped by the backend so sessions can be restored
        ...(sarsa ? { initial_policy: els.sarsaInitPolicy.value } : {}),
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
    renderCorridor();
    updateStats();
    await refreshSessions(data.experiment_id);
    els.sessionStatus.textContent = "New session registered — train it to enable AI.";
    els.sessionStatus.className = "ok";
    refreshSessionUI();
    renderCurrentModelCard();
    setStatus(
      `Experiment ready (id ${S.expId.slice(0, 8)}). ` +
      `Length ${S.env.n_states}, start ${S.state}, hole ${S.env.hole}, goal ${S.env.goal}.`,
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
  if (isSarsa()) return trainSarsa();
  if (!S.expId) { setError("Initialize the experiment first."); return; }
  setError("");
  els.trainBtn.disabled = true;
  els.trainBtn.textContent = "Training...";
  setStatus("Training agent (policy iteration)...");
  try {
    const data = await api("/train", { experiment_id: S.expId, train_params: readTrainParams() });
    S.value = data.value;

    // Fetch the greedy policy once so the corridor can draw the arrows.
    const policy = [];
    for (let s = 0; s < S.env.n_states; s++) {
      const p = await api("/predict", { experiment_id: S.expId, state: s });
      policy[s] = p.action;
    }
    S.policy = policy;
    S.trained = true;
    plotValue(S.value);
    renderCorridor();
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
    els.trainBtn.disabled = false;
    els.trainBtn.textContent = "Train & plot V(s)";
  }
}

// ---------- SARSA live training (WebSocket /ws/train_step) ----------
//
// The front end owns the rollouts: for every episode it starts an episode on
// the server, steps the local environment with the action the server returns,
// and sends (state, action, reward, next_state, done) back. Each reply gives
// the next action plus the live value function, plotted per episode as soon
// as it arrives. The frontend also enforces max_steps and the episode count.

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

async function trainSarsa() {
  if (!S.expId) { setError("Initialize the experiment first."); return; }
  if (S.bulkRunning) { setError("Wait for bulk prediction to finish first."); return; }
  if (S.sarsaRunning) {           // second click while running = stop
    S.sarsaAbort = true;
    els.trainBtn.disabled = true;
    els.trainBtn.textContent = "Stopping...";
    return;
  }

  stopAI();
  setError("");
  const tp = readSarsaTrainParams();
  S.sarsaRunning = true;
  S.sarsaAbort = false;
  S.busy = true;
  setBusyGame(true);
  els.trainBtn.disabled = false;
  els.trainBtn.textContent = "Stop training";
  els.sarsaLive.classList.remove("hidden");
  initLiveChart();

  let ws = null;
  let lastEpisode = 0;
  try {
    ws = await openTrainSocket();
    let reply = await wsTrain(ws, { type: "start", experiment_id: S.expId, train_params: tp });
    pushLivePoint(0, reply.value);
    let action = reply.action;
    S.value = reply.value;

    for (let ep = 0; ep < tp.n_episodes && !S.sarsaAbort; ep++) {
      lastEpisode = ep;
      if (ep > 0) {                       // ask the backend to reset to the start state
        reply = await wsTrain(ws, { type: "episode" });
        action = reply.action;
        S.value = reply.value;
        pushLivePoint(ep, S.value);
      }
      let state = tp.initial_state;
      for (let step = 0; step < tp.max_steps && !S.sarsaAbort; step++) {
        const outcome = sampleOutcome(S.env.P[state][action]);
        reply = await wsTrain(ws, {
          type: "step",
          state,
          action,
          reward: outcome.reward,
          next_state: outcome.next_state,
          done: !!outcome.done,
        });
        action = reply.action;
        state = outcome.next_state;
        S.value = reply.value;
        updateLiveStats(reply);
        if (outcome.done) break;          // reached goal/hole → next episode
      }

      const doneEp = ep + 1;
      if (doneEp === tp.n_episodes || doneEp % 10 === 0) {
        renderCorridor();                 // refresh the live V= labels on the cells
        setStatus(`SARSA training — episode ${doneEp} / ${tp.n_episodes}.`, "ok");
        await new Promise((r) => setTimeout(r, 0));  // let the UI repaint
      }
    }

    pushLivePoint(lastEpisode + 1, S.value);   // final value snapshot

    if (typeof Chart === "undefined" && S.value) {
      els.chartFallback.innerHTML = S.value
        .map((v, s) => `V(${s}) = ${Number(v).toFixed(4)}`)
        .join("<br>");
      els.chartFallback.classList.remove("hidden");
    }

    // Only a fully completed run is marked trained on the backend; an
    // interrupted one stays "not trained" everywhere (sessions, model card).
    const completed = !S.sarsaAbort;
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
    renderCorridor();
    await refreshSessions(S.expId);
    setSessionStatus({ experiment_name: "this session", trained: completed });
    renderCurrentModelCard();
    const range = `V ranges ${Math.min(...S.value).toFixed(3)} .. ${Math.max(...S.value).toFixed(3)}`;
    setStatus(
      completed
        ? `SARSA trained for ${tp.n_episodes} episodes. ${range}.`
        : `SARSA stopped at episode ${lastEpisode + 1} / ${tp.n_episodes} — partially trained, not marked as trained. ${range}.`,
      "ok"
    );
  } catch (err) {
    setError(err.message);
    setStatus("SARSA training failed.", "bad");
  } finally {
    if (ws) { try { ws.close(); } catch (_) { /* noop */ } }
    S.sarsaRunning = false;
    S.sarsaAbort = false;
    S.busy = false;
    els.sarsaLive.classList.add("hidden");
    els.trainBtn.disabled = !S.env;
    els.trainBtn.textContent = isSarsa() ? "Train SARSA (live)" : "Train & plot V(s)";
    setBusyGame(false);
  }
}

// ---------- Corridor rendering --------------------------------------

function markerFor(s) {
  if (s === S.env.goal) return "G";
  if (s === S.env.hole) return "H";
  return "·";
}

function arrowFor(s) {
  if (!S.policy || s === S.env.goal || s === S.env.hole) return "";
  const names = S.env.actions; // {"LEFT": 0, "RIGHT": 1}
  if (S.policy[s] === names.RIGHT) return "→ RIGHT";
  return "← LEFT";
}

function renderCorridor() {
  els.corridor.innerHTML = "";
  for (let s = 0; s < S.env.n_states; s++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    if (s === S.env.goal) cell.classList.add("goal");
    if (s === S.env.hole) cell.classList.add("hole");
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
    els.corridor.appendChild(cell);
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
  const isRight = S.env.actions.RIGHT === actionCode;
  const intent = isRight ? +1 : -1;
  const actual = to - from;
  const dirName = (d) => (d > 0 ? "RIGHT" : "LEFT");
  let text, cls;
  if (actual === 0) {
    text = `Pressed ${isRight ? "RIGHT" : "LEFT"} → bounced off the wall (stayed at s${to})`;
    cls = "bounce";
  } else if (Math.sign(actual) === intent) {
    text = `Pressed ${dirName(intent)} → moved ${dirName(actual)} as intended (no slip)`;
    cls = "ok";
  } else {
    text = `Pressed ${dirName(intent)} → SLIPPED → moved ${dirName(actual)}`;
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
  renderCorridor();
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
  renderCorridor();
  updateStats();
  setError("");
}

// ---------- AI predict mode -----------------------------------------

function startAI() {
  stopAI();
  if (S.sarsaRunning) { setError("Stop SARSA training first."); return; }
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
    ws.onerror = () =>
      reject(new Error("WebSocket prediction channel could not be opened."));
  });
}

// Request a single greedy action over the persistent socket; each message is
// tagged with an id so concurrent/in-flight requests can be matched up.
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
  if (S.sarsaRunning) { setError("Stop SARSA training first."); return; }
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
  const maxSteps = Math.max(200, env.n_states * 50); // safety cap per episode
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
        await new Promise((r) => setTimeout(r, 0)); // let the UI repaint
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
  els.aiRunBtn.disabled = !S.env;
  els.aiRunBtn.textContent = "Start AI";
  if (!ai) setBusyGame(false);
}

els.initBtn.addEventListener("click", initExperiment);
els.trainBtn.addEventListener("click", trainAgent);
els.agentType.addEventListener("change", () => {
  setAgentUI();
  // The agent type is chosen at registration; re-register so the backend
  // instantiates the newly selected agent for the current environment.
  if (S.env && !S.sarsaRunning && !S.bulkRunning) initExperiment();
});
els.leftBtn.addEventListener("click", () => takeAction(0));
els.rightBtn.addEventListener("click", () => takeAction(1));
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

// Keyboard support: arrows play the walk in Manual mode.
document.addEventListener("keydown", (e) => {
  if (S.mode !== "manual" || S.busy) return;
  if (e.key === "ArrowLeft") {
    e.preventDefault();
    if (S.env && !S.finished) takeAction(0);
  } else if (e.key === "ArrowRight") {
    e.preventDefault();
    if (S.env && !S.finished) takeAction(1);
  }
});

// ---------- Init UI --------------------------------------------------

setAgentUI();
setBusyGame(true);
renderCorridor();
updateStats();
refreshSessions();
