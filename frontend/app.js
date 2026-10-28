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
  sessionStatus: $("sessionStatus"),
  lastMove: $("lastMove"),
};

els.slipProb.addEventListener("blur", () => clampField(els.slipProb, 0, 1, 0.2));
els.nStates.addEventListener("change", clampEnvFromLength);

// ---------- App state ----------------------------------------------

const S = {
  expId: null,
  env: null,           // env descriptor returned by /register_experiment
  state: null,         // current walk position
  steps: 0,
  episodeReward: 0,
  finished: false,     // episode ended (goal or hole)
  busy: false,         // guard against overlapping requests
  training: false,
  value: null,         // V(s) array returned by /train
  policy: null,        // learned action per state (for the corridor arrows)
  mode: "manual",
  aiTimer: null,
};

// ---------- Chart ---------------------------------------------------

let chart = null;

function plotValue(V) {
  if (typeof Chart === "undefined") {
    const lines = V
      .map((v, s) => `V(${s}) = ${Number(v).toFixed(4)}`)
      .join("<br>");
    els.chartFallback.innerHTML = lines;
    els.chartFallback.classList.remove("hidden");
    return;
  }
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
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = V;
    chart.update();
  }
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
  els.leftBtn.disabled = busy || !S.env || S.finished || S.mode !== "manual";
  els.rightBtn.disabled = els.leftBtn.disabled;
  els.resetEpisodeBtn.disabled = !S.env || S.busy;
}

function refreshSessionUI() {
  const active = !!S.env;
  els.trainBtn.disabled = !active;
  els.aiRunBtn.disabled = !active || S.mode !== "ai";
  els.refreshSessionsBtn.disabled = false;
  setBusyGame(false);
}

function clearChart() {
  if (chart) {
    chart.data.datasets[0].data = [];
    chart.data.labels = [];
    chart.update();
  }
  els.chartFallback.classList.add("hidden");
}

function setSessionStatus(data) {
  if (!data) return;
  els.sessionStatus.textContent = data.trained
    ? `"${data.experiment_name}" — trained, ready for AI prediction`
    : `"${data.experiment_name}" — not trained yet`;
  els.sessionStatus.className = data.trained ? "ok" : "";
}

// ---------- Session library (experiment registry) -------------------

async function refreshSessions(selectId) {
  try {
    const data = await apiGet("/experiments");
    const list = data.experiments || [];
    els.sessionSelect.innerHTML = "";
    if (list.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "— no sessions yet —";
      els.sessionSelect.appendChild(opt);
      els.sessionStatus.textContent = "No sessions yet";
      els.sessionStatus.className = "";
    } else {
      list.forEach((session) => {
        const opt = document.createElement("option");
        opt.value = session.experiment_id;
        opt.textContent = `${session.experiment_name} · ${session.experiment_id.slice(0, 8)}${session.trained ? " ✓" : ""}`;
        els.sessionSelect.appendChild(opt);
      });
      if (selectId) els.sessionSelect.value = selectId;
      els.sessionStatus.textContent = `${list.length} session(s) in registry`;
      els.sessionStatus.className = "ok";
    }
    els.sessionSelect.disabled = false;
    els.refreshSessionsBtn.disabled = false;
    return data;
  } catch (err) {
    els.sessionStatus.textContent = "Could not load sessions";
    els.sessionStatus.className = "bad";
    els.sessionSelect.disabled = true;
    els.refreshSessionsBtn.disabled = false;
    setError(err.message);
    return null;
  }
}

async function selectSession(experimentId) {
  stopAI();
  setError("");
  if (!experimentId) return;
  try {
    const data = await apiGet(`/experiments/${experimentId}`);
    S.expId = data.experiment_id;
    S.env = data.env;
    S.value = data.value || null;
    S.policy = data.policy || null;

    S.state = S.env.start;
    S.steps = 0;
    S.episodeReward = 0;
    S.finished = false;

    if (S.value) plotValue(S.value); else clearChart();
    renderCorridor();
    updateStats();
    setSessionStatus(data);
    setStatus(
      `Session loaded: ${data.experiment_name} (${S.expId.slice(0, 8)}).`,
      data.trained ? "ok" : ""
    );
    refreshSessionUI();
  } catch (err) {
    setError(err.message);
    setStatus("Failed to load the selected session.", "bad");
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
    gamma: parseFloat(els.gamma.value),
    tolerance: parseFloat(els.tol.value),
    max_iters: 1000,
    eval_max_iters: 10000,
  };
}

async function initExperiment() {
  stopAI();
  setError("");
  els.initBtn.disabled = true;
  els.initBtn.textContent = "Initializing...";
  try {
    clampEnvFromLength();
    const data = await api("/register_experiment", {
      agent_type: "PolicyIteration",
      experiment_name: "bandit_slippery_walk",
      agent_params: {},
      env_config: readEnvConfig(),
    });
    S.expId = data.experiment_id;
    S.env = data.env;
    S.state = data.env.start;
    S.steps = 0;
    S.episodeReward = 0;
    S.finished = false;
    S.value = null;
    S.policy = null;

    clearChart();
    renderCorridor();
    updateStats();
    await refreshSessions(data.experiment_id);
    els.sessionStatus.textContent = "New session registered — train it to enable AI.";
    els.sessionStatus.className = "ok";
    refreshSessionUI();
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
    plotValue(S.value);
    renderCorridor();
    await refreshSessions(S.expId);
    setSessionStatus({ experiment_name: "this session", trained: true });
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
els.leftBtn.addEventListener("click", () => takeAction(0));
els.rightBtn.addEventListener("click", () => takeAction(1));
els.resetEpisodeBtn.addEventListener("click", resetEpisode);

els.sessionSelect.addEventListener("change", (e) => selectSession(e.target.value));
els.refreshSessionsBtn.addEventListener("click", () => refreshSessions(S.expId));

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

setBusyGame(true);
renderCorridor();
updateStats();
refreshSessions();
