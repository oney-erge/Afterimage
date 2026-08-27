import { state } from "./state.js";
import { $, api, badge, esc, progressValue, statusLabel, toast, watchJob } from "./shared.js";

let activeResearchJob = null;
let stopWatching = null;

function readyOptions(selected = "") {
  const models = state.models.filter((model) => model.state === "ready");
  if (!models.length) return '<option value="">No ready models</option>';
  return models.map((model) => `<option value="${esc(model.model_id)}"${model.model_id === selected ? " selected" : ""}>${esc(model.model_id)}</option>`).join("");
}

export function updateResearchModels() {
  const researchValue = $("research-model").value;
  const compareValue = $("compare-model").value;
  $("research-model").innerHTML = readyOptions(researchValue);
  $("compare-model").innerHTML = readyOptions(compareValue);
}

function hypothesisCode(id) { return String(id).split("-")[0].toUpperCase(); }

function measuredSummary(hypothesis) {
  const measured = hypothesis.measured;
  if (!measured) return { title: "No recorded result", detail: "This experiment has no stored project result." };
  const effect = measured.effect_pct == null ? "" : `${measured.effect_pct > 0 ? "+" : ""}${measured.effect_pct}%`;
  return { title: `${statusLabel(measured.verdict)}${effect ? ` · ${effect}` : ""}`, detail: measured.plain_language };
}

function renderExperiments() {
  const node = $("experiment-list");
  node.classList.remove("loading-line");
  const query = $("research-query").value.trim().toLowerCase();
  const hypotheses = state.experiments.filter((item) => !query || `${item.id} ${item.title} ${item.statement}`.toLowerCase().includes(query));
  if (!hypotheses.length) { node.innerHTML = '<div class="empty-inline">No experiments match this search.</div>'; return; }
  node.innerHTML = hypotheses.map((hypothesis) => {
    const measured = measuredSummary(hypothesis);
    const required = hypothesis.required_inputs || [];
    const nonDraft = required.filter((value) => value !== "draft_model_id");
    const canRun = nonDraft.length === 0;
    return `<details class="experiment-row" data-experiment="${esc(hypothesis.id)}"><summary class="experiment-summary"><span class="experiment-code">${esc(hypothesisCode(hypothesis.id))}</span><span class="experiment-title"><strong>${esc(hypothesis.title)}</strong><small>${esc(hypothesis.statement)}</small></span><span class="experiment-result"><strong>${esc(measured.title)}</strong>${esc(measured.detail)}</span><span class="chevron">›</span></summary><div class="experiment-detail"><div><h4>Question</h4><p>${esc(hypothesis.statement)}</p></div><div><h4>Primary measure</h4><p>${esc(hypothesis.primary_metric)} · minimum effect ${esc(hypothesis.minimum_effect)}</p></div><div><h4>Control</h4><p>${esc(hypothesis.control_profile)}</p></div><div><h4>Treatment</h4><p>${esc(hypothesis.candidate_profile)}</p></div><div class="full"><h4>Recorded evidence</h4><p>${esc(hypothesis.measured?.detail || "No result has been recorded in the project evidence registry.")}</p></div>${required.includes("draft_model_id") ? '<label class="full select-label">Draft model ID<input class="experiment-draft" placeholder="Required for this experiment"></label>' : ""}<div class="experiment-actions full"><button class="button primary" data-run-experiment="${esc(hypothesis.id)}" ${canRun ? "" : "disabled"}>Run experiment</button>${nonDraft.length ? `<span class="prerequisites">Requires prepared inputs: ${esc(nonDraft.join(", "))}</span>` : `<span class="prerequisites">${hypothesis.runner === "generation" ? "Uses the selected ready model." : "Runs from the registered protocol."}</span>`}</div></div></details>`;
  }).join("");
  node.querySelectorAll("[data-run-experiment]").forEach((button) => button.addEventListener("click", (event) => {
    event.preventDefault(); event.stopPropagation(); startExperiment(button.dataset.runExperiment, button.closest("details"));
  }));
}

function renderActive(job) {
  const node = $("active-research");
  if (!job) { node.hidden = true; node.innerHTML = ""; return; }
  node.hidden = false;
  const progress = job.progress || {};
  const amount = progressValue(progress);
  node.innerHTML = `<div class="section-heading"><div><span class="kicker">Active run</span><h2>${esc(job.kind || "Research job")}</h2></div>${badge(job.status)}</div><div class="progress-copy"><span>${esc(progress.message || progress.phase || progress.stage || "Running registered protocol")}</span><span>${Math.round(amount * 100)}%</span></div><div class="progress-track"><span style="width:${Math.round(amount * 100)}%"></span></div><div class="row-actions"><button class="button secondary" data-research-pause>${["paused", "pause_requested"].includes(job.status) ? "Resume" : "Pause"}</button><button class="button danger" data-research-cancel>Cancel</button></div>`;
  node.querySelector("[data-research-pause]").addEventListener("click", async (event) => {
    const action = ["paused", "pause_requested"].includes(job.status) ? "resume" : "pause";
    event.currentTarget.disabled = true;
    try { await api(`/api/jobs/${encodeURIComponent(job.id)}/${action}`, { method: "POST" }); }
    catch (error) { toast(error.message, "error"); event.currentTarget.disabled = false; }
  });
  node.querySelector("[data-research-cancel]").addEventListener("click", async (event) => {
    event.currentTarget.disabled = true;
    try { await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, { method: "POST" }); }
    catch (error) { toast(error.message, "error"); event.currentTarget.disabled = false; }
  });
}

function monitorResearch(jobId, kind, onResult = null) {
  activeResearchJob = jobId;
  stopWatching?.();
  stopWatching = watchJob(jobId, {
    onUpdate: (job) => renderActive({ ...job, kind }),
    onDone: (job) => {
      renderActive(null); activeResearchJob = null; stopWatching = null;
      if (job.status === "done") { toast("Research run completed."); if (onResult) onResult(job.result); }
      else toast(job.error || `Research run ${job.status}.`, "error");
    },
  });
}

async function startExperiment(id, row) {
  if (activeResearchJob) { toast("Finish or cancel the active research run first.", "error"); return; }
  const hypothesis = state.experiments.find((item) => item.id === id);
  if (!hypothesis) return;
  const body = {
    model_id: $("research-model").value || null,
    repeats: Math.max(1, Number(hypothesis.minimum_repeats || 1)),
    max_new_tokens: Math.max(8, Number(hypothesis.minimum_new_tokens || 1)),
  };
  const draft = row.querySelector(".experiment-draft")?.value.trim();
  if ((hypothesis.required_inputs || []).includes("draft_model_id") && !draft) { toast("This experiment requires a draft model ID.", "error"); return; }
  if (draft) body.draft_model_id = draft;
  if (hypothesis.runner === "generation" && !body.model_id) { toast("Prepare and select a model first.", "error"); return; }
  const button = row.querySelector("[data-run-experiment]"); button.disabled = true;
  try {
    const result = await api(`/api/experiments/${encodeURIComponent(id)}/runs`, { method: "POST", body });
    monitorResearch(result.job_id, `${hypothesisCode(id)} · ${hypothesis.title}`);
  } catch (error) { toast(error.message, "error"); button.disabled = false; }
}

function renderComparison(result) {
  const rows = result?.rows || [];
  $("compare-result").innerHTML = rows.length ? `<table class="comparison-table"><thead><tr><th>Profile</th><th>Seconds/token</th><th>Peak VRAM</th><th>Relative speed</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${esc(row.profile)}</td><td>${Number(row.seconds_per_token).toFixed(2)}</td><td>${row.peak_vram_gb == null ? "Unavailable" : `${Number(row.peak_vram_gb).toFixed(2)} GB`}</td><td>${row.speedup_vs_min_memory == null ? "Control" : `${Number(row.speedup_vs_min_memory).toFixed(2)}×`}</td></tr>`).join("")}</tbody></table>` : '<div class="empty-inline">The run completed without comparison rows.</div>';
}

async function startComparison(event) {
  event.preventDefault();
  if (activeResearchJob) { toast("Finish or cancel the active research run first.", "error"); return; }
  const modelId = $("compare-model").value;
  if (!modelId) { toast("Prepare a model before running a comparison.", "error"); return; }
  const button = event.currentTarget.querySelector("button"); button.disabled = true;
  $("compare-result").innerHTML = '<div class="empty-inline">Starting comparison…</div>';
  try {
    const result = await api("/api/compare", { method: "POST", body: { model_id: modelId, prompt: $("compare-prompt").value, max_new_tokens: 12 } });
    monitorResearch(result.job_id, "Execution profile comparison", renderComparison);
  } catch (error) { toast(error.message, "error"); }
  finally { button.disabled = false; }
}

export async function loadResearch({ quiet = false } = {}) {
  try {
    const payload = await api("/api/experiments");
    state.experiments = payload.hypotheses || [];
    renderExperiments(); updateResearchModels();
  } catch (error) {
    if (!quiet) toast(`Could not load experiments: ${error.message}`, "error");
    $("experiment-list").classList.remove("loading-line");
    $("experiment-list").innerHTML = '<div class="empty-inline">Experiment registry unavailable.</div>';
  }
}

export function initResearch() {
  $("research-query").addEventListener("input", renderExperiments);
  $("compare-form").addEventListener("submit", startComparison);
  updateResearchModels();
}
