import { state } from "./state.js";
import { $, $$, api, badge, copyText, esc, progressValue, statusLabel, toast, watchJob } from "./shared.js";

let activeResearchJob = null;
let stopWatching = null;
const uploadedArtifacts = new Map();
const uploadedInputs = new Map();

const GOALS = {
  "h0-joint-oracle-gap": "Check whether one fixed execution setting is already close to the best setting for every request.",
  "h1-critical-path": "Keep the weights that measured timing shows are actually delaying generation.",
  "h2-hazard-cost": "Change speculative draft length using a previously calibrated rejection model.",
  "h3-contextual-bandit": "Choose among execution profiles from request context using held-out calibration data.",
  "h4-feedback-prefetch": "Adjust how far ahead weights are read while the model is generating.",
  "h5-certified-mips": "Avoid output-head work only when mathematical bounds prove the chosen token cannot change.",
  "h6-representations": "Plan which exact physical representation each tensor should use under memory limits.",
  "h7-xor-reference": "Check whether related MoE experts compress better as exact XOR differences.",
  "h8-model-based-rl": "Test a trace simulator before using it to choose execution settings.",
  "h9-ram-overlay-head": "Keep the decoded output head in RAM to reduce repeated disk and decode work.",
  "h10-replay-cem": "Use measured traces to search complete weight-residency plans instead of ranking weights independently.",
  "h11-neural-utility-spec": "Choose speculative stopping points with a calibrated small utility model.",
  "h12-bayesian-prefetch": "Use uncertainty in read timing to choose a safer prefetch depth.",
  "h13-qubo-residency": "Search weight placement while accounting for pairs of choices that interact.",
  "h14-coalesced-storage": "Read adjacent compressed arrays together to reduce storage requests.",
  "h15-extent-qubo-residency": "Search placement using bounded physical storage groups.",
  "h16-spec-critical-path": "Combine a measured critical-path placement with fixed speculative decoding.",
  "h17-tensor-extents": "Coalesce reads within each tensor while preserving overlap between tensors.",
  "h18-rollback-cached-spec": "Reuse the accepted target-model cache during speculative decoding.",
};

function readyModels() { return state.models.filter((model) => model.state === "ready"); }

function readyOptions(selected = "") {
  const models = readyModels();
  if (!models.length) return '<option value="">No prepared models</option>';
  return models.map((model) => `<option value="${esc(model.model_id)}"${model.model_id === selected ? " selected" : ""}>${esc(model.model_id)}</option>`).join("");
}

export function updateResearchModels() {
  const select = $("research-model");
  const previous = select.value;
  const models = readyModels();
  select.innerHTML = readyOptions(previous);
  if (previous && models.some((model) => model.model_id === previous)) select.value = previous;
  const help = $("research-model-help");
  help.hidden = Boolean(models.length);
  help.innerHTML = models.length ? "" : 'No prepared model is available. <button type="button" class="text-button">Browse models</button>';
  help.querySelector("button")?.addEventListener("click", () => { location.hash = "models"; });
  updateReadiness();
}

function selectedExperiment() {
  return state.experiments.find((item) => item.id === $("research-experiment").value) || null;
}

function hypothesisIndexLabel(id) {
  // Hypothesis ids are already "h6-representations" etc.; the leading
  // number is the thing people actually say out loud ("H6") when
  // discussing results, but the dropdown previously showed only the
  // spelled-out title, so there was no way to find "H6" in the UI without
  // already knowing its title.
  const match = /^h(\d+)/i.exec(id || "");
  return match ? `H${match[1]}` : "";
}

function optionMarkup(rows) {
  return rows.map((item) => {
    const index = hypothesisIndexLabel(item.id);
    const label = index ? `${index}: ${item.title}` : item.title;
    return `<option value="${esc(item.id)}">${esc(label)}</option>`;
  }).join("");
}

function renderExperimentOptions() {
  const ready = state.experiments.filter((item) => item.runner === "generation" && !(item.required_inputs || []).length);
  const setup = state.experiments.filter((item) => item.runner === "generation" && (item.required_inputs || []).length);
  const data = state.experiments.filter((item) => item.runner !== "generation");
  $("research-experiment").innerHTML = `${ready.length ? `<optgroup label="Ready to run">${optionMarkup(ready)}</optgroup>` : ""}${setup.length ? `<optgroup label="Needs a calibration file or draft model">${optionMarkup(setup)}</optgroup>` : ""}${data.length ? `<optgroup label="Analyze an existing dataset">${optionMarkup(data)}</optgroup>` : ""}`;
  renderSelectedExperiment();
}

function requirementLabel(value) {
  return statusLabel(value).replace(" Id", " ID").replace(" Vram", " VRAM").replace(" Ram", " RAM");
}

function artifactInput(field) {
  return `<label class="artifact-field"><span>${esc(requirementLabel(field))}</span><small>Upload the JSON artifact produced by a separate calibration run.</small><span class="file-picker"><input type="file" accept="application/json,.json" data-artifact="${esc(field)}"><button class="button secondary" type="button" data-pick-artifact="${esc(field)}">Choose file</button><strong data-artifact-name="${esc(field)}">No file selected</strong></span></label>`;
}

function dataInput(field) {
  return `<label class="artifact-field"><span>${esc(requirementLabel(field))}</span><small>Choose a JSON file containing this measured input. Synthetic values are not supplied because they would not produce meaningful evidence.</small><span class="file-picker"><input type="file" accept="application/json,.json" data-dataset="${esc(field)}"><button class="button secondary" type="button" data-pick-dataset="${esc(field)}">Choose data</button><strong data-dataset-name="${esc(field)}">No file selected</strong></span></label>`;
}

function renderSelectedExperiment() {
  const experiment = selectedExperiment();
  if (!experiment) { $("research-explanation").innerHTML = ""; $("research-inputs").innerHTML = ""; return; }
  const requirements = experiment.required_inputs || [];
  const mode = experiment.runner === "generation" ? "Runs the model" : "Analyzes supplied measurements";
  const index = hypothesisIndexLabel(experiment.id);
  const goal = GOALS[experiment.id] || experiment.statement;
  $("research-explanation").innerHTML = `<div><span class="strategy-state ${requirements.length ? "setup" : "ready"}">${requirements.length ? "Setup required" : "Ready to run"}</span><strong>${esc(index ? `${index}: ${goal}` : goal)}</strong><p>${esc(mode)}. The candidate setting is compared with the registered control under the same run settings.</p></div><dl><div><dt>Standard</dt><dd>${esc(statusLabel(experiment.control_profile))}</dd></div><div><dt>Candidate</dt><dd>${esc(statusLabel(experiment.candidate_profile))}</dd></div></dl>`;
  $("research-tokens").value = String(Math.max(8, Number(experiment.minimum_new_tokens || 16)));
  $("research-repeats").value = String(Math.max(1, Number(experiment.minimum_repeats || 3)));
  const fields = [];
  for (const field of requirements) {
    if (field === "draft_model_id") fields.push('<label class="field-label wide">Draft model ID<input id="research-draft-model" value="Qwen/Qwen3-0.6B"><small>A small resident model proposes tokens. It does not need an Afterimage store.</small></label>');
    else if (["critical_path_profile", "spec_policy_state", "replay_plan_state"].includes(field)) fields.push(artifactInput(field));
    else fields.push(dataInput(field));
  }
  $("research-inputs").innerHTML = fields.join("");
  bindInputFiles(); updateReadiness();
}

function readFile(file, mode, field) {
  const reader = new FileReader();
  reader.onerror = () => toast(`Could not read ${file.name}.`, "error");
  reader.onload = async () => {
    try {
      if (mode === "dataset") {
        const payload = JSON.parse(String(reader.result));
        uploadedInputs.set(field, Object.hasOwn(payload, field) ? payload[field] : payload);
        document.querySelector(`[data-dataset-name="${field}"]`).textContent = file.name;
      } else {
        const encoded = String(reader.result).split(",", 2)[1];
        const saved = await api("/api/research/artifacts", { method: "POST", body: { name: file.name, content_base64: encoded, kind: field } });
        uploadedArtifacts.set(field, saved.path);
        document.querySelector(`[data-artifact-name="${field}"]`).textContent = file.name;
      }
      updateReadiness();
    } catch (error) { toast(`${file.name}: ${error.message}`, "error"); }
  };
  if (mode === "dataset") reader.readAsText(file); else reader.readAsDataURL(file);
}

function bindInputFiles() {
  $$('[data-pick-artifact]').forEach((button) => button.addEventListener("click", () => document.querySelector(`[data-artifact="${button.dataset.pickArtifact}"]`).click()));
  $$('[data-artifact]').forEach((input) => input.addEventListener("change", () => input.files[0] && readFile(input.files[0], "artifact", input.dataset.artifact)));
  $$('[data-pick-dataset]').forEach((button) => button.addEventListener("click", () => document.querySelector(`[data-dataset="${button.dataset.pickDataset}"]`).click()));
  $$('[data-dataset]').forEach((input) => input.addEventListener("change", () => input.files[0] && readFile(input.files[0], "dataset", input.dataset.dataset)));
}

function missingRequirements() {
  const experiment = selectedExperiment();
  if (!experiment) return ["test"];
  const missing = [];
  if (experiment.runner === "generation" && !$("research-model").value) missing.push("prepared model");
  for (const field of experiment.required_inputs || []) {
    if (field === "draft_model_id") { if (!$("research-draft-model")?.value.trim()) missing.push("draft model"); }
    else if (["critical_path_profile", "spec_policy_state", "replay_plan_state"].includes(field)) { if (!uploadedArtifacts.has(field)) missing.push(requirementLabel(field)); }
    else if (!uploadedInputs.has(field)) missing.push(requirementLabel(field));
  }
  return missing;
}

function updateReadiness() {
  const missing = missingRequirements();
  $("research-run").disabled = Boolean(activeResearchJob) || Boolean(missing.length);
  $("quick-compare").disabled = Boolean(activeResearchJob) || !$("research-model").value;
  $("research-readiness").textContent = activeResearchJob ? "A run is active." : missing.length ? `Needs ${missing.join(", ")}.` : "Ready. Results will be saved as a new local report.";
}

function renderActive(job) {
  const node = $("active-research");
  if (!job) { node.hidden = true; node.innerHTML = ""; return; }
  node.hidden = false;
  const progress = job.progress || {};
  const amount = progressValue(progress);
  node.innerHTML = `<div class="active-run-title"><div><span class="kicker">Running now</span><h2>${esc(job.label || "Research test")}</h2></div>${badge(job.status)}</div><div class="progress-copy"><span>${esc(progress.message || progress.phase || progress.stage || "Starting the controlled run")}</span><span>${Math.round(amount * 100)}%</span></div><div class="progress-track"><span style="width:${Math.round(amount * 100)}%"></span></div><div class="row-actions"><button class="button secondary" data-research-pause>${["paused", "pause_requested"].includes(job.status) ? "Resume" : "Pause"}</button><button class="button danger" data-research-cancel>Cancel</button></div>`;
  node.querySelector("[data-research-pause]").addEventListener("click", async (event) => {
    const action = ["paused", "pause_requested"].includes(job.status) ? "resume" : "pause";
    event.currentTarget.disabled = true;
    try { await api(`/api/jobs/${encodeURIComponent(job.id)}/${action}`, { method: "POST" }); } catch (error) { toast(error.message, "error"); event.currentTarget.disabled = false; }
  });
  node.querySelector("[data-research-cancel]").addEventListener("click", async (event) => {
    event.currentTarget.disabled = true;
    try { await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`, { method: "POST" }); } catch (error) { toast(error.message, "error"); event.currentTarget.disabled = false; }
  });
}

function monitorResearch(jobId, label, onResult) {
  activeResearchJob = jobId; updateReadiness(); stopWatching?.();
  stopWatching = watchJob(jobId, {
    onUpdate: (job) => renderActive({ ...job, label }),
    onDone: async (job) => {
      renderActive(null); activeResearchJob = null; stopWatching = null; updateReadiness();
      if (job.status === "done") { toast("Run completed and saved."); onResult?.(job.result); await loadReports(); }
      else if (job.status === "cancelled") toast("Run stopped.");
      else toast(job.error || `Run ${job.status}.`, "error");
    },
  });
}

function summaryRows(summary = {}) {
  const entries = Object.entries(summary).filter(([, value]) => ["number", "string", "boolean"].includes(typeof value)).slice(0, 8);
  if (!entries.length) return '<div class="empty-inline">No scalar measurements were recorded.</div>';
  return `<div class="report-metrics">${entries.map(([key, value]) => `<div><span>${esc(statusLabel(key))}</span><strong>${esc(typeof value === "number" ? Number(value).toPrecision(4) : value)}</strong></div>`).join("")}</div>`;
}

function profileConfig(run, variant) {
  const trials = variant === "candidate" ? run.candidate_trials : run.control_trials;
  const config = trials?.find((trial) => trial.config)?.config;
  if (!config) return null;
  const value = { ...config };
  if (run.metadata?.draft_model_id) value._draft_model_id = run.metadata.draft_model_id;
  return value;
}

function reportMarkup(run, { expanded = false } = {}) {
  const experiment = state.experiments.find((item) => item.id === run.hypothesis_id);
  const modelId = run.metadata?.model_id;
  const candidate = profileConfig(run, "candidate");
  const control = profileConfig(run, "control");
  const time = run.completed_at ? new Date(run.completed_at * 1000).toLocaleString() : "Completed locally";
  const index = hypothesisIndexLabel(run.hypothesis_id);
  const title = experiment?.title || "Research report";
  return `<article class="report-card${expanded ? " featured" : ""}" data-report="${esc(run.id)}"><div class="report-card-head"><div><span class="kicker">${esc(time)}</span><h2>${esc(index ? `${index}: ${title}` : title)}</h2><p>${esc(modelId || "Dataset analysis")}</p></div>${badge(run.verdict || run.status)}</div>${summaryRows(run.summary)}<div class="report-actions">${candidate && modelId ? `<button class="button primary" data-use-report="candidate">Use candidate in Chat</button>` : ""}${control && modelId ? `<button class="button secondary" data-use-report="control">Use standard in Chat</button>` : ""}<button class="button secondary" data-copy-endpoint>Copy API example</button></div><details class="report-details" ${expanded ? "open" : ""}><summary>Configuration and endpoint</summary><div class="endpoint-card"><code>POST http://127.0.0.1:8420/v1/chat/completions</code><p>Model: ${esc(modelId || "Not applicable")}</p></div><pre>${esc(JSON.stringify({ summary: run.summary, candidate_config: candidate, control_config: control }, null, 2))}</pre></details></article>`;
}

function endpointExample(modelId, profileId = null) {
  return JSON.stringify({ model: modelId || "your/model", messages: [{ role: "user", content: "Hello" }], stream: true, ...(profileId ? { runtime_profile_id: profileId } : { execution_profile: "auto" }) }, null, 2);
}

async function useReport(run, variant, button) {
  const modelId = run.metadata?.model_id;
  const config = profileConfig(run, variant);
  if (!modelId || !config) return;
  button.disabled = true;
  try {
    const experiment = state.experiments.find((item) => item.id === run.hypothesis_id);
    const profile = await api("/api/runtime-profiles", { method: "POST", body: { name: `${experiment?.title || "Research"} ${variant}`, model_id: modelId, config, source_run_id: run.id } });
    state.runtimeProfiles.unshift(profile);
    sessionStorage.setItem("afterimage.chatModel", modelId);
    sessionStorage.setItem("afterimage.chatProfile", `saved:${profile.id}`);
    location.hash = "chat";
  } catch (error) { toast(error.message, "error"); button.disabled = false; }
}

function bindReportActions(root, runs) {
  root.querySelectorAll("[data-report]").forEach((card) => {
    const run = runs.find((item) => item.id === card.dataset.report);
    card.querySelectorAll("[data-use-report]").forEach((button) => button.addEventListener("click", () => useReport(run, button.dataset.useReport, button)));
    card.querySelector("[data-copy-endpoint]")?.addEventListener("click", async () => {
      await copyText(endpointExample(run.metadata?.model_id)); toast("API request copied.");
    });
  });
}

function renderSingleResult(run) {
  const node = $("research-result");
  node.hidden = false; node.innerHTML = `<div class="reports-heading"><div><span class="kicker">New report</span><h2>Run completed</h2></div></div>${reportMarkup(run, { expanded: true })}`;
  bindReportActions(node, [run]); node.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderReports() {
  const node = $("research-reports"); node.classList.remove("loading-line");
  $("report-count").textContent = String(state.researchRuns.length);
  if (!state.researchRuns.length) { node.innerHTML = '<div class="empty-report"><strong>No reports yet</strong><p>Run a test. Its measurements and exact settings will appear here.</p></div>'; return; }
  node.innerHTML = state.researchRuns.map((run) => reportMarkup(run)).join(""); bindReportActions(node, state.researchRuns);
}

async function loadReports() {
  try {
    const [runs, profiles] = await Promise.all([api("/api/experiment-runs"), api("/api/runtime-profiles")]);
    state.researchRuns = runs.runs || []; state.runtimeProfiles = profiles.profiles || []; renderReports();
  } catch (error) { $("research-reports").classList.remove("loading-line"); $("research-reports").innerHTML = `<div class="notice error">${esc(error.message)}</div>`; }
}

async function startExperiment(event) {
  event.preventDefault();
  if (activeResearchJob) return;
  const experiment = selectedExperiment();
  const missing = missingRequirements();
  if (!experiment || missing.length) { toast(`Complete: ${missing.join(", ")}.`, "error"); return; }
  const body = { model_id: experiment.runner === "generation" ? $("research-model").value : null, prompt: $("research-prompt").value, repeats: Number($("research-repeats").value), max_new_tokens: Number($("research-tokens").value), config_overrides: {}, inputs: {} };
  const vram = Number($("research-vram").value); const ram = Number($("research-ram").value);
  if (vram > 0) body.config_overrides.vram_budget_gb = vram;
  if (ram > 0) body.config_overrides.ram_budget_gb = ram;
  for (const field of experiment.required_inputs || []) {
    if (field === "draft_model_id") body.draft_model_id = $("research-draft-model").value.trim();
    else if (uploadedArtifacts.has(field)) body.config_overrides[field] = uploadedArtifacts.get(field);
    else if (uploadedInputs.has(field)) body.inputs[field] = uploadedInputs.get(field);
  }
  try {
    const result = await api(`/api/experiments/${encodeURIComponent(experiment.id)}/runs`, { method: "POST", body });
    monitorResearch(result.job_id, experiment.title, (payload) => renderSingleResult(payload.run));
  } catch (error) { toast(error.message, "error"); }
}

function renderComparison(result) {
  const rows = result?.rows || [];
  const node = $("research-result"); node.hidden = false;
  node.innerHTML = `<div class="reports-heading"><div><span class="kicker">Quick comparison</span><h2>Measured on this computer</h2></div></div>${rows.length ? `<div class="comparison-cards">${rows.map((row) => `<article><span>${esc(statusLabel(row.profile))}</span><strong>${Number(row.seconds_per_token).toFixed(2)} s/token</strong><small>${row.peak_vram_gb == null ? "Peak VRAM unavailable" : `${Number(row.peak_vram_gb).toFixed(2)} GB peak VRAM`}</small><button class="button secondary" data-chat-built-in="${esc(row.profile)}">Use in Chat</button></article>`).join("")}</div>` : '<div class="empty-inline">The run completed without comparison rows.</div>'}`;
  node.querySelectorAll("[data-chat-built-in]").forEach((button) => button.addEventListener("click", () => { sessionStorage.setItem("afterimage.chatModel", $("research-model").value); sessionStorage.setItem("afterimage.chatProfile", button.dataset.chatBuiltIn); location.hash = "chat"; }));
  node.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function startComparison() {
  const modelId = $("research-model").value;
  if (!modelId) { toast("Prepare and select a model first.", "error"); return; }
  try {
    const result = await api("/api/compare", { method: "POST", body: { model_id: modelId, prompt: $("research-prompt").value, max_new_tokens: Math.max(8, Number($("research-tokens").value)) } });
    monitorResearch(result.job_id, "Memory profile comparison", renderComparison);
  } catch (error) { toast(error.message, "error"); }
}

function showTab(name) {
  $$('[data-research-tab]').forEach((button) => button.classList.toggle("is-active", button.dataset.researchTab === name));
  $("research-run-view").hidden = name !== "run"; $("research-reports-view").hidden = name !== "reports";
  if (name === "reports") loadReports();
}

export async function loadResearch({ quiet = false } = {}) {
  try {
    const [payload] = await Promise.all([api("/api/experiments"), loadReports()]);
    state.experiments = payload.hypotheses || []; renderExperimentOptions(); updateResearchModels();
  } catch (error) { if (!quiet) toast(`Could not load research workspace: ${error.message}`, "error"); }
}

export function initResearch() {
  $("research-form").addEventListener("submit", startExperiment);
  $("research-model").addEventListener("change", updateReadiness);
  $("research-experiment").addEventListener("change", () => { uploadedArtifacts.clear(); uploadedInputs.clear(); renderSelectedExperiment(); });
  $("quick-compare").addEventListener("click", startComparison);
  $("refresh-reports").addEventListener("click", loadReports);
  $$('[data-research-tab]').forEach((button) => button.addEventListener("click", () => showTab(button.dataset.researchTab)));
  updateResearchModels();
}
