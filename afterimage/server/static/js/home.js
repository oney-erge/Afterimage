import { state, updateState } from "./state.js";
import {
  $, afterimageFit, api, badge, confidenceBadge, esc, fitBadge, fmtGib, toast,
} from "./shared.js";
import { renderJobs } from "./jobs.js";

function renderMachineLine() {
  const hardware = state.hardware;
  const dot = $("machine-status-dot");
  const body = $("machine-line-body");
  if (!hardware) {
    dot.className = "machine-status-dot bad";
    body.textContent = "Hardware details are unavailable.";
    return;
  }
  const gpu = hardware.gpu || {};
  const gpuName = gpu.name || "No GPU detected";
  const vram = hardware.vram_total_gb != null ? fmtGib(hardware.vram_total_gb) : null;
  const ram = hardware.memory?.total_gib != null ? fmtGib(hardware.memory.total_gib) : null;
  const free = hardware.disk?.free_gib != null ? `${fmtGib(hardware.disk.free_gib)} free` : null;
  dot.className = `machine-status-dot ${vram ? "ok" : "bad"}`;
  const parts = [`<strong>${esc(gpuName)}</strong>`];
  if (vram) parts.push(`<span>${esc(vram)} VRAM</span>`);
  if (ram) parts.push(`<span>${esc(ram)} RAM</span>`);
  if (free) parts.push(`<span>${esc(free)}</span>`);
  body.innerHTML = parts.join('<span class="sep">·</span>');
}

function renderReference() {
  const reference = state.capability?.measured_reference;
  const node = $("reference-card");
  node.classList.remove("loading-line");
  if (!reference) {
    node.innerHTML = '<div class="empty-inline">Measured reference unavailable.</div>';
    return;
  }
  const bf16 = reference.params_b * reference.bf16_gb_per_b_params;
  const store = reference.params_b * reference.compressed_gb_per_b_params;
  // Deliberately no speed/s-per-token figure here: that number was
  // measured on one specific machine (an RTX 3080 Laptop) and presenting
  // it on a first run, before this viewer has measured anything on their
  // own hardware, reads as a promise about their machine rather than a
  // fact about a different one. Storage size is hardware-independent --
  // the compressed store is the same size on any machine -- so it stays.
  node.innerHTML = `<div class="reference-model"><strong>${esc(reference.model)}</strong><span>measured storage footprint</span></div><div class="reference-stats"><div><strong>${bf16.toFixed(1)} GB</strong><small>Original BF16</small></div><div><strong>${store.toFixed(1)} GB</strong><small>Afterimage store</small></div></div>`;
}

const CAMPAIGN_METHOD_LABELS = {
  "airllm": "AirLLM", "accelerate": "HF Accelerate", "dfloat11": "DFloat11",
  "exact-min": "Afterimage (min memory)", "exact-resident": "Afterimage (resident)",
  "spec-fixed": "Afterimage (speculative)",
};

function methodLabel(id) { return CAMPAIGN_METHOD_LABELS[id] || id; }

function renderCampaign() {
  const panel = $("campaign-panel");
  const node = $("campaign-card");
  const lengths = state.campaign?.lengths || [];
  if (!lengths.length) { panel.hidden = true; return; }
  panel.hidden = false;
  const active = lengths.find((length) => length.status === "in_progress") || lengths[lengths.length - 1];
  const doneCount = lengths.filter((length) => length.status === "complete").length;
  const rows = active.methods.map((method) => {
    const spt = method.mean_seconds_per_token != null ? `${method.mean_seconds_per_token.toFixed(2)} s/tok` : "—";
    return `<div class="campaign-method campaign-method-${esc(method.marker)}"><span>${esc(methodLabel(method.method_id))}</span><span>${method.blocks_done}/${active.blocks_requested} blocks</span><span>${spt}</span></div>`;
  }).join("");
  const pct = active.cells_required ? Math.round((active.cells_done / active.cells_required) * 100) : 0;
  node.innerHTML = `
    <div class="campaign-head">
      <div><strong>${active.max_new_tokens}-token pass</strong><span>${esc(active.prompt_suite || "")} · ${doneCount}/${lengths.length} lengths complete</span></div>
      <span class="campaign-status campaign-status-${active.status}">${active.status === "in_progress" ? "Running" : "Complete"}</span>
    </div>
    <div class="campaign-progress"><div class="campaign-progress-bar" style="width:${pct}%"></div></div>
    <div class="campaign-progress-label">${active.cells_done}/${active.cells_required} cells${active.eta_human ? ` · ~${active.eta_human} remaining` : ""}</div>
    <div class="campaign-methods">${rows}</div>`;
}

async function loadCampaign() {
  try {
    const data = await api("/api/campaigns");
    updateState({ campaign: data });
    renderCampaign();
  } catch (_) {
    $("campaign-panel").hidden = true;
  }
}

function activeAcquireJob() {
  const active = new Set(["queued", "running", "pause_requested", "paused", "cancelling"]);
  return state.jobs.find((job) => job.kind === "acquire" && active.has(job.status)) || null;
}

function readyModels() {
  return state.models.filter((model) => model.state === "ready");
}

function goToChat(modelId) {
  sessionStorage.setItem("afterimage.chatModel", modelId);
  location.hash = "chat";
}

// The four home states, in the order a new user actually moves through
// them: nothing installed yet, a model is being prepared, one model is
// ready (the common case right after first setup), several models are
// ready (the library has grown). Each state answers a different question,
// so each gets its own layout rather than one page trying to answer all
// of them at once.
function renderHomeStatus() {
  const node = $("home-status");
  const preparing = activeAcquireJob();
  const ready = readyModels();
  const vramTotal = state.hardware?.vram_total_gb;

  if (preparing) {
    node.innerHTML = `<div class="home-preparing"><span class="kicker">Preparing</span><h2>Preparing ${esc(preparing.model_id)}</h2><div id="home-preparing-job"></div></div>`;
    renderJobs($("home-preparing-job"), [preparing], { onChanged: () => loadHome() });
    return;
  }

  if (!ready.length) {
    node.innerHTML = `<div class="home-hero-compact">
      <h2>Run larger models than your GPU alone.</h2>
      <p>Afterimage streams exact model weights through the memory you have, losslessly. Larger models need more storage and generally run more slowly, but you are not limited to what fits in VRAM.</p>
      <button class="button primary" data-go="models">Browse models <span aria-hidden="true">→</span></button>
    </div>`;
    return;
  }

  if (ready.length === 1) {
    const model = ready[0];
    const fit = afterimageFit(model.comp_gb, vramTotal);
    node.innerHTML = `<div class="home-ready">
      <span class="kicker">Ready</span>
      <h2>${esc(model.model_id)} is ready</h2>
      <div class="badge-row">${fitBadge(fit)}${badge("ready", "Prepared")}</div>
      <div class="home-ready-meta">
        ${model.orig_gb ? `<div><strong>${model.orig_gb.toFixed(1)} GB</strong><small>Native size</small></div>` : ""}
        ${model.comp_gb ? `<div><strong>${model.comp_gb.toFixed(1)} GB</strong><small>Afterimage store</small></div>` : ""}
        ${vramTotal ? `<div><strong>${fmtGib(vramTotal)}</strong><small>Your GPU</small></div>` : ""}
      </div>
      <div class="home-ready-actions">
        <button class="button primary" data-home-chat="${encodeURIComponent(model.model_id)}">Chat</button>
        <button class="text-button" data-go="models">Model details</button>
      </div>
    </div>`;
    node.querySelector("[data-home-chat]")?.addEventListener("click", (event) => {
      goToChat(decodeURIComponent(event.currentTarget.dataset.homeChat));
    });
    return;
  }

  const rows = ready.slice(0, 6).map((model) => {
    const fit = afterimageFit(model.comp_gb, vramTotal);
    return `<div class="compact-model"><div><strong title="${esc(model.model_id)}">${esc(model.model_id)}</strong><div class="model-meta">${fitBadge(fit)}${model.comp_gb ? `${model.comp_gb.toFixed(1)} GB` : ""}</div></div><button class="text-button" data-home-chat="${encodeURIComponent(model.model_id)}">Chat →</button></div>`;
  }).join("");
  node.innerHTML = `<div class="home-multi">
    <span class="kicker">Your models</span>
    <div class="compact-list">${rows}</div>
    <div class="home-multi-actions"><button class="button secondary" data-go="models">Browse larger models</button></div>
  </div>`;
  node.querySelectorAll("[data-home-chat]").forEach((button) => button.addEventListener("click", () => {
    goToChat(decodeURIComponent(button.dataset.homeChat));
  }));
}

export async function loadHome({ quiet = false } = {}) {
  try {
    const [hardware, capability, models] = await Promise.all([
      api("/api/hardware"), api("/api/capability"), api("/api/models"),
    ]);
    updateState({ hardware, capability, models: models.models || [] });
    renderMachineLine(); renderReference(); renderHomeStatus();
  } catch (error) {
    if (!quiet) toast(`Could not load system status: ${error.message}`, "error");
    $("machine-line-body").textContent = "Hardware details are unavailable.";
    $("machine-status-dot").className = "machine-status-dot bad";
  }
  loadCampaign();
}

export function initHome() {
  $("refresh-hardware").addEventListener("click", () => loadHome());
  // A campaign started outside the UI (the paper benchmark's own CLI/
  // background process, not a job this server queued) has no job-status
  // websocket to push updates through, so Home polls its own read-only
  // snapshot independently of the job-driven refresh in app.js.
  setInterval(() => { if (state.route === "home") loadCampaign(); }, 15000);
}

export function refreshHomeModels() { renderHomeStatus(); }
