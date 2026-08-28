import { state, updateState } from "./state.js";
import {
  $, $$, afterimageFit, api, badge, compactNumber, confidenceBadge, esc, fitBadge,
  fmtBytes, hfUrl, statusLabel, toast, watchJob,
} from "./shared.js";
import { isActiveJob, renderJobs } from "./jobs.js";

function vramTotalGb() { return state.hardware?.vram_total_gb; }

const watchedJobs = new Set();
let libraryChanged = () => {};
let showAllComputerResults = false;

function modelCompatibility(model) { return model.metadata?.compatibility || {}; }
function activeFor(modelId) { return state.jobs.find((job) => job.model_id === modelId && isActiveJob(job)); }
function encodedModel(modelId) { return encodeURIComponent(modelId); }
function compactViewport() { return window.matchMedia("(max-width: 760px)").matches; }

function chatWith(modelId) {
  sessionStorage.setItem("afterimage.chatModel", modelId);
  location.hash = "chat";
}

function renderLibrary() {
  const node = $("local-models");
  node.classList.remove("loading-line");
  $("model-count").textContent = `${state.models.length} ${state.models.length === 1 ? "model" : "models"}`;
  if (!state.models.length) {
    node.innerHTML = '<div class="empty-inline">Nothing has been added to Afterimage yet.</div>';
    return;
  }
  node.innerHTML = state.models.map((model) => {
    const compatibility = modelCompatibility(model);
    const job = activeFor(model.model_id);
    const stateValue = job?.status || model.state;
    const size = model.comp_gb ? `${model.comp_gb.toFixed(1)} GB prepared` : model.metadata?.source_bytes ? `${fmtBytes(model.metadata.source_bytes)} source` : statusLabel(model.stage || model.state);
    const actions = [];
    if (model.state === "ready") actions.push(`<button class="button primary" data-chat-model="${encodedModel(model.model_id)}">Chat</button>`);
    else if (!job) actions.push(`<button class="button primary" data-get-model="${encodedModel(model.model_id)}">${model.state === "downloaded" ? "Prepare" : "Continue"}</button>`);
    if (model.loaded) actions.push(`<button class="button secondary" data-unload-model="${encodedModel(model.model_id)}">Unload</button>`);
    actions.push(`<button class="button secondary" data-model-detail="${encodedModel(model.model_id)}">Details</button>`);
    if (!job) actions.push(`<button class="text-button" data-remove-model="${encodedModel(model.model_id)}">Remove</button>`);
    const fit = model.comp_gb ? fitBadge(afterimageFit(model.comp_gb, vramTotalGb())) : "";
    // "loaded" (this engine is resident in VRAM/RAM right now) is a
    // separate fact from "ready" (a compressed store exists on disk) --
    // both badges can be true at once, and only "loaded" is ever untrue
    // for a model nobody has chatted with yet.
    return `<article class="model-row compact-model-row">
      <div class="model-primary"><strong class="model-name" title="${esc(model.model_id)}">${esc(model.model_id)}</strong><div class="badge-row">${badge(stateValue)}${model.loaded ? badge("loaded", "Loaded") : ""}${fit}${compatibility.modality === "vision-text" ? badge("vision", "Vision") : ""}${compatibility.mixture_of_experts ? badge("moe", "MoE") : ""}</div></div>
      <div class="model-description">${esc(model.error || size)}</div><div class="row-actions">${actions.join("")}</div>
    </article>`;
  }).join("");
  bindModelActions(node);
}

function bindModelActions(node) {
  node.querySelectorAll("[data-chat-model]").forEach((button) => button.addEventListener("click", () => chatWith(decodeURIComponent(button.dataset.chatModel))));
  node.querySelectorAll("[data-get-model]").forEach((button) => button.addEventListener("click", () => startGet(decodeURIComponent(button.dataset.getModel), button)));
  node.querySelectorAll("[data-model-detail]").forEach((button) => button.addEventListener("click", () => openModelDialog(decodeURIComponent(button.dataset.modelDetail))));
  node.querySelectorAll("[data-remove-model]").forEach((button) => button.addEventListener("click", () => removeModel(decodeURIComponent(button.dataset.removeModel), button)));
  node.querySelectorAll("[data-unload-model]").forEach((button) => button.addEventListener("click", () => unloadModel(decodeURIComponent(button.dataset.unloadModel), button)));
}

// The segmented control's server-side counterpart: All/Text/Vision search
// different Hugging Face pipeline_tag values (there is no single tag that
// means "any runnable chat model"), so the filter has to trigger a new
// /api/catalog/models fetch, not just a client-side re-render of results
// that were never fetched with that architecture in mind. MoE has no
// pipeline_tag of its own -- it stays a client-side refinement (below) on
// top of whichever task-filtered page came back.
function taskForFilter(filter) {
  if (filter === "vision") return "image-text-to-text";
  if (filter === "all") return "";
  return "text-generation"; // "text" and "moe" both search generative LLMs
}

function filteredCatalog() {
  const rows = state.catalog.models || [];
  if (state.catalogFilter === "moe") return rows.filter((row) => row.mixture_of_experts);
  return rows;
}

function computerAction(model) {
  const local = state.models.find((row) => row.model_id === model.model_id);
  const job = activeFor(model.model_id);
  if (local?.state === "ready") return `<button class="button primary" data-chat-model="${encodedModel(model.model_id)}">Chat</button>`;
  if (job) return badge(job.status);
  if (model.source === "huggingface-cache" && model.can_prepare) return `<button class="button primary" data-import-cache="${encodedModel(model.model_id)}">Prepare</button>`;
  if (model.source === "ollama") return `<a class="button secondary" href="${esc(model.external_url)}" target="_blank" rel="noreferrer">Open LocalDeploy</a>`;
  return "";
}

function renderComputerResults() {
  const node = $("computer-results");
  node.classList.remove("loading-line");
  const rows = state.localDiscovery.models || [];
  if (!rows.length) { node.innerHTML = ""; return; }
  const previewLimit = compactViewport() ? 2 : 4;
  const visible = showAllComputerResults ? rows : rows.slice(0, previewLimit);
  const disclosure = rows.length > previewLimit
    ? `<button type="button" class="text-button computer-disclosure" id="computer-disclosure">${showAllComputerResults ? "Show fewer" : `Show all ${rows.length}`}</button>`
    : "";
  node.innerHTML = `<div class="computer-result-head"><div><strong>Already on this computer</strong><span>Found in the Hugging Face cache or a running Ollama installation.</span></div><div class="computer-result-actions"><span>${rows.length}</span>${disclosure}</div></div><div class="computer-grid">${visible.map((model) => `<article class="computer-card"><div><strong title="${esc(model.model_id)}">${esc(model.model_id)}</strong><div class="badge-row">${badge(model.source, model.source_label)}${badge(model.format)}</div></div><p>${esc(model.message)}</p><div class="computer-card-foot"><span>${model.size_bytes ? fmtBytes(model.size_bytes) : "Size unavailable"}</span>${computerAction(model)}</div></article>`).join("")}</div>`;
  bindModelActions(node);
  node.querySelectorAll("[data-import-cache]").forEach((button) => button.addEventListener("click", () => importCached(decodeURIComponent(button.dataset.importCache), button)));
  $("computer-disclosure")?.addEventListener("click", () => { showAllComputerResults = !showAllComputerResults; renderComputerResults(); });
}

function renderPagination() {
  const current = Number(state.catalog.page || 1);
  $("catalog-page").textContent = `Page ${current}`;
  $("catalog-prev").disabled = current <= 1;
  $("catalog-next").disabled = !state.catalog.next_cursor && state.catalog.exhausted !== false;
  const windowPages = state.catalog.page_window?.length ? state.catalog.page_window : [current, ...(state.catalog.next_cursor ? [current + 1] : [])];
  const pieces = [];
  if (windowPages[0] > 1) pieces.push('<button type="button" data-catalog-page="1">1</button><span>…</span>');
  for (const page of windowPages) pieces.push(`<button type="button" data-catalog-page="${page}" class="${page === current ? "is-current" : ""}" ${page === current ? 'aria-current="page"' : ""}>${page}</button>`);
  $("catalog-pages").innerHTML = pieces.join("");
  $("catalog-pages").querySelectorAll("[data-catalog-page]").forEach((button) => button.addEventListener("click", () => searchModels(Number(button.dataset.catalogPage), { focus: true })));
}

function renderCatalog() {
  const node = $("catalog-results");
  node.classList.remove("loading-line");
  const rows = filteredCatalog();
  const query = state.catalogQuery || "";
  $("catalog-heading").textContent = query ? `Models matching “${query}”` : "Popular models";
  renderPagination();
  if (!rows.length) {
    node.innerHTML = `<div class="empty-inline">${state.catalogFilter === "all" ? "No online models matched this search." : "No models of this type are on this page."}</div>`;
    return;
  }
  node.innerHTML = rows.map((model) => {
    const local = model.local;
    const running = activeFor(model.model_id);
    let action = `<button class="button primary" data-catalog-get="${encodedModel(model.model_id)}">Get</button>`;
    if (local?.state === "ready") action = `<button class="button primary" data-chat-model="${encodedModel(model.model_id)}">Chat</button>`;
    else if (running) action = badge(running.status);
    const details = [model.params_b ? `${model.params_b}B parameters` : null, model.estimated_source_gb ? `about ${model.estimated_source_gb} GB source` : null, model.downloads != null ? `${compactNumber(model.downloads)} downloads` : null].filter(Boolean).join(" · ");
    // Fit (does this run well here) and support confidence (has this
    // architecture been validated) are deliberately separate badges, not
    // one merged status -- see shared.js's afterimageFit()/confidenceBadge()
    // docstrings. A download-only architecture can't run at all regardless
    // of memory fit, so it gets its own explicit label instead of a Fit
    // badge that would otherwise imply it works.
    const fit = model.execution === "download-only" ? "" : fitBadge(afterimageFit(model.estimated_store_gb, vramTotalGb()));
    const support = confidenceBadge(model.execution) || (model.execution === "download-only" ? badge("unsupported", "Not runnable yet") : "");
    return `<article class="catalog-card"><div class="catalog-card-head"><a class="model-name" href="${hfUrl(model.model_id)}" target="_blank" rel="noreferrer">${esc(model.model_id)} ↗</a><div class="badge-row">${fit}${support}${model.modality === "vision-text" ? badge("vision", "Vision") : badge("text", "Text")}${model.mixture_of_experts ? badge("moe", "MoE") : ""}</div></div><div class="catalog-card-body"><strong>${esc(details || "Metadata unavailable")}</strong><p>${esc(model.execution_reason || "Afterimage will inspect the local snapshot before preparing it.")}</p></div><div class="catalog-card-actions">${action}<button class="button secondary" data-catalog-detail="${encodedModel(model.model_id)}">Details</button></div></article>`;
  }).join("");
  node.querySelectorAll("[data-catalog-get]").forEach((button) => button.addEventListener("click", () => startGet(decodeURIComponent(button.dataset.catalogGet), button)));
  node.querySelectorAll("[data-chat-model]").forEach((button) => button.addEventListener("click", () => chatWith(decodeURIComponent(button.dataset.chatModel))));
  node.querySelectorAll("[data-catalog-detail]").forEach((button) => button.addEventListener("click", () => openModelDialog(decodeURIComponent(button.dataset.catalogDetail))));
}

function openModelDialog(modelId) {
  const model = state.models.find((row) => row.model_id === modelId) || state.catalog.models.find((row) => row.model_id === modelId);
  if (!model) return;
  const compatibility = modelCompatibility(model).execution ? modelCompatibility(model) : model;
  $("model-dialog-content").innerHTML = `<span class="kicker">Model details</span><h2>${esc(model.model_id)}</h2><div class="detail-grid"><div><small>Availability</small><strong>${esc(statusLabel(model.state || model.availability))}</strong></div><div><small>Execution</small><strong>${esc(statusLabel(compatibility.execution || model.compatibility || "unknown"))}</strong></div><div><small>Modality</small><strong>${esc(statusLabel(compatibility.modality || "text"))}</strong></div><div><small>Architecture</small><strong>${esc((compatibility.architectures || []).join(", ") || "Not reported")}</strong></div><div><small>Format</small><strong>${esc(model.format || (model.store_path ? "Afterimage store" : "Not reported"))}</strong></div><div><small>Revision</small><strong>${esc(model.revision || "Resolved during Get")}</strong></div></div><p class="quiet-note">${esc(compatibility.execution_reason || model.error || "Afterimage verifies the local snapshot before marking it ready.")}</p>`;
  $("model-dialog").showModal();
}

async function removeModel(modelId, button) {
  if (!window.confirm(`Remove the prepared Afterimage store for ${modelId}? The shared Hugging Face cache will be kept.`)) return;
  button.disabled = true;
  try {
    await api(`/api/models/${modelId.split("/").map(encodeURIComponent).join("/")}?confirm_model_id=${encodeURIComponent(modelId)}`, { method: "DELETE" });
    toast(`${modelId} removed from Afterimage.`); await loadLibrary();
  } catch (error) { toast(error.message, "error"); button.disabled = false; }
}

async function unloadModel(modelId, button) {
  button.disabled = true;
  try {
    await api(`/api/models/${modelId.split("/").map(encodeURIComponent).join("/")}/unload`, { method: "POST" });
    toast(`${modelId} unloaded — VRAM and RAM freed.`); await loadLibrary();
  } catch (error) { toast(error.message, "error"); button.disabled = false; }
}

function watchAcquisition(jobId, modelId) {
  if (watchedJobs.has(jobId)) return;
  watchedJobs.add(jobId);
  watchJob(jobId, {
    onUpdate: (job) => {
      const index = state.jobs.findIndex((value) => value.id === jobId);
      if (index >= 0) state.jobs[index] = { ...state.jobs[index], ...job }; else state.jobs.unshift(job);
      renderActivity(); renderCatalog(); renderComputerResults();
    },
    onDone: async (job) => {
      watchedJobs.delete(jobId); await loadLibrary();
      if (job.status === "done") toast(`${modelId} is ${job.result?.state || "ready"}.`);
      else if (job.status === "cancelled") toast(`${modelId} stopped. Its partial download can be resumed later.`);
      else toast(job.error || `${modelId} ${job.status}.`, "error");
    },
  });
}

async function queueModel(path, body, modelId, button) {
  if (button) button.disabled = true;
  try {
    const result = await api(path, { method: "POST", body });
    toast(result.existing ? `${modelId} is already in progress.` : `Getting ${modelId}.`);
    await loadLibrary(); watchAcquisition(result.job_id, modelId);
  } catch (error) { toast(error.message, "error"); if (button) button.disabled = false; }
}

export function startGet(modelId, button = null) { return queueModel("/api/models/acquire", { model_id: modelId, prepare: true }, modelId, button); }
function importCached(modelId, button) { return queueModel("/api/models/import-cache", { model_id: modelId }, modelId, button); }

function renderActivity() {
  renderJobs($("jobs-list"), state.jobs.filter((job) => job.kind !== "chat"), { onChanged: loadLibrary, onRetry: (modelId) => startGet(modelId), limit: 5 });
}

export async function loadLibrary({ quiet = false } = {}) {
  try {
    const [models, jobs] = await Promise.all([api("/api/models"), api("/api/jobs")]);
    updateState({ models: models.models || [], jobs: jobs.jobs || [] });
    renderLibrary(); renderActivity(); renderCatalog(); renderComputerResults(); libraryChanged(state.models);
    for (const job of state.jobs.filter(isActiveJob)) if (job.kind === "acquire") watchAcquisition(job.id, job.model_id);
  } catch (error) { if (!quiet) toast(`Could not load the model library: ${error.message}`, "error"); }
}

export async function searchModels(page = 1, { focus = false } = {}) {
  const node = $("catalog-results");
  const computer = $("computer-results");
  node.classList.add("loading-line"); computer.classList.add("loading-line");
  node.innerHTML = ""; computer.innerHTML = ""; $("catalog-error").hidden = true;
  const query = $("catalog-query").value.trim();
  state.catalogQuery = query;
  showAllComputerResults = false;
  const parameters = new URLSearchParams({ q: query, page_size: compactViewport() ? "6" : "12", sort: $("catalog-sort").value, page: String(page) });
  const task = taskForFilter(state.catalogFilter);
  if (task) parameters.set("task", task);
  try {
    const [payload, local] = await Promise.all([api(`/api/catalog/models?${parameters}`), api(`/api/models/discover?q=${encodeURIComponent(query)}`)]);
    state.catalog = payload; state.localDiscovery = local;
    if (payload.error) { $("catalog-error").textContent = payload.error; $("catalog-error").hidden = false; }
    renderComputerResults(); renderCatalog();
    if (focus) document.querySelector(".search-results").scrollIntoView({ block: "start", behavior: "smooth" });
  } catch (error) {
    node.classList.remove("loading-line"); computer.classList.remove("loading-line"); node.innerHTML = '<div class="empty-inline">Online catalog unavailable.</div>';
    $("catalog-error").textContent = error.message; $("catalog-error").hidden = false;
  }
}

// Afterimage Fit badges need state.hardware.vram_total_gb, which loads via
// home.js's loadHome() -- run concurrently with (not before) this module's
// own loadLibrary()/searchModels() during startup (see app.js's
// initialize()), so the very first catalog/library render can happen
// before hardware data exists. Re-rendering from already-fetched state
// once every startup call has settled fixes that without an extra
// network round trip.
export function refreshFitBadges() { renderLibrary(); renderCatalog(); renderComputerResults(); }

export function initModels({ onLibraryChanged = () => {} } = {}) {
  libraryChanged = onLibraryChanged;
  $("catalog-search").addEventListener("submit", (event) => { event.preventDefault(); searchModels(1, { focus: true }); });
  $("catalog-sort").addEventListener("change", () => searchModels(1));
  $("catalog-prev").addEventListener("click", () => searchModels(Math.max(1, Number(state.catalog.page || 1) - 1), { focus: true }));
  $("catalog-next").addEventListener("click", () => searchModels(Number(state.catalog.page || 1) + 1, { focus: true }));
  $$("#catalog-filters button").forEach((button) => button.addEventListener("click", () => {
    if (state.catalogFilter === button.dataset.filter) return;
    state.catalogFilter = button.dataset.filter;
    $$("#catalog-filters button").forEach((value) => value.classList.toggle("is-active", value === button));
    // All/Text/Vision each search a different HF pipeline_tag server-side
    // (see taskForFilter) -- MoE re-fetches the same "text-generation"
    // task as Text and refines client-side, but going through
    // searchModels either way keeps one code path instead of two.
    searchModels(1);
  }));
  $("model-dialog").querySelector(".dialog-close").addEventListener("click", () => $("model-dialog").close());
}
