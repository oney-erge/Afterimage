import { state, updateState } from "./state.js";
import { $, $$, api, badge, compactNumber, esc, hfUrl, statusLabel, toast, watchJob } from "./shared.js";
import { isActiveJob, renderJobs } from "./jobs.js";

const watchedJobs = new Set();
let libraryChanged = () => {};

function modelCompatibility(model) {
  return model.metadata?.compatibility || {};
}

function activeFor(modelId) {
  return state.jobs.find((job) => job.model_id === modelId && isActiveJob(job));
}

function renderLibrary() {
  const node = $("local-models");
  node.classList.remove("loading-line");
  $("model-count").textContent = `${state.models.length} ${state.models.length === 1 ? "model" : "models"}`;
  if (!state.models.length) {
    node.innerHTML = '<div class="empty-inline">No local models yet. Search the catalog and choose Get.</div>';
    return;
  }
  node.innerHTML = state.models.map((model) => {
    const compatibility = modelCompatibility(model);
    const job = activeFor(model.model_id);
    const stateValue = job?.status || model.state;
    const source = model.metadata?.source_bytes ? `${(model.metadata.source_bytes / 1e9).toFixed(1)} GB source` : null;
    const prepared = model.comp_gb ? `${model.comp_gb.toFixed(1)} GB prepared` : null;
    const actions = [];
    if (model.state === "ready") actions.push(`<button class="button primary" data-chat-model="${encodeURIComponent(model.model_id)}">Chat</button>`);
    else if (!job) actions.push(`<button class="button primary" data-get-model="${encodeURIComponent(model.model_id)}">${model.state === "downloaded" ? "Prepare" : "Continue"}</button>`);
    actions.push(`<button class="button secondary" data-model-detail="${encodeURIComponent(model.model_id)}">Details</button>`);
    if (!job) actions.push(`<button class="text-button" data-remove-model="${encodeURIComponent(model.model_id)}">Remove</button>`);
    return `<article class="model-row">
      <div class="model-primary"><strong class="model-name" title="${esc(model.model_id)}">${esc(model.model_id)}</strong><div class="badge-row">${badge(stateValue)}${compatibility.execution ? badge(compatibility.execution) : ""}${compatibility.modality === "vision-text" ? badge("vision", "Vision") : ""}${compatibility.mixture_of_experts ? badge("moe", "MoE") : ""}</div></div>
      <div class="model-description">${esc([source, prepared].filter(Boolean).join(" · ") || model.error || statusLabel(model.stage || model.state))}</div>
      <div class="row-actions">${actions.join("")}</div>
    </article>`;
  }).join("");
  bindLibraryActions(node);
}

function bindLibraryActions(node) {
  node.querySelectorAll("[data-chat-model]").forEach((button) => button.addEventListener("click", () => {
    sessionStorage.setItem("afterimage.chatModel", decodeURIComponent(button.dataset.chatModel));
    location.hash = "chat";
  }));
  node.querySelectorAll("[data-get-model]").forEach((button) => button.addEventListener("click", () => startGet(decodeURIComponent(button.dataset.getModel), button)));
  node.querySelectorAll("[data-model-detail]").forEach((button) => button.addEventListener("click", () => openModelDialog(decodeURIComponent(button.dataset.modelDetail))));
  node.querySelectorAll("[data-remove-model]").forEach((button) => button.addEventListener("click", () => removeModel(decodeURIComponent(button.dataset.removeModel), button)));
}

function filteredCatalog() {
  const rows = state.catalog.models || [];
  if (state.catalogFilter === "vision") return rows.filter((row) => row.modality === "vision-text");
  if (state.catalogFilter === "moe") return rows.filter((row) => row.mixture_of_experts);
  if (state.catalogFilter === "text") return rows.filter((row) => row.modality !== "vision-text");
  return rows;
}

function renderCatalog() {
  const node = $("catalog-results");
  node.classList.remove("loading-line");
  const rows = filteredCatalog();
  $("catalog-page").textContent = `Page ${state.catalog.page || 1}`;
  $("catalog-prev").disabled = !state.catalog.previous_cursor;
  $("catalog-next").disabled = !state.catalog.next_cursor;
  const query = $("catalog-query").value.trim();
  $("catalog-heading").textContent = query ? `Results for “${query}”` : "Popular models";
  if (!rows.length) {
    node.innerHTML = `<div class="empty-inline">${state.catalogFilter === "all" ? "No models found." : "No models of this type are on the current page. Try the next page or All."}</div>`;
    return;
  }
  node.innerHTML = rows.map((model) => {
    const local = model.local;
    const running = activeFor(model.model_id);
    let action = `<button class="button primary" data-catalog-get="${encodeURIComponent(model.model_id)}">Get</button>`;
    if (local?.state === "ready") action = `<button class="button secondary" data-chat-model="${encodeURIComponent(model.model_id)}">Chat</button>`;
    else if (running) action = badge(running.status);
    const details = [
      model.params_b ? `${model.params_b}B parameters` : null,
      model.estimated_source_gb ? `about ${model.estimated_source_gb} GB source` : null,
      model.downloads != null ? `${compactNumber(model.downloads)} downloads` : null,
    ].filter(Boolean).join(" · ");
    return `<article class="catalog-row">
      <div class="model-primary"><a class="model-name" href="${hfUrl(model.model_id)}" target="_blank" rel="noreferrer">${esc(model.model_id)} ↗</a><div class="badge-row">${badge(model.execution)}${model.modality === "vision-text" ? badge("vision", "Vision") : badge("text", "Text")}${model.mixture_of_experts ? badge("moe", "MoE") : badge("dense", "Dense")}${model.gated ? badge("gated") : ""}</div></div>
      <div class="model-description">${esc(details || "Catalog metadata unavailable")}<br>${esc(model.execution_reason || "Compatibility has not been inspected locally.")}</div>
      <div class="row-actions">${action}<button class="button secondary" data-catalog-detail="${encodeURIComponent(model.model_id)}">Details</button></div>
    </article>`;
  }).join("");
  node.querySelectorAll("[data-catalog-get]").forEach((button) => button.addEventListener("click", () => startGet(decodeURIComponent(button.dataset.catalogGet), button)));
  node.querySelectorAll("[data-chat-model]").forEach((button) => button.addEventListener("click", () => {
    sessionStorage.setItem("afterimage.chatModel", decodeURIComponent(button.dataset.chatModel)); location.hash = "chat";
  }));
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
    toast(`${modelId} removed from the local library.`);
    await loadLibrary();
  } catch (error) { toast(error.message, "error"); button.disabled = false; }
}

function watchAcquisition(jobId, modelId) {
  if (watchedJobs.has(jobId)) return;
  watchedJobs.add(jobId);
  watchJob(jobId, {
    onUpdate: async (job) => {
      const index = state.jobs.findIndex((value) => value.id === jobId);
      if (index >= 0) state.jobs[index] = { ...state.jobs[index], ...job };
      else state.jobs.unshift(job);
      renderActivity();
    },
    onDone: async (job) => {
      watchedJobs.delete(jobId);
      await loadLibrary();
      if (job.status === "done") toast(`${modelId} is ${job.result?.state || "ready"}.`);
      else toast(job.error || `${modelId} ${job.status}.`, "error");
    },
  });
}

export async function startGet(modelId, button = null) {
  if (button) button.disabled = true;
  try {
    const result = await api("/api/models/acquire", { method: "POST", body: { model_id: modelId, prepare: true } });
    toast(result.existing ? `${modelId} is already in progress.` : `Getting ${modelId}.`);
    await loadLibrary();
    watchAcquisition(result.job_id, modelId);
  } catch (error) {
    toast(error.message, "error");
    if (button) button.disabled = false;
  }
}

function renderActivity() {
  renderJobs($("jobs-list"), state.jobs.filter((job) => job.kind !== "chat"), {
    onChanged: loadLibrary,
    onRetry: (modelId) => startGet(modelId),
  });
}

async function refreshModelsOnly() {
  const payload = await api("/api/models");
  state.models = payload.models || [];
  renderLibrary(); renderCatalog(); libraryChanged(state.models);
}

export async function loadLibrary({ quiet = false } = {}) {
  try {
    const [models, jobs] = await Promise.all([api("/api/models"), api("/api/jobs")]);
    updateState({ models: models.models || [], jobs: jobs.jobs || [] });
    renderLibrary(); renderActivity(); renderCatalog(); libraryChanged(state.models);
    for (const job of state.jobs.filter(isActiveJob)) {
      if (job.kind === "acquire") watchAcquisition(job.id, job.model_id);
    }
  } catch (error) {
    if (!quiet) toast(`Could not load the model library: ${error.message}`, "error");
  }
}

export async function searchModels(cursor = null) {
  const node = $("catalog-results");
  node.classList.add("loading-line"); node.innerHTML = "";
  $("catalog-error").hidden = true;
  const parameters = new URLSearchParams({
    q: $("catalog-query").value.trim(), page_size: "24", sort: $("catalog-sort").value,
  });
  if (cursor) parameters.set("cursor", cursor);
  try {
    const payload = await api(`/api/catalog/models?${parameters}`);
    state.catalog = payload;
    if (payload.error) {
      $("catalog-error").textContent = payload.error; $("catalog-error").hidden = false;
    }
    renderCatalog();
  } catch (error) {
    node.classList.remove("loading-line"); node.innerHTML = '<div class="empty-inline">Catalog unavailable.</div>';
    $("catalog-error").textContent = error.message; $("catalog-error").hidden = false;
  }
}

export function initModels({ onLibraryChanged = () => {} } = {}) {
  libraryChanged = onLibraryChanged;
  $("catalog-search").addEventListener("submit", (event) => { event.preventDefault(); searchModels(); });
  $("catalog-sort").addEventListener("change", () => searchModels());
  $("catalog-prev").addEventListener("click", () => searchModels(state.catalog.previous_cursor));
  $("catalog-next").addEventListener("click", () => searchModels(state.catalog.next_cursor));
  $$("#catalog-filters button").forEach((button) => button.addEventListener("click", () => {
    state.catalogFilter = button.dataset.filter;
    $$("#catalog-filters button").forEach((value) => value.classList.toggle("is-active", value === button));
    renderCatalog();
  }));
  $("model-dialog").querySelector(".dialog-close").addEventListener("click", () => $("model-dialog").close());
}
