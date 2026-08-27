import { state, updateState } from "./state.js";
import { $, api, badge, esc, fmtGib, toast } from "./shared.js";

function metric(icon, label, value, note) {
  return `<article class="metric-card"><span class="metric-icon" aria-hidden="true">${icon}</span><strong title="${esc(value)}">${esc(value)}</strong><small>${esc(label)}${note ? ` · ${esc(note)}` : ""}</small></article>`;
}

function renderHardware() {
  const hardware = state.hardware;
  const grid = $("hardware-grid");
  if (!hardware) return;
  const gpu = hardware.gpu || {};
  grid.innerHTML = [
    metric("G", "GPU", gpu.name || "GPU unavailable", gpu.vendor && gpu.vendor !== "none" ? gpu.vendor.toUpperCase() : "Not detected"),
    metric("V", "GPU memory", fmtGib(hardware.vram_total_gb), hardware.vram_free_gb != null ? `${fmtGib(hardware.vram_free_gb)} free` : "Availability unavailable"),
    metric("R", "System RAM", fmtGib(hardware.memory?.total_gib), hardware.memory?.available_gib != null ? `${fmtGib(hardware.memory.available_gib)} available` : "Availability unavailable"),
    metric("D", "Model storage", fmtGib(hardware.disk?.free_gib), hardware.disk?.total_gib != null ? `${fmtGib(hardware.disk.total_gib)} total` : hardware.disk?.path || "Capacity unavailable"),
  ].join("");
  $("hardware-note").textContent = hardware.disk?.error || "Storage is reported for the filesystem that holds Afterimage model stores.";
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
  const speed = reference.params_b * reference.fast_s_per_token_per_b;
  node.innerHTML = `<div class="reference-model"><strong>${esc(reference.model)}</strong><span>RTX 3080 Laptop · cold cache</span></div><div class="reference-stats"><div><strong>${bf16.toFixed(1)} GB</strong><small>Original BF16</small></div><div><strong>${store.toFixed(1)} GB</strong><small>Lossless store</small></div><div><strong>${speed.toFixed(2)} s/token</strong><small>Measured faster profile</small></div></div>`;
}

function renderModels() {
  const models = state.models.filter((model) => model.state === "ready").slice(0, 4);
  const node = $("home-models");
  node.classList.remove("loading-line");
  if (!models.length) {
    node.innerHTML = '<div class="empty-inline">No models are ready yet. Browse the catalog to get one.</div>';
    return;
  }
  node.innerHTML = models.map((model) => `<div class="compact-model"><div><strong title="${esc(model.model_id)}">${esc(model.model_id)}</strong><div class="model-meta">${badge("ready")}${model.comp_gb ? `${model.comp_gb.toFixed(1)} GB` : ""}</div></div><button class="text-button" data-home-chat="${encodeURIComponent(model.model_id)}">Chat →</button></div>`).join("");
  node.querySelectorAll("[data-home-chat]").forEach((button) => button.addEventListener("click", () => {
    sessionStorage.setItem("afterimage.chatModel", decodeURIComponent(button.dataset.homeChat));
    location.hash = "chat";
  }));
}

export async function loadHome({ quiet = false } = {}) {
  try {
    const [hardware, capability, models] = await Promise.all([
      api("/api/hardware"), api("/api/capability"), api("/api/models"),
    ]);
    updateState({ hardware, capability, models: models.models || [] });
    renderHardware(); renderReference(); renderModels();
  } catch (error) {
    if (!quiet) toast(`Could not load system status: ${error.message}`, "error");
    $("hardware-grid").innerHTML = '<div class="empty-inline">Hardware details are unavailable.</div>';
  }
}

export function initHome() {
  $("refresh-hardware").addEventListener("click", () => loadHome());
}

export function refreshHomeModels() { renderModels(); }
