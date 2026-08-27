export const $ = (id) => document.getElementById(id);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export async function api(path, options = {}) {
  const request = { ...options, headers: { ...(options.headers || {}) } };
  if (request.body && typeof request.body !== "string") {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(request.body);
  }
  const response = await fetch(path, request);
  let payload = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) payload = await response.json();
  else payload = await response.text();
  if (!response.ok) {
    const detail = payload?.detail || payload?.error || payload || `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

export function esc(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

export function fmtGib(value, fallback = "Unavailable") {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(Number(value) >= 100 ? 0 : 1)} GB` : fallback;
}

export function fmtBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return "Size unavailable";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let current = Math.max(0, bytes);
  let unit = 0;
  while (current >= 1000 && unit < units.length - 1) { current /= 1000; unit += 1; }
  return `${current.toFixed(unit > 1 ? 1 : 0)} ${units[unit]}`;
}

export function compactNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(number);
}

export function statusLabel(value) {
  return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function badge(value, label = null) {
  const safeValue = String(value || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "-");
  return `<span class="badge ${safeValue}">${esc(label || statusLabel(value))}</span>`;
}

// The Afterimage Fit vocabulary: not fitting in VRAM is the entire reason
// Afterimage exists, so "does this run" needs its own words instead of
// borrowing generic compatibility language. Thresholds are relative to
// this machine's own GPU memory, computed from the same
// estimated_store_gb the catalog and local library already return -- no
// separate backend call, this is real data, just not yet phrased for a
// beginner. A dedicated /api/recommendations (a real per-machine
// suggestion engine, not just this per-model classification) is future
// work, not something this function pretends to be.
export const FIT_LABELS = {
  native: "Native",
  "afterimage-ready": "Afterimage Ready",
  "storage-bound": "Storage Bound",
  extreme: "Extreme",
};

export function afterimageFit(storeGb, vramTotalGb) {
  const store = Number(storeGb);
  const vram = Number(vramTotalGb);
  if (!Number.isFinite(store) || !Number.isFinite(vram) || vram <= 0 || store <= 0) return null;
  const ratio = store / vram;
  if (ratio <= 0.9) return "native";
  if (ratio <= 3) return "afterimage-ready";
  if (ratio <= 8) return "storage-bound";
  return "extreme";
}

export function fitBadge(fitClass) {
  if (!fitClass) return "";
  return `<span class="badge fit-${esc(fitClass)}">${esc(FIT_LABELS[fitClass] || statusLabel(fitClass))}</span>`;
}

// Support confidence is a separate axis from Fit: "will this run well on
// your machine" (Fit) is not the same claim as "has this architecture
// been validated at all" (confidence, from classify_config's own
// execution field -- see afterimage/runtime/adapters.py). Mixing them
// into one badge would blur "this will be slow" with "this hasn't been
// checked", which are different things a user needs to weigh differently.
const CONFIDENCE_LABELS = {
  verified: "✓ Verified", expected: "◐ Expected", experimental: "⚠ Experimental",
};

export function confidenceBadge(execution) {
  const label = CONFIDENCE_LABELS[execution];
  if (!label) return "";
  return `<span class="badge ${esc(execution)}">${esc(label)}</span>`;
}

export function hfUrl(modelId) {
  return `https://huggingface.co/${String(modelId).split("/").map(encodeURIComponent).join("/")}`;
}

export async function copyText(value) {
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(value); return; } catch (_) { /* fallback */ }
  }
  const area = document.createElement("textarea");
  area.value = value; area.setAttribute("readonly", "");
  area.style.position = "fixed"; area.style.opacity = "0";
  document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove();
}

let toastTimer;
export function toast(message, kind = "info") {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show${kind === "error" ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.className = "toast"; }, 4200);
}

export function progressValue(progress = {}) {
  const direct = Number(progress.progress);
  if (Number.isFinite(direct)) return Math.max(0, Math.min(1, direct));
  const done = Number(progress.bytes_done ?? progress.completed);
  const total = Number(progress.bytes_total ?? progress.total);
  return Number.isFinite(done) && Number.isFinite(total) && total > 0 ? Math.max(0, Math.min(1, done / total)) : 0;
}

export function watchJob(jobId, { onUpdate = () => {}, onDone = () => {} } = {}) {
  let stopped = false;
  let socket;
  let pollTimer;
  const terminal = new Set(["done", "error", "cancelled", "interrupted"]);

  async function finish(snapshot) {
    if (stopped) return;
    stopped = true;
    if (socket && socket.readyState < 2) socket.close();
    clearTimeout(pollTimer);
    try {
      const full = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      onUpdate(full);
      onDone(full);
    } catch (error) {
      onDone({ id: jobId, ...snapshot, error: error.message });
    }
  }

  function accept(snapshot) {
    if (stopped) return;
    const value = { id: jobId, ...snapshot };
    onUpdate(value);
    if (terminal.has(value.status)) finish(value);
  }

  async function poll() {
    if (stopped) return;
    try { accept(await api(`/api/jobs/${encodeURIComponent(jobId)}`)); }
    catch (error) { onUpdate({ id: jobId, status: "error", error: error.message }); }
    if (!stopped) pollTimer = setTimeout(poll, 1000);
  }

  try {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${scheme}://${location.host}/ws/jobs/${encodeURIComponent(jobId)}`);
    socket.onmessage = (event) => accept(JSON.parse(event.data));
    socket.onerror = () => { if (!pollTimer && !stopped) poll(); };
    socket.onclose = () => { if (!stopped && !pollTimer) poll(); };
  } catch (_) { poll(); }

  return () => {
    stopped = true;
    clearTimeout(pollTimer);
    if (socket && socket.readyState < 2) socket.close();
  };
}
