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

export function hfUrl(modelId) {
  return `https://huggingface.co/${String(modelId).split("/").map(encodeURIComponent).join("/")}`;
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
