import { api, badge, esc, fmtBytes, progressValue, statusLabel, toast } from "./shared.js";

const ACTIVE = new Set(["queued", "running", "pause_requested", "paused", "cancelling"]);

export function isActiveJob(job) { return ACTIVE.has(job?.status); }

function progressCopy(job) {
  const progress = job.progress || {};
  const value = progressValue(progress);
  const message = progress.message || progress.stage || statusLabel(job.status);
  let amount = `${Math.round(value * 100)}%`;
  if (progress.bytes_total) amount = `${fmtBytes(progress.bytes_done || 0)} / ${fmtBytes(progress.bytes_total)}`;
  else if (progress.completed != null && progress.total != null) amount = `${progress.completed} / ${progress.total}`;
  return { value, message, amount };
}

export function renderJobs(container, jobs, { onChanged = () => {}, onRetry = null, limit = 8 } = {}) {
  const visible = [...jobs]
    .sort((a, b) => Number(b.created_at || 0) - Number(a.created_at || 0))
    .slice(0, limit);
  container.classList.remove("loading-line");
  if (!visible.length) {
    container.innerHTML = '<div class="empty-inline">No active or recent work.</div>';
    return;
  }
  container.innerHTML = visible.map((job) => {
    const progress = progressCopy(job);
    const encoded = encodeURIComponent(job.id);
    const controls = [];
    if (["queued", "running"].includes(job.status)) controls.push(`<button class="button secondary" data-job-action="pause" data-job="${encoded}">Pause</button>`);
    if (["paused", "pause_requested"].includes(job.status)) controls.push(`<button class="button secondary" data-job-action="resume" data-job="${encoded}">Resume</button>`);
    if (isActiveJob(job)) controls.push(`<button class="button danger" data-job-action="cancel" data-job="${encoded}">Cancel</button>`);
    if (job.status === "interrupted" && job.kind === "acquire" && onRetry) controls.push(`<button class="button secondary" data-job-action="retry" data-model="${encodeURIComponent(job.model_id || "")}">Retry</button>`);
    if (!isActiveJob(job)) controls.push(`<button class="text-button" data-job-action="delete" data-job="${encoded}">Delete</button>`);
    return `<article class="job-row">
      <div class="job-title"><strong>${esc(job.model_id || job.kind)}</strong><small>${esc(job.kind)}</small></div>
      <div class="job-progress"><div class="progress-copy"><span>${esc(progress.message)}</span><span>${esc(progress.amount)}</span></div><div class="progress-track"><span style="width:${Math.round(progress.value * 100)}%"></span></div>${job.error ? `<small class="progress-copy">${esc(job.error)}</small>` : ""}</div>
      <div class="row-actions">${badge(job.status)}${controls.join("")}</div>
    </article>`;
  }).join("");

  container.querySelectorAll("[data-job-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      const action = button.dataset.jobAction;
      if (action === "retry") {
        onRetry(decodeURIComponent(button.dataset.model));
        return;
      }
      button.disabled = true;
      try {
        const jobId = decodeURIComponent(button.dataset.job);
        if (action === "delete") await api(`/api/jobs/${jobId}`, { method: "DELETE" });
        else await api(`/api/jobs/${jobId}/${action}`, { method: "POST" });
        await onChanged();
      } catch (error) {
        toast(error.message, "error");
        button.disabled = false;
      }
    });
  });
}
