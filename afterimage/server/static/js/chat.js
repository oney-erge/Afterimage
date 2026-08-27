import { state } from "./state.js";
import { $, api, badge, copyText, esc, statusLabel, toast, watchJob } from "./shared.js";

let stopWatching = null;

function readyModels() { return state.models.filter((model) => model.state === "ready"); }

function selectedModel() {
  return readyModels().find((model) => model.model_id === $("chat-model").value) || null;
}

function capability(model) { return model?.metadata?.compatibility || {}; }

function isVision(model) { return capability(model).modality === "vision-text"; }

export function updateChatModels() {
  const select = $("chat-model");
  const previous = select.value || sessionStorage.getItem("afterimage.chatModel");
  const models = readyModels();
  select.innerHTML = models.length
    ? models.map((model) => `<option value="${esc(model.model_id)}">${esc(model.model_id)}</option>`).join("")
    : '<option value="">No ready models</option>';
  if (models.some((model) => model.model_id === previous)) select.value = previous;
  else if (models.length) select.value = models[0].model_id;
  sessionStorage.removeItem("afterimage.chatModel");
  updateProfileOptions();
  updateModelContext();
}

function updateProfileOptions() {
  const select = $("chat-profile");
  const requested = sessionStorage.getItem("afterimage.chatProfile") || select.value || "auto";
  const modelId = $("chat-model").value;
  const saved = state.runtimeProfiles.filter((profile) => profile.model_id === modelId);
  select.innerHTML = `<option value="auto">Auto</option><option value="fast">Faster</option><option value="balanced">Balanced</option><option value="min-memory">Lowest memory</option>${saved.length ? `<optgroup label="Research profiles">${saved.map((profile) => `<option value="saved:${esc(profile.id)}">${esc(profile.name)}</option>`).join("")}</optgroup>` : ""}`;
  if ([...select.options].some((option) => option.value === requested)) select.value = requested;
  sessionStorage.removeItem("afterimage.chatProfile");
}

function updateModelContext() {
  const model = selectedModel();
  const attachmentButton = $("attach-button");
  if (!model) {
    $("chat-model-context").innerHTML = `${badge("unknown", "No model ready")} <button class="text-button" data-chat-find>Find a model</button>`;
    attachmentButton.disabled = true;
    $("chat-send").disabled = true;
    $("chat-endpoint").hidden = true;
    $("chat-model-context").querySelector("[data-chat-find]").addEventListener("click", () => { location.hash = "models"; });
    return;
  }
  const meta = capability(model);
  const pieces = [badge("ready"), meta.execution ? badge(meta.execution) : "", isVision(model) ? badge("vision", "Images enabled") : badge("text", "Text only")];
  $("chat-model-context").innerHTML = `${pieces.join(" ")} <span>${esc(model.comp_gb ? `${model.comp_gb.toFixed(1)} GB lossless store` : "Prepared locally")}</span>`;
  const selectedProfile = $("chat-profile").value;
  const saved = selectedProfile.startsWith("saved:") ? state.runtimeProfiles.find((profile) => `saved:${profile.id}` === selectedProfile) : null;
  $("chat-endpoint").hidden = false;
  $("chat-endpoint").innerHTML = `<div><span>OpenAI-compatible endpoint</span><code>http://127.0.0.1:8420/v1/chat/completions</code>${saved ? `<small>Using ${esc(saved.name)}</small>` : ""}</div><button class="text-button" data-copy-chat-endpoint>Copy request</button>`;
  $("chat-endpoint").querySelector("[data-copy-chat-endpoint]").addEventListener("click", async () => {
    const payload = { model: model.model_id, messages: [{ role: "user", content: "Hello" }], stream: true, ...(saved ? { runtime_profile_id: saved.id } : { execution_profile: selectedProfile }) };
    await copyText(JSON.stringify(payload, null, 2)); toast("API request copied.");
  });
  attachmentButton.disabled = !isVision(model) || Boolean(state.chat.activeJob);
  attachmentButton.title = isVision(model) ? "Attach images" : "The selected model is text only";
  $("chat-send").disabled = Boolean(state.chat.activeJob);
  if (!isVision(model) && state.chat.attachments.length) {
    state.chat.attachments = [];
    renderAttachments();
  }
}

function renderAttachments() {
  const tray = $("attachment-tray");
  tray.hidden = !state.chat.attachments.length;
  tray.innerHTML = state.chat.attachments.map((image, index) => `<div class="attachment"><img src="${image.url}" alt="${esc(image.name)}"><button type="button" data-remove-attachment="${index}" aria-label="Remove ${esc(image.name)}">×</button></div>`).join("");
  tray.querySelectorAll("[data-remove-attachment]").forEach((button) => button.addEventListener("click", () => {
    state.chat.attachments.splice(Number(button.dataset.removeAttachment), 1); renderAttachments();
  }));
}

function renderConversation() {
  const node = $("conversation");
  if (!state.chat.messages.length) {
    node.innerHTML = '<div class="empty-state" id="chat-empty"><div class="empty-orb"><span></span></div><h2>Start a local conversation</h2><p>Select a ready model, then ask a question. Generation and cancellation run through the same model job system.</p><button class="button secondary" data-chat-find>Find a model</button></div>';
    node.querySelector("[data-chat-find]").addEventListener("click", () => { location.hash = "models"; });
    return;
  }
  node.innerHTML = state.chat.messages.map((message) => `<article class="message ${message.role}"><div class="message-label"><i></i>${message.role === "assistant" ? "Afterimage" : "You"}</div><div class="message-body${message.pending ? " typing-cursor" : ""}">${esc(message.text)}</div>${message.images?.length ? `<div class="message-images">${message.images.map((image) => `<img src="${image.url}" alt="${esc(image.name)}">`).join("")}</div>` : ""}${message.error ? `<div class="notice error">${esc(message.error)}</div>` : ""}</article>`).join("");
  node.scrollTop = node.scrollHeight;
}

function apiMessages() {
  return state.chat.messages.filter((message) => !message.pending && !message.error).map((message) => {
    if (message.role !== "user" || !message.images?.length) return { role: message.role, content: message.text };
    return {
      role: "user",
      content: [
        { type: "text", text: message.text },
        ...message.images.map((image) => ({ type: "image_url", image_url: { url: image.url } })),
      ],
    };
  });
}

function setRunning(running) {
  $("chat-stop").disabled = false;
  $("chat-stop").hidden = !running;
  $("chat-send").hidden = running;
  $("chat-input").disabled = running;
  $("chat-model").disabled = running;
  $("chat-profile").disabled = running;
  updateModelContext();
}

async function submitChat() {
  const model = selectedModel();
  const text = $("chat-input").value.trim();
  if (!model) { toast("Get and prepare a model before chatting.", "error"); return; }
  if (!text && !state.chat.attachments.length) return;
  const images = state.chat.attachments.map((image) => ({ ...image }));
  state.chat.messages.push({ role: "user", text, images });
  state.chat.attachments = [];
  $("chat-input").value = "";
  resizeInput(); renderAttachments();
  state.chat.messages.push({ role: "assistant", text: "", pending: true });
  renderConversation();
  const body = {
    model: model.model_id,
    messages: apiMessages(),
    max_tokens: Number($("chat-tokens").value) || 128,
    stream: false,
  };
  const profile = $("chat-profile").value;
  if (profile.startsWith("saved:")) body.runtime_profile_id = profile.slice(6);
  else body.execution_profile = profile;
  const vram = Number($("chat-vram").value);
  const ram = Number($("chat-ram").value);
  const draft = $("chat-draft").value.trim();
  if (vram > 0) body.vram_budget_gb = vram;
  if (ram > 0) body.ram_budget_gb = ram;
  if (draft) body.draft_model = draft;
  try {
    const result = await api("/api/chat", { method: "POST", body });
    state.chat.activeJob = result.job_id;
    setRunning(true);
    stopWatching = watchJob(result.job_id, {
      onUpdate: (job) => {
        const message = state.chat.messages.at(-1);
        if (message?.pending && job.progress?.text != null) message.text = job.progress.text;
        renderConversation();
      },
      onDone: (job) => {
        const message = state.chat.messages.at(-1);
        if (message?.pending) {
          message.pending = false;
          if (job.status === "done") message.text = job.result?.text ?? message.text;
          else {
            message.error = job.error || (job.status === "cancelled" ? "Generation stopped." : `Generation ${job.status}.`);
            if (!message.text) message.text = "No response was completed.";
          }
        }
        state.chat.activeJob = null;
        stopWatching = null;
        setRunning(false); renderConversation();
      },
    });
  } catch (error) {
    const message = state.chat.messages.at(-1);
    message.pending = false; message.error = error.message; message.text = "The request could not start.";
    state.chat.activeJob = null; setRunning(false); renderConversation();
  }
}

async function stopChat() {
  if (!state.chat.activeJob) return;
  $("chat-stop").disabled = true;
  try { await api(`/api/jobs/${encodeURIComponent(state.chat.activeJob)}/cancel`, { method: "POST" }); }
  catch (error) { toast(error.message, "error"); $("chat-stop").disabled = false; }
}

function readImages(files) {
  const available = 4 - state.chat.attachments.length;
  [...files].slice(0, available).forEach((file) => {
    if (!file.type.startsWith("image/")) return;
    if (file.size > 10 * 1024 * 1024) { toast(`${file.name} is larger than 10 MiB.`, "error"); return; }
    const reader = new FileReader();
    reader.onload = () => { state.chat.attachments.push({ name: file.name, url: reader.result }); renderAttachments(); };
    reader.readAsDataURL(file);
  });
  if (files.length > available) toast("A message can contain at most 4 images.", "error");
}

function resizeInput() {
  const input = $("chat-input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

export function initChat() {
  $("chat-model").addEventListener("change", () => { updateProfileOptions(); updateModelContext(); });
  $("chat-profile").addEventListener("change", updateModelContext);
  $("attach-button").addEventListener("click", () => $("chat-images").click());
  $("chat-images").addEventListener("change", (event) => { readImages(event.target.files); event.target.value = ""; });
  $("chat-form").addEventListener("submit", (event) => { event.preventDefault(); submitChat(); });
  $("chat-stop").addEventListener("click", stopChat);
  $("chat-input").addEventListener("input", resizeInput);
  $("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitChat(); }
  });
  updateChatModels(); renderConversation();
}
