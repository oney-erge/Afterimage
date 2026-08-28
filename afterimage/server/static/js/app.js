import { state } from "./state.js";
import { $, $$, api } from "./shared.js";
import { initHome, loadHome, refreshHomeModels } from "./home.js";
import { initModels, loadLibrary, refreshFitBadges, searchModels } from "./models.js";
import { initChat, updateChatModels } from "./chat.js";
import { initResearch, loadResearch, updateResearchModels } from "./research.js";

const ROUTES = new Set(["home", "models", "chat", "research"]);

function routeFromHash() {
  const value = location.hash.replace(/^#/, "");
  return ROUTES.has(value) ? value : "home";
}

function showRoute(route) {
  state.route = ROUTES.has(route) ? route : "home";
  $$("[data-page]").forEach((page) => page.classList.toggle("is-active", page.dataset.page === state.route));
  $$("[data-route]").forEach((button) => button.classList.toggle("is-active", button.dataset.route === state.route));
  document.body.classList.remove("menu-open");
  $("mobile-menu").setAttribute("aria-expanded", "false");
  if (location.hash !== `#${state.route}`) history.replaceState(null, "", `#${state.route}`);
  if (state.route === "chat") updateChatModels();
  if (state.route === "research") updateResearchModels();
  window.scrollTo({ top: 0, behavior: "instant" });
}

async function checkRuntime() {
  try {
    const [health, version] = await Promise.all([api("/health"), api("/api/version")]);
    $("health-dot").className = "ok";
    // A model load can take minutes (streaming a compressed store off
    // disk) with nothing else in the UI showing progress during it --
    // this poll is what fills that gap.
    if (health.loading_model) $("health-label").textContent = `Loading ${health.loading_model}…`;
    else $("health-label").textContent = health.model_loaded ? "Runtime ready · model loaded" : "Runtime ready";
    $("version-label").textContent = `Afterimage ${version.version}`;
  } catch (_) {
    $("health-dot").className = "bad";
    $("health-label").textContent = "Runtime unavailable";
  }
}

function libraryChanged() {
  updateChatModels(); updateResearchModels(); refreshHomeModels();
}

function bindNavigation() {
  const navigate = (route) => { location.hash = route; showRoute(route); };
  $$("[data-route]").forEach((button) => button.addEventListener("click", () => navigate(button.dataset.route)));
  $$("[data-route-link]").forEach((link) => link.addEventListener("click", (event) => { event.preventDefault(); navigate(link.dataset.routeLink); }));
  // Delegated at the document level, not bound per-button at startup:
  // [data-go] buttons are routinely injected later by innerHTML (the
  // empty-library hero's "Browse models" CTA, chat's "Find a model"
  // empty state, ...), and a one-time forEach here at bindNavigation()
  // time never sees those -- they rendered with no listener at all.
  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-go]");
    if (button) navigate(button.dataset.go);
  });
  $("mobile-menu").addEventListener("click", () => {
    const open = document.body.classList.toggle("menu-open");
    $("mobile-menu").setAttribute("aria-expanded", String(open));
  });
  window.addEventListener("hashchange", () => showRoute(routeFromHash()));
}

async function initialize() {
  bindNavigation();
  initHome();
  initModels({ onLibraryChanged: libraryChanged });
  initChat();
  initResearch();
  showRoute(routeFromHash());
  checkRuntime();
  await Promise.allSettled([loadHome({ quiet: true }), loadLibrary({ quiet: true }), loadResearch({ quiet: true }), searchModels()]);
  libraryChanged();
  refreshFitBadges();
  setInterval(() => {
    checkRuntime();
    if (state.jobs.some((job) => ["queued", "running", "pause_requested", "paused", "cancelling"].includes(job.status))) loadLibrary({ quiet: true });
  }, 5000);
}

initialize();
