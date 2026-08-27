export const state = {
  route: "home",
  hardware: null,
  capability: null,
  models: [],
  jobs: [],
  catalog: { models: [], page: 1, cursor: null, next_cursor: null, previous_cursor: null },
  catalogQuery: "",
  // "text" (not "all") is the default so the catalog lands on runnable
  // chat/instruct models -- an unfiltered HF search sorted by downloads
  // is dominated by embedding/encoder checkpoints (all-MiniLM,
  // bert-base, ...) that Afterimage can only download, never run, which
  // made a first-time user's landing view of the catalog look empty of
  // anything usable. "All" stays one click away for deliberate browsing.
  catalogFilter: "text",
  localDiscovery: { models: [], sources: {} },
  campaign: { lengths: [] },
  chat: { messages: [], attachments: [], activeJob: null },
  experiments: [],
  researchRuns: [],
  runtimeProfiles: [],
};

const listeners = new Set();

export function updateState(values) {
  Object.assign(state, values);
  for (const listener of listeners) listener(state);
}

export function notify() {
  for (const listener of listeners) listener(state);
}

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
