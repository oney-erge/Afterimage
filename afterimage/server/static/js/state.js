export const state = {
  route: "home",
  hardware: null,
  capability: null,
  models: [],
  jobs: [],
  catalog: { models: [], page: 1, cursor: null, next_cursor: null, previous_cursor: null },
  catalogQuery: "",
  catalogFilter: "all",
  localDiscovery: { models: [], sources: {} },
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
