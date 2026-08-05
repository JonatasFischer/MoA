const state = {
  config: null,
  generation: 0,
  persisted: false,
  selectedFlow: null,
  selectedTarget: null,
  dirty: false,
  modelOptions: {},
  simulation: {
    active: false,
    controller: null,
    profile: null,
    requestId: null,
    events: [],
    nodeStates: {},
    message: "",
  },
};

const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const clone = (value) => JSON.parse(JSON.stringify(value));

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.className = "toast", error ? 6500 : 2600);
}

function markDirty() {
  state.dirty = true;
  $("#apply-button").disabled = false;
  $("#dirty-label").textContent = "UNAPPLIED";
}

function targetTemplate(provider = "") {
  return { provider, model: "", role: "general" };
}

function currentProfile() {
  return state.config?.profiles[state.selectedFlow];
}

function allTargets(profile) {
  if (profile.strategy === "direct") return [{ provider: profile.provider, model: profile.model }];
  return [...(profile.proposers || []), ...(profile.contributors || []), profile.aggregator, profile.tool_dispatch].filter(Boolean);
}

function flowGlyph(strategy) {
  return strategy === "direct" ? "1>1" : strategy === "classic" ? "N>1" : "C>1";
}

function render() {
  if (!state.config) return;
  renderFlows();
  renderCanvas();
  renderInspector();
  appendRunOutput();
  renderRuntime();
  renderSimulationStatus();
}

function renderRuntime() {
  $("#runtime-label").textContent = `Generation ${state.generation} / ${state.persisted ? "saved to YAML" : "runtime only"}`;
  $("#apply-button").disabled = !state.dirty;
  $("#dirty-label").textContent = state.dirty ? "UNAPPLIED" : "";
}

function renderFlows() {
  const list = $("#flow-list");
  list.innerHTML = Object.entries(state.config.profiles).map(([name, profile]) => `
    <button class="flow-card ${name === state.selectedFlow ? "active" : ""}" data-flow="${esc(name)}" type="button" ${state.simulation.active ? "disabled" : ""}>
      <span class="flow-glyph">${flowGlyph(profile.strategy)}</span>
      <span class="flow-name"><strong>${esc(name)}</strong><small>${esc(profile.strategy)}</small></span>
      ${name === state.config.default_profile ? '<span class="default-tag">LIVE</span>' : ""}
    </button>`).join("");
  list.querySelectorAll("[data-flow]").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.flow !== state.selectedFlow) {
      state.simulation.events = [];
      state.simulation.nodeStates = {};
      state.simulation.message = "";
      state.simulation.profile = null;
    }
    state.selectedFlow = button.dataset.flow;
    state.selectedTarget = null;
    render();
  }));
}

function nodeHtml(target, kind, locator, index = "") {
  if (!target) return "";
  const selected = state.selectedTarget && state.selectedTarget.locator === locator && String(state.selectedTarget.index ?? "") === String(index);
  const runKey = `target:${locator}:${index}`;
  const run = state.simulation.nodeStates[runKey];
  return `<button class="node ${kind} ${selected ? "selected" : ""} ${run ? `run-${run.status}` : ""}" data-target="${locator}" data-index="${index}" data-run-key="${runKey}" type="button">
    ${run ? `<i class="node-run-indicator" title="${esc(run.status)}"></i>` : ""}
    <span class="node-type">${kind}</span><strong>${esc(target.model || "Choose model")}</strong>
    <small>${esc(target.provider || "No provider")}${target.family ? ` / ${esc(target.family)}` : ""}</small>
  </button>`;
}

function systemNodeHtml(id, title, detail, kind = "system", status = "always") {
  const selected = state.selectedTarget?.locator === "system" && state.selectedTarget.id === id;
  const runKey = `system:${id}`;
  const run = state.simulation.nodeStates[runKey];
  return `<button class="node ${kind} ${selected ? "selected" : ""} ${run ? `run-${run.status}` : ""}" data-system="${id}" data-run-key="${runKey}" type="button">
    ${run ? `<i class="node-run-indicator" title="${esc(run.status)}"></i>` : ""}
    <span class="node-type">${esc(status)}</span><strong>${esc(title)}</strong><small>${esc(detail)}</small>
  </button>`;
}

function renderCanvas() {
  const profile = currentProfile();
  const canvas = $("#flow-canvas");
  const canvasWrap = canvas.parentElement;
  const scrollLeft = canvasWrap.scrollLeft;
  const scrollTop = canvasWrap.scrollTop;
  if (!profile) {
    $("#flow-title").textContent = "Select a flow";
    canvas.innerHTML = '<div class="empty-node">Create a flow to begin an experiment.</div>';
    return;
  }
  $("#flow-title").textContent = state.selectedFlow;
  $("#flow-strategy").textContent = `${profile.strategy.toUpperCase()} ACTUAL REQUEST PATH`;
  $("#duplicate-flow").disabled = false;
  $("#delete-flow").disabled = Object.keys(state.config.profiles).length <= 1;

  let mainPath = "";
  let branches = "";
  if (profile.strategy === "direct") {
    mainPath = `
      <div class="stage"><span class="stage-label">1. Route</span>${systemNodeHtml("routing", "Profile resolution", "Direct strategy selected")}</div>
      <div class="stage"><span class="stage-label">2. Execute</span>${nodeHtml(profile, "aggregator", "profile")}</div>
      <div class="stage"><span class="stage-label">3. Return</span>${systemNodeHtml("response", "Client response", "No filter, quorum, or enforcement", "response")}</div>`;
  } else {
    const collection = profile.strategy === "classic" ? "proposers" : "contributors";
    const label = profile.strategy === "classic" ? "Proposers" : "Council contributors";
    const targets = profile[collection] || [];
    const aggregator = profile.aggregator || {};
    const enforcement = state.config.tool_enforcement || {};
    const enforcementDetail = enforcement.enabled
      ? `${enforcement.enforcement_mode} / ${(enforcement.required_tools || []).join(", ")}`
      : "Disabled";
    mainPath = `
      <div class="stage"><span class="stage-label">1. Route</span>${systemNodeHtml("routing", "Request routing", "Initial MoA turn")}</div>
      <div class="stage"><span class="stage-label">2. Preflight</span>${systemNodeHtml("enforcement", "Tool preflight", enforcementDetail, enforcement.enabled ? "enforcement" : "system", enforcement.enabled ? "global policy" : "inactive")}</div>
      <div class="stage"><span class="stage-label">3. Analyze</span>${systemNodeHtml("filter", "Request filter", `${aggregator.model || "No model"} via ${aggregator.provider || "none"}`, "filter", "model call")}</div>
      <div class="stage wide-stage"><span class="stage-label">4. ${label}</span>
      ${systemNodeHtml("quorum", "Parallel quorum", `${profile.min_quorum}/${targets.length} required / concurrency ${profile.max_concurrency}`, "system", "orchestration")}
      ${targets.map((target, index) => nodeHtml(target, "contributor", collection, index)).join("")}
      <button class="node add-node" data-add-target="${collection}" type="button">+ Add ${profile.strategy === "classic" ? "proposer" : "contributor"}</button></div>
      <div class="stage"><span class="stage-label">5. Synthesize</span>${nodeHtml(profile.aggregator, "aggregator", "aggregator")}</div>
      <div class="stage"><span class="stage-label">6. Validate output</span>${systemNodeHtml("output-gate", "Tool-call gate", enforcementDetail, enforcement.enabled ? "enforcement" : "system", enforcement.enabled ? "conditional" : "pass-through")}</div>
      <div class="stage"><span class="stage-label">7. Return</span>${systemNodeHtml("response", "Client response", profile.strategy === "council" ? "Text response is scored" : "Final aggregator only", "response")}</div>`;

    branches = `<div class="execution-branches">
      <div class="branch-lane dispatch-lane"><span class="branch-title">Conditional bypass: tool result, recent tool call, delegated investigation, or OpenCode maintenance</span>
        <div class="branch-nodes">${systemNodeHtml("dispatch-route", "Route bypass", "Skips filter, contributors, and aggregation", "dispatch", "conditional")}
        ${profile.tool_dispatch ? nodeHtml(profile.tool_dispatch, "dispatch", "tool_dispatch") : systemNodeHtml("dispatch-missing", "No dispatcher", "Conditional turns use the MoA path", "warning", "not configured")}
        ${systemNodeHtml("dispatch-response", "Client response", "Direct model output", "response")}</div>
      </div>
      <div class="branch-lane recovery-lane"><span class="branch-title">Aggregation recovery: only when output has no text and no tool calls</span>
        <div class="branch-nodes">${systemNodeHtml("empty-check", "Empty output?", "Text and tool calls both absent", "recovery", "conditional")}
        ${systemNodeHtml("recovery", "Retry aggregator", "Same model / 2x tokens / thinking off", "recovery", "attempt 2")}
        ${systemNodeHtml("fallback", "Best contribution", "Fallback if retry is still empty", "recovery", "last resort")}</div>
      </div>
      <div class="branch-lane limitation-lane"><span class="branch-title">Refinement reality</span>
        <div class="branch-nodes">${systemNodeHtml("refinement", "No critique/revision round", "Current code has recovery, not semantic refinement", "warning", "not implemented")}</div>
      </div>
    </div>`;
  }
  canvas.innerHTML = `<div class="execution-map"><div class="pipeline">${mainPath}</div>${branches}</div>`;
  canvas.querySelectorAll("[data-target]").forEach((button) => button.addEventListener("click", () => {
    state.selectedTarget = { locator: button.dataset.target, index: button.dataset.index === "" ? null : Number(button.dataset.index) };
    render();
  }));
  canvas.querySelectorAll("[data-system]").forEach((button) => button.addEventListener("click", () => {
    state.selectedTarget = { locator: "system", id: button.dataset.system };
    render();
  }));
  canvas.querySelectorAll("[data-add-target]").forEach((button) => button.addEventListener("click", () => {
    const collection = button.dataset.addTarget;
    profile[collection].push(targetTemplate(Object.keys(state.config.providers)[0] || ""));
    state.selectedTarget = { locator: collection, index: profile[collection].length - 1 };
    markDirty();
    render();
  }));
  canvasWrap.scrollLeft = scrollLeft;
  canvasWrap.scrollTop = scrollTop;
}

function profileField(profile, field, value) {
  if (field === "aliases") profile.aliases = value.split(",").map((item) => item.trim()).filter(Boolean);
  else if (["min_quorum", "max_concurrency", "proposer_max_tokens", "contributor_max_tokens", "reference_token_budget", "contributor_history_chars", "num_ctx"].includes(field)) profile[field] = value === "" ? null : Number(value);
  else if (field === "contributor_deadline_seconds") profile[field] = value === "" ? null : Number(value);
  else if (field === "think") profile[field] = value === "" ? null : value === "true";
  else if (field === "reasoning_reserve") {
    profile.reasoning_reserve = Object.fromEntries(value.split(",").map((item) => item.trim()).filter(Boolean).map((item) => {
      const [family, tokens] = item.split(":");
      return [family.trim(), Number(tokens)];
    }));
  }
  else profile[field] = value;
}

function transformStrategy(profile, strategy) {
  const provider = Object.keys(state.config.providers)[0] || "";
  profile.strategy = strategy;
  if (strategy === "direct") {
    Object.assign(profile, { provider: profile.provider || provider, model: profile.model || "" });
    delete profile.proposers; delete profile.contributors; delete profile.aggregator; delete profile.tool_dispatch;
  } else if (strategy === "classic") {
    delete profile.provider; delete profile.model; delete profile.contributors;
    profile.proposers = profile.proposers?.length ? profile.proposers : [targetTemplate(provider)];
    profile.aggregator = profile.aggregator || targetTemplate(provider);
    profile.min_quorum = Math.min(profile.min_quorum || 1, profile.proposers.length);
  } else {
    delete profile.provider; delete profile.model; delete profile.proposers;
    profile.contributors = profile.contributors?.length >= 3 ? profile.contributors : [
      { ...targetTemplate(provider), family: "family-a" },
      { ...targetTemplate(provider), family: "family-b" },
      { ...targetTemplate(provider), family: "family-c" },
    ];
    profile.aggregator = profile.aggregator || targetTemplate(provider);
    profile.min_quorum = Math.min(Math.max(profile.min_quorum || 1, 1), profile.contributors.length);
  }
  state.selectedTarget = null;
}

function renderInspector() {
  const container = $("#inspector-content");
  const profile = currentProfile();
  if (!profile) {
    container.innerHTML = '<p class="empty-inspector">Select a flow to inspect its strategy and model targets.</p>';
    return;
  }
  if (state.selectedTarget?.locator === "system") {
    renderSystemInspector(container, profile, state.selectedTarget.id);
    return;
  }
  if (state.selectedTarget && state.selectedTarget.locator !== "profile") {
    renderTargetInspector(container, profile);
    return;
  }
  const direct = profile.strategy === "direct";
  container.innerHTML = `
    <h3 class="inspector-title">Flow settings</h3>
    <div class="field"><label>Flow name</label><input id="flow-name-input" value="${esc(state.selectedFlow)}"></div>
    <div class="field"><label>Public aliases</label><input data-profile-field="aliases" value="${esc((profile.aliases || []).join(", "))}"></div>
    <div class="field"><span class="field-label">Strategy</span><div class="segmented">
      ${["direct", "classic", "council"].map((strategy) => `<button type="button" data-strategy="${strategy}" class="${profile.strategy === strategy ? "active" : ""}">${strategy}</button>`).join("")}
    </div></div>
    ${direct ? `
      <hr class="inspector-divider"><h3 class="inspector-title">Direct model</h3>
      ${providerSelect("provider", profile.provider)}
      <div class="field"><label>Model ID</label><input data-profile-field="model" list="model-options" value="${esc(profile.model)}"></div>
      <div class="field-row">
        <div class="field"><label>Context size</label><input data-profile-field="num_ctx" type="number" min="1" value="${esc(profile.num_ctx ?? "")}"></div>
        <div class="field"><label>Keep alive</label><input data-profile-field="keep_alive" value="${esc(profile.keep_alive ?? "")}"></div>
      </div>
      <div class="field"><label>Thinking</label><select data-profile-field="think"><option value="" ${profile.think == null ? "selected" : ""}>Provider default</option><option value="true" ${profile.think === true ? "selected" : ""}>Enabled</option><option value="false" ${profile.think === false ? "selected" : ""}>Disabled</option></select></div>` : `
      <div class="field-row">
        <div class="field"><label>Minimum quorum</label><input data-profile-field="min_quorum" type="number" min="1" value="${esc(profile.min_quorum)}"></div>
        <div class="field"><label>Concurrency</label><input data-profile-field="max_concurrency" type="number" min="1" value="${esc(profile.max_concurrency)}"></div>
      </div>
      <div class="field"><label>Contributor deadline (seconds)</label><input data-profile-field="contributor_deadline_seconds" type="number" min="0" step="1" value="${esc(profile.contributor_deadline_seconds ?? "")}" placeholder="No deadline"></div>
      <hr class="inspector-divider"><h3 class="inspector-title">Context and budgets</h3>
      ${profile.strategy === "classic" ? `<div class="field-row">
        <div class="field"><label>Proposer max tokens</label><input data-profile-field="proposer_max_tokens" type="number" min="1" value="${esc(profile.proposer_max_tokens)}"></div>
        <div class="field"><label>Filter max tokens</label><input data-profile-field="contributor_max_tokens" type="number" min="1" value="${esc(profile.contributor_max_tokens)}"></div>
      </div>` : `<div class="field"><label>Contributor and filter max tokens</label><input data-profile-field="contributor_max_tokens" type="number" min="1" value="${esc(profile.contributor_max_tokens)}"></div><p class="hint">The current backend uses one shared budget for the request-filter call and every council contributor.</p>`}
      <div class="field"><label>History characters per contributor</label><input data-profile-field="contributor_history_chars" type="number" min="1" value="${esc(profile.contributor_history_chars)}"></div>
      ${profile.strategy === "classic" ? `<div class="field"><label>Reference token budget</label><input data-profile-field="reference_token_budget" type="number" min="1" value="${esc(profile.reference_token_budget)}"></div>` : '<p class="hint">Council aggregation includes complete contributor answers; reference_token_budget is not used on this path.</p>'}
      <div class="field"><label>Aggregator reasoning reserve</label><input data-profile-field="reasoning_reserve" value="${esc(Object.entries(profile.reasoning_reserve || {}).map(([family, tokens]) => `${family}:${tokens}`).join(", "))}" placeholder="qwen:4096, gemma:2048"></div>
      ${profile.strategy === "council" ? `<div class="field"><label>Contributor output format</label><select data-profile-field="contributor_format"><option value="text" ${profile.contributor_format === "text" ? "selected" : ""}>Text</option><option value="json-schema" ${profile.contributor_format === "json-schema" ? "selected" : ""}>Five-perspective JSON schema</option></select></div>` : ""}
      <p class="hint">Tool policy is fixed by the backend: only the final aggregator may emit client-visible tool calls.</p>
      <button class="button inspector-action" id="toggle-dispatch" type="button">${profile.tool_dispatch ? "Remove" : "Add"} tool dispatch</button>`}
    <hr class="inspector-divider">
    <button class="button inspector-action" id="make-default" type="button" ${state.config.default_profile === state.selectedFlow ? "disabled" : ""}>${state.config.default_profile === state.selectedFlow ? "Current live default" : "Make default flow"}</button>`;

  container.querySelectorAll("[data-profile-field]").forEach((input) => {
    input.addEventListener("input", () => { profileField(profile, input.dataset.profileField, input.value); markDirty(); });
    input.addEventListener("change", render);
  });
  container.querySelectorAll("[data-strategy]").forEach((button) => button.addEventListener("click", () => {
    transformStrategy(profile, button.dataset.strategy); markDirty(); render();
  }));
  $("#flow-name-input").addEventListener("change", (event) => renameFlow(event.target.value.trim()));
  $("#make-default").addEventListener("click", () => { state.config.default_profile = state.selectedFlow; markDirty(); render(); });
  const dispatch = $("#toggle-dispatch");
  if (dispatch) dispatch.addEventListener("click", () => {
    if (profile.tool_dispatch) delete profile.tool_dispatch;
    else profile.tool_dispatch = targetTemplate(Object.keys(state.config.providers)[0] || "");
    markDirty(); render();
  });
}

function renderSystemInspector(container, profile, id) {
  const enforcement = state.config.tool_enforcement || {};
  const aggregator = profile.aggregator || {};
  const descriptions = {
    routing: ["Request routing", profile.strategy === "direct"
      ? "The selected direct profile calls its model immediately."
      : "Initial turns enter the MoA path. Tool results, recent assistant tool calls, delegated investigations, and OpenCode maintenance can bypass it."],
    filter: ["Request filter", `A real non-streaming call to ${aggregator.model || "the aggregator"} on ${aggregator.provider || "an unconfigured provider"}. Its analysis is appended as untrusted context to every contributor and the final aggregator.`],
    quorum: ["Parallel contribution quorum", "All configured model calls are scheduled, bounded by max_concurrency and the shared deadline. Pending calls are cancelled at the deadline; aggregation starts only when min_quorum responses succeeded."],
    "output-gate": ["Tool-call output gate", "In block and auto modes, aggregator output is withheld until the required investigation tool call is validated. Text emitted beside that call is removed."],
    response: ["Response and scoring", profile.strategy === "council" ? "Text responses score council contributors after aggregation. Tool-call responses skip scoring." : "The final model response is returned with panel usage accounted separately."],
    "dispatch-route": ["Conditional tool dispatch", "This route bypasses request filtering, contributors, and aggregation. It is selected for tool-result turns, recent assistant tool calls, delegated investigations, and OpenCode maintenance prompts."],
    "dispatch-missing": ["Tool dispatcher missing", "No direct dispatcher is configured for conditional continuation turns."],
    "dispatch-response": ["Direct continuation response", "The tool dispatcher output is returned directly to the client."],
    "empty-check": ["Empty completion check", "Recovery starts only when the aggregator returns neither non-whitespace text nor tool calls."],
    recovery: ["Aggregation retry", "The same aggregator is called once more. max_tokens is doubled and thinking is forced off. This behavior is currently fixed in code."],
    fallback: ["Contributor fallback", "If both aggregator attempts are empty, MoA prefers a complete non-empty contribution, then the longest response. An upstream error is raised when none exists."],
    refinement: ["No semantic refinement stage", "The current gateway does not ask contributors to critique, revise, or rank earlier answers. The second aggregation attempt is empty-output recovery, not refinement."],
  };

  if (id === "enforcement" || id === "output-gate") {
    container.innerHTML = `
      <h3 class="inspector-title">Tool enforcement</h3>
      <p class="inspector-copy">Global policy used on initial non-direct aggregation turns. The preflight chooses an available required tool; the output gate validates the aggregator call.</p>
      <label class="toggle-row"><input id="enforcement-enabled" type="checkbox" ${enforcement.enabled ? "checked" : ""}><span>Enable enforcement</span></label>
      <div class="field"><label>Required tools, in priority order</label><input data-enforcement-field="required_tools" value="${esc((enforcement.required_tools || []).join(", "))}" placeholder="task"></div>
      <div class="field"><label>Mode</label><select data-enforcement-field="enforcement_mode"><option value="warn" ${enforcement.enforcement_mode === "warn" ? "selected" : ""}>Warn: instruct but do not reject</option><option value="block" ${enforcement.enforcement_mode === "block" ? "selected" : ""}>Block: require emitted call</option><option value="auto" ${enforcement.enforcement_mode === "auto" ? "selected" : ""}>Auto: force named tool choice</option></select></div>
      <div class="field"><label>Minimum delegated investigations</label><input data-enforcement-field="min_investigation_calls" type="number" min="1" max="8" value="${esc(enforcement.min_investigation_calls || 1)}"></div>
      <p class="hint">When the required tool is task, MoA expands it into this many architecture, implementation, and verification investigations.</p>`;
    $("#enforcement-enabled").addEventListener("change", (event) => {
      enforcement.enabled = event.target.checked;
      if (enforcement.enabled && !(enforcement.required_tools || []).length) enforcement.required_tools = ["task"];
      markDirty(); render();
    });
    container.querySelectorAll("[data-enforcement-field]").forEach((input) => input.addEventListener("input", () => {
      const field = input.dataset.enforcementField;
      if (field === "required_tools") enforcement[field] = input.value.split(",").map((tool) => tool.trim()).filter(Boolean);
      else if (field === "min_investigation_calls") enforcement[field] = Number(input.value);
      else enforcement[field] = input.value;
      markDirty();
    }));
    return;
  }

  if (id === "quorum") {
    container.innerHTML = `
      <h3 class="inspector-title">Parallel contribution quorum</h3>
      <p class="inspector-copy">All configured targets are scheduled. The gateway waits until they finish or the shared deadline expires, then cancels pending calls and checks quorum.</p>
      <div class="field-row">
        <div class="field"><label>Minimum successful</label><input data-profile-field="min_quorum" type="number" min="1" value="${esc(profile.min_quorum)}"></div>
        <div class="field"><label>Maximum concurrent</label><input data-profile-field="max_concurrency" type="number" min="1" value="${esc(profile.max_concurrency)}"></div>
      </div>
      <div class="field"><label>Shared deadline (seconds)</label><input data-profile-field="contributor_deadline_seconds" type="number" min="0" value="${esc(profile.contributor_deadline_seconds ?? "")}" placeholder="No deadline"></div>
      <div class="field"><label>History characters per call</label><input data-profile-field="contributor_history_chars" type="number" min="1" value="${esc(profile.contributor_history_chars)}"></div>
      <div class="field"><label>${profile.strategy === "council" ? "Contributor and filter" : "Proposer"} max tokens</label><input data-profile-field="${profile.strategy === "council" ? "contributor_max_tokens" : "proposer_max_tokens"}" type="number" min="1" value="${esc(profile.strategy === "council" ? profile.contributor_max_tokens : profile.proposer_max_tokens)}"></div>`;
    container.querySelectorAll("[data-profile-field]").forEach((input) => {
      input.addEventListener("input", () => { profileField(profile, input.dataset.profileField, input.value); markDirty(); });
      input.addEventListener("change", render);
    });
    return;
  }

  const [title, body] = descriptions[id] || ["Gateway stage", "This stage is part of the current request path."];
  container.innerHTML = `<h3 class="inspector-title">${esc(title)}</h3><p class="inspector-copy">${esc(body)}</p>
    ${id === "filter" ? `<div class="fact-card"><span>EXECUTED BY</span><strong>${esc(aggregator.model || "Not configured")}</strong><small>${esc(aggregator.provider || "No provider")}</small></div>
      <div class="field"><label>Filter maximum tokens</label><input data-profile-field="contributor_max_tokens" type="number" min="1" value="${esc(profile.contributor_max_tokens)}"></div>
      <button class="button inspector-action" id="inspect-aggregator" type="button">Inspect shared aggregator target</button>` : ""}`;
  container.querySelectorAll("[data-profile-field]").forEach((input) => input.addEventListener("input", () => {
    profileField(profile, input.dataset.profileField, input.value); markDirty();
  }));
  const inspect = $("#inspect-aggregator");
  if (inspect) inspect.addEventListener("click", () => { state.selectedTarget = { locator: "aggregator", index: null }; render(); });
}

function providerSelect(field, selected, target = false) {
  return `<div class="field"><label>Provider</label><select ${target ? "data-target-field" : "data-profile-field"}="${field}">
    ${Object.keys(state.config.providers).map((name) => `<option value="${esc(name)}" ${name === selected ? "selected" : ""}>${esc(name)}</option>`).join("")}
  </select></div>`;
}

function selectedTarget(profile) {
  const { locator, index } = state.selectedTarget;
  return index === null ? profile[locator] : profile[locator][index];
}

function renderTargetInspector(container, profile) {
  const target = selectedTarget(profile);
  if (!target) { state.selectedTarget = null; renderInspector(); return; }
  const { locator, index } = state.selectedTarget;
  const removable = index !== null;
  const runtimeRole = locator === "aggregator"
    ? "This target runs the request-filter call, aggregation attempt 1, and the conditional empty-output retry."
    : locator === "tool_dispatch"
      ? "This target is called only on continuation and maintenance turns that bypass the MoA panel."
      : profile.strategy === "council"
        ? "This target runs one complete five-perspective council response with no client tools."
        : "This target runs one independent proposal with no client tools.";
  container.innerHTML = `
    <h3 class="inspector-title">${esc(locator.replace("_", " "))}${index !== null ? ` ${index + 1}` : ""}</h3>
    <p class="inspector-copy">${esc(runtimeRole)}</p>
    ${providerSelect("provider", target.provider, true)}
    <div class="field"><label>Model ID</label><input data-target-field="model" list="model-options" value="${esc(target.model)}"></div>
    <button class="button inspector-action" id="discover-models" type="button">Discover models</button>
    <p class="model-result" id="model-result"></p>
    <hr class="inspector-divider">
    <div class="field"><label>Role</label><input data-target-field="role" value="${esc(target.role || "general")}"></div>
    <div class="field"><label>Model family</label><input data-target-field="family" value="${esc(target.family || "")}" placeholder="Required and unique for council contributors"></div>
    <div class="field-row">
      <div class="field"><label>Temperature</label><input data-target-field="temperature" type="number" min="0" max="2" step="0.1" value="${esc(target.temperature ?? "")}"></div>
      <div class="field"><label>Context size</label><input data-target-field="num_ctx" type="number" min="1" value="${esc(target.num_ctx ?? "")}"></div>
    </div>
    <div class="field"><label>Keep alive</label><input data-target-field="keep_alive" value="${esc(target.keep_alive ?? "")}" placeholder="Provider default"></div>
    <div class="field"><label>Thinking</label><select data-target-field="think"><option value="" ${target.think == null ? "selected" : ""}>Provider default</option><option value="true" ${target.think === true ? "selected" : ""}>Enabled</option><option value="false" ${target.think === false ? "selected" : ""}>Disabled</option></select></div>
    ${removable ? '<hr class="inspector-divider"><button class="remove-target" id="remove-target" type="button">Remove target</button>' : ""}`;
  container.querySelectorAll("[data-target-field]").forEach((input) => {
    input.addEventListener("input", () => {
      let value = input.value;
      if (["temperature", "num_ctx"].includes(input.dataset.targetField)) value = value === "" ? null : Number(value);
      if (input.dataset.targetField === "think") value = value === "" ? null : value === "true";
      if (value === null || value === "") delete target[input.dataset.targetField]; else target[input.dataset.targetField] = value;
      markDirty();
    });
    input.addEventListener("change", render);
  });
  $("#discover-models").addEventListener("click", () => discoverModels(target.provider));
  const remove = $("#remove-target");
  if (remove) remove.addEventListener("click", () => {
    profile[locator].splice(index, 1); state.selectedTarget = null; markDirty(); render();
  });
}

function selectedRunKey() {
  if (!state.selectedTarget) return null;
  if (state.selectedTarget.locator === "system") return `system:${state.selectedTarget.id}`;
  return `target:${state.selectedTarget.locator}:${state.selectedTarget.index ?? ""}`;
}

function appendRunOutput() {
  const key = selectedRunKey();
  const run = key ? state.simulation.nodeStates[key] : null;
  if (!run) return;
  const container = $("#inspector-content");
  const outputs = run.events.filter((event) =>
    event.content || event.tool_calls?.length || event.error || event.event === "stage_progress"
  );
  const outputHtml = outputs.length ? outputs.map((event) => {
    const content = event.content ? `<pre>${esc(event.content)}</pre>` : "";
    const calls = event.tool_calls?.length ? `<pre>${esc(JSON.stringify(event.tool_calls, null, 2))}</pre>` : "";
    const error = event.error ? `<pre class="error-output">${esc(event.error)}</pre>` : "";
    const progress = event.event === "stage_progress"
      ? `<p>${esc(`${event.successes} succeeded / ${event.failures} failed / ${event.pending} pending`)}</p>`
      : "";
    return `<div class="run-event-output"><span>${esc(event.event.replaceAll("_", " "))}${event.attempt ? ` / attempt ${event.attempt}` : ""}</span>${content}${calls}${error}${progress}</div>`;
  }).join("") : '<p class="run-empty">This stage has status events but no model output.</p>';
  container.insertAdjacentHTML("beforeend", `
    <hr class="inspector-divider">
    <section class="run-output">
      <div class="run-output-heading"><span>LIVE RUN</span><strong class="status-${esc(run.status)}">${esc(run.status)}</strong></div>
      ${outputHtml}
      <details><summary>Raw trace events (${run.events.length})</summary><pre>${esc(JSON.stringify(run.events, null, 2))}</pre></details>
    </section>`);
}

function targetRunKey(event, profile) {
  if (event.stage === "filter") return "system:filter";
  if (event.stage === "aggregator") return "target:aggregator:";
  if (event.stage === "direct") {
    return profile.strategy === "direct" ? "target:profile:" : "target:tool_dispatch:";
  }
  if (!["contributor", "proposer"].includes(event.stage)) return null;
  if (event.event.startsWith("stage_")) return "system:quorum";
  const collection = event.stage === "contributor" ? "contributors" : "proposers";
  const targets = profile[collection] || [];
  let index = targets.findIndex((target) =>
    target.model === event.model && target.provider === event.provider
      && (!event.role || target.role === event.role)
      && (!event.family || target.family === event.family)
  );
  if (index < 0) index = targets.findIndex((target) => target.model === event.model);
  return index < 0 ? "system:quorum" : `target:${collection}:${index}`;
}

function simulationEventTarget(event) {
  const profile = state.config?.profiles[state.simulation.profile];
  if (!profile) return null;
  if (event.event === "model_retrying") return "system:recovery";
  if (event.event === "model_fallback") return "system:fallback";
  if (event.event === "request_started") return "system:routing";
  if (["investigation_tool_selected", "investigation_tool_unavailable", "investigation_tool_fallback"].includes(event.event)) return "system:enforcement";
  if (["investigation_tool_validated", "investigation_tool_missing", "investigation_tool_fanout", "investigation_tool_grounded"].includes(event.event)) return "system:output-gate";
  if (["request_completed", "request_failed", "simulation_completed", "simulation_failed"].includes(event.event)) return "system:response";
  if (event.stage) return targetRunKey(event, profile);
  return null;
}

function simulationEventStatus(event) {
  if (["request_failed", "simulation_failed", "model_failed", "stage_failed", "investigation_tool_missing"].includes(event.event)) return "failed";
  if (["model_cancelled", "stage_cancelled"].includes(event.event)) return "skipped";
  if (["request_completed", "simulation_completed", "model_completed", "stage_completed", "investigation_tool_selected", "investigation_tool_validated", "model_fallback"].includes(event.event)) return "complete";
  return "running";
}

function handleSimulationEvent(event) {
  state.simulation.events.push(event);
  if (event.request_id) state.simulation.requestId = event.request_id;
  const key = simulationEventTarget(event);
  if (key) {
    const run = state.simulation.nodeStates[key] || { status: "running", events: [] };
    run.events.push(event);
    run.status = simulationEventStatus(event);
    state.simulation.nodeStates[key] = run;
  }
  if (event.event === "stage_started" && ["filter", "direct"].includes(event.stage)) {
    state.simulation.nodeStates["system:routing"] = {
      status: "complete",
      events: [...(state.simulation.nodeStates["system:routing"]?.events || []), event],
    };
    if (event.stage === "filter" && !state.config.tool_enforcement?.enabled) {
      state.simulation.nodeStates["system:enforcement"] = { status: "complete", events: [event] };
    }
  }
  if (event.event === "request_completed" && !state.config.tool_enforcement?.enabled) {
    state.simulation.nodeStates["system:output-gate"] = { status: "complete", events: [event] };
  }
  if (event.event === "simulation_started") {
    state.simulation.message = `Generation ${event.generation} started / request ${event.request_id.slice(0, 8)}`;
    state.simulation.nodeStates["system:routing"] = { status: "running", events: [event] };
  } else if (event.event === "stage_progress") {
    state.simulation.message = `${event.stage}: ${event.successes} complete, ${event.pending} pending`;
  } else if (event.event === "model_started") {
    state.simulation.message = `${event.stage}: running ${event.model}`;
  } else if (event.event === "simulation_completed") {
    state.simulation.message = event.tool_calls?.length
      ? `Stopped at client tool boundary / ${event.tool_calls.length} call(s) requested`
      : `Completed with ${event.usage?.output_tokens || 0} output tokens`;
  } else if (event.event === "simulation_failed") {
    state.simulation.message = event.error;
  }
  render();
}

function renderSimulationStatus() {
  const simulation = state.simulation;
  const status = $("#simulation-status");
  const button = $("#run-simulation");
  if (!status || !button) return;
  status.classList.toggle("active", simulation.active);
  status.classList.toggle("failed", Object.values(simulation.nodeStates).some((node) => node.status === "failed"));
  status.querySelector("span:last-child").textContent = simulation.message || "Runs the real models and streams their trace.";
  button.textContent = simulation.active ? "Stop run" : "Run flow";
  button.classList.toggle("active", simulation.active);
}

async function runSimulation() {
  if (state.simulation.active) {
    state.simulation.message = "Stopping run...";
    state.simulation.controller?.abort();
    renderSimulationStatus();
    return;
  }
  const input = $("#simulation-input").value.trim();
  if (!input) {
    toast("Enter a prompt before running the flow.", true);
    $("#simulation-input").focus();
    return;
  }
  const controller = new AbortController();
  state.simulation = {
    active: true,
    controller,
    profile: state.selectedFlow,
    requestId: null,
    events: [],
    nodeStates: {},
    message: "Connecting to the live gateway...",
  };
  state.selectedTarget = { locator: "system", id: "routing" };
  render();
  try {
    const response = await fetch("/api/simulations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile: state.selectedFlow,
        input,
        max_tokens: Number($("#simulation-max-tokens").value),
      }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      const payload = await response.json();
      throw new Error(formatError(payload.detail));
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        const data = block.split("\n").filter((line) => line.startsWith("data: ")).map((line) => line.slice(6)).join("");
        if (data) handleSimulationEvent(JSON.parse(data));
      }
    }
  } catch (error) {
    if (error.name === "AbortError") state.simulation.message = "Run stopped by user.";
    else {
      state.simulation.message = error.message;
      toast(error.message, true);
    }
  } finally {
    state.simulation.active = false;
    state.simulation.controller = null;
    render();
  }
}

function renameFlow(name) {
  if (!name || name === state.selectedFlow) return;
  if (state.config.profiles[name]) { toast("A flow with that name already exists.", true); renderInspector(); return; }
  const previous = state.selectedFlow;
  const entries = Object.entries(state.config.profiles).map(([key, value]) => [key === previous ? name : key, value]);
  state.config.profiles = Object.fromEntries(entries);
  if (state.config.default_profile === previous) state.config.default_profile = name;
  state.selectedFlow = name; markDirty(); render();
}

function addFlow() {
  const base = "new-flow";
  let name = base; let suffix = 2;
  while (state.config.profiles[name]) name = `${base}-${suffix++}`;
  const provider = Object.keys(state.config.providers)[0] || "";
  state.config.profiles[name] = { aliases: [name], strategy: "direct", provider, model: "" };
  state.selectedFlow = name; state.selectedTarget = null; markDirty(); render();
}

function duplicateFlow() {
  if (!currentProfile()) return;
  let name = `${state.selectedFlow}-copy`; let suffix = 2;
  while (state.config.profiles[name]) name = `${state.selectedFlow}-copy-${suffix++}`;
  const copied = clone(currentProfile()); copied.aliases = copied.aliases.map((alias) => `${alias}-copy`);
  state.config.profiles[name] = copied; state.selectedFlow = name; state.selectedTarget = null; markDirty(); render();
}

function deleteFlow() {
  if (Object.keys(state.config.profiles).length <= 1 || !currentProfile()) return;
  if (!confirm(`Delete flow "${state.selectedFlow}"?`)) return;
  const wasDefault = state.config.default_profile === state.selectedFlow;
  delete state.config.profiles[state.selectedFlow];
  state.selectedFlow = Object.keys(state.config.profiles)[0];
  if (wasDefault) state.config.default_profile = state.selectedFlow;
  state.selectedTarget = null; markDirty(); render();
}

function openProviders() {
  renderProviders();
  $("#drawer-backdrop").hidden = false;
  $("#provider-drawer").classList.add("open");
  $("#provider-drawer").setAttribute("aria-hidden", "false");
}
function closeProviders() {
  $("#provider-drawer").classList.remove("open");
  $("#provider-drawer").setAttribute("aria-hidden", "true");
  setTimeout(() => $("#drawer-backdrop").hidden = true, 200);
}

function renderProviders() {
  $("#provider-list").innerHTML = Object.entries(state.config.providers).map(([name, provider]) => `
    <div class="provider-card" data-provider-card="${esc(name)}">
      <div class="provider-card-heading"><strong>${esc(name)}</strong><div class="provider-actions">
        <button class="mini-button" data-provider-discover="${esc(name)}" type="button">DISCOVER</button>
        <button class="mini-button remove" data-provider-remove="${esc(name)}" type="button">REMOVE</button>
      </div></div>
      <div class="field"><label>Name</label><input data-provider-name="${esc(name)}" value="${esc(name)}"></div>
      <div class="field"><label>Type</label><select data-provider-field="type" data-provider="${esc(name)}">
        ${["ollama", "openai", "deepseek", "openai-compatible"].map((kind) => `<option ${provider.type === kind ? "selected" : ""}>${kind}</option>`).join("")}
      </select></div>
      <div class="field"><label>Base URL</label><input data-provider-field="base_url" data-provider="${esc(name)}" value="${esc(provider.base_url || "")}"></div>
      <div class="field-row">
        <div class="field"><label>API key env</label><input data-provider-field="api_key_env" data-provider="${esc(name)}" value="${esc(provider.api_key_env || "")}" placeholder="None"></div>
        <div class="field"><label>Timeout (s)</label><input data-provider-field="timeout_seconds" data-provider="${esc(name)}" type="number" min="1" value="${esc(provider.timeout_seconds || 1800)}"></div>
      </div>
      <div class="model-result" id="provider-result-${esc(name)}"></div>
    </div>`).join("");
  document.querySelectorAll("[data-provider-field]").forEach((input) => input.addEventListener("input", () => {
    const provider = state.config.providers[input.dataset.provider];
    let value = input.value;
    if (input.dataset.providerField === "timeout_seconds") value = Number(value);
    if (input.dataset.providerField === "api_key_env" && value === "") value = null;
    provider[input.dataset.providerField] = value; markDirty();
  }));
  document.querySelectorAll("[data-provider-name]").forEach((input) => input.addEventListener("change", () => renameProvider(input.dataset.providerName, input.value.trim())));
  document.querySelectorAll("[data-provider-remove]").forEach((button) => button.addEventListener("click", () => removeProvider(button.dataset.providerRemove)));
  document.querySelectorAll("[data-provider-discover]").forEach((button) => button.addEventListener("click", () => discoverModels(button.dataset.providerDiscover, `#provider-result-${CSS.escape(button.dataset.providerDiscover)}`)));
}

function renameProvider(previous, name) {
  if (!name || name === previous) return;
  if (state.config.providers[name]) { toast("A provider with that name already exists.", true); renderProviders(); return; }
  state.config.providers = Object.fromEntries(Object.entries(state.config.providers).map(([key, value]) => [key === previous ? name : key, value]));
  Object.values(state.config.profiles).forEach((profile) => {
    if (profile.strategy === "direct" && profile.provider === previous) profile.provider = name;
    if (profile.strategy !== "direct") allTargets(profile).forEach((target) => { if (target.provider === previous) target.provider = name; });
  });
  markDirty(); renderProviders(); render();
}

function removeProvider(name) {
  const used = Object.entries(state.config.profiles).filter(([, profile]) => allTargets(profile).some((target) => target.provider === name)).map(([flow]) => flow);
  if (used.length) { toast(`Provider is used by: ${used.join(", ")}`, true); return; }
  if (Object.keys(state.config.providers).length <= 1) { toast("At least one provider is required.", true); return; }
  delete state.config.providers[name]; markDirty(); renderProviders(); render();
}

function addProvider() {
  let name = "provider"; let suffix = 2;
  while (state.config.providers[name]) name = `provider-${suffix++}`;
  state.config.providers[name] = { type: "ollama", base_url: "http://127.0.0.1:11434", timeout_seconds: 1800 };
  markDirty(); renderProviders();
}

async function discoverModels(provider, resultSelector = "#model-result") {
  const result = $(resultSelector);
  if (result) result.textContent = "Querying provider...";
  try {
    const response = await fetch("/api/providers/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.config.providers[provider]),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(formatError(payload.detail));
    state.modelOptions[provider] = payload.models;
    $("#model-options").innerHTML = payload.models.map((model) => `<option value="${esc(model)}"></option>`).join("");
    if (result) result.textContent = `${payload.models.length} model${payload.models.length === 1 ? "" : "s"} available`;
    toast(`Discovered ${payload.models.length} models from ${provider}.`);
  } catch (error) {
    if (result) result.textContent = "Discovery failed";
    toast(error.message, true);
  }
}

function formatError(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => `${(item.loc || []).slice(1).join(" > ") || "config"}: ${item.msg}`).join("\n");
  return JSON.stringify(detail);
}

async function applyConfig() {
  if (!state.dirty) return;
  const button = $("#apply-button");
  button.disabled = true;
  button.querySelector("span").textContent = "Validating...";
  try {
    const response = await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.config),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(formatError(payload.detail));
    state.config = payload.config;
    state.generation = payload.generation;
    state.persisted = payload.persisted;
    state.dirty = false;
    toast(`Generation ${state.generation} is live${state.persisted ? " and saved" : ""}.`);
    render();
  } catch (error) {
    toast(error.message, true);
    button.disabled = false;
  } finally {
    button.querySelector("span").textContent = "Apply live";
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error("Could not load runtime configuration.");
    const payload = await response.json();
    state.config = payload.config;
    state.generation = payload.generation;
    state.persisted = payload.persisted;
    state.selectedFlow = state.config.default_profile || Object.keys(state.config.profiles)[0];
    render();
  } catch (error) {
    $("#runtime-label").textContent = "Runtime unavailable";
    $(".pulse").classList.add("error");
    toast(error.message, true);
  }
}

$("#add-flow").addEventListener("click", addFlow);
$("#duplicate-flow").addEventListener("click", duplicateFlow);
$("#delete-flow").addEventListener("click", deleteFlow);
$("#apply-button").addEventListener("click", applyConfig);
$("#run-simulation").addEventListener("click", runSimulation);
$("#policy-button").addEventListener("click", () => {
  state.selectedTarget = { locator: "system", id: "enforcement" };
  render();
});
$("#providers-button").addEventListener("click", openProviders);
$("#close-providers").addEventListener("click", closeProviders);
$("#drawer-backdrop").addEventListener("click", closeProviders);
$("#add-provider").addEventListener("click", addProvider);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") { event.preventDefault(); applyConfig(); }
  if (event.key === "Escape") closeProviders();
});

initialize();
