const state = {
  config: null,
  generation: 0,
  persisted: false,
  selectedFlow: null,
  selectedStep: null,
  dirty: false,
  applying: false,
  unsupportedV1: false,
  editorErrors: new Map(),
  modelOptions: {},
  simulation: {
    active: false,
    controller: null,
    flow: null,
    start: null,
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
const startConditions = ["always", "skill_result", "investigation_result", "tool_continuation", "opencode_maintenance", "delegated_investigation", "simple_request"];
const targetConditions = ["always", "has_tool_calls", "no_tool_calls"];

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.className = `toast show${error ? " error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.className = "toast", error ? 6500 : 2600);
}

function markDirty() {
  state.dirty = true;
  $("#apply-button").disabled = state.applying || state.simulation.active;
  $("#dirty-label").textContent = "UNAPPLIED";
}

function currentFlow() {
  return state.config?.flows?.[state.selectedFlow];
}

function selectedStep() {
  return currentFlow()?.steps.find((step) => step.id === state.selectedStep) || null;
}

function optionList(values, selected, emptyLabel = null) {
  const options = emptyLabel == null ? [] : [`<option value="">${esc(emptyLabel)}</option>`];
  return options.concat(values.map((value) => `<option value="${esc(value)}" ${value === selected ? "selected" : ""}>${esc(value)}</option>`)).join("");
}

function uniqueId(base, values) {
  let value = base;
  let suffix = 2;
  while (values.includes(value)) value = `${base}-${suffix++}`;
  return value;
}

function parseList(value) {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function parseOptionalNumber(value) {
  return value === "" ? null : Number(value);
}

function parseMaxTokens(value) {
  const trimmed = value.trim();
  if (!trimmed) return null;
  return trimmed === "request" ? "request" : Number(trimmed);
}

function setOptional(object, field, value) {
  if (value === null || value === "" || Number.isNaN(value)) delete object[field];
  else object[field] = value;
}

function ordinaryStart(flow) {
  return orderedStarts(flow).find((start) => start.when === "always")?.step || null;
}

function startPriority(start, index) {
  return start.priority ?? index + 1;
}

function orderedStarts(flow) {
  return (flow?.starts || [])
    .map((start, index) => ({ ...start, priority: startPriority(start, index), declarationIndex: index }))
    .sort((left, right) => left.priority - right.priority || left.declarationIndex - right.declarationIndex);
}

function nextStartPriority(flow) {
  return Math.max(0, ...flow.starts.map((start, index) => startPriority(start, index))) + 1;
}

function startChainHtml(flow) {
  const routes = orderedStarts(flow);
  return `<div class="entry-chain" aria-label="Entry route priority chain">
    <span class="entry-chain-source">REQUEST</span>
    ${routes.map((start) => `<span class="entry-chain-arrow">&rarr;</span><span class="entry-chain-route ${start.when === "always" ? "fallback" : ""}">
      <strong>P${esc(start.priority)}</strong><span>${esc(start.when)}</span><small>${esc(start.step)}</small>
    </span>`).join("")}
  </div>`;
}

function ancestorGateIds(flow, stepId) {
  const ancestors = new Set();
  const pending = [stepId];
  while (pending.length) {
    const targetId = pending.pop();
    flow.steps.forEach((candidate) => {
      if (!ancestors.has(candidate.id) && candidate.targets.some((target) => target.step === targetId)) {
        ancestors.add(candidate.id);
        pending.push(candidate.id);
      }
    });
  }
  return flow.steps.filter((step) => step.type === "gate" && ancestors.has(step.id)).map((step) => step.id);
}

function commitActiveEditor() {
  const active = document.activeElement;
  if (!(active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement || active instanceof HTMLSelectElement)) return;
  active.dispatchEvent(new Event("input", { bubbles: true }));
  if (active.isConnected) active.dispatchEvent(new Event("change", { bubbles: true }));
  if (active.isConnected) active.blur();
}

function hasEditorErrors() {
  for (const [key, input] of state.editorErrors) {
    if (!input.isConnected) state.editorErrors.delete(key);
  }
  return state.editorErrors.size > 0;
}

function render() {
  if (!state.config) return;
  if (state.unsupportedV1) {
    renderUnsupported();
    return;
  }
  renderFlows();
  renderCanvas();
  renderInspector();
  appendRunOutput();
  renderRuntime();
  renderSimulationStatus();
}

function renderUnsupported() {
  $("#runtime-label").textContent = "Unsupported v1 configuration";
  $("#flow-list").innerHTML = '<div class="unsupported-card">No v2 flows available</div>';
  $("#flow-title").textContent = "Flow editor unavailable";
  $("#flow-structure").textContent = "UNSUPPORTED CONFIGURATION";
  $("#flow-canvas").innerHTML = '<div class="unsupported-message"><strong>Version 2 flows are required</strong><p>This runtime is using a legacy profile configuration or has no flows. Migrate the configuration to <code>flows</code> before using Flow Lab.</p></div>';
  $("#inspector-content").innerHTML = '<p class="empty-inspector">Legacy v1 profiles cannot be edited safely in this interface.</p>';
  ["#add-flow", "#duplicate-flow", "#delete-flow", "#add-ai-step", "#add-gate-step", "#providers-button", "#apply-button", "#run-simulation"].forEach((selector) => { $(selector).disabled = true; });
  $("#dirty-label").textContent = "";
}

function renderRuntime() {
  $("#runtime-label").textContent = `Generation ${state.generation} / ${state.persisted ? "saved to YAML" : "runtime only"}`;
  $("#apply-button").disabled = !state.dirty || state.applying || state.simulation.active;
  $("#dirty-label").textContent = state.dirty ? "UNAPPLIED" : "";
}

function renderFlows() {
  const list = $("#flow-list");
  const flows = Object.entries(state.config.flows || {});
  list.innerHTML = flows.map(([name, flow]) => {
    const aiCount = flow.steps.filter((step) => step.type === "ai").length;
    const gateCount = flow.steps.length - aiCount;
    return `<button class="flow-card ${name === state.selectedFlow ? "active" : ""}" data-flow="${esc(name)}" type="button" ${state.simulation.active ? "disabled" : ""}>
      <span class="flow-glyph">${aiCount}<i>/${gateCount}</i></span>
      <span class="flow-name"><strong>${esc(name)}</strong><small>${flow.steps.length} steps / ${flow.starts.length} routes</small></span>
      ${name === state.config.default_flow ? '<span class="default-tag">LIVE</span>' : ""}
    </button>`;
  }).join("");
  list.querySelectorAll("[data-flow]").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.flow !== state.selectedFlow) clearSimulationTrace();
    state.selectedFlow = button.dataset.flow;
    state.selectedStep = null;
    render();
  }));
}

function clearSimulationTrace() {
  state.simulation.events = [];
  state.simulation.nodeStates = {};
  state.simulation.message = "";
  state.simulation.flow = null;
  state.simulation.start = null;
}

function graphLevels(flow) {
  const levels = Object.fromEntries(flow.steps.map((step) => [step.id, 0]));
  const starts = new Set(flow.starts.map((start) => start.step));
  flow.steps.forEach((step) => { if (!starts.has(step.id)) levels[step.id] = 1; });
  for (let pass = 0; pass < flow.steps.length; pass += 1) {
    let changed = false;
    flow.steps.forEach((step) => step.targets.forEach((target) => {
      if (target.step === "$return" || levels[target.step] == null) return;
      const next = Math.max(levels[target.step], levels[step.id] + 1);
      if (next !== levels[target.step] && next < flow.steps.length) {
        levels[target.step] = next;
        changed = true;
      }
    }));
    if (!changed) break;
  }
  return levels;
}

function stepNodeHtml(step, starts) {
  const selected = state.selectedStep === step.id;
  const run = state.simulation.nodeStates[`step:${step.id}`];
  const startTags = orderedStarts({ starts }).filter((start) => start.step === step.id);
  const detail = step.type === "ai"
    ? `${step.role || step.family || "AI"} / ${step.provider || "No provider"}`
    : `${step.min_success} required / ${step.max_concurrency} concurrent`;
  return `<button class="graph-node ${step.type} ${selected ? "selected" : ""} ${run ? `run-${run.status}` : ""}" data-step-id="${esc(step.id)}" type="button">
    ${run ? `<i class="node-run-indicator" title="${esc(run.status)}"></i>` : ""}
    <span class="node-type">${esc(step.type)} STEP</span>
    <strong>${esc(step.id)}</strong>
    <small>${esc(detail)}</small>
    ${startTags.length ? `<span class="start-tags">${startTags.map((start) => `P${esc(start.priority)} / START / ${esc(start.when)}`).join("<br>")}</span>` : ""}
  </button>`;
}

function renderCanvas() {
  const flow = currentFlow();
  const canvas = $("#flow-canvas");
  const wrap = canvas.parentElement;
  const scrollLeft = wrap.scrollLeft;
  const scrollTop = wrap.scrollTop;
  $("#duplicate-flow").disabled = !flow || state.simulation.active;
  $("#delete-flow").disabled = !flow || Object.keys(state.config.flows).length <= 1 || state.simulation.active;
  $("#add-flow").disabled = state.simulation.active;
  $("#add-ai-step").disabled = !flow || state.simulation.active;
  $("#add-gate-step").disabled = !flow || state.simulation.active;
  if (!flow) {
    $("#flow-title").textContent = "Select a flow";
    $("#flow-structure").textContent = "FLOW STRUCTURE";
    canvas.innerHTML = '<div class="empty-node">Create a flow to begin an experiment.</div>';
    return;
  }

  $("#flow-title").textContent = state.selectedFlow;
  $("#flow-structure").textContent = `${flow.steps.length} STEPS / ${flow.starts.length} START ROUTES`;
  const levels = graphLevels(flow);
  const maxLevel = Math.max(0, ...Object.values(levels));
  const columns = Array.from({ length: maxLevel + 2 }, () => []);
  flow.steps.forEach((step) => columns[levels[step.id]].push(step));
  canvas.innerHTML = `<div class="graph-board" style="--graph-columns:${columns.length}">
    ${startChainHtml(flow)}
    <svg class="graph-edges" aria-hidden="true"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs></svg>
    <div class="graph-columns">
      ${columns.map((steps, index) => `<div class="graph-column" data-level="${index}">
        <span class="column-label">${index === columns.length - 1 ? "TERMINAL" : `STAGE ${index + 1}`}</span>
        ${steps.map((step) => stepNodeHtml(step, flow.starts)).join("")}
        ${index === columns.length - 1 ? `<button class="graph-node return ${state.selectedStep === "$return" ? "selected" : ""} ${state.simulation.nodeStates["step:$return"] ? `run-${state.simulation.nodeStates["step:$return"].status}` : ""}" data-step-id="$return" type="button">
          ${state.simulation.nodeStates["step:$return"] ? `<i class="node-run-indicator" title="${esc(state.simulation.nodeStates["step:$return"].status)}"></i>` : ""}
          <span class="node-type">RETURN TERMINAL</span><strong>Client response</strong><small>Flow output boundary</small>
        </button>` : ""}
      </div>`).join("")}
    </div>
  </div>`;
  canvas.querySelectorAll("[data-step-id]").forEach((button) => button.addEventListener("click", () => {
    state.selectedStep = button.dataset.stepId;
    const revealInspector = window.matchMedia("(max-width: 900px)").matches;
    render();
    if (revealInspector) requestAnimationFrame(() => $(".inspector").scrollIntoView({ behavior: "smooth", block: "start" }));
  }));
  requestAnimationFrame(drawEdges);
  wrap.scrollLeft = scrollLeft;
  wrap.scrollTop = scrollTop;
}

function drawEdges() {
  const flow = currentFlow();
  const board = $(".graph-board");
  const svg = board?.querySelector(".graph-edges");
  if (!flow || !board || !svg) return;
  const boardRect = board.getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${boardRect.width} ${boardRect.height}`);
  svg.setAttribute("width", boardRect.width);
  svg.setAttribute("height", boardRect.height);
  svg.querySelectorAll(".edge").forEach((edge) => edge.remove());
  const nodes = new Map([...board.querySelectorAll("[data-step-id]")].map((node) => [node.dataset.stepId, node]));
  flow.steps.forEach((step) => step.targets.forEach((target, index) => {
    const source = nodes.get(step.id);
    const destination = nodes.get(target.step);
    if (!source || !destination) return;
    const from = source.getBoundingClientRect();
    const to = destination.getBoundingClientRect();
    const x1 = from.right - boardRect.left;
    const y1 = from.top + from.height / 2 - boardRect.top + (index - (step.targets.length - 1) / 2) * 7;
    const x2 = to.left - boardRect.left;
    const y2 = to.top + to.height / 2 - boardRect.top;
    const bend = Math.max(26, (x2 - x1) * 0.45);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("class", `edge edge-${target.when}`);
    path.setAttribute("d", `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`);
    path.setAttribute("marker-end", "url(#arrow)");
    svg.appendChild(path);
    if (target.when !== "always") {
      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("class", "edge edge-label");
      label.setAttribute("x", String((x1 + x2) / 2));
      label.setAttribute("y", String((y1 + y2) / 2 - 5));
      label.textContent = target.when.replaceAll("_", " ");
      svg.appendChild(label);
    }
  }));
}

function renderInspector() {
  const container = $("#inspector-content");
  state.editorErrors.clear();
  const flow = currentFlow();
  if (!flow) {
    container.innerHTML = '<p class="empty-inspector">Select a flow to inspect its routes and steps.</p>';
    return;
  }
  if (state.selectedStep === "$return") {
    renderReturnInspector(container, flow);
  } else {
    const step = selectedStep();
    if (step?.type === "ai") renderAiInspector(container, flow, step);
    else if (step?.type === "gate") renderGateInspector(container, flow, step);
    else renderFlowInspector(container, flow);
  }
  if (state.simulation.active) {
    container.insertAdjacentHTML("afterbegin", '<p class="inspector-lock">Editing is locked while this simulation is active.</p>');
    container.querySelectorAll("input, select, textarea, button").forEach((control) => { control.disabled = true; });
  }
}

function renderFlowInspector(container, flow) {
  const stepIds = flow.steps.map((step) => step.id);
  const hasSimpleRouting = flow.starts.some((start) => start.when === "simple_request");
  const routing = flow.routing || {
    max_latest_user_chars: 800,
    max_conversation_chars: 4000,
    max_messages: 4,
    require_no_tools: true,
  };
  container.innerHTML = `
    <h3 class="inspector-title">Flow settings</h3>
    <div class="field"><label>Flow name</label><input id="flow-name-input" value="${esc(state.selectedFlow)}"></div>
    <div class="field"><label>Public aliases</label><input id="flow-aliases" value="${esc(flow.aliases.join(", "))}"></div>
    <hr class="inspector-divider">
    <div class="subheading"><span>START ROUTES</span><button class="mini-button" id="add-start" type="button">+ ADD</button></div>
    <div class="route-list">
      ${orderedStarts(flow).map((start) => `<div class="route-row start-route-row">
        <input data-start-priority="${start.declarationIndex}" type="number" min="1" value="${start.priority}" title="Priority">
        <select data-start-step="${start.declarationIndex}">${optionList(stepIds, start.step)}</select>
        <select data-start-when="${start.declarationIndex}">${optionList(startConditions, start.when)}</select>
        <button class="route-remove" data-start-remove="${start.declarationIndex}" type="button" ${flow.starts.length <= 1 ? "disabled" : ""}>x</button>
      </div>`).join("")}
      <p class="hint route-priority-hint">Lower priority numbers are evaluated first. The first matching route handles the request.</p>
    </div>
    ${hasSimpleRouting ? `<div class="subheading"><span>SIMPLE REQUEST LIMITS</span></div>
    <div class="field"><label>Latest user characters</label><input id="routing-user-chars" type="number" min="1" value="${routing.max_latest_user_chars}"></div>
    <div class="field"><label>Conversation characters</label><input id="routing-conversation-chars" type="number" min="1" value="${routing.max_conversation_chars}"></div>
    <div class="field"><label>Message count</label><input id="routing-message-count" type="number" min="1" value="${routing.max_messages}"></div>
    <label class="toggle-row"><input id="routing-no-tools" type="checkbox" ${routing.require_no_tools ? "checked" : ""}><span>Route only requests without tools</span></label>` : ""}
    <hr class="inspector-divider">
    <h3 class="inspector-title compact">Output</h3>
    <div class="field"><label>Output step</label><select id="output-step">${optionList(stepIds, flow.output.step)}</select></div>
    <label class="toggle-row"><input id="output-passthrough" type="checkbox" ${flow.output.passthrough_input_on_no_tool_calls ? "checked" : ""}><span>Pass through input when output has no tool calls</span></label>
    <button class="button inspector-action" id="make-default" type="button" ${state.config.default_flow === state.selectedFlow ? "disabled" : ""}>${state.config.default_flow === state.selectedFlow ? "Current live default" : "Make default flow"}</button>`;

  $("#flow-name-input").addEventListener("change", (event) => renameFlow(event.target.value.trim()));
  $("#flow-aliases").addEventListener("input", (event) => { flow.aliases = parseList(event.target.value); markDirty(); });
  $("#add-start").addEventListener("click", () => {
    flow.starts.push({ step: stepIds[0], when: "always", priority: nextStartPriority(flow) });
    markDirty(); render();
  });
  container.querySelectorAll("[data-start-priority]").forEach((input) => input.addEventListener("change", () => {
    flow.starts[Number(input.dataset.startPriority)].priority = Number(input.value); markDirty(); render();
  }));
  container.querySelectorAll("[data-start-step]").forEach((input) => input.addEventListener("change", () => {
    flow.starts[Number(input.dataset.startStep)].step = input.value; markDirty(); render();
  }));
  container.querySelectorAll("[data-start-when]").forEach((input) => input.addEventListener("change", () => {
    flow.starts[Number(input.dataset.startWhen)].when = input.value;
    if (input.value === "simple_request" && !flow.routing) flow.routing = clone(routing);
    markDirty(); render();
  }));
  container.querySelectorAll("[data-start-remove]").forEach((button) => button.addEventListener("click", () => {
    flow.starts.splice(Number(button.dataset.startRemove), 1); markDirty(); render();
  }));
  $("#output-step").addEventListener("change", (event) => { flow.output.step = event.target.value; markDirty(); render(); });
  $("#output-passthrough").addEventListener("change", (event) => { flow.output.passthrough_input_on_no_tool_calls = event.target.checked; markDirty(); });
  if (hasSimpleRouting) {
    flow.routing = routing;
    $("#routing-user-chars").addEventListener("input", (event) => { routing.max_latest_user_chars = Number(event.target.value); markDirty(); });
    $("#routing-conversation-chars").addEventListener("input", (event) => { routing.max_conversation_chars = Number(event.target.value); markDirty(); });
    $("#routing-message-count").addEventListener("input", (event) => { routing.max_messages = Number(event.target.value); markDirty(); });
    $("#routing-no-tools").addEventListener("change", (event) => { routing.require_no_tools = event.target.checked; markDirty(); });
  }
  $("#make-default").addEventListener("click", () => { state.config.default_flow = state.selectedFlow; markDirty(); render(); });
}

function targetsHtml(step, flow) {
  const destinations = [...flow.steps.map((item) => item.id).filter((id) => id !== step.id), "$return"];
  return `<div class="subheading"><span>TARGETS</span><button class="mini-button" id="add-target" type="button">+ ADD</button></div>
    <div class="route-list">
      ${step.targets.map((target, index) => `<div class="route-row">
        <select data-target-step="${index}">${optionList(destinations, target.step)}</select>
        <select data-target-when="${index}">${optionList(targetConditions, target.when || "always")}</select>
        <button class="route-remove" data-target-remove="${index}" type="button">x</button>
      </div>`).join("") || '<p class="hint route-empty">No outgoing route. This step cannot complete the flow.</p>'}
    </div>`;
}

function bindTargets(container, flow, step) {
  $("#add-target").addEventListener("click", () => {
    step.targets.push({ step: "$return", when: "always" }); markDirty(); render();
  });
  container.querySelectorAll("[data-target-step]").forEach((input) => input.addEventListener("change", () => {
    step.targets[Number(input.dataset.targetStep)].step = input.value; markDirty(); render();
  }));
  container.querySelectorAll("[data-target-when]").forEach((input) => input.addEventListener("change", () => {
    step.targets[Number(input.dataset.targetWhen)].when = input.value; markDirty(); render();
  }));
  container.querySelectorAll("[data-target-remove]").forEach((button) => button.addEventListener("click", () => {
    step.targets.splice(Number(button.dataset.targetRemove), 1); markDirty(); render();
  }));
}

function triStateOptions(value) {
  return `<option value="" ${value == null ? "selected" : ""}>Provider default</option><option value="true" ${value === true ? "selected" : ""}>Enabled</option><option value="false" ${value === false ? "selected" : ""}>Disabled</option>`;
}

function renderAiInspector(container, flow, step) {
  const prompts = Object.keys(state.config.prompts || {});
  const schemas = Object.keys(state.config.schemas || {});
  const validators = Object.keys(state.config.tool_validators || {});
  const gates = ancestorGateIds(flow, step.id);
  const repair = step.repair;
  const retry = step.retry;
  const fallback = step.fallback;
  container.innerHTML = `
    <div class="inspector-kicker">AI STEP</div><h3 class="inspector-title">${esc(step.id)}</h3>
    <div class="field"><label>Step ID</label><input id="step-id" value="${esc(step.id)}"></div>
    <div class="field"><label>Prompt</label><select data-ai-field="prompt">${optionList(prompts, step.prompt, "No linked prompt")}</select></div>
    <div class="field"><label>Prompt variables (JSON string map)</label><textarea id="prompt-variables" aria-describedby="prompt-variables-error">${esc(JSON.stringify(step.prompt_variables || {}, null, 2))}</textarea><p class="field-error" id="prompt-variables-error"></p></div>
    <hr class="inspector-divider">
    <div class="field"><label>Provider</label><select data-ai-field="provider">${optionList(Object.keys(state.config.providers), step.provider)}</select></div>
    <div class="field"><label>Model ID</label><input data-ai-field="model" list="model-options" value="${esc(step.model)}"></div>
    <button class="button inspector-action" id="discover-models" type="button">Discover models</button>
    <p class="model-result" id="model-result"></p>
    <div class="field-row">
      <div class="field"><label>Role</label><input data-ai-field="role" value="${esc(step.role || "general")}"></div>
      <div class="field"><label>Family</label><input data-ai-field="family" value="${esc(step.family || "")}" placeholder="Optional"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Conversation</label><select data-ai-field="conversation">${optionList(["none", "advisory", "full"], step.conversation || "none")}</select></div>
      <div class="field"><label>Activation</label><select data-ai-field="activation">${optionList(["single", "first"], step.activation || "single")}</select></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Max tokens</label><input id="max-tokens" value="${esc(step.max_tokens ?? "")}" placeholder="request or integer"></div>
      <div class="field"><label>Reasoning reserve</label><input data-number-field="reasoning_reserve" type="number" min="0" value="${esc(step.reasoning_reserve ?? 0)}"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Temperature</label><input data-optional-number="temperature" type="number" min="0" max="2" step="0.1" value="${esc(step.temperature ?? "")}"></div>
      <div class="field"><label>Context size</label><input data-optional-number="num_ctx" type="number" min="1" value="${esc(step.num_ctx ?? "")}"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Thinking</label><select id="think-field">${triStateOptions(step.think)}</select></div>
      <div class="field"><label>Keep alive</label><input id="keep-alive" value="${esc(step.keep_alive ?? "")}" placeholder="Provider default"></div>
    </div>
    <div class="field"><label>Response schema</label><select data-ai-field="response_schema">${optionList(schemas, step.response_schema, "None")}</select></div>
    <hr class="inspector-divider">
    <h3 class="inspector-title compact">Tool policy</h3>
    <div class="field"><label>Mode</label><select data-tools-field="mode">${optionList(["none", "client"], step.tools?.mode || "none")}</select></div>
    <div class="field-row">
      <div class="field"><label>Include</label><input data-tools-list="include" value="${esc((step.tools?.include || []).join(", "))}" placeholder="tool-a, tool-b"></div>
      <div class="field"><label>Exclude</label><input data-tools-list="exclude" value="${esc((step.tools?.exclude || []).join(", "))}" placeholder="tool-a, tool-b"></div>
    </div>
    <div class="field-row">
      <div class="field"><label>Validator</label><select data-tools-field="validator">${optionList(validators, step.tools?.validator, "None")}</select></div>
      <div class="field"><label>Max calls</label><input id="tools-max-calls" type="number" min="1" value="${esc(step.tools?.max_calls ?? "")}"></div>
    </div>
    <hr class="inspector-divider">
    <details class="advanced" ${repair ? "open" : ""}><summary>Repair</summary>
      <label class="toggle-row"><input id="repair-enabled" type="checkbox" ${repair ? "checked" : ""}><span>Enable schema repair</span></label>
      ${repair ? `<div class="field"><label>Repair prompt</label><select data-repair-field="prompt">${optionList(prompts, repair.prompt)}</select></div><div class="field"><label>Attempts</label><input data-repair-field="attempts" type="number" min="1" max="4" value="${esc(repair.attempts)}"></div>` : ""}
    </details>
    <details class="advanced" ${retry ? "open" : ""}><summary>Retry</summary>
      <label class="toggle-row"><input id="retry-enabled" type="checkbox" ${retry ? "checked" : ""}><span>Retry empty completions</span></label>
      ${retry ? `<div class="field-row"><div class="field"><label>Attempts</label><input data-retry-number="attempts" type="number" min="1" max="4" value="${esc(retry.attempts)}"></div><div class="field"><label>Token multiplier</label><input data-retry-number="max_tokens_multiplier" type="number" min="0.1" step="0.1" value="${esc(retry.max_tokens_multiplier)}"></div></div><div class="field"><label>Thinking on retry</label><select id="retry-think">${triStateOptions(retry.think)}</select></div>` : ""}
    </details>
    <details class="advanced" ${fallback ? "open" : ""}><summary>Fallback</summary>
      <label class="toggle-row"><input id="fallback-enabled" type="checkbox" ${fallback ? "checked" : ""} ${!gates.length ? "disabled" : ""}><span>Use best non-empty gate result</span></label>
      ${fallback ? `<div class="field"><label>Source gate</label><select data-fallback-field="gate">${optionList(gates, fallback.gate, gates.length ? null : "No ancestor gates")}</select>${gates.includes(fallback.gate) ? "" : '<p class="field-error visible">The configured fallback is not an ancestor of this step.</p>'}</div><div class="field"><label>Strategy</label><select data-fallback-field="strategy">${optionList(["best-nonempty"], fallback.strategy)}</select></div>` : ""}
    </details>
    <hr class="inspector-divider">
    ${targetsHtml(step, flow)}
    ${linkedPromptHtml(step)}
    <hr class="inspector-divider"><button class="remove-target" id="delete-step" type="button">Delete step</button>`;

  bindStepId(flow, step);
  container.querySelectorAll("[data-ai-field]").forEach((input) => input.addEventListener("change", () => {
    const field = input.dataset.aiField;
    if (["prompt", "family", "response_schema"].includes(field)) setOptional(step, field, input.value);
    else step[field] = input.value;
    markDirty(); render();
  }));
  container.querySelectorAll("[data-number-field]").forEach((input) => input.addEventListener("input", () => {
    step[input.dataset.numberField] = Number(input.value); markDirty();
  }));
  container.querySelectorAll("[data-optional-number]").forEach((input) => input.addEventListener("input", () => {
    setOptional(step, input.dataset.optionalNumber, parseOptionalNumber(input.value)); markDirty();
  }));
  $("#prompt-variables").addEventListener("input", (event) => validatePromptVariables(step, event.target));
  $("#max-tokens").addEventListener("change", (event) => { setOptional(step, "max_tokens", parseMaxTokens(event.target.value)); markDirty(); render(); });
  $("#think-field").addEventListener("change", (event) => { setOptional(step, "think", event.target.value === "" ? null : event.target.value === "true"); markDirty(); render(); });
  $("#keep-alive").addEventListener("change", (event) => {
    const value = event.target.value.trim();
    const parsed = value !== "" && Number.isFinite(Number(value)) ? Number(value) : value;
    setOptional(step, "keep_alive", parsed); markDirty(); render();
  });
  step.tools ||= { mode: "none", include: [], exclude: [] };
  container.querySelectorAll("[data-tools-field]").forEach((input) => input.addEventListener("change", () => {
    setOptional(step.tools, input.dataset.toolsField, input.value); markDirty(); render();
  }));
  container.querySelectorAll("[data-tools-list]").forEach((input) => input.addEventListener("input", () => {
    step.tools[input.dataset.toolsList] = parseList(input.value); markDirty();
  }));
  $("#tools-max-calls").addEventListener("change", (event) => { setOptional(step.tools, "max_calls", parseOptionalNumber(event.target.value)); markDirty(); render(); });
  $("#discover-models").addEventListener("click", () => discoverModels(step.provider));
  bindOptionalConfigs(container, step, prompts, gates);
  bindTargets(container, flow, step);
  bindLinkedPrompt(container, step);
  $("#delete-step").addEventListener("click", () => deleteStep(flow, step));
}

function linkedPromptHtml(step) {
  const prompt = step.prompt && state.config.prompts[step.prompt];
  if (!prompt) return "";
  return `<hr class="inspector-divider"><div class="linked-heading"><span>LINKED PROMPT</span><strong>${esc(step.prompt)}</strong></div>
    <p class="hint">Prompt edits affect every step linked to this prompt.</p>
    <div class="field"><label>System</label><textarea class="prompt-editor" data-prompt-field="system">${esc(prompt.system)}</textarea></div>
    <div class="field"><label>Context</label><textarea class="prompt-editor" data-prompt-field="context" placeholder="Optional">${esc(prompt.context || "")}</textarea></div>`;
}

function bindLinkedPrompt(container, step) {
  container.querySelectorAll("[data-prompt-field]").forEach((input) => input.addEventListener("input", () => {
    const prompt = state.config.prompts[step.prompt];
    if (input.dataset.promptField === "system") prompt.system = input.value;
    else setOptional(prompt, "context", input.value);
    markDirty();
  }));
}

function bindOptionalConfigs(container, step, prompts, gates) {
  $("#repair-enabled").addEventListener("change", (event) => {
    if (event.target.checked) step.repair = { prompt: prompts[0] || "", attempts: 1 };
    else delete step.repair;
    markDirty(); render();
  });
  container.querySelectorAll("[data-repair-field]").forEach((input) => input.addEventListener("change", () => {
    step.repair[input.dataset.repairField] = input.dataset.repairField === "attempts" ? Number(input.value) : input.value; markDirty(); render();
  }));
  $("#retry-enabled").addEventListener("change", (event) => {
    if (event.target.checked) step.retry = { attempts: 1, condition: "empty", max_tokens_multiplier: 2 };
    else delete step.retry;
    markDirty(); render();
  });
  container.querySelectorAll("[data-retry-number]").forEach((input) => input.addEventListener("input", () => {
    step.retry[input.dataset.retryNumber] = Number(input.value); markDirty();
  }));
  const retryThink = $("#retry-think");
  if (retryThink) retryThink.addEventListener("change", () => {
    setOptional(step.retry, "think", retryThink.value === "" ? null : retryThink.value === "true"); markDirty(); render();
  });
  $("#fallback-enabled").addEventListener("change", (event) => {
    if (event.target.checked) step.fallback = { gate: gates[0], strategy: "best-nonempty" };
    else delete step.fallback;
    markDirty(); render();
  });
  container.querySelectorAll("[data-fallback-field]").forEach((input) => input.addEventListener("change", () => {
    step.fallback[input.dataset.fallbackField] = input.value; markDirty(); render();
  }));
}

function validatePromptVariables(step, input) {
  const errorElement = $("#prompt-variables-error");
  const errorKey = `${state.selectedFlow}:${step.id}:prompt_variables`;
  try {
    const parsed = JSON.parse(input.value || "{}");
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("Expected a JSON object.");
    const invalidKey = Object.keys(parsed).find((key) => typeof parsed[key] !== "string");
    if (invalidKey) throw new Error(`Value for ${invalidKey} must be a string.`);
    step.prompt_variables = parsed;
    input.removeAttribute("aria-invalid");
    errorElement.textContent = "";
    errorElement.classList.remove("visible");
    state.editorErrors.delete(errorKey);
    markDirty();
  } catch (error) {
    input.setAttribute("aria-invalid", "true");
    errorElement.textContent = error.message;
    errorElement.classList.add("visible");
    state.editorErrors.set(errorKey, input);
  }
}

function renderGateInspector(container, flow, step) {
  container.innerHTML = `
    <div class="inspector-kicker">GATE STEP</div><h3 class="inspector-title">${esc(step.id)}</h3>
    <div class="field"><label>Step ID</label><input id="step-id" value="${esc(step.id)}"></div>
    <div class="field-row">
      <div class="field"><label>Minimum success</label><input data-gate-number="min_success" type="number" min="1" value="${esc(step.min_success)}"></div>
      <div class="field"><label>Max concurrency</label><input data-gate-number="max_concurrency" type="number" min="1" value="${esc(step.max_concurrency)}"></div>
    </div>
    <div class="field"><label>Deadline seconds</label><input id="gate-deadline" type="number" min="0.1" step="0.1" value="${esc(step.deadline_seconds ?? "")}" placeholder="No deadline"></div>
    <div class="field"><label>Completion</label><select data-gate-field="completion">${optionList(["all-or-deadline"], step.completion)}</select></div>
    <div class="field"><label>On failure</label><select data-gate-field="on_failure">${optionList(["fail"], step.on_failure)}</select></div>
    <hr class="inspector-divider">
    ${targetsHtml(step, flow)}
    <hr class="inspector-divider"><button class="remove-target" id="delete-step" type="button">Delete step</button>`;
  bindStepId(flow, step);
  container.querySelectorAll("[data-gate-number]").forEach((input) => input.addEventListener("input", () => {
    step[input.dataset.gateNumber] = Number(input.value); markDirty();
  }));
  container.querySelectorAll("[data-gate-field]").forEach((input) => input.addEventListener("change", () => {
    step[input.dataset.gateField] = input.value; markDirty(); render();
  }));
  $("#gate-deadline").addEventListener("change", (event) => { setOptional(step, "deadline_seconds", parseOptionalNumber(event.target.value)); markDirty(); render(); });
  bindTargets(container, flow, step);
  $("#delete-step").addEventListener("click", () => deleteStep(flow, step));
}

function renderReturnInspector(container, flow) {
  const run = state.simulation.nodeStates["step:$return"];
  container.innerHTML = `<div class="inspector-kicker">TERMINAL</div><h3 class="inspector-title">Client response</h3>
    <p class="inspector-copy">Every target to <code>$return</code> exits the graph. The configured output step is <strong>${esc(flow.output.step)}</strong>${flow.output.passthrough_input_on_no_tool_calls ? " and may pass through its input when it emits no tool calls" : ""}.</p>
    ${run ? '<p class="hint">The latest completed simulation output is shown below.</p>' : ""}`;
}

function bindStepId(flow, step) {
  $("#step-id").addEventListener("change", (event) => renameStep(flow, step, event.target.value.trim()));
}

function renameStep(flow, step, name) {
  if (!name || name === step.id) return;
  if (name === "$return" || flow.steps.some((item) => item.id === name)) {
    toast("Step IDs must be unique and cannot be $return.", true); renderInspector(); return;
  }
  const previous = step.id;
  step.id = name;
  flow.starts.forEach((start) => { if (start.step === previous) start.step = name; });
  flow.steps.forEach((item) => {
    item.targets.forEach((target) => { if (target.step === previous) target.step = name; });
    if (item.fallback?.gate === previous) item.fallback.gate = name;
  });
  const oldPlaceholder = `{{steps.${previous}}}`;
  const newPlaceholder = `{{steps.${name}}}`;
  rewriteFlowPromptReferences(flow, oldPlaceholder, newPlaceholder);
  if (flow.output.step === previous) flow.output.step = name;
  if (state.simulation.nodeStates[`step:${previous}`]) {
    state.simulation.nodeStates[`step:${name}`] = state.simulation.nodeStates[`step:${previous}`];
    delete state.simulation.nodeStates[`step:${previous}`];
  }
  state.selectedStep = name;
  markDirty(); render();
}

function rewriteFlowPromptReferences(flow, oldPlaceholder, newPlaceholder) {
  const prompts = state.config.prompts || {};
  const referenced = new Set(flow.steps.flatMap((item) => [item.prompt, item.repair?.prompt]).filter(Boolean));
  referenced.forEach((promptId) => {
    const prompt = prompts[promptId];
    if (!prompt || ![prompt.system, prompt.context].some((value) => typeof value === "string" && value.includes(oldPlaceholder))) return;
    const sharedOutsideFlow = Object.entries(state.config.flows).some(([flowName, otherFlow]) => flowName !== state.selectedFlow && otherFlow.steps.some((item) => item.prompt === promptId || item.repair?.prompt === promptId));
    let targetPrompt = prompt;
    let targetId = promptId;
    if (sharedOutsideFlow) {
      targetId = uniqueId(`${promptId}-${state.selectedFlow}`, Object.keys(prompts));
      targetPrompt = clone(prompt);
      prompts[targetId] = targetPrompt;
      flow.steps.forEach((item) => {
        if (item.prompt === promptId) item.prompt = targetId;
        if (item.repair?.prompt === promptId) item.repair.prompt = targetId;
      });
    }
    ["system", "context"].forEach((field) => {
      if (typeof targetPrompt[field] === "string") targetPrompt[field] = targetPrompt[field].split(oldPlaceholder).join(newPlaceholder);
    });
  });
}

function extendTerminalRoute(source, stepId) {
  let replaced = false;
  source.targets = source.targets.map((target) => {
    if (target.step !== "$return") return target;
    replaced = true;
    return { ...target, step: stepId };
  });
  if (!replaced) source.targets.push({ step: stepId, when: "always" });
}

function addAiStep() {
  const flow = currentFlow();
  if (!flow) return;
  const id = uniqueId("ai-step", flow.steps.map((step) => step.id));
  const step = {
    id,
    type: "ai",
    provider: Object.keys(state.config.providers)[0] || "",
    model: "",
    role: "general",
    conversation: "none",
    activation: "single",
    reasoning_reserve: 0,
    tools: { mode: "none", include: [], exclude: [] },
    targets: [{ step: "$return", when: "always" }],
  };
  const source = selectedStep() || flow.steps.find((item) => item.id === flow.output.step) || flow.steps.at(-1);
  flow.steps.push(step);
  if (source && source.id !== id) {
    const extendsOutput = source.id === flow.output.step;
    extendTerminalRoute(source, id);
    if (extendsOutput) {
      flow.output.step = id;
      flow.output.passthrough_input_on_no_tool_calls = false;
    }
  }
  else flow.starts.push({ step: id, when: "always", priority: nextStartPriority(flow) });
  state.selectedStep = id;
  markDirty(); render();
}

function addGateStep() {
  const flow = currentFlow();
  if (!flow) return;
  const source = selectedStep()?.type === "ai" ? selectedStep() : flow.steps.find((item) => item.type === "ai");
  if (!source) { toast("Add an AI step before adding a gate; gates require an AI source.", true); return; }
  const targetSteps = new Map(flow.steps.map((item) => [item.id, item]));
  if (source.targets.some((target) => targetSteps.get(target.step)?.type === "gate")) {
    toast(`Cannot add a gate: AI step "${source.id}" already directly targets a gate.`, true);
    return;
  }
  if (source.targets.some((target) => target.step === "$return" && (target.when || "always") !== "always")) {
    toast(`Cannot add a gate: AI step "${source.id}" has a conditional terminal route, but gate sources must be unconditional.`, true);
    return;
  }
  const id = uniqueId("gate", flow.steps.map((step) => step.id));
  const step = { id, type: "gate", min_success: 1, max_concurrency: 1, completion: "all-or-deadline", on_failure: "fail", targets: [{ step: "$return", when: "always" }] };
  flow.steps.push(step);
  const extendsOutput = source.id === flow.output.step;
  extendTerminalRoute(source, id);
  if (extendsOutput) {
    flow.output.step = id;
    flow.output.passthrough_input_on_no_tool_calls = false;
  }
  state.selectedStep = id;
  markDirty(); render();
}

function deleteStep(flow, step) {
  if (flow.steps.length <= 1) { toast("Cannot delete the only step in a flow.", true); return; }
  if (!confirm(`Delete step "${step.id}"?`)) return;
  const candidate = clone(flow);
  const removed = candidate.steps.find((item) => item.id === step.id);
  const incomingIds = candidate.steps
    .filter((item) => item.targets.some((target) => target.step === step.id))
    .map((item) => item.id);
  const uniqueIncoming = [...new Set(incomingIds)];
  const terminal = removed.targets.some((target) => target.step === "$return");
  const startReplacement = removed.targets.filter((target) => target.step !== "$return" && (target.when || "always") === "always");

  if (candidate.starts.some((start) => start.step === step.id) && startReplacement.length !== 1) {
    toast(`Cannot delete "${step.id}": its start route has no single unconditional successor.`, true);
    return;
  }
  if (candidate.output.step === step.id) {
    if (terminal && uniqueIncoming.length === 1) candidate.output.step = uniqueIncoming[0];
    else if (!terminal && startReplacement.length === 1) candidate.output.step = startReplacement[0].step;
    else {
      toast(`Cannot delete output step "${step.id}": a unique safe output replacement is not available.`, true);
      return;
    }
  }

  candidate.starts.forEach((start) => {
    if (start.step === step.id) start.step = startReplacement[0].step;
  });
  candidate.steps = candidate.steps.filter((item) => item.id !== step.id);
  for (const item of candidate.steps) {
    const nextTargets = [];
    for (const target of item.targets) {
      if (target.step !== step.id) {
        nextTargets.push(target);
        continue;
      }
      if (!removed.targets.length) {
        toast(`Cannot delete "${step.id}": incoming route from "${item.id}" would become terminal without a return.`, true);
        return;
      }
      for (const replacement of removed.targets) {
        const incomingWhen = target.when || "always";
        const outgoingWhen = replacement.when || "always";
        if (incomingWhen !== "always" && outgoingWhen !== "always" && incomingWhen !== outgoingWhen) {
          toast(`Cannot delete "${step.id}": conditional routes from "${item.id}" cannot be combined safely.`, true);
          return;
        }
        nextTargets.push({ step: replacement.step, when: incomingWhen === "always" ? outgoingWhen : incomingWhen });
      }
    }
    item.targets = nextTargets.filter((target, index, targets) => targets.findIndex((other) => other.step === target.step && other.when === target.when) === index);
    if (item.fallback?.gate === step.id) delete item.fallback;
  }

  const invalidReason = validateRunnableFlow(candidate);
  if (invalidReason) {
    toast(`Cannot delete "${step.id}": ${invalidReason}`, true);
    return;
  }
  state.config.flows[state.selectedFlow] = candidate;
  state.selectedStep = null;
  markDirty(); render();
}

function validateRunnableFlow(flow) {
  const steps = new Map(flow.steps.map((step) => [step.id, step]));
  if (!steps.has(flow.output.step)) return "the output would reference a missing step.";
  if (!flow.starts.length || !flow.starts.some((start) => start.when === "always")) return "the flow would have no ordinary start route.";
  if (flow.starts.some((start) => !steps.has(start.step))) return "a start route would reference a missing step.";
  const predecessors = new Map(flow.steps.map((step) => [step.id, new Set()]));
  for (const source of flow.steps) {
    const ownedGates = new Set();
    for (const target of source.targets) {
      if (target.step === "$return") continue;
      if (!steps.has(target.step)) return `route from "${source.id}" would reference a missing step.`;
      predecessors.get(target.step).add(source.id);
      if (steps.get(target.step).type === "gate") {
        if ((target.when || "always") !== "always") return `gate "${target.step}" would have a conditional source.`;
        ownedGates.add(target.step);
      }
    }
    if (ownedGates.size > 1) return `step "${source.id}" would feed multiple gates.`;
  }
  for (const gate of flow.steps.filter((item) => item.type === "gate")) {
    const sourceCount = predecessors.get(gate.id).size;
    if (!sourceCount) return `gate "${gate.id}" would have no source.`;
    if (gate.min_success > sourceCount) return `gate "${gate.id}" would require ${gate.min_success} successes from only ${sourceCount} source(s).`;
  }
  for (const aiStep of flow.steps.filter((item) => item.type === "ai")) {
    if (predecessors.get(aiStep.id).size > 1 && aiStep.activation !== "first") return `AI step "${aiStep.id}" would require activation "first" for explicit fan-in.`;
  }
  const visiting = new Set();
  const visited = new Set();
  let reachesReturn = false;
  const visit = (stepId) => {
    if (visiting.has(stepId)) throw new Error(`the flow would contain a cycle at "${stepId}".`);
    if (visited.has(stepId)) return;
    visiting.add(stepId);
    for (const target of steps.get(stepId).targets) {
      if (target.step === "$return") reachesReturn = true;
      else visit(target.step);
    }
    visiting.delete(stepId);
    visited.add(stepId);
  };
  try {
    flow.starts.forEach((start) => visit(start.step));
  } catch (error) {
    return error.message;
  }
  if (visited.size !== steps.size) return "one or more steps would become unreachable.";
  if (!reachesReturn) return "no reachable route would return a client response.";
  const returnable = new Map();
  const canReturn = (stepId) => {
    if (returnable.has(stepId)) return returnable.get(stepId);
    const result = steps.get(stepId).targets.some((target) => target.step === "$return" || canReturn(target.step));
    returnable.set(stepId, result);
    return result;
  };
  if (flow.starts.some((start) => !canReturn(start.step))) return "a start route would no longer reach the return terminal.";
  return null;
}

function renameFlow(name) {
  if (!name || name === state.selectedFlow) return;
  if (state.config.flows[name]) { toast("A flow with that name already exists.", true); renderInspector(); return; }
  const previous = state.selectedFlow;
  state.config.flows = Object.fromEntries(Object.entries(state.config.flows).map(([key, value]) => [key === previous ? name : key, value]));
  if (state.config.default_flow === previous) state.config.default_flow = name;
  if (Array.isArray(state.config.warmup_flows)) state.config.warmup_flows = state.config.warmup_flows.map((flowName) => flowName === previous ? name : flowName);
  state.selectedFlow = name;
  markDirty(); render();
}

function addFlow() {
  const name = uniqueId("new-flow", Object.keys(state.config.flows));
  const provider = Object.keys(state.config.providers)[0] || "";
  state.config.flows[name] = {
    aliases: [`${name}-alias`],
    starts: [{ step: "answer", when: "always", priority: 1 }],
    output: { step: "answer", passthrough_input_on_no_tool_calls: false },
    steps: [{ id: "answer", type: "ai", provider, model: "", role: "general", conversation: "full", activation: "single", reasoning_reserve: 0, tools: { mode: "none", include: [], exclude: [] }, targets: [{ step: "$return", when: "always" }] }],
  };
  state.selectedFlow = name;
  state.selectedStep = null;
  markDirty(); render();
}

function duplicateFlow() {
  const flow = currentFlow();
  if (!flow) return;
  const name = uniqueId(`${state.selectedFlow}-copy`, Object.keys(state.config.flows));
  const copied = clone(flow);
  const usedAliases = Object.entries(state.config.flows).flatMap(([flowName, item]) => [flowName, ...item.aliases]);
  copied.aliases = [uniqueId(`${name}-alias`, usedAliases)];
  state.config.flows[name] = copied;
  state.selectedFlow = name;
  state.selectedStep = null;
  markDirty(); render();
}

function deleteFlow() {
  if (Object.keys(state.config.flows).length <= 1 || !currentFlow()) return;
  if (!confirm(`Delete flow "${state.selectedFlow}"?`)) return;
  const removed = state.selectedFlow;
  const wasDefault = state.config.default_flow === removed;
  delete state.config.flows[removed];
  if (Array.isArray(state.config.warmup_flows)) state.config.warmup_flows = state.config.warmup_flows.filter((flowName) => flowName !== removed);
  state.selectedFlow = Object.keys(state.config.flows)[0];
  if (wasDefault) state.config.default_flow = state.selectedFlow;
  state.selectedStep = null;
  markDirty(); render();
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
    <div class="provider-card">
      <div class="provider-card-heading"><strong>${esc(name)}</strong><div class="provider-actions">
        <button class="mini-button" data-provider-discover="${esc(name)}" type="button">DISCOVER</button>
        <button class="mini-button remove" data-provider-remove="${esc(name)}" type="button">REMOVE</button>
      </div></div>
      <div class="field"><label>Name</label><input data-provider-name="${esc(name)}" value="${esc(name)}"></div>
      <div class="field"><label>Type</label><select data-provider-field="type" data-provider="${esc(name)}">${optionList(["ollama", "openai", "deepseek", "openai-compatible"], provider.type)}</select></div>
      <div class="field"><label>Base URL</label><input data-provider-field="base_url" data-provider="${esc(name)}" value="${esc(provider.base_url || "")}"></div>
      <div class="field-row">
        <div class="field"><label>API key env</label><input data-provider-field="api_key_env" data-provider="${esc(name)}" value="${esc(provider.api_key_env || "")}" placeholder="None"></div>
        <div class="field"><label>Timeout (s)</label><input data-provider-field="timeout_seconds" data-provider="${esc(name)}" type="number" min="1" value="${esc(provider.timeout_seconds || 1800)}"></div>
      </div>
      <div class="model-result" id="provider-result-${esc(name)}"></div>
    </div>`).join("");
  document.querySelectorAll("[data-provider-field]").forEach((input) => input.addEventListener("input", () => {
    const provider = state.config.providers[input.dataset.provider];
    const value = input.dataset.providerField === "timeout_seconds" ? Number(input.value) : input.value;
    setOptional(provider, input.dataset.providerField, value); markDirty();
  }));
  document.querySelectorAll("[data-provider-name]").forEach((input) => input.addEventListener("change", () => renameProvider(input.dataset.providerName, input.value.trim())));
  document.querySelectorAll("[data-provider-remove]").forEach((button) => button.addEventListener("click", () => removeProvider(button.dataset.providerRemove)));
  document.querySelectorAll("[data-provider-discover]").forEach((button) => button.addEventListener("click", () => discoverModels(button.dataset.providerDiscover, `#provider-result-${CSS.escape(button.dataset.providerDiscover)}`)));
}

function renameProvider(previous, name) {
  if (!name || name === previous) return;
  if (state.config.providers[name]) { toast("A provider with that name already exists.", true); renderProviders(); return; }
  state.config.providers = Object.fromEntries(Object.entries(state.config.providers).map(([key, value]) => [key === previous ? name : key, value]));
  Object.values(state.config.flows).forEach((flow) => flow.steps.forEach((step) => {
    if (step.type === "ai" && step.provider === previous) step.provider = name;
  }));
  markDirty(); renderProviders(); render();
}

function removeProvider(name) {
  const used = Object.entries(state.config.flows).flatMap(([flowName, flow]) => flow.steps
    .filter((step) => step.type === "ai" && step.provider === name)
    .map((step) => `${flowName}/${step.id}`));
  if (used.length) { toast(`Provider is used by: ${used.join(", ")}`, true); return; }
  if (Object.keys(state.config.providers).length <= 1) { toast("At least one provider is required.", true); return; }
  delete state.config.providers[name]; markDirty(); renderProviders(); render();
}

function addProvider() {
  const name = uniqueId("provider", Object.keys(state.config.providers));
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

function selectedRunKey() {
  return state.selectedStep ? `step:${state.selectedStep}` : null;
}

function appendRunOutput() {
  const key = selectedRunKey();
  const run = key ? state.simulation.nodeStates[key] : null;
  if (!run) return;
  const outputs = run.events.filter((event) => event.content || event.tool_calls?.length || event.tools?.length || event.error || event.event === "gate_progress");
  const outputHtml = outputs.length ? outputs.map((event) => {
    const content = event.content ? `<pre>${esc(event.content)}</pre>` : "";
    const calls = event.tool_calls?.length ? `<pre>${esc(JSON.stringify(event.tool_calls, null, 2))}</pre>` : "";
    const tools = event.tools?.length ? `<p>Validated tools: ${esc(event.tools.join(", "))}</p>` : "";
    const error = event.error ? `<pre class="error-output">${esc(event.error)}</pre>` : "";
    const progress = event.event === "gate_progress" ? `<p>${esc(`${event.successes} succeeded / ${event.failures} failed / ${event.pending} pending`)}</p>` : "";
    return `<div class="run-event-output"><span>${esc(event.event.replaceAll("_", " "))}${event.attempt ? ` / attempt ${event.attempt}` : ""}</span>${content}${calls}${tools}${error}${progress}</div>`;
  }).join("") : '<p class="run-empty">This node has status events but no model output.</p>';
  $("#inspector-content").insertAdjacentHTML("beforeend", `<hr class="inspector-divider"><section class="run-output">
    <div class="run-output-heading"><span>LIVE RUN</span><strong class="status-${esc(run.status)}">${esc(run.status)}</strong></div>
    ${outputHtml}<details><summary>Raw trace events (${run.events.length})</summary><pre>${esc(JSON.stringify(run.events, null, 2))}</pre></details>
  </section>`);
}

function simulationEventTarget(event) {
  const flow = state.config?.flows?.[state.simulation.flow];
  if (event.event === "simulation_completed") return "step:$return";
  if (event.node_id && flow?.steps.some((step) => step.id === event.node_id)) return `step:${event.node_id}`;
  if (event.stage && flow?.steps.some((step) => step.id === event.stage)) return `step:${event.stage}`;
  if (["request_started", "simulation_started"].includes(event.event)) {
    const start = state.simulation.start || ordinaryStart(flow);
    return start ? `step:${start}` : null;
  }
  if (["request_completed", "request_failed", "simulation_failed"].includes(event.event)) return "step:$return";
  return null;
}

function simulationEventStatus(event) {
  if (event.event === "step_failed" || event.event.endsWith("_failed") || event.event === "request_failed") return "failed";
  if (event.event.endsWith("_cancelled")) return "skipped";
  if (event.event.endsWith("_completed") || event.event === "tool_calls_validated") return "complete";
  return "running";
}

function handleSimulationEvent(event) {
  state.simulation.events.push(event);
  if (event.request_id) state.simulation.requestId = event.request_id;
  if (event.event === "flow_started" && event.node_id) {
    const flow = state.config?.flows?.[state.simulation.flow];
    if (flow?.steps.some((step) => step.id === event.node_id)) {
      const actualKey = `step:${event.node_id}`;
      const provisionalKey = state.simulation.start ? `step:${state.simulation.start}` : null;
      if (provisionalKey && provisionalKey !== actualKey) delete state.simulation.nodeStates[provisionalKey];
      state.simulation.start = event.node_id;
      state.selectedStep = event.node_id;
    }
  }
  const key = simulationEventTarget(event);
  if (key) {
    const run = state.simulation.nodeStates[key] || { status: "running", events: [] };
    run.events.push(event);
    run.status = simulationEventStatus(event);
    state.simulation.nodeStates[key] = run;
  }
  if (event.event === "simulation_started") {
    state.simulation.message = `Generation ${event.generation} started / request ${event.request_id.slice(0, 8)}`;
  } else if (event.event === "flow_started") {
    state.simulation.message = `Flow started at ${event.node_id}`;
  } else if (event.event === "gate_progress") {
    state.simulation.message = `${event.node_id || "gate"}: ${event.successes} complete, ${event.pending} pending`;
  } else if (event.event === "model_started") {
    state.simulation.message = `${event.node_id || "model"}: running ${event.model}`;
  } else if (event.event === "model_retrying") {
    state.simulation.message = `${event.node_id}: retrying empty completion`;
  } else if (event.event === "tool_calls_validated") {
    state.simulation.message = `${event.node_id}: validated ${event.tools.join(", ")}`;
  } else if (event.event === "simulation_completed") {
    state.simulation.message = event.tool_calls?.length
      ? `Stopped at client tool boundary / ${event.tool_calls.length} call(s) requested`
      : `Completed with ${event.usage?.output_tokens || 0} output tokens`;
  } else if (["simulation_failed", "request_failed"].includes(event.event)) {
    state.simulation.message = event.error;
  }
  render();
}

function renderSimulationStatus() {
  const simulation = state.simulation;
  const status = $("#simulation-status");
  const button = $("#run-simulation");
  status.classList.toggle("active", simulation.active);
  status.classList.toggle("failed", Object.values(simulation.nodeStates).some((node) => node.status === "failed"));
  status.querySelector("span:last-child").textContent = simulation.message || "Runs the real models and streams their trace.";
  button.textContent = simulation.active ? "Stop run" : "Run flow";
  button.classList.toggle("active", simulation.active);
  button.disabled = !currentFlow() || state.applying;
}

async function runSimulation() {
  if (state.simulation.active) {
    state.simulation.message = "Stopping run...";
    state.simulation.controller?.abort();
    renderSimulationStatus();
    return;
  }
  commitActiveEditor();
  if (hasEditorErrors()) { toast("Fix the highlighted inspector value before running the flow.", true); return; }
  const input = $("#simulation-input").value.trim();
  if (!input) { toast("Enter a prompt before running the flow.", true); $("#simulation-input").focus(); return; }
  if (state.dirty) {
    state.simulation.message = "Applying configuration before run...";
    renderSimulationStatus();
    if (!await applyConfig()) {
      state.simulation.message = "Run cancelled because the configuration could not be applied.";
      renderSimulationStatus();
      return;
    }
  }
  const controller = new AbortController();
  const start = ordinaryStart(currentFlow());
  state.simulation = { active: true, controller, flow: state.selectedFlow, start, requestId: null, events: [], nodeStates: {}, message: "Connecting to the live gateway..." };
  if (start) state.simulation.nodeStates[`step:${start}`] = { status: "running", events: [] };
  state.selectedStep = start || null;
  render();
  try {
    const response = await fetch("/api/simulations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: state.selectedFlow, input, max_tokens: Number($("#simulation-max-tokens").value) }),
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
    else { state.simulation.message = error.message; toast(error.message, true); }
  } finally {
    state.simulation.active = false;
    state.simulation.controller = null;
    render();
  }
}

function formatError(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => `${(item.loc || []).slice(1).join(" > ") || "config"}: ${item.msg}`).join("\n");
  return JSON.stringify(detail);
}

async function applyConfig() {
  if (state.applying) return false;
  commitActiveEditor();
  if (hasEditorErrors()) {
    toast("Fix the highlighted inspector value before applying configuration.", true);
    return false;
  }
  if (!state.dirty) return true;
  state.applying = true;
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
    const flows = state.config.flows;
    if (!flows || !Object.keys(flows).length) {
      state.unsupportedV1 = true;
      state.selectedFlow = null;
    } else if (!flows[state.selectedFlow]) {
      state.selectedFlow = state.config.default_flow || Object.keys(flows)[0];
    }
    if (state.selectedStep !== "$return" && !currentFlow()?.steps.some((step) => step.id === state.selectedStep)) state.selectedStep = null;
    toast(`Generation ${state.generation} is live${state.persisted ? " and saved" : ""}.`);
    render();
    if ($("#provider-drawer").classList.contains("open")) renderProviders();
    return true;
  } catch (error) {
    toast(error.message, true);
    button.disabled = false;
    return false;
  } finally {
    state.applying = false;
    button.querySelector("span").textContent = "Apply live";
    renderRuntime();
    renderSimulationStatus();
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) throw new Error("Could not load runtime configuration.");
    const payload = await response.json();
    state.config = payload.config;
    state.config.prompts ||= {};
    state.config.schemas ||= {};
    state.config.tool_validators ||= {};
    state.generation = payload.generation;
    state.persisted = payload.persisted;
    const flows = state.config.flows;
    if (!flows || !Object.keys(flows).length) {
      state.unsupportedV1 = true;
      state.selectedFlow = null;
    } else {
      state.selectedFlow = state.config.default_flow || Object.keys(flows)[0];
    }
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
$("#add-ai-step").addEventListener("click", addAiStep);
$("#add-gate-step").addEventListener("click", addGateStep);
$("#apply-button").addEventListener("click", applyConfig);
$("#run-simulation").addEventListener("click", runSimulation);
$("#providers-button").addEventListener("click", openProviders);
$("#close-providers").addEventListener("click", closeProviders);
$("#drawer-backdrop").addEventListener("click", closeProviders);
$("#add-provider").addEventListener("click", addProvider);
window.addEventListener("resize", drawEdges);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") { event.preventDefault(); applyConfig(); }
  if (event.key === "Escape") closeProviders();
});

initialize();
