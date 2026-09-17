"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  config: null,
  groups: [],
  checked: new Set(),   // image ids marked for deletion
  sizes: new Map(),     // image id -> bytes
  scanMode: null,       // mode of the scan the current groups came from
  polling: null,
};

function humanBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || response.statusText);
  return body;
}

function postJSON(path, payload) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function showError(message) {
  const box = $("setupError");
  box.textContent = message;
  box.hidden = !message;
}

function currentMode() {
  return document.querySelector('input[name="mode"]:checked').value;
}

function thresholdParam() {
  // The radio says what to scan *next*; the groups on screen came from whatever
  // mode actually ran, and only that mode has scores in the database. Grouping
  // on the radio would return nothing after flipping to Smart without rescanning.
  const mode = state.scanMode || currentMode();
  if (mode !== "smart") return `?mode=${mode}`;
  const threshold = (Number($("similarity").value) / 100).toFixed(2);
  return `?mode=${mode}&threshold=${threshold}`;
}

// --- scanning ---------------------------------------------------------------

async function browse() {
  try {
    const { folder } = await postJSON("/api/pick-folder", {});
    if (folder) $("folder").value = folder;
  } catch (err) {
    showError(err.message);
  }
}

async function scan() {
  const folder = $("folder").value.trim();
  showError("");
  if (!folder) { showError("Choose a folder first."); return; }

  try {
    await postJSON("/api/scan", { folder, mode: currentMode() });
  } catch (err) {
    showError(err.message);
    return;
  }

  $("review").hidden = true;
  $("footer").hidden = true;
  $("progress").hidden = false;
  $("scan").disabled = true;
  poll();
}

function poll() {
  clearInterval(state.polling);
  state.polling = setInterval(async () => {
    let progress;
    try {
      progress = await api("/api/progress");
    } catch {
      return;
    }
    renderProgress(progress);
    if (progress.phase === "done" || progress.phase === "error") {
      clearInterval(state.polling);
      $("scan").disabled = false;
      $("progress").hidden = true;
      if (progress.phase === "error") { showError(progress.message); return; }
      state.scanMode = progress.mode;
      await loadGroups(progress);
    }
  }, 400);
}

const PHASE_LABELS = {
  walking: "Looking for images…",
  hashing: "Hashing images…",
  matching: "Comparing…",
  embedding: "Running the CNN on candidates…",
  done: "Done",
  error: "Failed",
};

function renderProgress(progress) {
  $("phaseLabel").textContent = PHASE_LABELS[progress.phase] || progress.phase;
  const pct = progress.total ? (progress.done / progress.total) * 100 : 0;
  $("barFill").style.width = `${progress.phase === "matching" ? 100 : pct}%`;
  const parts = [];
  if (progress.total) parts.push(`${progress.done} / ${progress.total} files`);
  if (progress.cached_hits) parts.push(`${progress.cached_hits} from cache`);
  if (progress.errors) parts.push(`${progress.errors} unreadable`);
  $("progressCounts").textContent = parts.join(" · ");
}

// --- review -----------------------------------------------------------------

async function loadGroups(progress) {
  let data;
  try {
    data = await api(`/api/groups${thresholdParam()}`);
  } catch (err) {
    showError(err.message);
    return;
  }

  state.groups = data.groups;
  state.checked = new Set();
  state.sizes = new Map();

  $("review").hidden = false;
  $("emptyState").hidden = data.groups.length > 0;
  $("summary").textContent = summarize(data, progress);
  renderErrors(progress);
  renderGroups(data.groups);
  $("footer").hidden = data.groups.length === 0;
  updateCounter();
}

function summarize(data, progress) {
  const bits = [
    `${data.group_count} group${data.group_count === 1 ? "" : "s"}`,
    `${data.file_count} files`,
    `${humanBytes(data.reclaimable)} reclaimable`,
  ];
  if (progress) {
    bits.push(`${progress.image_count} images scanned in ${progress.elapsed}s`);
  }
  return bits.join(" · ");
}

function renderErrors(progress) {
  const box = $("errorBox");
  if (!progress || !progress.errors) { box.hidden = true; return; }
  box.hidden = false;
  $("errorSummary").textContent =
    `${progress.errors} file${progress.errors === 1 ? "" : "s"} could not be read`;
  const list = $("errorList");
  list.replaceChildren();
  for (const line of progress.error_files) {
    const item = document.createElement("li");
    item.textContent = line;
    list.append(item);
  }
}

function renderGroups(groups) {
  const container = $("groups");
  container.replaceChildren();

  for (const [index, group] of groups.entries()) {
    const card = document.createElement("section");
    card.className = "group";

    const head = document.createElement("header");
    head.innerHTML =
      `<span>Group ${index + 1} &middot; ${group.files.length} copies</span>` +
      `<span class="muted">${humanBytes(group.reclaimable)} reclaimable</span>`;
    card.append(head);

    const grid = document.createElement("div");
    grid.className = "tiles";
    for (const file of group.files) {
      state.sizes.set(file.id, file.size);
      if (!file.keep) state.checked.add(file.id);
      grid.append(renderTile(file));
    }
    card.append(grid);
    container.append(card);
  }
}

function renderTile(file) {
  const tile = document.createElement("label");
  tile.className = "tile";

  // The best copy starts unchecked, everything else checked. That is the only
  // thing that marks it out -- no badge, no colour.
  const box = document.createElement("input");
  box.type = "checkbox";
  box.checked = !file.keep;
  box.addEventListener("change", () => {
    if (box.checked) state.checked.add(file.id);
    else state.checked.delete(file.id);
    updateCounter();
  });

  // loading="lazy" lets the browser fetch each thumbnail only once its
  // card is near the viewport.
  const img = document.createElement("img");
  img.alt = file.name;
  img.loading = "lazy";
  img.src = `/api/thumb/${file.id}`;

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.innerHTML =
    `<span class="name" title="${escapeHTML(file.path)}">${escapeHTML(file.name)}</span>` +
    `<span>${file.width}&times;${file.height} &middot; ${humanBytes(file.size)}</span>` +
    `<span>${(file.similarity * 100).toFixed(1)}% similar</span>`;

  tile.append(box, img, meta);
  return tile;
}

function escapeHTML(text) {
  const node = document.createElement("span");
  node.textContent = text;
  return node.innerHTML;
}

function updateCounter() {
  let bytes = 0;
  for (const id of state.checked) bytes += state.sizes.get(id) || 0;
  $("counter").textContent =
    `${state.checked.size} file${state.checked.size === 1 ? "" : "s"} · ${humanBytes(bytes)}`;
  $("delete").disabled = state.checked.size === 0;
}

async function remove() {
  const ids = [...state.checked];
  if (!ids.length) return;
  $("delete").disabled = true;
  try {
    const result = await postJSON("/api/delete", { image_ids: ids });
    if (result.failed.length) {
      showError(
        `${result.failed.length} file(s) could not be moved to the Recycle Bin: ` +
        result.failed.map((f) => f.error).join("; ")
      );
    }
    const data = await api(`/api/groups${thresholdParam()}`);
    state.groups = data.groups;
    state.checked = new Set();
    state.sizes = new Map();
    $("emptyState").hidden = data.groups.length > 0;
    $("summary").textContent = summarize(data, null);
    renderGroups(data.groups);
    $("footer").hidden = data.groups.length === 0;
  } catch (err) {
    showError(err.message);
  }
  updateCounter();
}

// --- boot -------------------------------------------------------------------

async function init() {
  try {
    state.config = await api("/api/config");
  } catch {
    state.config = null;
  }
  if (state.config) {
    const slider = $("similarity");
    slider.min = Math.round(state.config.cosine_min * 100);
    slider.max = Math.round(state.config.cosine_max * 100);
    slider.value = Math.round(state.config.cosine_default * 100);
    $("similarityOut").textContent = `${slider.value}%`;
    $("smartMode").classList.remove("is-disabled");
    $("smartMode").querySelector("input").disabled = false;
  }

  $("browse").addEventListener("click", browse);
  $("scan").addEventListener("click", scan);
  $("delete").addEventListener("click", remove);
  $("folder").addEventListener("keydown", (e) => { if (e.key === "Enter") scan(); });
  $("similarity").addEventListener("input", (e) => {
    $("similarityOut").textContent = `${e.target.value}%`;
  });
  $("similarity").addEventListener("change", () => {
    if (state.groups.length) loadGroups(null);
  });
  for (const radio of document.querySelectorAll('input[name="mode"]')) {
    radio.addEventListener("change", () => {
      $("sliderRow").hidden = currentMode() !== "smart";
    });
  }
}

init();
