const state = {
  project: null,
  frameIndex: -1,
  image: null,
  selectedId: null,
  mode: "select",
  drag: null,
  dirty: false,
  polygonDraft: [],
  autosaveTimer: null,
  labelModalOpen: false,
  aiEnabled: false,
  aiDevice: localStorage.getItem("autolabel.aiDevice") || "cuda:0",
  selectedAiModels: loadSelectedAiModels(),
  aiBusy: false,
};

const AI_MODELS = [
  {
    id: "sam3.1",
    label: "SAM3.1",
    description: "实例分割精修与文本查找同类",
    available: true,
  },
];

const CATEGORY_COLORS = [
  "#14746f",
  "#b85c38",
  "#4f46e5",
  "#b45309",
  "#0f766e",
  "#c2410c",
  "#7c3aed",
  "#be123c",
  "#1d4ed8",
  "#15803d",
];

const els = {
  uploadForm: document.getElementById("uploadForm"),
  projectNameInput: document.getElementById("projectNameInput"),
  videoInput: document.getElementById("videoInput"),
  serverVideoPathInput: document.getElementById("serverVideoPathInput"),
  uploadButton: document.getElementById("uploadButton"),
  exportButton: document.getElementById("exportButton"),
  downloadButton: document.getElementById("downloadButton"),
  status: document.getElementById("status"),
  frameCount: document.getElementById("frameCount"),
  annotationCount: document.getElementById("annotationCount"),
  samplerBadge: document.getElementById("samplerBadge"),
  projectMeta: document.getElementById("projectMeta"),
  selectedMeta: document.getElementById("selectedMeta"),
  frameLabelSummary: document.getElementById("frameLabelSummary"),
  videoLabelSummary: document.getElementById("videoLabelSummary"),
  annotationCategoryPicker: document.getElementById("annotationCategoryPicker"),
  newLabelInput: document.getElementById("newLabelInput"),
  addLabelButton: document.getElementById("addLabelButton"),
  frameList: document.getElementById("frameList"),
  annotationList: document.getElementById("annotationList"),
  labelModal: document.getElementById("labelModal"),
  labelModalMeta: document.getElementById("labelModalMeta"),
  labelModalOptions: document.getElementById("labelModalOptions"),
  modalNewLabelInput: document.getElementById("modalNewLabelInput"),
  modalAddLabelButton: document.getElementById("modalAddLabelButton"),
  canvas: document.getElementById("canvas"),
  emptyState: document.getElementById("emptyState"),
  selectTool: document.getElementById("selectTool"),
  rectTool: document.getElementById("rectTool"),
  polygonTool: document.getElementById("polygonTool"),
  circleTool: document.getElementById("circleTool"),
  finishPolygonButton: document.getElementById("finishPolygonButton"),
  deleteButton: document.getElementById("deleteButton"),
  saveButton: document.getElementById("saveButton"),
  aiAssistToggle: document.getElementById("aiAssistToggle"),
  samPromptInput: document.getElementById("samPromptInput"),
  aiModelPicker: document.getElementById("aiModelPicker"),
  aiDeviceSelect: document.getElementById("aiDeviceSelect"),
  samRefineButton: document.getElementById("samRefineButton"),
  samFindButton: document.getElementById("samFindButton"),
  aiStatus: document.getElementById("aiStatus"),
};

const ctx = els.canvas.getContext("2d");

boot();

function boot() {
  els.uploadForm.addEventListener("submit", createProject);
  els.exportButton.addEventListener("click", exportProject);
  els.downloadButton.addEventListener("click", downloadProjectBundle);
  els.addLabelButton.addEventListener("click", addLabelFromInput);
  els.modalAddLabelButton.addEventListener("click", addLabelFromModal);
  els.modalNewLabelInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addLabelFromModal();
    }
  });
  els.labelModal.addEventListener("click", (event) => {
    if (event.target === els.labelModal) {
      closeLabelModal();
    }
  });
  els.newLabelInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addLabelFromInput();
    }
  });
  els.selectTool.addEventListener("click", () => setMode("select"));
  els.rectTool.addEventListener("click", () => setMode("rectangle"));
  els.polygonTool.addEventListener("click", () => setMode("polygon"));
  els.circleTool.addEventListener("click", () => setMode("circle"));
  els.finishPolygonButton.addEventListener("click", finishPolygonDraft);
  els.deleteButton.addEventListener("click", deleteSelectedAnnotation);
  els.saveButton.addEventListener("click", saveCurrentFrame);
  els.aiDeviceSelect.addEventListener("change", () => {
    state.aiDevice = els.aiDeviceSelect.value;
    localStorage.setItem("autolabel.aiDevice", state.aiDevice);
    setStatus(`AI 推理设备已切换到 ${selectedAiDeviceLabel()}`, selectedAiDeviceLabel());
  });
  els.aiAssistToggle.addEventListener("change", () => {
    syncAiEnabledFromDom();
    setStatus(
      state.aiEnabled
        ? "AI 辅助标注已开启：Ctrl+J 精修当前对象，Ctrl+K 查找当前帧同类"
        : "AI 辅助标注已关闭，当前为纯手工标注",
      state.aiEnabled ? "AI 已开启" : "未开启",
    );
    render();
  });
  els.samRefineButton.addEventListener("click", refineSelectedWithSam31);
  els.samFindButton.addEventListener("click", findSimilarWithSam31);
  els.canvas.addEventListener("mousedown", onMouseDown);
  els.canvas.addEventListener("mousemove", onMouseMove);
  els.canvas.addEventListener("dblclick", onCanvasDoubleClick);
  window.addEventListener("mouseup", onMouseUp);
  window.addEventListener("keydown", onKeyDown);
  loadAiDevices();
  renderAiModelPicker();
  render();
}

function loadSelectedAiModels() {
  try {
    const saved = JSON.parse(localStorage.getItem("autolabel.aiModels") || "[]");
    if (Array.isArray(saved) && saved.length) return saved;
  } catch {
    // Fall back to the default model.
  }
  return ["sam3.1"];
}

async function loadAiDevices() {
  try {
    const response = await fetch("/api/ai/sam31/status");
    if (!response.ok) return;
    const payload = await response.json();
    const devices = payload.devices?.filter((item) => item.available) || [];
    if (!devices.length) return;
    els.aiDeviceSelect.innerHTML = "";
    devices.forEach((device) => {
      const option = document.createElement("option");
      option.value = device.id;
      option.textContent = device.label || device.id;
      els.aiDeviceSelect.appendChild(option);
    });
    if (!devices.some((device) => device.id === state.aiDevice)) {
      state.aiDevice = devices.find((device) => device.id !== "cpu")?.id || devices[0].id;
      localStorage.setItem("autolabel.aiDevice", state.aiDevice);
    }
    els.aiDeviceSelect.value = state.aiDevice;
    renderAiControls();
  } catch {
    // Device discovery is best-effort; the static defaults remain usable.
  }
}

async function createProject(event) {
  event.preventDefault();
  const file = els.videoInput.files?.[0];
  const serverVideoPath = els.serverVideoPathInput.value.trim();
  if (!file && !serverVideoPath) {
    els.status.textContent = "请上传视频，或填写服务器视频路径";
    return;
  }

  setBusy(true, "创建中");
  try {
    const formData = new FormData(els.uploadForm);
    const response = await fetch("/api/projects", { method: "POST", body: formData });
    if (!response.ok) throw new Error(await response.text());
    state.project = await response.json();
    state.frameIndex = 0;
    state.selectedId = currentFrame()?.annotations[0]?.id || null;
    state.polygonDraft = [];
    state.dirty = false;
    state.project.label_catalog = state.project.label_catalog || [];
    await loadFrameImage();
    els.status.textContent = `${state.project.video_name} · 已抽帧，可直接手工标注 · 存储于 ${state.project.project_dir}`;
    render();
  } catch (error) {
    els.status.textContent = `处理失败：${cleanErrorMessage(error.message)}`;
  } finally {
    setBusy(false);
  }
}

async function exportProject() {
  if (!state.project) return;
  if (state.dirty) await saveCurrentFrame();
  window.open(`/api/projects/${state.project.id}/export/coco`, "_blank");
}

async function downloadProjectBundle() {
  if (!state.project) return;
  if (state.dirty) await saveCurrentFrame();
  window.open(`/api/projects/${state.project.id}/export/bundle`, "_blank");
}

function setBusy(isBusy, label = "创建手工标注项目") {
  els.uploadButton.disabled = isBusy;
  els.uploadButton.textContent = isBusy ? label : "创建手工标注项目";
}

function setMode(mode) {
  state.mode = mode;
  if (mode !== "polygon") {
    state.polygonDraft = [];
  }
  els.selectTool.classList.toggle("active", mode === "select");
  els.rectTool.classList.toggle("active", mode === "rectangle");
  els.polygonTool.classList.toggle("active", mode === "polygon");
  els.circleTool.classList.toggle("active", mode === "circle");
  renderCanvas();
}

function currentFrame() {
  return state.project?.frames[state.frameIndex] || null;
}

async function selectFrame(index) {
  if (!state.project || index === state.frameIndex) return;
  if (state.dirty) await saveCurrentFrame();
  state.frameIndex = index;
  state.selectedId = currentFrame()?.annotations[0]?.id || null;
  state.polygonDraft = [];
  await loadFrameImage();
  render();
}

async function loadFrameImage() {
  const frame = currentFrame();
  if (!state.project || !frame) {
    state.image = null;
    return;
  }
  state.image = await loadImage(`/api/projects/${state.project.id}/frames/${frame.id}/image`);
  els.canvas.width = frame.width;
  els.canvas.height = frame.height;
}

async function saveCurrentFrame() {
  const frame = currentFrame();
  if (!state.project || !frame) return;
  clearTimeout(state.autosaveTimer);
  const response = await fetch(`/api/projects/${state.project.id}/frames/${frame.id}/annotations`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ annotations: frame.annotations }),
  });
  if (!response.ok) {
    els.status.textContent = `保存失败：${await response.text()}`;
    return;
  }
  const updated = await response.json();
  state.project.frames[state.frameIndex] = updated;
  syncLabelCatalogWithFrames();
  state.dirty = false;
  els.status.textContent = `已保存 ${updated.id}`;
  render();
}

function deleteSelectedAnnotation() {
  const frame = currentFrame();
  if (!frame || !state.selectedId) return;
  frame.annotations = frame.annotations.filter((item) => item.id !== state.selectedId);
  state.selectedId = frame.annotations[0]?.id || null;
  markDirty();
  closeLabelModal();
  render();
}

function render() {
  renderProjectMeta();
  renderFrames();
  renderLabelSummaries();
  renderAnnotationCategoryPicker();
  renderAnnotations();
  renderCanvas();
  renderLabelModal();
  els.frameCount.textContent = state.project?.frames.length || 0;
  els.annotationCount.textContent = currentFrame()?.annotations.length || 0;
  els.samplerBadge.textContent = samplerLabel(state.project?.frame_sampler || "segmentation_diverse");
  els.selectedMeta.textContent = state.selectedId ? state.selectedId : "0 selected";
  renderAiControls();
}

function renderAiModelPicker() {
  els.aiModelPicker.innerHTML = "";
  AI_MODELS.forEach((model) => {
    const option = document.createElement("label");
    const checked = state.selectedAiModels.includes(model.id);
    option.className = `ai-model-option${checked ? " active" : ""}${model.available ? "" : " disabled"}`;
    option.title = model.description;
    option.innerHTML = `
      <input type="checkbox" value="${escapeAttr(model.id)}" ${checked ? "checked" : ""} ${model.available ? "" : "disabled"} />
      <span>${escapeHtml(model.label)}</span>
    `;
    const input = option.querySelector("input");
    input.addEventListener("change", () => {
      if (input.checked) {
        state.selectedAiModels = [...new Set([...state.selectedAiModels, model.id])];
      } else {
        state.selectedAiModels = state.selectedAiModels.filter((id) => id !== model.id);
      }
      localStorage.setItem("autolabel.aiModels", JSON.stringify(state.selectedAiModels));
      renderAiModelPicker();
      renderAiControls();
    });
    els.aiModelPicker.appendChild(option);
  });
}

function renderAiControls() {
  const frame = currentFrame();
  const annotation = frame?.annotations.find((item) => item.id === state.selectedId);
  const sam31Selected = aiModelSelected("sam3.1");
  els.aiAssistToggle.checked = state.aiEnabled;
  els.samPromptInput.disabled = !state.aiEnabled || state.aiBusy;
  els.aiDeviceSelect.disabled = state.aiBusy;
  els.aiDeviceSelect.value = state.aiDevice;
  els.aiModelPicker.querySelectorAll("input").forEach((input) => {
    input.disabled = state.aiBusy || !AI_MODELS.find((model) => model.id === input.value)?.available;
  });
  els.samRefineButton.disabled = !state.aiEnabled || !sam31Selected || !state.project || !annotation || state.aiBusy;
  els.samFindButton.disabled = !state.aiEnabled || !sam31Selected || !state.project || state.aiBusy;
  els.samRefineButton.textContent = state.aiBusy ? "AI 处理中" : "AI 精修";
  els.samFindButton.textContent = state.aiBusy ? "AI 处理中" : "查找同类";
  if (!state.aiEnabled) {
    els.aiStatus.textContent = "未开启";
  } else if (!state.selectedAiModels.length) {
    els.aiStatus.textContent = "未选模型";
  } else if (!state.project) {
    els.aiStatus.textContent = "等待项目";
  } else if (state.aiBusy) {
    els.aiStatus.textContent = "推理中";
  } else if (!sam31Selected) {
    els.aiStatus.textContent = "SAM3.1 未选";
  } else {
    els.aiStatus.textContent = annotation ? `${selectedAiDeviceLabel()} 可用` : `${selectedAiDeviceLabel()} 可查找`;
  }
  if (annotation && !els.samPromptInput.value.trim()) {
    els.samPromptInput.placeholder = annotation.category || "object";
  }
}

function aiModelSelected(modelId) {
  return state.selectedAiModels.includes(modelId);
}

function renderProjectMeta() {
  if (!state.project) {
    els.projectMeta.textContent = "等待项目";
    return;
  }
  const frame = currentFrame();
  els.projectMeta.textContent = `${state.project.name || state.project.id} · ${frame ? `${frame.timestamp_sec}s` : ""}`;
}

function renderFrames() {
  els.frameList.innerHTML = "";
  const frames = state.project?.frames || [];
  frames.forEach((frame, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `frame-item${index === state.frameIndex ? " active" : ""}`;
    button.title = `切换到 ${frame.id}；A 上一帧，S 下一帧`;
    button.innerHTML = `
      <img src="/api/projects/${state.project.id}/frames/${frame.id}/image" alt="" />
      <span class="frame-meta">
        <strong>${frame.id}</strong>
        <span>${frame.timestamp_sec}s · src ${frame.source_frame_index}</span>
        <span>${frame.annotations.length} anns</span>
      </span>
    `;
    button.addEventListener("click", () => selectFrame(index));
    els.frameList.appendChild(button);
    if (index === state.frameIndex) {
      queueMicrotask(() => {
        button.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
      });
    }
  });
}

function renderAnnotations() {
  els.annotationList.innerHTML = "";
  const frame = currentFrame();
  if (!frame) return;
  frame.annotations.forEach((annotation, index) => {
    const card = document.createElement("div");
    card.className = `annotation-card${annotation.id === state.selectedId ? " active" : ""}`;
    const swatch = categoryColor(annotation.category);
    card.innerHTML = `
      <div class="annotation-head">
        <strong><span class="label-swatch" style="background:${swatch}"></span>#${index + 1} ${escapeHtml(annotation.category)}</strong>
        <span>${shapeLabel(annotation.shape_type)}</span>
      </div>
      <div class="annotation-source">${escapeHtml(annotation.source || "manual")}</div>
      ${annotation.score !== undefined && annotation.source?.startsWith("sam3.1") ? `<div class="annotation-source">score ${Number(annotation.score).toFixed(3)}</div>` : ""}
      <div class="fields">
        <label>类别<input data-field="category" type="text" value="${escapeAttr(annotation.category)}" /></label>
      </div>
    `;
    card.addEventListener("click", () => {
      state.selectedId = annotation.id;
      render();
    });
    card.querySelectorAll("input").forEach((input) => {
      input.addEventListener("change", (evt) => updateAnnotation(annotation.id, evt.target.dataset.field, evt.target.value));
    });
    els.annotationList.appendChild(card);
  });
}

async function refineSelectedWithSam31() {
  syncAiEnabledFromDom();
  if (!state.aiEnabled) {
    setStatus("请先开启 AI 辅助标注", "未开启");
    return;
  }
  if (!aiModelSelected("sam3.1")) {
    setStatus("请先在 AI 模型中勾选 SAM3.1", "未选模型");
    return;
  }
  const frame = currentFrame();
  const annotation = frame?.annotations.find((item) => item.id === state.selectedId);
  if (!state.project || !frame || !annotation) {
    setStatus("请先选择一个需要精修的对象", "未选对象");
    return;
  }
  setStatus("已触发 SAM3.1 精修，正在准备当前帧", "准备精修");
  if (state.dirty) await saveCurrentFrame();
  const prompt = samPrompt(annotation);
  await runAiTask(async () => {
    const response = await fetch(`/api/projects/${state.project.id}/frames/${frame.id}/ai/sam31/refine`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ annotation, prompt, device: state.aiDevice }),
    });
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    const refined = payload.annotations?.[0];
    if (!refined) {
      setStatus("SAM3.1 没有返回可用 mask", "无结果");
      return;
    }
    const index = frame.annotations.findIndex((item) => item.id === annotation.id);
    refined.id = annotation.id;
    refined.category = annotation.category || refined.category || prompt || "object";
    frame.annotations[index] = refined;
    state.selectedId = refined.id;
    mergeLabels([refined.category]);
    markDirty();
    setStatus(`SAM3.1 已精修 ${refined.category}`, "精修完成");
    render();
  }, `SAM3.1 正在用 ${selectedAiDeviceLabel()} 精修当前对象，首次加载模型会比较慢`);
}

async function findSimilarWithSam31() {
  syncAiEnabledFromDom();
  if (!state.aiEnabled) {
    setStatus("请先开启 AI 辅助标注", "未开启");
    return;
  }
  if (!aiModelSelected("sam3.1")) {
    setStatus("请先在 AI 模型中勾选 SAM3.1", "未选模型");
    return;
  }
  const frame = currentFrame();
  if (!state.project || !frame) return;
  setStatus("已触发 SAM3.1 查找同类，正在准备当前帧", "准备查找");
  if (state.dirty) await saveCurrentFrame();
  const selected = frame.annotations.find((item) => item.id === state.selectedId);
  const prompt = samPrompt(selected);
  await runAiTask(async () => {
    const response = await fetch(`/api/projects/${state.project.id}/frames/${frame.id}/ai/sam31/find`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, category: selected?.category || prompt, max_results: 8, threshold: 0.28, device: state.aiDevice }),
    });
    if (!response.ok) throw new Error(await response.text());
    const payload = await response.json();
    const incoming = payload.annotations || [];
    const additions = incoming.filter((candidate) => !frame.annotations.some((item) => bboxIoU(item.bbox, candidate.bbox) > 0.75));
    additions.forEach((candidate) => {
      candidate.id = candidate.id || nextAnnotationId();
      candidate.category = selected?.category || candidate.category || prompt;
      candidate.source = candidate.source || "sam3.1_suggestion";
      frame.annotations.push(candidate);
    });
    if (additions.length) {
      state.selectedId = additions[0].id;
      mergeLabels(additions.map((item) => item.category));
      markDirty();
    }
    setStatus(`SAM3.1 返回 ${incoming.length} 个候选，新增 ${additions.length} 个`, `新增 ${additions.length}`);
    render();
  }, `SAM3.1 正在用 ${selectedAiDeviceLabel()} 查找当前帧同类，文本提示越准确结果越稳`);
}

async function runAiTask(task, message) {
  if (state.aiBusy) return;
  state.aiBusy = true;
  renderAiControls();
  setStatus(message || "SAM3.1 正在推理，首次加载模型会比较慢", "推理中");
  try {
    await task();
  } catch (error) {
    setStatus(`SAM3.1 失败：${cleanErrorMessage(error.message)}`, "失败");
  } finally {
    state.aiBusy = false;
    renderAiControls();
  }
}

function samPrompt(annotation) {
  return els.samPromptInput.value.trim() || annotation?.category || "object";
}

function syncAiEnabledFromDom() {
  state.aiEnabled = Boolean(els.aiAssistToggle.checked);
}

function setStatus(message, aiMessage) {
  els.status.textContent = message;
  if (aiMessage) {
    els.aiStatus.textContent = aiMessage;
  }
}

function selectedAiDeviceLabel() {
  return els.aiDeviceSelect.selectedOptions[0]?.textContent || state.aiDevice;
}

function renderLabelSummaries() {
  renderLabelChipList(els.frameLabelSummary, summarizeFrameLabels());
  renderLabelChipList(els.videoLabelSummary, summarizeVideoLabels());
}

function renderAnnotationCategoryPicker() {
  const frame = currentFrame();
  const annotation = frame?.annotations.find((item) => item.id === state.selectedId);
  els.annotationCategoryPicker.innerHTML = "";
  if (!annotation) return;

  const title = document.createElement("div");
  title.className = "picker-title";
  title.textContent = "当前标注可选 Label";
  els.annotationCategoryPicker.appendChild(title);

  const optionList = document.createElement("div");
  optionList.className = "category-option-list";
  const labels = ensureLabelCatalog();
  const options = labels.length ? labels : [annotation.category || "object"];
  options.forEach((label) => {
    const option = document.createElement("label");
    option.className = "category-option";
    option.style.borderColor = `${categoryColor(label)}44`;
    option.style.background = hexToRgba(categoryColor(label), annotation.category === label ? 0.16 : 0.08);
    option.innerHTML = `
      <input type="checkbox" ${annotation.category === label ? "checked" : ""} />
      <span class="label-swatch" style="background:${categoryColor(label)}"></span>
      <span>${escapeHtml(label)}</span>
    `;
    const input = option.querySelector("input");
    input.addEventListener("change", () => setAnnotationCategory(annotation.id, label));
    option.addEventListener("click", (event) => {
      if (event.target !== input) {
        event.preventDefault();
        setAnnotationCategory(annotation.id, label);
      }
    });
    optionList.appendChild(option);
  });
  els.annotationCategoryPicker.appendChild(optionList);
}

function renderLabelModal() {
  const frame = currentFrame();
  const annotation = frame?.annotations.find((item) => item.id === state.selectedId);
  els.labelModalOptions.innerHTML = "";
  els.labelModal.classList.toggle("hidden", !state.labelModalOpen || !annotation);
  els.labelModal.setAttribute("aria-hidden", String(!state.labelModalOpen || !annotation));
  if (!state.labelModalOpen || !annotation) return;

  els.labelModalMeta.textContent = `${annotation.id} · ${shapeLabel(annotation.shape_type)}`;
  const labels = ensureLabelCatalog();
  const options = labels.length ? labels : [annotation.category || "object"];
  options.forEach((label) => {
    const option = document.createElement("label");
    const active = annotation.category === label;
    option.className = `modal-option${active ? " active" : ""}`;
    option.style.color = categoryColor(label);
    option.style.borderColor = `${categoryColor(label)}44`;
    option.style.background = hexToRgba(categoryColor(label), active ? 0.16 : 0.08);
    option.innerHTML = `
      <input type="checkbox" ${active ? "checked" : ""} />
      <span class="label-swatch" style="background:${categoryColor(label)}"></span>
      <span>${escapeHtml(label)}</span>
    `;
    const input = option.querySelector("input");
    input.addEventListener("change", () => {
      setAnnotationCategory(annotation.id, label);
      closeLabelModal();
    });
    option.addEventListener("click", (event) => {
      if (event.target !== input) {
        event.preventDefault();
        setAnnotationCategory(annotation.id, label);
        closeLabelModal();
      }
    });
    els.labelModalOptions.appendChild(option);
  });
}

function summarizeFrameLabels() {
  const frame = currentFrame();
  if (!frame) return [];
  const counts = new Map();
  frame.annotations.forEach((annotation) => {
    const key = annotation.category || "object";
    counts.set(key, (counts.get(key) || 0) + 1);
  });
  return [...counts.entries()].map(([name, count]) => ({ name, count }));
}

function summarizeVideoLabels() {
  const counts = new Map();
  ensureLabelCatalog().forEach((label) => counts.set(label, 0));
  const frames = state.project?.frames || [];
  frames.forEach((frame) => {
    frame.annotations.forEach((annotation) => {
      const key = annotation.category || "object";
      counts.set(key, (counts.get(key) || 0) + 1);
    });
  });
  return [...counts.entries()]
    .sort((a, b) => a[0].localeCompare(b[0], "zh-CN"))
    .map(([name, count]) => ({ name, count }));
}

function renderLabelChipList(container, items) {
  container.innerHTML = "";
  if (!items.length) {
    const chip = document.createElement("div");
    chip.className = "label-chip empty";
    chip.textContent = "暂无";
    container.appendChild(chip);
    return;
  }

  items.forEach((item) => {
    const chip = document.createElement("div");
    chip.className = "label-chip";
    chip.style.borderColor = `${categoryColor(item.name)}33`;
    chip.style.background = hexToRgba(categoryColor(item.name), 0.12);
    chip.style.color = categoryColor(item.name);
    chip.textContent = `${item.name} · ${item.count}`;
    container.appendChild(chip);
  });
}

function renderCanvas() {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  const frame = currentFrame();
  els.emptyState.classList.toggle("hidden", Boolean(frame));
  if (!frame || !state.image) return;

  ctx.drawImage(state.image, 0, 0, frame.width, frame.height);
  frame.annotations.forEach((annotation) => drawAnnotation(annotation, annotation.id === state.selectedId));
  drawDraft();
}

function drawAnnotation(annotation, selected) {
  const color = categoryColor(annotation.category);
  const fill = hexToRgba(color, selected ? 0.28 : 0.16);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = fill;
  ctx.lineWidth = selected ? 4 : 2;

  if (annotation.shape_type === "polygon" && annotation.points?.length >= 3) {
    ctx.beginPath();
    ctx.moveTo(annotation.points[0][0], annotation.points[0][1]);
    annotation.points.slice(1).forEach((point) => ctx.lineTo(point[0], point[1]));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  } else if (annotation.shape_type === "circle" && annotation.points?.length >= 2) {
    const center = annotation.points[0];
    const edge = annotation.points[1];
    const radius = Math.hypot(edge[0] - center[0], edge[1] - center[1]);
    ctx.beginPath();
    ctx.arc(center[0], center[1], radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  } else {
    const [x, y, w, h] = annotation.bbox;
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
  }

  const label = `${annotation.category} · ${shapeLabel(annotation.shape_type)}`;
  const labelX = annotation.bbox[0];
  const labelY = Math.max(0, annotation.bbox[1] - 22);
  ctx.font = "13px ui-sans-serif, system-ui, sans-serif";
  const width = Math.max(96, ctx.measureText(label).width + 12);
  ctx.fillStyle = color;
  ctx.fillRect(labelX, labelY, width, 20);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, labelX + 6, labelY + 14);
  ctx.restore();
}

function drawDraft() {
  if (state.drag?.preview) {
    drawAnnotation(state.drag.preview, true);
  }
  if (state.mode === "polygon" && state.polygonDraft.length > 0) {
    ctx.save();
    ctx.strokeStyle = "#ff8a00";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(state.polygonDraft[0][0], state.polygonDraft[0][1]);
    state.polygonDraft.slice(1).forEach((point) => ctx.lineTo(point[0], point[1]));
    ctx.stroke();
    state.polygonDraft.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
      ctx.fillStyle = "#ff8a00";
      ctx.fill();
    });
    ctx.restore();
  }
}

function onMouseDown(event) {
  const frame = currentFrame();
  if (!frame) return;
  const point = canvasPoint(event);

  if (state.mode === "polygon") {
    state.polygonDraft.push([point.x, point.y]);
    renderCanvas();
    return;
  }

  if (state.mode === "rectangle") {
    state.drag = {
      type: "rectangle",
      start: point,
      preview: makeRectangleAnnotation(point.x, point.y, point.x, point.y),
    };
    return;
  }

  if (state.mode === "circle") {
    state.drag = {
      type: "circle",
      start: point,
      preview: makeCircleAnnotation(point, point),
    };
    return;
  }

  const hit = [...frame.annotations].reverse().find((annotation) => pointHitsAnnotation(point, annotation));
  state.selectedId = hit?.id || null;
  if (hit) {
    state.drag = {
      type: "move",
      start: point,
      id: hit.id,
      snapshot: cloneAnnotation(hit),
    };
  }
  render();
}

function onMouseMove(event) {
  const frame = currentFrame();
  if (!frame || !state.drag) return;
  const point = canvasPoint(event);

  if (state.drag.type === "rectangle") {
    state.drag.preview = makeRectangleAnnotation(state.drag.start.x, state.drag.start.y, point.x, point.y);
    renderCanvas();
    return;
  }

  if (state.drag.type === "circle") {
    state.drag.preview = makeCircleAnnotation(state.drag.start, point);
    renderCanvas();
    return;
  }

  if (state.drag.type === "move") {
    const annotation = frame.annotations.find((item) => item.id === state.drag.id);
    if (!annotation) return;
    const dx = point.x - state.drag.start.x;
    const dy = point.y - state.drag.start.y;
    applyAnnotationMove(annotation, state.drag.snapshot, dx, dy, frame);
    annotation.source = "manual";
    state.dirty = true;
    renderCanvas();
  }
}

function onMouseUp() {
  const frame = currentFrame();
  if (!frame || !state.drag) return;
  if (state.drag.type === "move") {
    markDirty();
  }
  if (state.drag.type === "rectangle" || state.drag.type === "circle") {
    const annotation = state.drag.preview;
    if (annotation.bbox[2] > 8 && annotation.bbox[3] > 8) {
      frame.annotations.push(annotation);
      state.selectedId = annotation.id;
      mergeLabels([annotation.category]);
      markDirty();
      openLabelModal();
    }
  }
  state.drag = null;
  render();
}

function onCanvasDoubleClick() {
  if (state.mode === "polygon") {
    finishPolygonDraft();
  }
}

function onKeyDown(event) {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f") {
    event.preventDefault();
    openLabelModal();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "j") {
    event.preventDefault();
    refineSelectedWithSam31();
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    findSimilarWithSam31();
    return;
  }
  if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target?.tagName)) {
    return;
  }
  if (event.key === "Escape") {
    closeLabelModal();
    state.polygonDraft = [];
    state.drag = null;
    renderCanvas();
  }
  if (event.key === "Enter" && state.mode === "polygon") {
    finishPolygonDraft();
  }
  if (event.key === "Delete" || event.key === "Backspace") {
    event.preventDefault();
    deleteSelectedAnnotation();
  }
  if (event.key.toLowerCase() === "a") {
    event.preventDefault();
    cycleFrame(-1);
  }
  if (event.key.toLowerCase() === "s") {
    event.preventDefault();
    cycleFrame(1);
  }
}

async function cycleFrame(step) {
  if (!state.project?.frames?.length) return;
  const total = state.project.frames.length;
  const nextIndex = (state.frameIndex + step + total) % total;
  await selectFrame(nextIndex);
}

function openLabelModal() {
  if (!state.project || !state.selectedId) return;
  state.labelModalOpen = true;
  renderLabelModal();
  queueMicrotask(() => els.modalNewLabelInput.focus());
}

function closeLabelModal() {
  state.labelModalOpen = false;
  els.modalNewLabelInput.value = "";
  renderLabelModal();
}

function finishPolygonDraft() {
  const frame = currentFrame();
  if (!frame || state.polygonDraft.length < 3) return;
  frame.annotations.push(makePolygonAnnotation(state.polygonDraft));
  state.selectedId = frame.annotations.at(-1)?.id || null;
  mergeLabels([frame.annotations.at(-1)?.category || "object"]);
  state.polygonDraft = [];
  markDirty();
  openLabelModal();
  render();
}

function updateAnnotation(id, field, value) {
  const frame = currentFrame();
  if (!frame) return;
  const annotation = frame.annotations.find((item) => item.id === id);
  if (!annotation) return;
  if (field === "category") {
    annotation.category = value.trim() || "object";
    mergeLabels([annotation.category]);
  }
  annotation.source = "manual";
  markDirty();
  render();
}

function setAnnotationCategory(id, label) {
  const frame = currentFrame();
  if (!frame) return;
  const annotation = frame.annotations.find((item) => item.id === id);
  if (!annotation) return;
  annotation.category = label;
  annotation.source = "manual";
  mergeLabels([label]);
  markDirty();
  render();
}

function addLabelFromInput() {
  const value = els.newLabelInput.value.trim();
  if (!value || !state.project) return;
  mergeLabels([value]);
  persistLabelCatalog();
  els.newLabelInput.value = "";
  render();
}

function addLabelFromModal() {
  const value = els.modalNewLabelInput.value.trim();
  if (!value || !state.project || !state.selectedId) return;
  mergeLabels([value]);
  persistLabelCatalog();
  setAnnotationCategory(state.selectedId, value);
  els.modalNewLabelInput.value = "";
  closeLabelModal();
}

function makeRectangleAnnotation(x1, y1, x2, y2) {
  const frame = currentFrame();
  const bbox = clampBBox(pointsToBBox([x1, y1], [x2, y2]), frame);
  return {
    id: nextAnnotationId(),
    category: "object",
    shape_type: "rectangle",
    bbox,
    points: [[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]],
    segmentation: [bboxPolygon(bbox)],
    score: 1,
    source: "manual",
  };
}

function makePolygonAnnotation(points) {
  const frame = currentFrame();
  const bbox = clampBBox(polygonBBox(points), frame);
  return {
    id: nextAnnotationId(),
    category: "object",
    shape_type: "polygon",
    bbox,
    points: points.map((point) => [point[0], point[1]]),
    segmentation: [points.flatMap((point) => [point[0], point[1]])],
    score: 1,
    source: "manual",
  };
}

function makeCircleAnnotation(center, edge) {
  const frame = currentFrame();
  const radius = Math.hypot(edge.x - center.x, edge.y - center.y);
  const bbox = clampBBox([center.x - radius, center.y - radius, radius * 2, radius * 2], frame);
  const lockedCenter = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2];
  const lockedRadius = Math.min(bbox[2], bbox[3]) / 2;
  return {
    id: nextAnnotationId(),
    category: "object",
    shape_type: "circle",
    bbox,
    points: [lockedCenter, [lockedCenter[0] + lockedRadius, lockedCenter[1]]],
    segmentation: [circlePolygon(lockedCenter, lockedRadius)],
    score: 1,
    source: "manual",
  };
}

function nextAnnotationId() {
  return `ann_${Math.random().toString(36).slice(2, 10)}`;
}

function markDirty() {
  state.dirty = true;
  queueAutosave();
}

function queueAutosave() {
  if (!state.project || state.frameIndex < 0) return;
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = setTimeout(() => {
    saveCurrentFrame();
  }, 500);
}

function mergeLabels(labels) {
  if (!state.project) return;
  const merged = ensureLabelCatalog();
  labels.forEach((item) => {
    const label = String(item || "").trim();
    if (label && !merged.includes(label)) {
      merged.push(label);
    }
  });
  state.project.label_catalog = merged;
}

function ensureLabelCatalog() {
  if (!state.project) return [];
  state.project.label_catalog = Array.from(
    new Set((state.project.label_catalog || []).map((item) => String(item || "").trim()).filter(Boolean)),
  );
  return state.project.label_catalog;
}

function syncLabelCatalogWithFrames() {
  if (!state.project) return;
  const labels = [];
  state.project.frames.forEach((frame) => {
    frame.annotations.forEach((annotation) => labels.push(annotation.category));
  });
  mergeLabels(labels);
  persistLabelCatalog();
}

async function persistLabelCatalog() {
  if (!state.project) return;
  await fetch(`/api/projects/${state.project.id}/labels`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ labels: ensureLabelCatalog() }),
  });
}

function cloneAnnotation(annotation) {
  return JSON.parse(JSON.stringify(annotation));
}

function applyAnnotationMove(annotation, snapshot, dx, dy, frame) {
  annotation.points = (snapshot.points || []).map((point) => [point[0] + dx, point[1] + dy]);
  if (annotation.shape_type === "polygon" && annotation.points.length >= 3) {
    annotation.bbox = clampBBox(polygonBBox(annotation.points), frame);
    annotation.segmentation = [annotation.points.flatMap((point) => [point[0], point[1]])];
    return;
  }
  if (annotation.shape_type === "circle" && annotation.points.length >= 2) {
    const center = annotation.points[0];
    const edge = annotation.points[1];
    const radius = Math.hypot(edge[0] - center[0], edge[1] - center[1]);
    annotation.bbox = clampBBox([center[0] - radius, center[1] - radius, radius * 2, radius * 2], frame);
    const box = annotation.bbox;
    const lockedCenter = [box[0] + box[2] / 2, box[1] + box[3] / 2];
    const lockedRadius = Math.min(box[2], box[3]) / 2;
    annotation.points = [lockedCenter, [lockedCenter[0] + lockedRadius, lockedCenter[1]]];
    annotation.segmentation = [circlePolygon(lockedCenter, lockedRadius)];
    return;
  }
  annotation.bbox = clampBBox([snapshot.bbox[0] + dx, snapshot.bbox[1] + dy, snapshot.bbox[2], snapshot.bbox[3]], frame);
  annotation.points = [[annotation.bbox[0], annotation.bbox[1]], [annotation.bbox[0] + annotation.bbox[2], annotation.bbox[1] + annotation.bbox[3]]];
  annotation.segmentation = [bboxPolygon(annotation.bbox)];
}

function pointHitsAnnotation(point, annotation) {
  if (annotation.shape_type === "circle" && annotation.points?.length >= 2) {
    const center = annotation.points[0];
    const edge = annotation.points[1];
    const radius = Math.hypot(edge[0] - center[0], edge[1] - center[1]);
    return Math.hypot(point.x - center[0], point.y - center[1]) <= radius;
  }
  return pointInBBox(point, annotation.bbox);
}

function canvasPoint(event) {
  const rect = els.canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * els.canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * els.canvas.height,
  };
}

function pointInBBox(point, bbox) {
  const [x, y, w, h] = bbox;
  return point.x >= x && point.x <= x + w && point.y >= y && point.y <= y + h;
}

function bboxIoU(a, b) {
  if (!a || !b) return 0;
  const [ax, ay, aw, ah] = a.map(Number);
  const [bx, by, bw, bh] = b.map(Number);
  const ax2 = ax + aw;
  const ay2 = ay + ah;
  const bx2 = bx + bw;
  const by2 = by + bh;
  const ix1 = Math.max(ax, bx);
  const iy1 = Math.max(ay, by);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const intersection = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const union = aw * ah + bw * bh - intersection;
  return union > 0 ? intersection / union : 0;
}

function clampBBox(bbox, frame) {
  let [x, y, w, h] = bbox.map(Number);
  w = Math.max(1, Math.min(w, frame.width));
  h = Math.max(1, Math.min(h, frame.height));
  x = Math.max(0, Math.min(x, frame.width - w));
  y = Math.max(0, Math.min(y, frame.height - h));
  return [x, y, w, h];
}

function bboxPolygon([x, y, w, h]) {
  return [x, y, x + w, y, x + w, y + h, x, y + h];
}

function pointsToBBox(a, b) {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])];
}

function polygonBBox(points) {
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const xMin = Math.min(...xs);
  const yMin = Math.min(...ys);
  return [xMin, yMin, Math.max(...xs) - xMin, Math.max(...ys) - yMin];
}

function circlePolygon(center, radius, sides = 24) {
  const coords = [];
  for (let index = 0; index < sides; index += 1) {
    const angle = (Math.PI * 2 * index) / sides;
    coords.push(center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle));
  }
  return coords;
}

function shapeLabel(shapeType) {
  return { rectangle: "矩形", polygon: "多边形", circle: "圆形" }[shapeType || "rectangle"];
}

function samplerLabel(value) {
  return {
    segmentation_diverse: "平衡关键帧",
  }[value] || "平衡关键帧";
}

function categoryColor(category) {
  const key = String(category || "object");
  let hash = 0;
  for (let index = 0; index < key.length; index += 1) {
    hash = (hash * 31 + key.charCodeAt(index)) >>> 0;
  }
  return CATEGORY_COLORS[hash % CATEGORY_COLORS.length];
}

function hexToRgba(hex, alpha) {
  const clean = hex.replace("#", "");
  const value = clean.length === 3
    ? clean.split("").map((char) => char + char).join("")
    : clean;
  const bigint = Number.parseInt(value, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = `${src}?t=${Date.now()}`;
  });
}

function cleanErrorMessage(text) {
  return String(text).replace(/^"|"$/g, "");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("'", "&#039;");
}
