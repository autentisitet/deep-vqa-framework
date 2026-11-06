const API = "/api/v1";

const $ = (id) => document.getElementById(id);
const IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "bmp", "tiff", "webp"]);
const VIDEO_EXTENSIONS = new Set(["mp4", "avi", "mov", "mkv", "webm", "wmv"]);

const state = {
  capabilities: {
    authRequired: false,
    historyEnabled: false,
    visualizationsEnabled: false,
  },
  file: null,
  history: [],
  hasEvaluation: false,
  lang: localStorage.getItem("deep-vqa-lang") || "zh",
  modelId: null,
  models: [],
  objectUrl: null,
  selected: new Set(),
};

const i18n = {
  zh: {
    apiFailed: "无法连接 API",
    brandSubtitle: "图像与视频质量评估工作台",
    chooseFile: "选择文件",
    connecting: "正在连接服务",
    dataset: "数据集",
    datasetMos: "数据集 MOS",
    device: "设备",
    done: "评估完成。",
    doubleClickHint: "双击任意图像可放大查看。",
    dropMedia: "拖放文件到这里",
    evaluate: "开始质量评估",
    evaluating: "评估中...",
    evaluatingHint: "视频推理可能需要更长时间。",
    evaluatingMedia: "模型正在评估媒体",
    evaluationFailed: "评估失败",
    exportAll: "导出全部",
    exportSelected: "导出已选",
    feature: "原图 vs 特征图",
    generating: "生成中...",
    gradcam: "原图 vs Grad-CAM",
    heroDescription:
      "系统根据文件类型自动匹配 IQA 或 VQA 模型，返回归一化分数、数据集 MOS、推理延迟和媒体信息。图像还可以生成特征图与 Grad-CAM。",
    heroTitle: "从媒体上传到质量评估与响应可视化，一屏完成。",
    history: "评估日志",
    historyDescription: "当前浏览器会话的服务器端 SQLite 记录。",
    historyEmpty: "还没有评估记录。",
    historyPreview: "服务器记录",
    imageRoute: "IQA 图像路由",
    interpretation: "03 · 响应可视化",
    latency: "推理延迟",
    levels: "层",
    loadedModels: "已加载模型",
    logout: "退出登录",
    mediaSize: "媒体尺寸",
    model: "模型",
    noModel: "对应的服务模型尚未加载",
    normalizedScore: "归一化分数",
    notEvaluated: "尚未评估",
    original: "原图",
    ready: "服务正常",
    refresh: "刷新",
    resultDescription: "字段与 FastAPI EvaluationResponse 保持一致。",
    resultTitle: "02 · 评估结果",
    selectAll: "全选",
    selectMedia: "01 · 选择媒体",
    selectMediaDescription: "模型会根据图片或视频自动路由。",
    serviceOverview: "服务概览",
    supportedFormats: "支持 JPG、PNG、WEBP、MP4、MOV、MKV、AVI 等格式",
    unavailable: "服务不可用",
    unsupportedMedia: "不支持该文件格式",
    videoRoute: "VQA 视频路由",
    visualizationFailed: "可视化失败",
    visualize: "生成特征图与 Grad-CAM",
    vizDone: "解释性可视化已生成。",
    waitingForEvaluation: "等待质量评估",
    waitingForFile: "等待文件",
    waitingHint: "上传媒体并点击“开始质量评估”。",
  },
  en: {
    apiFailed: "Unable to connect to API",
    brandSubtitle: "Image & video quality assessment workspace",
    chooseFile: "Choose file",
    connecting: "Connecting",
    dataset: "Dataset",
    datasetMos: "Dataset MOS",
    device: "Device",
    done: "Evaluation completed.",
    doubleClickHint: "Double-click any image to enlarge it.",
    dropMedia: "Drop a file here",
    evaluate: "Evaluate quality",
    evaluating: "Evaluating...",
    evaluatingHint: "Video inference may take longer.",
    evaluatingMedia: "The model is evaluating this media",
    evaluationFailed: "Evaluation failed",
    exportAll: "Export all",
    exportSelected: "Export selected",
    feature: "Original vs feature map",
    generating: "Generating...",
    gradcam: "Original vs Grad-CAM",
    heroDescription:
      "Images are routed to IQA and videos to VQA. The workspace shows " +
      "normalized score, dataset MOS, latency, media metadata, feature maps and Grad-CAM.",
    heroTitle: "From media upload to quality assessment and response visualization.",
    history: "Evaluation history",
    historyDescription: "Server-side SQLite records for the current browser session.",
    historyEmpty: "No evaluations have been recorded yet.",
    historyPreview: "Server records",
    imageRoute: "IQA image route",
    interpretation: "03 · Response visualization",
    latency: "Inference latency",
    levels: "layers",
    loadedModels: "Loaded models",
    logout: "Sign out",
    mediaSize: "Media size",
    model: "Model",
    noModel: "The matching serving model is not loaded",
    normalizedScore: "Normalized score",
    notEvaluated: "Not evaluated",
    original: "Original",
    ready: "Service ready",
    refresh: "Refresh",
    resultDescription: "Fields are aligned with FastAPI EvaluationResponse.",
    resultTitle: "02 · Evaluation result",
    selectAll: "Select all",
    selectMedia: "01 · Select media",
    selectMediaDescription: "The model is selected automatically from the media type.",
    serviceOverview: "Service overview",
    supportedFormats: "Supports JPG, PNG, WEBP, MP4, MOV, MKV and AVI",
    unavailable: "Service unavailable",
    unsupportedMedia: "Unsupported media format",
    videoRoute: "VQA video route",
    visualizationFailed: "Visualization failed",
    visualize: "Generate feature maps & Grad-CAM",
    vizDone: "Interpretability visuals generated.",
    waitingForEvaluation: "Waiting for evaluation",
    waitingForFile: "Waiting for file",
    waitingHint: "Upload media and select Evaluate quality.",
  },
};

const t = (key) => i18n[state.lang][key] || key;

function setText(selector, value) {
  const node = document.querySelector(selector);
  if (node) node.textContent = value;
}

function notify(message, error = false, target = "notice") {
  const node = $(target);
  node.textContent = message;
  node.className = `notice show${error ? " error" : ""}`;
}

function clearNotice(target = "notice") {
  $(target).className = "notice";
}

async function readJson(response) {
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : {};
}

function apiError(response, data) {
  const error = new Error(data.error || data.detail || `HTTP ${response.status}`);
  error.code = Number.isInteger(data.code) ? data.code : null;
  error.status = response.status;
  return error;
}

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    credentials: "include",
    // The anonymous session is a server-issued HttpOnly cookie; the client must
    // not claim an identity of its own, so no session header is sent.
    headers: { ...(options.headers || {}) },
  });
  const data = await readJson(response);
  if (response.status === 401) {
    window.location.replace("/login.html");
    throw apiError(response, data);
  }
  if (!response.ok) {
    throw apiError(response, data);
  }
  return data;
}

async function loadCapabilities() {
  const response = await fetch(`${API}/frontend-settings`, {
    cache: "no-store",
    credentials: "include",
  });
  const capabilities = await readJson(response);
  if (!response.ok) {
    throw apiError(response, capabilities);
  }
  if (capabilities.auth_required && !capabilities.authenticated) {
    window.location.replace("/login.html");
    return false;
  }
  state.capabilities = {
    authRequired: Boolean(capabilities.auth_required),
    historyEnabled: Boolean(capabilities.evaluation_history_enabled),
    visualizationsEnabled: Boolean(capabilities.visualizations_enabled),
  };
  $("logoutButton").hidden = !state.capabilities.authRequired;
  $("accountEmail").textContent = capabilities.email || "";
  $("accountEmail").hidden = !capabilities.email;
  $("historySection").hidden = !state.capabilities.historyEnabled;
  $("visualizeButton").hidden = !state.capabilities.visualizationsEnabled;
  $("apiBase").textContent = API;
  return true;
}

function applyLanguage() {
  document.documentElement.lang = state.lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  $("languageButton").textContent = state.lang === "zh" ? "EN" : "中文";
  if (!state.file) $("routeBadge").textContent = t("waitingForFile");
  renderModels();
  renderHistory();
}

function loadedModel(modelId) {
  return state.models.some((model) => model.model_id === modelId);
}

function updateActionAvailability() {
  const modelAvailable = Boolean(state.modelId && loadedModel(state.modelId));
  $("evaluateButton").disabled = !state.file || !modelAvailable;
  $("visualizeButton").disabled =
    !state.file ||
    state.modelId !== "iqa" ||
    !loadedModel("iqa") ||
    !state.capabilities.visualizationsEnabled;
}

function selectFile(file) {
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  const isImage = IMAGE_EXTENSIONS.has(extension);
  const isVideo = VIDEO_EXTENSIONS.has(extension);
  if (!isImage && !isVideo) {
    notify(`${t("unsupportedMedia")}: .${extension || "unknown"}`, true);
    return;
  }
  clearNotice();
  state.file = file;
  state.modelId = isImage ? "iqa" : "vqa";
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);

  $("routeBadge").textContent = isImage ? t("imageRoute") : t("videoRoute");
  $("fileName").textContent = file.name;
  $("fileMeta").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type || extension}`;

  const media = document.createElement(isImage ? "img" : "video");
  media.src = state.objectUrl;
  media.alt = file.name;
  if (!isImage) {
    media.controls = true;
    media.muted = true;
  }
  $("previewMedia").replaceChildren(media);
  $("mediaPreview").hidden = false;
  renderModels();
  updateActionAvailability();
  if (!loadedModel(state.modelId) && state.models.length) notify(t("noModel"), true);
}

function renderModels() {
  const box = $("models");
  box.replaceChildren();
  state.models.forEach((model) => {
    const card = document.createElement("article");
    card.className = `model${model.model_id === state.modelId ? " active" : ""}`;
    const head = document.createElement("div");
    head.className = "model-head";
    const title = document.createElement("h3");
    title.textContent = `${model.model_id.toUpperCase()} · ${model.model_name}`;
    const mediaType = document.createElement("small");
    mediaType.textContent = model.media_type;
    head.append(title, mediaType);
    const details = document.createElement("p");
    details.textContent = `${model.backbone} · ${model.dataset}`;
    const mosRange = document.createElement("span");
    mosRange.className = "model-mos";
    mosRange.textContent = `MOS ${model.mos_min}–${model.mos_max}`;
    details.append(mosRange);
    card.append(head, details);
    box.append(card);
  });
}

async function loadService() {
  const status = $("status");
  status.className = "status";
  setText("#status span", t("connecting"));
  try {
    const [health, models] = await Promise.all([
      jsonFetch(`${API}/health`),
      jsonFetch(`${API}/models`),
    ]);
    state.models = models.items || [];
    $("device").textContent = health.device;
    $("modelCount").textContent = models.count;
    status.className = "status ok";
    setText("#status span", t("ready"));
    renderModels();
    updateActionAvailability();
  } catch (error) {
    status.className = "status bad";
    setText("#status span", t("unavailable"));
    notify(`${t("apiFailed")}: ${error.message}`, true);
  }
}

function showEvaluation(data) {
  state.hasEvaluation = true;
  $("emptyResult").hidden = true;
  $("result").hidden = false;
  $("resultBadge").textContent = data.evaluation_id;
  $("mos").textContent = data.mos_score ?? "--";
  $("mosRange").textContent = `${data.mos_min} – ${data.mos_max}`;
  $("score").textContent = Number(data.score).toFixed(4);
  $("latency").textContent = `${data.latency_ms} ms`;
  $("modelName").textContent = data.model_name;
  $("backbone").textContent = data.backbone;
  $("dataset").textContent = data.dataset;
  $("mediaSize").textContent = data.media_size;
  const ratio = Math.max(0, Math.min(1, Number(data.score)));
  $("meter").style.width = `${ratio * 100}%`;
  $("meter").parentElement.setAttribute("aria-valuenow", ratio.toFixed(4));
}

function setEvaluationLoading(loading) {
  $("resultPanel").setAttribute("aria-busy", String(loading));
  $("evaluationLoading").hidden = !loading;
  if (loading) {
    $("emptyResult").hidden = true;
    $("result").hidden = true;
    $("resultBadge").textContent = t("evaluating");
    return;
  }
  $("emptyResult").hidden = state.hasEvaluation;
  $("result").hidden = !state.hasEvaluation;
  if (!state.hasEvaluation) $("resultBadge").textContent = t("notEvaluated");
}

async function evaluate() {
  if (!state.file || !state.modelId) return;
  const button = $("evaluateButton");
  button.disabled = true;
  button.textContent = t("evaluating");
  setEvaluationLoading(true);
  clearNotice();
  try {
    const body = new FormData();
    body.append("file", state.file);
    const data = await jsonFetch(
      `${API}/evaluations?model_id=${encodeURIComponent(state.modelId)}`,
      { body, headers: { "X-Deep-VQA-Client": "frontend" }, method: "POST" },
    );
    showEvaluation(data);
    if (state.capabilities.historyEnabled) await loadHistory();
    notify(t("done"));
  } catch (error) {
    notify(`${t("evaluationFailed")}: ${error.message}`, true);
  } finally {
    setEvaluationLoading(false);
    button.textContent = t("evaluate");
    updateActionAvailability();
  }
}

function generatedFileUrl(path) {
  if (typeof path !== "string") return "";
  const marker = "/reports/iqa-test/";
  const index = path.indexOf(marker);
  return index >= 0 ? `${API}/files/${path.slice(index + marker.length)}` : path;
}

function zoomImage(src, alt) {
  const image = document.createElement("img");
  image.src = src;
  image.alt = alt;
  $("lightboxContent").replaceChildren(image);
  $("lightbox").hidden = false;
  $("closeLightbox").focus();
}

function closeLightbox() {
  $("lightbox").hidden = true;
  $("lightboxContent").replaceChildren();
}

function comparisonRow(title, originalSrc, generatedSrc, generatedAlt) {
  const row = document.createElement("section");
  row.className = "comparison-row";
  const heading = document.createElement("h3");
  heading.textContent = title;
  row.append(heading);
  for (const [src, label] of [[originalSrc, t("original")], [generatedSrc, generatedAlt]]) {
    const item = document.createElement("article");
    item.className = "viz-item";
    const image = document.createElement("img");
    image.src = src;
    image.alt = label;
    image.title = "双击放大查看";
    const caption = document.createElement("strong");
    caption.textContent = label;
    item.append(image, caption);
    row.append(item);
  }
  return row;
}

async function visualize() {
  if (!state.file || state.modelId !== "iqa" || !state.capabilities.visualizationsEnabled) return;
  const button = $("visualizeButton");
  button.disabled = true;
  button.textContent = t("generating");
  clearNotice();
  try {
    const body = new FormData();
    body.append("file", state.file);
    const data = await jsonFetch(`${API}/visualizations?model_id=iqa`, { body, method: "POST" });
    const firstFeature = Object.entries(data.feature_maps || {})[0];
    if (!firstFeature || !data.gradcam) throw new Error("Visualization response is incomplete");
    $("comparisonRows").replaceChildren(
      comparisonRow(t("feature"), state.objectUrl, generatedFileUrl(firstFeature[1]), firstFeature[0]),
      comparisonRow(t("gradcam"), state.objectUrl, generatedFileUrl(data.gradcam), "Grad-CAM"),
    );
    $("visualizationMeta").textContent =
      `${data.layers.length} ${t("levels")} · score ` +
      `${Number(data.score).toFixed(4)} · ${data.visualization_id}`;
    $("visualization").hidden = false;
    notify(t("vizDone"));
    requestAnimationFrame(() =>
      $("visualization").scrollIntoView({ behavior: "smooth", block: "start" }),
    );
  } catch (error) {
    notify(`${t("visualizationFailed")}: ${error.message}`, true);
  } finally {
    button.textContent = t("visualize");
    updateActionAvailability();
  }
}

function recordId(record, index) {
  return `${record.timestamp || ""}:${record.file_hash || ""}:${index}`;
}

function selectedHistory() {
  return state.history.filter((record, index) => state.selected.has(recordId(record, index)));
}

function renderHistory() {
  const box = $("historyList");
  box.replaceChildren();
  if (!state.history.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = t("historyEmpty");
    box.append(empty);
  }
  state.history.forEach((record, index) => {
    const id = recordId(record, index);
    const row = document.createElement("label");
    row.className = "history-item";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = state.selected.has(id);
    check.addEventListener("change", () => {
      check.checked ? state.selected.add(id) : state.selected.delete(id);
      renderHistory();
    });
    const main = document.createElement("div");
    main.className = "history-item-main";
    const name = document.createElement("strong");
    name.textContent = record.file_name || "unknown";
    const meta = document.createElement("span");
    meta.className = "history-item-meta";
    meta.textContent =
      `${record.timestamp || "--"} · ${record.task_type || "--"} · ` +
      `${record.model_used || "--"} · MOS ${record.mos_interval?.join("–") || "--"} · ` +
      `${record.inference_time_ms ?? "--"} ms`;
    main.append(name, meta);
    const score = document.createElement("span");
    score.className = "history-score";
    score.textContent = record.mos_score ?? "--";
    row.append(check, main, score);
    box.append(row);
  });
  $("historyPreview").textContent = state.history.map((record) => JSON.stringify(record)).join("\n") || "(empty)";
  $("historyCount").textContent = String(state.history.length);
  $("exportSelectedButton").disabled = selectedHistory().length === 0;
  $("exportAllButton").disabled = state.history.length === 0;
  $("selectAllButton").disabled = state.history.length === 0;
}

async function loadHistory() {
  if (!state.capabilities.historyEnabled) return;
  try {
    const data = await jsonFetch(`${API}/frontend-evaluations?limit=200`);
    state.history = Array.isArray(data.items) ? data.items : [];
    state.selected.clear();
    renderHistory();
    clearNotice("historyNotice");
  } catch (error) {
    notify(error.message, true, "historyNotice");
  }
}

function downloadJsonl(records, filename) {
  const content = records.map((record) => JSON.stringify(record)).join("\n");
  const blob = new Blob([content + (records.length ? "\n" : "")], { type: "application/x-ndjson" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  document.body.append(link);
  link.click();
  const url = link.href;
  link.remove();
  URL.revokeObjectURL(url);
}

function toggleAll() {
  const allSelected =
    state.history.length > 0 &&
    state.history.every((record, index) => state.selected.has(recordId(record, index)));
  state.selected = allSelected
    ? new Set()
    : new Set(state.history.map((record, index) => recordId(record, index)));
  renderHistory();
}

async function logout() {
  try {
    await fetch(`${API}/auth/logout`, { credentials: "include", method: "POST" });
  } finally {
    window.location.replace("/login.html");
  }
}

function bindEvents() {
  const dropZone = $("dropZone");
  $("mediaFile").addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) selectFile(file);
  });
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      $("mediaFile").click();
    }
  });
  for (const eventName of ["dragenter", "dragover"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("drag");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("drag");
    });
  }
  dropZone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (file) selectFile(file);
  });
  $("evaluateButton").addEventListener("click", evaluate);
  $("visualizeButton").addEventListener("click", visualize);
  $("refreshButton").addEventListener("click", loadService);
  $("refreshHistoryButton").addEventListener("click", loadHistory);
  $("selectAllButton").addEventListener("click", toggleAll);
  $("exportSelectedButton").addEventListener("click", () => {
    downloadJsonl(selectedHistory(), "deep-vqa-history-selected.jsonl");
  });
  $("exportAllButton").addEventListener("click", () => {
    downloadJsonl(state.history, "deep-vqa-history-all.jsonl");
  });
  $("languageButton").addEventListener("click", () => {
    state.lang = state.lang === "zh" ? "en" : "zh";
    localStorage.setItem("deep-vqa-lang", state.lang);
    applyLanguage();
  });
  $("logoutButton").addEventListener("click", logout);
  $("closeLightbox").addEventListener("click", closeLightbox);
  $("lightbox").addEventListener("click", (event) => {
    if (event.target === $("lightbox")) closeLightbox();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !$("lightbox").hidden) closeLightbox();
  });
  $("previewMedia").addEventListener("dblclick", (event) => {
    if (event.target.matches("img")) {
      zoomImage(event.target.currentSrc || event.target.src, event.target.alt || "preview");
    }
  });
  $("comparisonRows").addEventListener("dblclick", (event) => {
    if (event.target.matches("img")) zoomImage(event.target.currentSrc || event.target.src, event.target.alt);
  });
  window.addEventListener("beforeunload", () => {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  });
}

async function init() {
  try {
    if (!(await loadCapabilities())) return;
    applyLanguage();
    bindEvents();
    $("workspace").hidden = false;
    const tasks = [loadService()];
    if (state.capabilities.historyEnabled) tasks.push(loadHistory());
    await Promise.all(tasks);
  } catch (error) {
    $("workspace").hidden = false;
    applyLanguage();
    notify(`${t("apiFailed")}: ${error.message}`, true);
  }
}

init();
