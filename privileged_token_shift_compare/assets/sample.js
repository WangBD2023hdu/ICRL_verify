"use strict";

const app = document.getElementById("app");
let sample = null;
let manifest = null;
let selectedIndex = -1;
let streamRoot = null;
let streamStatus = null;
let detailBody = null;
let classFilter = "ALL";
let minimumGap = 0;
let tokenQuery = "";
let renderedCount = 0;
const RENDER_BATCH = 320;

function node(tag, className, text) {
  const value = document.createElement(tag);
  if (className) value.className = className;
  if (text !== undefined) value.textContent = text;
  return value;
}

function metric(label, value, tone) {
  const root = node("div", "metric");
  root.append(node("div", "metric-label", label));
  root.append(node("div", `metric-value ${tone || ""}`, value));
  return root;
}

function fixed(value, digits = 4) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  if (numeric !== 0 && Math.abs(numeric) < 0.0001) return numeric.toExponential(3);
  return numeric.toFixed(digits);
}

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(3)}%`;
}

function visiblePiece(raw) {
  if (raw === "") return "∅";
  return raw.replace(/ /g, "·").replace(/\t/g, "⇥").replace(/\r/g, "␍").replace(/\n/g, "↵");
}

function badgeFor(label) {
  let tone = "";
  if (label === "TEACHER_REJECTS_STUDENT_TOKEN" || label === "EOS_SHIFT") tone = "reject";
  if (label === "TEACHER_PROMOTES_STUDENT_TOKEN") tone = "promote";
  if (label.includes("TOKENIZATION")) tone = "equivalent";
  return node("span", `badge ${tone}`, label);
}

function heatColor(gap) {
  const intensity = Math.min(Math.abs(Number(gap)) / 8, 1);
  if (Math.abs(gap) < 0.04) return "rgba(220, 227, 230, 0.54)";
  if (gap > 0) return `rgba(20, 125, 100, ${0.16 + intensity * 0.68})`;
  return `rgba(195, 79, 70, ${0.16 + intensity * 0.68})`;
}

function tokenMatches(token) {
  if (classFilter !== "ALL" && token.comparison_class !== classFilter) return false;
  if (Math.abs(Number(token.delta_logp_teacher_minus_student)) < minimumGap) return false;
  if (tokenQuery) {
    const haystack = `${token.token_id} ${token.raw_token} ${token.token_piece_repr}`.toLowerCase();
    if (!haystack.includes(tokenQuery)) return false;
  }
  return true;
}

function makeTokenChip(token) {
  const chip = node("button", "token-chip", visiblePiece(token.raw_token));
  chip.type = "button";
  chip.dataset.index = String(token.index);
  chip.style.backgroundColor = heatColor(token.delta_logp_teacher_minus_student);
  chip.title = `#${token.index} id=${token.token_id} pS=${fixed(token.student_probability_t1)} pT=${fixed(token.teacher_probability_t1)} Δ=${fixed(token.delta_logp_teacher_minus_student)}`;
  if (token.is_eos) chip.classList.add("eos");
  if (token.comparison_class.includes("TOKENIZATION")) chip.classList.add("equivalent");
  if (!tokenMatches(token)) chip.classList.add("dimmed");
  chip.addEventListener("click", () => selectToken(token.index));
  return chip;
}

function renderNextTokenBatch() {
  const end = Math.min(sample.tokens.length, renderedCount + RENDER_BATCH);
  const fragment = document.createDocumentFragment();
  for (let index = renderedCount; index < end; index += 1) {
    const token = sample.tokens[index];
    fragment.append(makeTokenChip(token));
    if (token.raw_token.includes("\n")) fragment.append(document.createElement("br"));
  }
  streamRoot.append(fragment);
  renderedCount = end;
  streamStatus.textContent = `已渲染 ${renderedCount}/${sample.tokens.length} token`;
  if (renderedCount < sample.tokens.length) {
    if ("requestIdleCallback" in window) {
      window.requestIdleCallback(renderNextTokenBatch, { timeout: 50 });
    } else {
      window.setTimeout(renderNextTokenBatch, 20);
    }
  }
}

function refreshTokenFocus() {
  streamRoot.querySelectorAll(".token-chip").forEach((chip) => {
    const token = sample.tokens[Number(chip.dataset.index)];
    chip.classList.toggle("dimmed", !tokenMatches(token));
  });
}

function field(label, value, wide = false) {
  const root = node("div", `detail-field ${wide ? "wide" : ""}`);
  const dl = node("dl");
  dl.append(node("dt", "", label));
  dl.append(node("dd", "", String(value)));
  root.append(dl);
  return root;
}

function candidates(title, values) {
  const root = node("div");
  root.append(node("h3", "", title));
  const table = node("table", "candidate-table");
  table.innerHTML = "<thead><tr><th>#</th><th>Token</th><th class='numeric'>p</th></tr></thead>";
  const body = node("tbody");
  values.forEach((item) => {
    const tr = node("tr");
    tr.append(node("td", "", String(item.rank)));
    const token = node("td", "", visiblePiece(item.raw_token));
    token.title = `id=${item.token_id} ${item.raw_token}`;
    tr.append(token);
    tr.append(node("td", "numeric", fixed(item.probability)));
    body.append(tr);
  });
  table.append(body);
  root.append(table);
  return root;
}

function selectToken(index, scroll = false) {
  selectedIndex = Math.max(0, Math.min(sample.tokens.length - 1, Number(index)));
  document.querySelectorAll(".token-chip.selected").forEach((chip) => chip.classList.remove("selected"));
  const chip = streamRoot.querySelector(`[data-index="${selectedIndex}"]`);
  if (chip) {
    chip.classList.add("selected");
    if (scroll) chip.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
  }
  const token = sample.tokens[selectedIndex];
  detailBody.replaceChildren();
  const labelLine = node("div");
  labelLine.append(badgeFor(token.comparison_class));
  detailBody.append(labelLine);

  const grid = node("div", "detail-grid");
  grid.append(field("位置", token.index));
  grid.append(field("Token ID", token.token_id));
  grid.append(field("Token piece", token.token_piece_repr, true));
  grid.append(field("单 token decode", token.single_decode_repr, true));
  grid.append(field("学生 p (T=1)", fixed(token.student_probability_t1, 6)));
  grid.append(field("教师 p (T=1)", fixed(token.teacher_probability_t1, 6)));
  grid.append(field("学生 logp", fixed(token.student_logprob_t1, 6)));
  grid.append(field("教师 logp", fixed(token.teacher_logprob_t1, 6)));
  grid.append(field("Δ logp (T-S)", fixed(token.delta_logp_teacher_minus_student, 6)));
  grid.append(field("pT / pS", fixed(token.teacher_to_student_probability_ratio, 4)));
  grid.append(field("学生 rank", token.student_rank_t1));
  grid.append(field("教师 rank", token.teacher_rank_t1));
  grid.append(field("学生 entropy", fixed(token.student_entropy_t1, 5)));
  grid.append(field("教师 entropy", fixed(token.teacher_entropy_t1, 5)));
  grid.append(field("采样 p", fixed(token.rollout_probability_t_sampling, 6)));
  grid.append(field("EOS", token.is_eos ? "是" : "否"));
  detailBody.append(grid);

  detailBody.append(node("h3", "", "上下文窗口"));
  detailBody.append(node("pre", "context-box", token.context_text));
  if (token.text_equivalent_candidate) {
    detailBody.append(node("h3", "", "分词等价候选"));
    detailBody.append(node("pre", "context-box", JSON.stringify(token.text_equivalent_candidate, null, 2)));
  }
  const columns = node("div", "candidate-columns");
  columns.append(candidates("学生 Top-k", token.student_top_candidates));
  columns.append(candidates("教师 Top-k", token.teacher_top_candidates));
  detailBody.append(columns);
}

function drawHistogram(tokens) {
  const canvas = document.getElementById("gap-histogram");
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(300, Math.floor(rect.width * ratio));
  canvas.height = Math.max(160, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = rect.width;
  const height = rect.height;
  ctx.clearRect(0, 0, width, height);
  const values = tokens.map((item) => Math.max(-10, Math.min(10, Number(item.delta_logp_teacher_minus_student))));
  const bins = 40;
  const counts = Array(bins).fill(0);
  values.forEach((value) => {
    const index = Math.min(bins - 1, Math.max(0, Math.floor(((value + 10) / 20) * bins)));
    counts[index] += 1;
  });
  const maxCount = Math.max(...counts, 1);
  const padding = { left: 38, right: 14, top: 12, bottom: 28 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const zeroX = padding.left + plotWidth / 2;
  ctx.strokeStyle = "#b4c0c5";
  ctx.beginPath();
  ctx.moveTo(zeroX, padding.top);
  ctx.lineTo(zeroX, padding.top + plotHeight);
  ctx.stroke();
  counts.forEach((count, index) => {
    const x = padding.left + (index / bins) * plotWidth;
    const barWidth = Math.max(1, plotWidth / bins - 1);
    const barHeight = (count / maxCount) * plotHeight;
    ctx.fillStyle = index < bins / 2 ? "rgba(195,79,70,.72)" : "rgba(20,125,100,.72)";
    ctx.fillRect(x, padding.top + plotHeight - barHeight, barWidth, barHeight);
  });
  ctx.fillStyle = "#617077";
  ctx.font = "11px system-ui";
  ctx.fillText("学生更高 ←", padding.left, height - 7);
  ctx.textAlign = "right";
  ctx.fillText("→ 教师更高", width - padding.right, height - 7);
  ctx.textAlign = "center";
  ctx.fillText("0", zeroX, height - 7);
}

function renderDifferenceTable(tokens, container) {
  const sorted = [...tokens]
    .sort((a, b) => Math.abs(b.delta_logp_teacher_minus_student) - Math.abs(a.delta_logp_teacher_minus_student))
    .slice(0, 200);
  const table = node("table", "difference-table");
  table.innerHTML = "<thead><tr><th>#</th><th>Token</th><th>类别</th><th class='numeric'>pS</th><th class='numeric'>pT</th><th class='numeric'>Δ(T-S)</th></tr></thead>";
  const body = node("tbody");
  sorted.forEach((token) => {
    const tr = node("tr");
    tr.addEventListener("click", () => selectToken(token.index, true));
    tr.append(node("td", "", String(token.index)));
    tr.append(node("td", "", visiblePiece(token.raw_token)));
    const classCell = node("td");
    classCell.append(badgeFor(token.comparison_class));
    tr.append(classCell);
    tr.append(node("td", "numeric", fixed(token.student_probability_t1)));
    tr.append(node("td", "numeric", fixed(token.teacher_probability_t1)));
    tr.append(node("td", "numeric", fixed(token.delta_logp_teacher_minus_student)));
    body.append(tr);
  });
  table.append(body);
  container.append(table);
}

function addDetailsSection(parent, title, value) {
  const details = node("details");
  details.append(node("summary", "", title));
  details.append(node("pre", "text-block", typeof value === "string" ? value : JSON.stringify(value, null, 2)));
  parent.append(details);
}

function navigationLinks() {
  const actions = node("nav", "nav-actions");
  const back = node("a", "nav-button", "批次首页");
  back.href = "../../index.html";
  const previous = node("a", "nav-button", "上一条");
  const next = node("a", "nav-button", "下一条");
  if (manifest) {
    const index = manifest.samples.findIndex((item) => item.slug === sample.slug);
    if (index > 0) previous.href = `../${manifest.samples[index - 1].slug}/index.html`;
    else previous.setAttribute("aria-disabled", "true");
    if (index >= 0 && index < manifest.samples.length - 1) next.href = `../${manifest.samples[index + 1].slug}/index.html`;
    else next.setAttribute("aria-disabled", "true");
  }
  actions.append(back, previous, next);
  return actions;
}

function render() {
  app.replaceChildren();
  document.title = `${sample.sample_id} · Token 差异`;
  const bar = node("header", "topbar");
  bar.append(node("div", "topbar-title", sample.sample_id));
  bar.append(navigationLinks());
  app.append(bar);

  const page = node("main", "page");
  const heading = node("div", "page-heading");
  const title = node("div");
  title.append(node("h1", "", sample.sample_id));
  title.append(node("p", "", `数据行 ${sample.line_number} · 响应 ${sample.response.token_count} token · teacher standalone text-only`));
  heading.append(title);
  page.append(heading);

  const summary = sample.summary;
  const strip = node("div", "metric-strip");
  strip.append(metric("响应 Token", String(summary.token_count)));
  strip.append(metric("平均 |Δlogp|", fixed(summary.mean_absolute_logprob_gap)));
  strip.append(metric("P90 |Δlogp|", fixed(summary.p90_absolute_logprob_gap)));
  strip.append(metric("P99 |Δlogp|", fixed(summary.p99_absolute_logprob_gap)));
  strip.append(metric("教师强烈反对", percent(summary.teacher_confidence_filtered_ratio), "student"));
  strip.append(metric("教师/学生均值 p", `${fixed(summary.mean_teacher_probability)} / ${fixed(summary.mean_student_probability)}`, "teacher"));
  page.append(strip);

  const layout = node("div", "sample-layout");
  const main = node("div", "main-column");
  const side = node("aside", "side-column");

  const streamSection = node("section", "section");
  const streamHeader = node("div", "section-header");
  streamHeader.append(node("h2", "", "完整生成序列"));
  streamHeader.append(node("div", "section-note", "点击 token 查看详情；颜色表示 teacher logp - student logp"));
  streamSection.append(streamHeader);
  const controls = node("div", "control-bar");
  const query = node("input");
  query.type = "search";
  query.placeholder = "搜索 token 文本或 ID";
  const classes = node("select");
  const labels = ["ALL", ...Object.keys(summary.class_counts || {}).sort()];
  labels.forEach((label) => {
    const option = node("option", "", label === "ALL" ? "全部类别" : label);
    option.value = label;
    classes.append(option);
  });
  const gap = node("input");
  gap.type = "number";
  gap.min = "0";
  gap.step = "0.1";
  gap.value = "0";
  gap.title = "最小绝对 logprob 差";
  gap.placeholder = "最小 |Δlogp|";
  const reset = node("button", "command-button", "重置筛选");
  controls.append(query, classes, gap, reset);
  streamSection.append(controls);
  const legend = node("div", "legend");
  legend.innerHTML = "<span><i class='legend-swatch student'></i>学生概率更高</span><span><i class='legend-swatch neutral'></i>接近</span><span><i class='legend-swatch teacher'></i>教师概率更高</span><span>虚线：EOS</span><span>下划线：分词等价</span>";
  streamSection.append(legend);
  streamRoot = node("div", "token-stream");
  streamRoot.setAttribute("aria-label", "逐 token 概率热力图");
  streamStatus = node("div", "stream-status", "准备渲染...");
  streamSection.append(streamRoot, streamStatus);
  main.append(streamSection);

  query.addEventListener("input", () => { tokenQuery = query.value.trim().toLowerCase(); refreshTokenFocus(); });
  classes.addEventListener("change", () => { classFilter = classes.value; refreshTokenFocus(); });
  gap.addEventListener("input", () => { minimumGap = Math.max(0, Number(gap.value || 0)); refreshTokenFocus(); });
  reset.addEventListener("click", () => {
    query.value = ""; classes.value = "ALL"; gap.value = "0";
    tokenQuery = ""; classFilter = "ALL"; minimumGap = 0; refreshTokenFocus();
  });

  const histogramSection = node("section", "section");
  const histogramHeader = node("div", "section-header");
  histogramHeader.append(node("h2", "", "Logprob 差异分布"));
  histogramHeader.append(node("div", "section-note", "Δ = teacher - student，显示范围裁剪为 [-10, 10]"));
  histogramSection.append(histogramHeader);
  const histogramWrap = node("div", "histogram-wrap");
  const canvas = node("canvas");
  canvas.id = "gap-histogram";
  histogramWrap.append(canvas);
  histogramSection.append(histogramWrap);
  main.append(histogramSection);

  const diffSection = node("section", "section");
  const diffHeader = node("div", "section-header");
  diffHeader.append(node("h2", "", "分歧最大的 Token"));
  diffHeader.append(node("div", "section-note", "按 |Δlogp| 排序，最多显示 200 条"));
  diffSection.append(diffHeader);
  renderDifferenceTable(sample.tokens, diffSection);
  main.append(diffSection);

  const sourceSection = node("section", "section");
  const sourceHeader = node("div", "section-header");
  sourceHeader.append(node("h2", "", "样本输入"));
  sourceSection.append(sourceHeader);
  if (sample.images?.length) {
    const media = node("div", "media-grid");
    sample.images.forEach((path, index) => {
      const image = node("img");
      image.src = path;
      image.alt = `输入图片 ${index + 1}`;
      image.loading = "lazy";
      media.append(image);
    });
    sourceSection.append(media);
  }
  addDetailsSection(sourceSection, "学生原始消息", sample.student_messages);
  addDetailsSection(sourceSection, "教师 standalone 消息", sample.teacher_messages);
  addDetailsSection(sourceSection, "特权文本", sample.privileged_text);
  addDetailsSection(sourceSection, "模型完整响应", sample.response.text);
  main.append(sourceSection);

  const panel = node("div", "detail-panel");
  const panelHeader = node("div", "detail-header");
  panelHeader.append(node("h2", "", "Token 详情"));
  detailBody = node("div", "detail-body");
  detailBody.append(node("div", "detail-placeholder", "从左侧完整序列或分歧表中选择一个 token。可使用键盘 ← → 切换相邻 token。"));
  panel.append(panelHeader, detailBody);
  side.append(panel);
  layout.append(main, side);
  page.append(layout);
  app.append(page);

  renderedCount = 0;
  renderNextTokenBatch();
  requestAnimationFrame(() => drawHistogram(sample.tokens));
  window.addEventListener("resize", () => drawHistogram(sample.tokens));
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    if (event.key === "ArrowLeft" && selectedIndex > 0) selectToken(selectedIndex - 1, true);
    if (event.key === "ArrowRight" && selectedIndex < sample.tokens.length - 1) selectToken(selectedIndex + 1, true);
  });
}

if (window.__PRIVILEGED_SAMPLE__) {
  sample = window.__PRIVILEGED_SAMPLE__;
  manifest = window.__PRIVILEGED_MANIFEST__ || null;
  render();
} else {
  app.replaceChildren(node("div", "error-state", "样本页面加载失败：data.js 不存在或未与 index.html 一起复制。"));
}
