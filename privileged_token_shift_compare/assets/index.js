"use strict";

const app = document.getElementById("app");

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

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(3)}%`;
}

function number(value, digits = 3) {
  return Number(value || 0).toFixed(digits);
}

function render(manifest) {
  app.replaceChildren();
  const bar = node("header", "topbar");
  bar.append(node("div", "topbar-title", "特权条件 Token 概率对比"));
  app.append(bar);

  const page = node("main", "page");
  const heading = node("div", "page-heading");
  const title = node("div");
  title.append(node("h1", "", "批次样本"));
  title.append(node("p", "", "学生仅生成一次；每条响应在原始多模态条件和 standalone 特权文本条件下分别打分。"));
  heading.append(title);
  page.append(heading);

  const strip = node("div", "metric-strip");
  strip.append(metric("样本", String(manifest.sample_count)));
  strip.append(metric("响应 Token", String(manifest.total_tokens)));
  strip.append(metric("强烈反对 Token", String(manifest.teacher_confidence_filtered_tokens), "student"));
  strip.append(metric("强烈反对比例", percent(manifest.teacher_confidence_filtered_ratio), "student"));
  strip.append(metric("缺失特权字段", String(manifest.scan?.missing_privileged_rows || 0)));
  strip.append(metric("数据错误", String(manifest.scan?.malformed_rows || 0)));
  page.append(strip);

  const controls = node("div", "index-controls");
  const search = node("input");
  search.type = "search";
  search.placeholder = "搜索 sample ID 或响应文本";
  const sort = node("select");
  [
    ["line", "按数据行号"],
    ["gap", "按平均绝对分歧"],
    ["reject", "按强烈反对比例"],
    ["length", "按响应长度"],
  ].forEach(([value, label]) => {
    const option = node("option", "", label);
    option.value = value;
    sort.append(option);
  });
  controls.append(search, sort);
  page.append(controls);

  const list = node("div", "sample-list");
  const table = node("table", "sample-table");
  table.innerHTML = "<thead><tr><th>图像</th><th>样本</th><th class='numeric'>Token</th><th class='numeric'>|Δlogp|</th><th class='numeric'>反对比例</th><th>响应预览</th></tr></thead>";
  const body = node("tbody");
  table.append(body);
  list.append(table);
  page.append(list);
  app.append(page);

  const draw = () => {
    const query = search.value.trim().toLowerCase();
    const rows = [...manifest.samples].filter((item) => {
      return !query || item.sample_id.toLowerCase().includes(query) || item.response_preview.toLowerCase().includes(query);
    });
    rows.sort((left, right) => {
      if (sort.value === "gap") return Number(right.summary.mean_absolute_logprob_gap || 0) - Number(left.summary.mean_absolute_logprob_gap || 0);
      if (sort.value === "reject") return Number(right.summary.teacher_confidence_filtered_ratio || 0) - Number(left.summary.teacher_confidence_filtered_ratio || 0);
      if (sort.value === "length") return Number(right.summary.token_count || 0) - Number(left.summary.token_count || 0);
      return left.line_number - right.line_number;
    });
    body.replaceChildren();
    rows.forEach((item) => {
      const tr = node("tr");
      tr.tabIndex = 0;
      tr.addEventListener("click", () => { window.location.href = item.url; });
      tr.addEventListener("keydown", (event) => {
        if (event.key === "Enter") window.location.href = item.url;
      });
      const imageCell = node("td");
      if (item.thumbnail) {
        const image = node("img", "sample-thumb");
        image.src = item.thumbnail;
        image.alt = "输入图片";
        image.loading = "lazy";
        imageCell.append(image);
      }
      const idCell = node("td", "sample-id", item.sample_id);
      idCell.append(node("div", "section-note", `line ${item.line_number}`));
      tr.append(imageCell, idCell);
      tr.append(node("td", "numeric", String(item.summary.token_count || 0)));
      tr.append(node("td", "numeric", number(item.summary.mean_absolute_logprob_gap)));
      tr.append(node("td", "numeric", percent(item.summary.teacher_confidence_filtered_ratio)));
      tr.append(node("td", "preview", item.response_preview));
      body.append(tr);
    });
  };
  search.addEventListener("input", draw);
  sort.addEventListener("change", draw);
  draw();
}

if (window.__PRIVILEGED_MANIFEST__) {
  render(window.__PRIVILEGED_MANIFEST__);
} else {
  app.replaceChildren(node("div", "error-state", "报告加载失败：manifest.js 不存在或未与 index.html 一起复制。"));
}
