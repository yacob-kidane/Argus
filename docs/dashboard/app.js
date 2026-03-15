const DATA_BASE = "../data";
const EST_COST_PER_GPU_HOUR = 4.0;

async function fetchText(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.text();
}

async function fetchJson(path) {
  const text = await fetchText(path);
  if (!text.trim()) return null;
  return JSON.parse(text);
}

function parseJsonLines(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

async function loadClaimed() {
  try {
    const data = await fetchJson(`${DATA_BASE}/claimed.json`);
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

async function loadResults() {
  try {
    const text = await fetchText(`${DATA_BASE}/results.jsonl`);
    return parseJsonLines(text).sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));
  } catch {
    return [];
  }
}

async function loadBest() {
  try {
    const data = await fetchJson(`${DATA_BASE}/global_best.json`);
    return data && data.best_fingerprint ? data : null;
  } catch {
    return null;
  }
}

async function loadPromotions() {
  try {
    const text = await fetchText(`${DATA_BASE}/promoted.jsonl`);
    return parseJsonLines(text).sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));
  } catch {
    return [];
  }
}

async function loadReasoning() {
  try {
    const text = await fetchText(`${DATA_BASE}/agent_reasoning.jsonl`);
    return parseJsonLines(text).sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));
  } catch {
    return [];
  }
}

async function loadPolicySummaries() {
  const items = await Promise.all([
    fetchJson(`${DATA_BASE}/ab_baseline_summary.json`).catch(() => null),
    fetchJson(`${DATA_BASE}/ab_history_summary.json`).catch(() => null),
  ]);
  return items.filter(Boolean);
}

function fmtNum(value, digits = 6) {
  if (value == null || value === "") return "—";
  const num = Number(value);
  return Number.isNaN(num) ? "—" : num.toFixed(digits);
}

function fmtCost(value) {
  if (value == null || value === "") return "—";
  const num = Number(value);
  return Number.isNaN(num) ? "—" : `$${num.toFixed(2)}`;
}

function fmtReward(value) {
  if (value == null || value === "") return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  return Math.abs(num) >= 0.001 ? num.toFixed(6) : num.toExponential(3);
}

function fmtTflops(value) {
  if (value == null || value === "") return "—";
  const num = Number(value);
  if (Number.isNaN(num)) return "—";
  return num >= 1000 ? num.toFixed(0) : num.toFixed(3);
}

function fmtTs(raw) {
  if (!raw) return "—";
  const date = new Date(raw);
  if (Number.isNaN(date.valueOf())) return "—";
  return date.toLocaleString("en", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function shortFp(fp) {
  if (!fp) return "—";
  return fp.length > 10 ? `${fp.slice(0, 10)}…` : fp;
}

function badgeClass(value) {
  const normalized = String(value || "").toLowerCase();
  if (["kept", "succeeded", "submitted"].includes(normalized)) return "badge badge-good";
  if (["rejected", "queued", "running"].includes(normalized)) return "badge badge-warn";
  if (["failed", "timed_out", "cancelled"].includes(normalized)) return "badge badge-bad";
  return "badge badge-neutral";
}

function getSuccessfulResults(results) {
  return results
    .filter((row) => row.outcome === "succeeded")
    .slice()
    .sort((a, b) => new Date(a.timestamp || 0) - new Date(b.timestamp || 0));
}

function getBaselineResult(results) {
  return results.length ? results[0] : null;
}

function getBestResult(results) {
  if (!results.length) return null;
  return results.reduce((best, row) => {
    const nextVal = Number(row.val_bpb);
    const bestVal = Number(best.val_bpb);
    if (Number.isNaN(nextVal)) return best;
    if (Number.isNaN(bestVal) || nextVal < bestVal) return row;
    return best;
  });
}

function computeEfficiency(results) {
  const successful = getSuccessfulResults(results);
  if (!successful.length) {
    return {
      baselineValBpb: null,
      bestImprovement: null,
      totalTflops: null,
      discoveryEfficiency: null,
    };
  }

  const baseline = getBaselineResult(successful);
  const best = getBestResult(successful);
  const baselineValBpb = Number(baseline?.val_bpb);
  const bestValBpb = Number(best?.val_bpb);
  const bestImprovement = baselineValBpb - bestValBpb;
  const totalTflops = successful.reduce((sum, row) => {
    const value = Number(row.total_tflops_consumed || 0);
    return sum + (Number.isFinite(value) ? value : 0);
  }, 0);

  return {
    baselineValBpb,
    bestImprovement,
    totalTflops,
    discoveryEfficiency: totalTflops > 0 ? bestImprovement / totalTflops : null,
  };
}

function renderSummary({ claims, results, best, promotions }) {
  document.getElementById("claims-count").textContent = claims.length;
  document.getElementById("results-count").textContent = results.length;
  document.getElementById("best-val").textContent = best ? fmtNum(best.best_val_bpb) : "—";
  document.getElementById("promotions-count").textContent = promotions.length;
}

function renderEfficiency(results) {
  const metrics = computeEfficiency(results);
  document.getElementById("baseline-val").textContent = fmtNum(metrics.baselineValBpb);
  document.getElementById("best-improvement").textContent = fmtNum(metrics.bestImprovement);
  document.getElementById("total-tflops").textContent = fmtTflops(metrics.totalTflops);
  document.getElementById("discovery-efficiency").textContent = fmtReward(metrics.discoveryEfficiency);
}

function renderPolicyComparison(summaries) {
  const tbody = document.getElementById("policy-comparison-body");
  if (!summaries.length) {
    tbody.innerHTML = `<tr><td colspan="9" class="empty-row">No policy comparison yet</td></tr>`;
    return;
  }

  const rows = summaries.map((summary) => {
    const label =
      summary.condition === "ab_baseline"
        ? "Baseline proposer"
        : summary.condition === "ab_history"
          ? "History-aware proposer"
          : summary.condition;

    return `
      <tr>
        <td>${label}</td>
        <td>${summary.runs}</td>
        <td>${fmtNum(summary.total_tflops, 2)}</td>
        <td>${fmtNum(summary.gpu_hours, 2)}</td>
        <td>${fmtNum(summary.best_val_bpb)}</td>
        <td>${fmtNum(summary.best_absolute_improvement, 6)}</td>
        <td>${fmtReward(summary.discovery_per_tflop)}</td>
        <td>${fmtCost(summary.estimated_cost_usd)}</td>
        <td>${fmtCost(summary.cost_per_discovery)}</td>
      </tr>
    `;
  });

  tbody.innerHTML = rows.join("");
}

function renderCurrentBest(best) {
  const el = document.getElementById("current-best");
  if (!best) {
    el.className = "panel-body empty-state";
    el.textContent = "No best run yet";
    return;
  }

  el.className = "panel-body";
  el.innerHTML = `
    <div class="current-best-grid">
      <div class="kv-row"><div class="kv-key">Fingerprint</div><div class="kv-value mono">${best.best_fingerprint || "—"}</div></div>
      <div class="kv-row"><div class="kv-key">val_bpb</div><div class="kv-value">${fmtNum(best.best_val_bpb)}</div></div>
      <div class="kv-row"><div class="kv-key">Reward</div><div class="kv-value">${fmtReward(best.best_reward)}</div></div>
      <div class="kv-row"><div class="kv-key">Job ID</div><div class="kv-value mono">${best.slurm_job_id || "—"}</div></div>
      <div class="kv-row"><div class="kv-key">Updated</div><div class="kv-value ts">${fmtTs(best.updated_at)}</div></div>
    </div>`;
}

function renderClaims(claims) {
  const el = document.getElementById("active-claims");
  if (!claims.length) {
    el.className = "panel-body empty-state";
    el.textContent = "No active claims";
    return;
  }

  el.className = "panel-body";
  el.innerHTML = `
    <div class="claims-list">${claims.map((claim) => `
      <div class="claim-item">
        <div class="mono claim-fp">${claim.fingerprint || "—"}</div>
        <div class="claim-meta">
          <span>worker: ${claim.claimed_by || "—"}</span>
          <span>state: <span class="${badgeClass(claim.state)}">${claim.state || "—"}</span></span>
          <span class="mono">job: ${claim.slurm_job_id || "—"}</span>
        </div>
      </div>`).join("")}
    </div>`;
}

function renderReasoning(reasoningRows) {
  const el = document.getElementById("latest-reasoning");
  if (!reasoningRows.length) {
    el.innerHTML = `<div class="empty-state">No reasoning trace yet</div>`;
    return;
  }

  const latest = reasoningRows[0];
  el.innerHTML = `
    <div class="current-best-grid">
      <div class="kv-row">
        <div class="kv-key">Fingerprint</div>
        <div class="kv-value mono">${escapeHtml(latest.fingerprint || "—")}</div>
      </div>
      <div class="kv-row">
        <div class="kv-key">Hypothesis</div>
        <div class="kv-value prewrap">${escapeHtml(latest.hypothesis || "—")}</div>
      </div>
      <div class="kv-row">
        <div class="kv-key">Reasoning</div>
        <div class="kv-value prewrap">${escapeHtml(latest.reasoning || "—")}</div>
      </div>
      <div class="kv-row">
        <div class="kv-key">Time</div>
        <div class="kv-value ts">${fmtTs(latest.timestamp)}</div>
      </div>
    </div>`;
}

function renderExperimentGraph(results, best) {
  const el = document.getElementById("experiment-graph");
  if (!results.length) {
    el.className = "trajectory-grid empty-state";
    el.textContent = "No experiment history yet";
    return;
  }

  const sorted = results.slice().sort((a, b) => new Date(a.timestamp || 0) - new Date(b.timestamp || 0));
  el.className = "trajectory-grid";
  el.innerHTML = sorted.map((row, index) => {
    const isBest = best && best.best_fingerprint === row.fingerprint;
    return `
      <div class="exp-node ${isBest ? "best" : ""}">
        <div class="exp-node-header">
          <div class="exp-node-title">${index === 0 ? "Baseline" : `Run ${index + 1}`}</div>
          ${isBest ? `<span class="badge badge-good">Best</span>` : ""}
        </div>
        <div class="exp-node-meta">
          <div class="mono" title="${escapeHtml(row.fingerprint || "")}">${escapeHtml(shortFp(row.fingerprint))}</div>
          <div>status: <span class="${badgeClass(row.status)}">${escapeHtml(row.status || "—")}</span></div>
          <div>outcome: <span class="${badgeClass(row.outcome)}">${escapeHtml(row.outcome || "—")}</span></div>
          <div>val_bpb: ${fmtNum(row.val_bpb)}</div>
          <div>reward: ${fmtReward(row.reward)}</div>
          <div class="mono">job: ${escapeHtml(row.slurm_job_id || "—")}</div>
          <div class="ts">${fmtTs(row.timestamp)}</div>
        </div>
      </div>
    `;
  }).join("");
}

function renderResults(results) {
  const tbody = document.getElementById("results-table-body");
  if (!results.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-row">No results yet</td></tr>`;
    return;
  }

  tbody.innerHTML = results.slice(0, 20).map((row) => `
    <tr>
      <td class="mono" title="${row.fingerprint || ""}">${shortFp(row.fingerprint)}</td>
      <td><span class="${badgeClass(row.status)}">${row.status || "—"}</span></td>
      <td><span class="${badgeClass(row.outcome)}">${row.outcome || "—"}</span></td>
      <td>${fmtNum(row.val_bpb)}</td>
      <td>${fmtReward(row.reward)}</td>
      <td class="mono">${row.slurm_job_id || "—"}</td>
      <td class="ts">${fmtTs(row.timestamp)}</td>
    </tr>`).join("");
}

function renderPromotions(promotions) {
  const tbody = document.getElementById("promotions-table-body");
  if (!promotions.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-row">No promotions yet</td></tr>`;
    return;
  }

  tbody.innerHTML = promotions.slice(0, 20).map((promotion) => `
    <tr>
      <td class="mono" title="${promotion.fingerprint || ""}">${shortFp(promotion.fingerprint)}</td>
      <td>${fmtNum(promotion.relative_improvement, 6)}</td>
      <td>${promotion.promotion_reason || "—"}</td>
      <td><span class="${badgeClass(promotion.tinker_status)}">${promotion.tinker_status || "—"}</span></td>
      <td class="mono">${promotion.tinker_request_id || "—"}</td>
      <td class="ts">${fmtTs(promotion.timestamp)}</td>
    </tr>`).join("");
}

function renderSnapshotMeta(results) {
  if (!results.length) {
    document.getElementById("last-refresh").textContent = "empty snapshot";
    return;
  }

  const latest = results[0];
  document.getElementById("last-refresh").textContent = `latest run ${fmtTs(latest.timestamp)}`;
}

async function renderDashboard() {
  const [claims, results, best, promotions, reasoningRows, summaries] = await Promise.all([
    loadClaimed(),
    loadResults(),
    loadBest(),
    loadPromotions(),
    loadReasoning(),
    loadPolicySummaries(),
  ]);

  renderSummary({ claims, results, best, promotions });
  renderEfficiency(results);
  renderPolicyComparison(summaries);
  renderCurrentBest(best);
  renderClaims(claims);
  renderReasoning(reasoningRows);
  renderExperimentGraph(results, best);
  renderResults(results);
  renderPromotions(promotions);
  renderSnapshotMeta(results);
}

renderDashboard();
