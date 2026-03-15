const STATE_BASE = "../state";
const LOGS_BASE = "../logs";
const REFRESH_MS = 5000;
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
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => {
      try { return JSON.parse(l); } catch { return null; }
    })
    .filter(Boolean);
}

async function loadClaimed() {
  try {
    const d = await fetchJson(`${STATE_BASE}/claimed.json`);
    return Array.isArray(d) ? d : [];
  } catch { return []; }
}

async function loadResults() {
  try {
    const text = await fetchText(`${STATE_BASE}/results.jsonl`);
    return parseJsonLines(text).sort((a, b) =>
      new Date(b.timestamp || 0) - new Date(a.timestamp || 0)
    );
  } catch { return []; }
}

async function loadBest() {
  try {
    const d = await fetchJson(`${STATE_BASE}/global_best.json`);
    return d && d.best_fingerprint ? d : null;
  } catch { return null; }
}

async function loadPromotions() {
  try {
    const text = await fetchText(`${STATE_BASE}/promoted.jsonl`);
    return parseJsonLines(text).sort((a, b) =>
      new Date(b.timestamp || 0) - new Date(a.timestamp || 0)
    );
  } catch { return []; }
}

async function loadReasoning() {
  try {
    const text = await fetchText(`${LOGS_BASE}/agent_reasoning.jsonl`);
    return parseJsonLines(text).sort((a, b) =>
      new Date(b.timestamp || 0) - new Date(a.timestamp || 0)
    );
  } catch { return []; }
}

// ── Formatters ──────────────────────────────────────

function fmtNum(v, digits = 6) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return Number.isNaN(n) ? "—" : n.toFixed(digits);
}

function fmtCost(v) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return Number.isNaN(n) ? "—" : `$${n.toFixed(2)}`;
}

function fmtReward(v) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return Math.abs(n) >= 0.001 ? n.toFixed(6) : n.toExponential(3);
}

function fmtTflops(v) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return "—";
  return n >= 1000 ? n.toFixed(0) : n.toFixed(3);
}

function fmtTs(raw) {
  if (!raw) return "—";
  const d = new Date(raw);
  if (isNaN(d)) return "—";
  const now = new Date();
  const diff = now - d;

  if (diff < 60_000) return "just now";
  if (diff < 3600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86400_000) return `${Math.floor(diff / 3600_000)}h ago`;

  const sameYear = d.getFullYear() === now.getFullYear();
  const mo = d.toLocaleString("en", { month: "short" });
  const day = d.getDate();
  const time = d.toLocaleTimeString("en", { hour: "2-digit", minute: "2-digit", hour12: false });

  return sameYear ? `${mo} ${day}, ${time}` : `${mo} ${day} ${d.getFullYear()}, ${time}`;
}

function fmtRefresh() {
  const d = new Date();
  return d.toLocaleTimeString("en", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
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
  return fp.length > 10 ? fp.slice(0, 10) + "…" : fp;
}

function badgeClass(v) {
  const s = String(v || "").toLowerCase();
  if (["kept", "succeeded", "submitted"].includes(s)) return "badge badge-good";
  if (["rejected", "queued", "running"].includes(s)) return "badge badge-warn";
  if (["failed", "timed_out", "cancelled"].includes(s)) return "badge badge-bad";
  return "badge badge-neutral";
}

function getSuccessfulResults(results) {
  return results
    .filter((r) => r.outcome === "succeeded")
    .slice()
    .sort((a, b) => new Date(a.timestamp || 0) - new Date(b.timestamp || 0));
}

function getBaselineResult(results) {
  return results.length ? results[0] : null;
}

function getBestResult(results) {
  if (!results.length) return null;
  return results.reduce((best, r) => {
    const nextVal = Number(r.val_bpb);
    const bestVal = Number(best.val_bpb);
    if (Number.isNaN(nextVal)) return best;
    if (Number.isNaN(bestVal) || nextVal < bestVal) return r;
    return best;
  });
}

// ── Research Efficiency ─────────────────────────────

function computeEfficiency(results) {
  const ok = getSuccessfulResults(results);

  if (!ok.length)
    return { baselineValBpb: null, bestImprovement: null, totalTflops: null, discoveryEfficiency: null };

  const baseline = getBaselineResult(ok);
  const best = getBestResult(ok);
  const baselineValBpb = Number(baseline?.val_bpb);
  const bestValBpb = Number(best?.val_bpb);

  const bestImprovement = baselineValBpb - bestValBpb;
  const totalTflops = ok.reduce((s, r) => {
    const v = Number(r.total_tflops_consumed || 0);
    return s + (Number.isNaN(v) ? 0 : v);
  }, 0);
  const discoveryEfficiency = totalTflops > 0 ? bestImprovement / totalTflops : null;

  return { baselineValBpb, bestImprovement, totalTflops, discoveryEfficiency };
}

function computeGpuHours(results) {
  return getSuccessfulResults(results).reduce((sum, r) => {
    const secs = Number(r.runtime_seconds || 0);
    return sum + (Number.isFinite(secs) ? secs : 0);
  }, 0) / 3600;
}

function computeEstimatedCostUsd(gpuHours) {
  if (gpuHours == null || Number.isNaN(Number(gpuHours))) return null;
  return gpuHours * EST_COST_PER_GPU_HOUR;
}

function computeCostPerDiscovery(estimatedCostUsd, bestImprovement) {
  if (bestImprovement == null || bestImprovement <= 0) return null;
  return estimatedCostUsd / bestImprovement;
}

function computePolicyReport(results, policyName = "AutoResearch") {
  const successful = getSuccessfulResults(results);
  const baseline = getBaselineResult(successful);
  const best = getBestResult(successful);
  const totalTflops = successful.reduce((sum, r) => {
    const v = Number(r.total_tflops_consumed || 0);
    return sum + (Number.isFinite(v) ? v : 0);
  }, 0);
  const gpuHours = computeGpuHours(successful);
  const estimatedCostUsd = computeEstimatedCostUsd(gpuHours);
  const baselineVal = baseline ? Number(baseline.val_bpb) : null;
  const bestVal = best ? Number(best.val_bpb) : null;
  const bestImprovement =
    baselineVal != null && bestVal != null ? baselineVal - bestVal : null;
  const discoveryPerTflop =
    bestImprovement != null && totalTflops > 0
      ? bestImprovement / totalTflops
      : null;
  const costPerDiscovery = computeCostPerDiscovery(
    estimatedCostUsd,
    bestImprovement
  );

  return {
    policyName,
    runs: successful.length,
    totalTflops,
    gpuHours,
    bestVal,
    bestImprovement,
    discoveryPerTflop,
    estimatedCostUsd,
    costPerDiscovery,
  };
}

function placeholderPolicy(policyName) {
  return {
    policyName,
    runs: "pending",
    totalTflops: null,
    gpuHours: null,
    bestVal: null,
    bestImprovement: null,
    discoveryPerTflop: null,
    estimatedCostUsd: null,
    costPerDiscovery: null,
  };
}

// ── Renderers ───────────────────────────────────────

function renderSummary({ claims, results, best, promotions }) {
  document.getElementById("claims-count").textContent = claims.length;
  document.getElementById("results-count").textContent = results.length;
  document.getElementById("best-val").textContent = best ? fmtNum(best.best_val_bpb) : "—";
  document.getElementById("promotions-count").textContent = promotions.length;
}

function renderEfficiency(results) {
  const m = computeEfficiency(results);
  document.getElementById("baseline-val").textContent = fmtNum(m.baselineValBpb);
  document.getElementById("best-improvement").textContent = fmtNum(m.bestImprovement);
  document.getElementById("total-tflops").textContent = fmtTflops(m.totalTflops);
  document.getElementById("discovery-efficiency").textContent = fmtReward(m.discoveryEfficiency);
}

function renderPolicyComparison(results) {
  const tbody = document.getElementById("policy-comparison-body");
  const reports = [
    placeholderPolicy("Random Search"),
    placeholderPolicy("Grid Search"),
    computePolicyReport(results, "AutoResearch"),
  ];

  tbody.innerHTML = reports.map((r) => `
    <tr>
      <td>${r.policyName}</td>
      <td>${r.runs}</td>
      <td>${r.totalTflops !== null ? fmtNum(r.totalTflops, 2) : "pending"}</td>
      <td>${r.gpuHours !== null ? fmtNum(r.gpuHours, 2) : "pending"}</td>
      <td>${r.bestVal !== null ? fmtNum(r.bestVal) : "pending"}</td>
      <td>${r.bestImprovement !== null ? `+${fmtNum(r.bestImprovement, 6)}` : "pending"}</td>
      <td>${r.discoveryPerTflop !== null ? fmtReward(r.discoveryPerTflop) : "pending"}</td>
      <td>${r.estimatedCostUsd !== null ? fmtCost(r.estimatedCostUsd) : "pending"}</td>
      <td>${r.costPerDiscovery !== null ? fmtCost(r.costPerDiscovery) : "pending"}</td>
    </tr>
  `).join("");
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
    <div class="claims-list">${claims.map((c) => `
      <div class="claim-item">
        <div class="mono claim-fp">${c.fingerprint || "—"}</div>
        <div class="claim-meta">
          <span>worker: ${c.claimed_by || "—"}</span>
          <span>state: <span class="${badgeClass(c.state)}">${c.state || "—"}</span></span>
          <span class="mono">job: ${c.slurm_job_id || "—"}</span>
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

  const sorted = results
    .slice()
    .sort((a, b) => new Date(a.timestamp || 0) - new Date(b.timestamp || 0));

  el.className = "trajectory-grid";
  el.innerHTML = sorted.map((r, idx) => {
    const isBest = best && best.best_fingerprint === r.fingerprint;
    return `
      <div class="exp-node ${isBest ? "best" : ""}">
        <div class="exp-node-header">
          <div class="exp-node-title">${idx === 0 ? "Baseline" : `Run ${idx + 1}`}</div>
          ${isBest ? `<span class="badge badge-good">Best</span>` : ""}
        </div>
        <div class="exp-node-meta">
          <div class="mono" title="${escapeHtml(r.fingerprint || "")}">${escapeHtml(shortFp(r.fingerprint))}</div>
          <div>status: <span class="${badgeClass(r.status)}">${escapeHtml(r.status || "—")}</span></div>
          <div>outcome: <span class="${badgeClass(r.outcome)}">${escapeHtml(r.outcome || "—")}</span></div>
          <div>val_bpb: ${fmtNum(r.val_bpb)}</div>
          <div>reward: ${fmtReward(r.reward)}</div>
          <div class="mono">job: ${escapeHtml(r.slurm_job_id || "—")}</div>
          <div class="ts">${fmtTs(r.timestamp)}</div>
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
  tbody.innerHTML = results.slice(0, 20).map((r) => `
    <tr>
      <td class="mono" title="${r.fingerprint || ""}">${shortFp(r.fingerprint)}</td>
      <td><span class="${badgeClass(r.status)}">${r.status || "—"}</span></td>
      <td><span class="${badgeClass(r.outcome)}">${r.outcome || "—"}</span></td>
      <td>${fmtNum(r.val_bpb)}</td>
      <td>${fmtReward(r.reward)}</td>
      <td class="mono">${r.slurm_job_id || "—"}</td>
      <td class="ts">${fmtTs(r.timestamp)}</td>
    </tr>`).join("");
}

function renderPromotions(promotions) {
  const tbody = document.getElementById("promotions-table-body");
  if (!promotions.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty-row">No promotions yet</td></tr>`;
    return;
  }
  tbody.innerHTML = promotions.slice(0, 20).map((p) => `
    <tr>
      <td class="mono" title="${p.fingerprint || ""}">${shortFp(p.fingerprint)}</td>
      <td>${fmtNum(p.relative_improvement, 6)}</td>
      <td>${p.promotion_reason || "—"}</td>
      <td><span class="${badgeClass(p.tinker_status)}">${p.tinker_status || "—"}</span></td>
      <td class="mono">${p.tinker_request_id || "—"}</td>
      <td class="ts">${fmtTs(p.timestamp)}</td>
    </tr>`).join("");
}

// ── Main loop ───────────────────────────────────────

async function refreshAll() {
  const [claims, results, best, promotions, reasoningRows] = await Promise.all([
    loadClaimed(), loadResults(), loadBest(), loadPromotions(), loadReasoning(),
  ]);

  renderSummary({ claims, results, best, promotions });
  renderEfficiency(results);
  renderPolicyComparison(results);
  renderCurrentBest(best);
  renderClaims(claims);
  renderReasoning(reasoningRows);
  renderExperimentGraph(results, best);
  renderResults(results);
  renderPromotions(promotions);

  document.getElementById("last-refresh").textContent = fmtRefresh();
}

refreshAll();
setInterval(refreshAll, REFRESH_MS);
