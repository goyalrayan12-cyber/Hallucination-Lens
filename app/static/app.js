(() => {
    const state = {
        mode: "single",          // 'single' | 'compare'
        alpha: 0.05,             // 0.05 => 95% conformal coverage
        models: [],
        history: [],
        activeHistoryId: null,
        activeRecord: null,
        filter: "all",
        search: "",
        isRunning: false,
        timerInterval: null,
        timerStart: 0,
    };

    // DOM Elements
    const el = {
        modeSingleBtn: document.getElementById("modeSingleBtn"),
        modeCompareBtn: document.getElementById("modeCompareBtn"),
        confidencePills: document.getElementById("confidencePills"),
        sidebarAlphaSub: document.getElementById("sidebarAlphaSub"),
        singleSelectorRow: document.getElementById("singleSelectorRow"),
        compareSelectorRow: document.getElementById("compareSelectorRow"),
        modelSelectSingle: document.getElementById("modelSelectSingle"),
        modelSelectA: document.getElementById("modelSelectA"),
        modelSelectB: document.getElementById("modelSelectB"),
        singleModelKpi: document.getElementById("singleModelKpi"),
        questionInput: document.getElementById("questionInput"),
        runBtn: document.getElementById("runBtn"),
        runBtnLabel: document.getElementById("runBtnLabel"),
        newQueryBtn: document.getElementById("newQueryBtn"),
        errorBanner: document.getElementById("errorBanner"),
        runningBanner: document.getElementById("runningBanner"),
        runningTitle: document.getElementById("runningTitle"),
        runningSub: document.getElementById("runningSub"),
        liveTimer: document.getElementById("liveTimer"),
        resultsStage: document.getElementById("resultsStage"),
        historyList: document.getElementById("historyList"),
        historyCount: document.getElementById("historyCount"),
        historySearch: document.getElementById("historySearch"),
        historyFilters: document.getElementById("historyFilters"),
        exportHistoryBtn: document.getElementById("exportHistoryBtn"),
        clearHistoryBtn: document.getElementById("clearHistoryBtn"),
    };

    // ==================== Utilities ====================
    function escapeHtml(str) {
        return String(str ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function formatLatency(ms) {
        if (ms == null || Number.isNaN(ms)) return "—";
        return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
    }

    function formatTimeAgo(iso) {
        const diffSec = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000));
        if (diffSec < 60) return "just now";
        if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
        if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
        return new Date(iso).toLocaleDateString();
    }

    function verdictCopy(verdict, alpha = state.alpha) {
        const pct = Math.round((1 - alpha) * 100);
        if (verdict === "TRUST") {
            return {
                title: "VERIFIED · NOT A HALLUCINATION",
                desc: `Single-label conformal set {Factual} passes the ${pct}% statistical guarantee.`,
                short: "Factual",
            };
        }
        if (verdict === "DONT_TRUST") {
            return {
                title: "FLAGGED · LIKELY HALLUCINATION",
                desc: `Single-label conformal set {Hallucinated} indicates high fabrication risk (${pct}% guarantee).`,
                short: "Hallucinated",
            };
        }
        return {
            title: "UNCERTAIN · HALLUCINATION RISK",
            desc: `Ambiguous conformal set {Factual, Hallucinated} — guardrail abstains at ${pct}% confidence.`,
            short: "Uncertain",
        };
    }

    function evaluateAtAlpha(res, alpha) {
        const model = state.models.find((m) => m.run_name === res.run_name);
        const qHat = model?.quantiles?.[alpha.toFixed(2)] ?? res.q_hat;
        if (qHat == null || res.prob_hallucination == null) {
            return { ...res, alpha };
        }

        const p1 = res.prob_hallucination;
        const predSet = [];
        if (p1 <= qHat) predSet.push(0);
        if (1 - p1 <= qHat) predSet.push(1);

        let verdict = "UNCERTAIN";
        if (predSet.length === 1 && predSet[0] === 0) verdict = "TRUST";
        else if (predSet.length === 1 && predSet[0] === 1) verdict = "DONT_TRUST";

        return {
            ...res,
            alpha,
            q_hat: qHat,
            prediction_set: predSet,
            set_size: predSet.length,
            verdict,
            is_hallucination_flagged: verdict !== "TRUST",
        };
    }

    function formatHighlightedAnswer(res) {
        const tokens = res.tokens;
        const lps = res.token_logprobs;
        if (Array.isArray(tokens) && Array.isArray(lps) && tokens.length > 0 && tokens.length === lps.length) {
            return tokens
                .map((tok, i) => {
                    const lp = lps[i];
                    const safe = escapeHtml(tok);
                    const prob = Math.round(Math.exp(lp) * 100);
                    if (lp < -3.5) {
                        return `<span class="tok-dip-deep" title="logp=${lp.toFixed(2)} (${prob}%)">${safe}</span>`;
                    }
                    if (lp < -0.693) {
                        return `<span class="tok-dip-mild" title="logp=${lp.toFixed(2)} (${prob}%)">${safe}</span>`;
                    }
                    return safe;
                })
                .join("");
        }
        return escapeHtml(res.answer || "(Empty response)");
    }

    function showError(msg) {
        if (!msg) {
            el.errorBanner.classList.add("hidden");
            el.errorBanner.textContent = "";
            return;
        }
        el.errorBanner.textContent = msg;
        el.errorBanner.classList.remove("hidden");
    }

    // ==================== Live Stopwatch ====================
    function startStopwatch(title, sub) {
        state.isRunning = true;
        el.runBtn.disabled = true;
        el.runningTitle.textContent = title;
        el.runningSub.textContent = sub;
        el.runningBanner.classList.remove("hidden");
        state.timerStart = performance.now();
        el.liveTimer.textContent = "0.00s";
        clearInterval(state.timerInterval);
        state.timerInterval = setInterval(() => {
            const elapsed = (performance.now() - state.timerStart) / 1000;
            el.liveTimer.textContent = `${elapsed.toFixed(2)}s`;
        }, 50);
    }

    function stopStopwatch() {
        state.isRunning = false;
        el.runBtn.disabled = false;
        clearInterval(state.timerInterval);
        el.runningBanner.classList.add("hidden");
    }

    // ==================== Models ====================
    async function loadModels() {
        try {
            const res = await fetch("/api/models");
            const data = await res.json();
            state.models = data.models || [];

            if (!state.models.length) {
                showError("No trained runs found in runs/. Run `hallucination-lens run-all` first.");
                return;
            }

            const optsHtml = state.models
                .map(
                    (m) =>
                        `<option value="${escapeHtml(m.run_name)}">${escapeHtml(m.display_name)} — ${escapeHtml(
                            m.backend_label
                        )} (${escapeHtml(m.run_name)})</option>`
                )
                .join("");

            el.modelSelectSingle.innerHTML = optsHtml;
            el.modelSelectA.innerHTML = optsHtml;
            el.modelSelectB.innerHTML = optsHtml;

            if (state.models.length >= 2) {
                el.modelSelectA.selectedIndex = 0;
                el.modelSelectB.selectedIndex = 1;
            }

            renderSingleModelKpi();
        } catch (err) {
            showError(`Failed to load local models: ${err.message}`);
        }
    }

    function renderSingleModelKpi() {
        const runName = el.modelSelectSingle.value;
        const m = state.models.find((x) => x.run_name === runName);
        if (!m) {
            el.singleModelKpi.innerHTML = "";
            return;
        }
        const items = [];
        if (m.metrics?.accuracy != null)
            items.push(`<div class="kpi-mini">Test Acc<strong>${(m.metrics.accuracy * 100).toFixed(1)}%</strong></div>`);
        if (m.metrics?.coverage != null)
            items.push(`<div class="kpi-mini">Coverage<strong>${(m.metrics.coverage * 100).toFixed(1)}%</strong></div>`);
        if (m.metrics?.avg_set_size != null)
            items.push(`<div class="kpi-mini">Avg |S|<strong>${Number(m.metrics.avg_set_size).toFixed(2)}</strong></div>`);
        items.push(`<div class="kpi-mini">Engine<strong>${escapeHtml(m.backend_label)}</strong></div>`);
        el.singleModelKpi.innerHTML = items.join("");
    }

    // ==================== Mode Switching ====================
    function setMode(mode) {
        state.mode = mode;
        el.modeSingleBtn.classList.toggle("active", mode === "single");
        el.modeCompareBtn.classList.toggle("active", mode === "compare");
        el.singleSelectorRow.classList.toggle("hidden", mode !== "single");
        el.compareSelectorRow.classList.toggle("hidden", mode !== "compare");
        el.runBtnLabel.textContent = mode === "compare" ? "Ask Both & Compare" : "Analyze Output";
        showError(null);
    }

    // ==================== SVG Token Confidence Sparkline ====================
    function buildSparklineSvg(logprobs) {
        if (!logprobs || logprobs.length < 2) return "";
        const w = 500;
        const h = 68;
        const pad = 6;
        const minVal = Math.min(-4.0, ...logprobs);
        const maxVal = 0.0;
        const range = Math.max(0.5, maxVal - minVal);

        const pts = logprobs.map((v, i) => {
            const x = pad + (i / (logprobs.length - 1)) * (w - pad * 2);
            const y = pad + ((maxVal - v) / range) * (h - pad * 2);
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        });

        const deepCount = logprobs.filter((v) => v < -3.5).length;
        const lowCount = logprobs.filter((v) => v < -0.693).length;
        const worstVal = Math.min(...logprobs);
        const worstIdx = logprobs.indexOf(worstVal) + 1;

        const threshY = pad + ((maxVal - -3.5) / range) * (h - pad * 2);
        const areaPts = `${pts[0].split(",")[0]},${h - pad} ${pts.join(" ")} ${pts[pts.length - 1].split(",")[0]
            },${h - pad}`;

        return `
      <div class="sparkline-box">
        <div class="section-head">
          <span class="section-label">Token Log-Probability Trajectory (${logprobs.length} tokens)</span>
          <span class="dip-legend">
            <span>Dips &lt; -3.5: <strong style="color:${deepCount ? "#f43f5e" : "#34d399"}">${deepCount}</strong></span>
            <span>Low (&lt;50%): <strong>${lowCount}</strong></span>
            <span>Min: <strong>#${worstIdx} (${worstVal.toFixed(2)})</strong></span>
          </span>
        </div>
        <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
          <defs>
            <linearGradient id="lpGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
              <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
            </linearGradient>
          </defs>
          <line x1="${pad}" y1="${threshY.toFixed(1)}" x2="${w - pad}" y2="${threshY.toFixed(1)}"
                stroke="#f59e0b" stroke-dasharray="4 4" stroke-opacity="0.65" stroke-width="1.2"/>
          <polygon points="${areaPts}" fill="url(#lpGrad)" />
          <polyline points="${pts.join(" ")}" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linejoin="round"/>
        </svg>
      </div>
    `;
    }

    // ==================== Result Card Renderer ====================
    function renderResultCard(res, badgePrefix = "") {
        const activeAlpha = res.alpha ?? state.alpha;
        const confPct = Math.round((1 - activeAlpha) * 100);
        const vc = verdictCopy(res.verdict, activeAlpha);
        const predSetStr =
            res.prediction_set && res.prediction_set.length
                ? `{${res.prediction_set.map((c) => (c === 0 ? "Factual" : "Hallucinated")).join(", ")}}`
                : "∅ (Abstain)";

        const sFactual = res.prob_hallucination != null ? res.prob_hallucination.toFixed(3) : "—";
        const sHalluc = res.prob_hallucination != null ? (1 - res.prob_hallucination).toFixed(3) : "—";
        const qHatLine =
            res.q_hat != null
                ? `<div class="mono" style="font-size:11px;opacity:0.82;margin-top:5px;">
             q̂(${confPct}%) = ${Number(res.q_hat).toFixed(3)} · s(Factual) = ${sFactual} · s(Halluc) = ${sHalluc}
           </div>`
                : "";

        const probPct =
            res.prob_hallucination != null ? Math.round(res.prob_hallucination * 1000) / 10 : null;
        const barColor =
            probPct == null ? "#38bdf8" : probPct >= 65 ? "#f43f5e" : probPct >= 35 ? "#f59e0b" : "#10b981";

        const f = res.features || {};
        const meanLp = f.mean_log_prob ?? f.mean_logprob;
        const minLp = f.min_log_prob ?? f.min_logprob;
        const varLp = f.var_log_prob ?? f.var_logprob;

        const metricsHtml = `
      <div class="telemetry-grid">
        <div class="metric-cell">
          <div class="metric-key">Latency</div>
          <div class="metric-val">${formatLatency(res.latency_ms)}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">Throughput</div>
          <div class="metric-val">${res.tokens_per_sec ? `${res.tokens_per_sec} tok/s` : "—"}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">Set Size |S|</div>
          <div class="metric-val">${res.set_size ?? "—"}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">Mean Log-Prob</div>
          <div class="metric-val">${meanLp != null ? Number(meanLp).toFixed(3) : "—"}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">Min Log-Prob</div>
          <div class="metric-val">${minLp != null ? Number(minLp).toFixed(3) : "—"}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-key">Variance</div>
          <div class="metric-val">${varLp != null ? Number(varLp).toFixed(3) : "—"}</div>
        </div>
      </div>
    `;

        const riskBarHtml =
            probPct != null
                ? `
        <div class="risk-meter-wrap">
          <div class="risk-meter-top">
            <span>Classifier P(Hallucination)</span>
            <strong class="mono" style="color:${barColor}">${probPct.toFixed(1)}%</strong>
          </div>
          <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:${Math.min(100, Math.max(2, probPct))}%;background:${barColor};"></div>
          </div>
        </div>
      `
                : "";

        const cardId = `ans_${Math.random().toString(36).slice(2, 9)}`;

        return `
      <article class="result-card">
        <div class="result-card-head">
          <div>
            <div class="model-title">${badgePrefix ? `<span style="color:var(--accent-indigo);margin-right:6px;">${badgePrefix}</span>` : ""}${escapeHtml(res.display_name)}</div>
            <div class="model-sub">${escapeHtml(res.backend_label)} · ${escapeHtml(res.run_name)}</div>
          </div>
          <div class="latency-pill">⏱ ${formatLatency(res.latency_ms)}</div>
        </div>

        <div class="verdict-banner ${escapeHtml(res.verdict)}">
          <div>
            <div class="verdict-title">${vc.title}</div>
            <div class="verdict-desc">${vc.desc}</div>
            ${qHatLine}
          </div>
          <div class="set-badge">${escapeHtml(predSetStr)}</div>
        </div>

        <div class="answer-section">
          <div class="section-head">
            <span class="section-label">Full Model Output</span>
            <button class="copy-btn" data-copy-target="${cardId}">Copy</button>
          </div>
          <div class="answer-text" id="${cardId}">${formatHighlightedAnswer(res)}</div>
        </div>

        ${riskBarHtml}
        ${metricsHtml}
        ${buildSparklineSvg(res.token_logprobs)}
      </article>
    `;
    }

    function renderRecord(record) {
        state.activeHistoryId = record.id;
        state.activeRecord = record;

        const qHtml = `
      <div class="question-echo">
        <span>Evaluated Prompt · ${new Date(record.created_at).toLocaleString()}</span>
        ${escapeHtml(record.question)}
      </div>
    `;

        if (record.mode === "compare") {
            const resA = evaluateAtAlpha(record.payload.result_a, state.alpha);
            const resB = evaluateAtAlpha(record.payload.result_b, state.alpha);
            const comparison = {
                ...record.payload.comparison,
                agreement: resA.verdict === resB.verdict,
            };

            const agreeColor = comparison.agreement ? "#34d399" : "#a5b4fc";
            const agreeText = comparison.agreement
                ? `Consensus: ${verdictCopy(resA.verdict, state.alpha).short}`
                : `Diverged: ${verdictCopy(resA.verdict, state.alpha).short} vs ${verdictCopy(resB.verdict, state.alpha).short}`;

            const riskDeltaText =
                comparison.risk_delta_pct != null
                    ? `${comparison.risk_delta_pct > 0 ? "+" : ""}${comparison.risk_delta_pct}% (B vs A)`
                    : "N/A";

            el.resultsStage.innerHTML = `
        ${qHtml}
        <div class="compare-summary-bar">
          <div class="summary-tile">
            <div class="summary-tile-label">Guardrail Alignment</div>
            <div class="summary-tile-val" style="color:${agreeColor}">${escapeHtml(agreeText)}</div>
          </div>
          <div class="summary-tile">
            <div class="summary-tile-label">Speed Winner</div>
            <div class="summary-tile-val" style="color:var(--accent-cyan)">
              ${escapeHtml(comparison.faster_model)} (${comparison.speedup_ratio}× faster)
            </div>
          </div>
          <div class="summary-tile">
            <div class="summary-tile-label">Time Saved</div>
            <div class="summary-tile-val mono">${formatLatency(comparison.time_saved_ms)}</div>
          </div>
          <div class="summary-tile">
            <div class="summary-tile-label">Hallucination Risk Delta</div>
            <div class="summary-tile-val mono">${escapeHtml(riskDeltaText)}</div>
          </div>
        </div>
        <div class="compare-grid">
          ${renderResultCard(resA, "MODEL A")}
          ${renderResultCard(resB, "MODEL B")}
        </div>
      `;
        } else {
            const resSingle = evaluateAtAlpha(record.payload.result, state.alpha);
            el.resultsStage.innerHTML = `
        ${qHtml}
        ${renderResultCard(resSingle)}
      `;
        }

        renderHistoryList();
    }

    // ==================== Execution ====================
    async function runQuery() {
        if (state.isRunning) return;
        const question = el.questionInput.value.trim();
        if (!question) {
            showError("Please enter a question first.");
            el.questionInput.focus();
            return;
        }

        showError(null);
        const pct = Math.round((1 - state.alpha) * 100);

        try {
            let res;
            if (state.mode === "compare") {
                const runA = el.modelSelectA.value;
                const runB = el.modelSelectB.value;
                if (runA === runB) {
                    showError("Choose two different local models in Arena Compare mode.");
                    return;
                }
                startStopwatch(
                    "Arena Evaluation Running (Model A → Model B)",
                    `Serialized on Apple Silicon GPU · ${pct}% conformal guarantee`
                );
                res = await fetch("/api/compare", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question, run_a: runA, run_b: runB, alpha: state.alpha }),
                });
            } else {
                const runName = el.modelSelectSingle.value;
                startStopwatch(
                    `Querying ${runName}...`,
                    `Extracting token log-probabilities & computing ${pct}% conformal set`
                );
                res = await fetch("/api/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question, run_name: runName, alpha: state.alpha }),
                });
            }

            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || `Server returned ${res.status}`);
            }

            state.history.unshift(data);
            renderRecord(data);
        } catch (err) {
            showError(err.message);
        } finally {
            stopStopwatch();
        }
    }

    // ==================== History ====================
    async function loadHistory() {
        const params = new URLSearchParams();
        if (state.search) params.set("q", state.search);
        if (state.filter === "compare") {
            params.set("mode", "compare");
        } else if (state.filter !== "all") {
            params.set("verdict", state.filter);
        }

        const res = await fetch(`/api/history?${params.toString()}`);
        const data = await res.json();
        state.history = data.items || [];
        renderHistoryList();
    }

    function renderHistoryList() {
        el.historyCount.textContent = `${state.history.length} ${state.history.length === 1 ? "query" : "queries"
            } saved`;

        if (!state.history.length) {
            el.historyList.innerHTML = `<div class="history-empty">No matching history entries found.</div>`;
            return;
        }

        el.historyList.innerHTML = state.history
            .map((item) => {
                const activeCls = item.id === state.activeHistoryId ? "active" : "";
                const modelLabel =
                    item.mode === "compare"
                        ? `${item.primary_model} vs ${item.secondary_model}`
                        : item.primary_model;
                const badgeLabel =
                    item.verdict_summary === "DIVERGENT"
                        ? "Divergent"
                        : verdictCopy(item.verdict_summary).short;

                return `
          <div class="history-item ${activeCls}" data-id="${item.id}">
            <div class="history-item-top">
              <span class="mini-badge ${escapeHtml(item.verdict_summary)}">${escapeHtml(badgeLabel)}</span>
              <button class="history-del" data-del="${item.id}" title="Delete entry">×</button>
            </div>
            <div class="history-q">${escapeHtml(item.question)}</div>
            <div class="history-foot">
              <span>${escapeHtml(modelLabel.slice(0, 22))}</span>
              <span>${formatLatency(item.total_latency_ms)} · ${formatTimeAgo(item.created_at)}</span>
            </div>
          </div>
        `;
            })
            .join("");
    }

    // ==================== Event Bindings ====================
    el.modeSingleBtn.addEventListener("click", () => setMode("single"));
    el.modeCompareBtn.addEventListener("click", () => setMode("compare"));
    el.modelSelectSingle.addEventListener("change", renderSingleModelKpi);
    el.runBtn.addEventListener("click", runQuery);

    el.confidencePills.addEventListener("click", (e) => {
        const btn = e.target.closest(".conf-btn");
        if (!btn) return;
        el.confidencePills.querySelectorAll(".conf-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.alpha = parseFloat(btn.dataset.alpha);
        if (el.sidebarAlphaSub) {
            el.sidebarAlphaSub.textContent = `Conformal SML · α = ${state.alpha.toFixed(2)}`;
        }
        if (state.activeRecord) {
            renderRecord(state.activeRecord);
        }
    });

    el.newQueryBtn.addEventListener("click", () => {
        el.questionInput.value = "";
        el.questionInput.focus();
        state.activeHistoryId = null;
        state.activeRecord = null;
        renderHistoryList();
    });

    el.questionInput.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault();
            runQuery();
        }
    });

    document.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
            e.preventDefault();
            el.questionInput.focus();
        }
    });

    document.querySelectorAll(".preset-chip").forEach((btn) => {
        btn.addEventListener("click", () => {
            el.questionInput.value = btn.dataset.q;
            el.questionInput.focus();
        });
    });

    el.resultsStage.addEventListener("click", async (e) => {
        const copyBtn = e.target.closest("[data-copy-target]");
        if (!copyBtn) return;
        const target = document.getElementById(copyBtn.dataset.copyTarget);
        if (!target) return;
        await navigator.clipboard.writeText(target.textContent);
        const old = copyBtn.textContent;
        copyBtn.textContent = "Copied ✓";
        setTimeout(() => (copyBtn.textContent = old), 1400);
    });

    el.historyList.addEventListener("click", async (e) => {
        const delBtn = e.target.closest("[data-del]");
        if (delBtn) {
            e.stopPropagation();
            const id = Number(delBtn.dataset.del);
            await fetch(`/api/history/${id}`, { method: "DELETE" });
            state.history = state.history.filter((x) => x.id !== id);
            if (state.activeHistoryId === id) {
                state.activeHistoryId = null;
                state.activeRecord = null;
            }
            renderHistoryList();
            return;
        }
        const itemEl = e.target.closest(".history-item");
        if (!itemEl) return;
        const id = Number(itemEl.dataset.id);
        const found = state.history.find((x) => x.id === id);
        if (found) {
            el.questionInput.value = found.question;
            renderRecord(found);
        }
    });

    let searchDebounce;
    el.historySearch.addEventListener("input", (e) => {
        state.search = e.target.value;
        clearTimeout(searchDebounce);
        searchDebounce = setTimeout(loadHistory, 220);
    });

    el.historyFilters.addEventListener("click", (e) => {
        const pill = e.target.closest(".pill");
        if (!pill) return;
        el.historyFilters.querySelectorAll(".pill").forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        state.filter = pill.dataset.filter;
        loadHistory();
    });

    el.exportHistoryBtn.addEventListener("click", () => {
        window.location.href = "/api/history/export";
    });

    el.clearHistoryBtn.addEventListener("click", async () => {
        if (!state.history.length) return;
        if (!confirm("Delete all saved query history from SQLite?")) return;
        await fetch("/api/history", { method: "DELETE" });
        state.history = [];
        state.activeHistoryId = null;
        state.activeRecord = null;
        renderHistoryList();
    });

    // Boot
    loadModels();
    loadHistory();
})();