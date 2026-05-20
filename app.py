from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agents.graph import build_sales_helper_graph
from agents.knowledge_retrieval_agent import clear_retrieval_cache, retrieval_cache_status
from agents.llm import (
    reset_llm_preferences,
    reset_request_started_at,
    set_llm_preferences,
    set_request_started_at,
)


load_dotenv()

app = FastAPI(title="Predikly Sales Helper")
graph = build_sales_helper_graph()


class QueryRequest(BaseModel):
    query: str
    force_db_search: bool = False
    use_gemini_llm: bool = True
    use_local_llm: bool = True
    verbose: bool = True
    answer_style: str = "short"


def seeded_internal_context(query: str) -> dict[str, list[dict[str, Any]]]:
    return {
        "internal_context": [],
        "qdrant_sources": [],
    }


def with_verbose_details(item: dict[str, Any], details: dict[str, Any], verbose: bool) -> dict[str, Any]:
    if not verbose:
        return item

    return {
        **item,
        "verbose": details,
    }


def build_agent_trace(result: dict[str, Any], *, verbose: bool = False) -> list[dict[str, Any]]:
    evaluations = {
        evaluation.get("agent_name"): evaluation
        for evaluation in result.get("evaluations", [])
    }
    selected_agents = result.get("selected_agents", [])
    trace_items = [
        with_verbose_details(
            {
                "agent": "Input Guardrail",
                "status": result.get("input_guardrail_status", "unknown"),
                "summary": result.get("input_guardrail_reason", ""),
            },
            {
                "what_it_checked": [
                    "prompt-injection attempts",
                    "system prompt / hidden reasoning requests",
                    "self-harm or harm-inducing intent",
                    "violence or credential extraction",
                    "sales-helper scope",
                    "sales-helper scope",
                ],
                "decision": result.get("input_guardrail_status", "unknown"),
                "safe_reasoning_summary": result.get("input_guardrail_reason", ""),
            },
            verbose,
        ),
        with_verbose_details(
            {
                "agent": "Main Orchestrator Agent",
                "status": "completed",
                "summary": (
                    f"Classified intent as '{result.get('intent', 'unknown')}' "
                    f"and selected route: {', '.join(selected_agents) or 'fallback'}."
                ),
                "model": result.get("orchestrator_llm_model"),
            },
            {
                "thinking_summary": (
                    "The orchestrator inspected the query intent, honored any forced DB-search "
                    "setting, then selected the retrieval and eval path for grounded answers."
                ),
                "inputs_considered": [
                    "user_query",
                    "force_db_search",
                ],
                "selected_agents": selected_agents,
                "tools_used": [
                    "Intent Classification Tool",
                    "Agent Selection Tool",
                    "Task Planning Tool",
                    "State Management Tool",
                ],
            },
            verbose,
        ),
    ]

    if "knowledge_retrieval" in selected_agents:
        evaluation = evaluations.get("knowledge_retrieval", {})
        trace_items.append(
            with_verbose_details(
                {
                    "agent": "Knowledge Retrieval Agent",
                    "status": evaluation.get("status", "not_evaluated"),
                    "summary": (
                        f"Searched Qdrant collection '{result.get('retrieval_collection', 'unknown')}' "
                        f"and returned {len(result.get('internal_context', []))} context item(s) "
                        f"with {len(result.get('qdrant_sources', []))} source record(s)."
                    ),
                    "confidence": evaluation.get("confidence"),
                },
                {
                    "thinking_summary": (
                        "The agent searched internal Qdrant knowledge only, ranked matching "
                        "chunks, extracted source metadata, and prepared grounded context for "
                        "downstream agents."
                    ),
                    "tools_used": [
                        "Qdrant Retrieval Tool",
                        "Metadata Filter Tool",
                        "Context Builder Tool",
                        "Source Ranking Tool",
                        "Source Citation Tool",
                    ],
                    "inputs_considered": ["user_query", "QDRANT_COLLECTION_NAME", "QDRANT_FALLBACK_COLLECTIONS"],
                    "outputs_created": ["internal_context", "qdrant_sources"],
                    "retrieval_collection": result.get("retrieval_collection", "unknown"),
                    "retrieval_error": result.get("retrieval_error", ""),
                    "retrieval_cache_status": result.get("retrieval_cache_status", "not_used"),
                },
                verbose,
            )
        )

    eval_evaluation = evaluations.get("eval", {})
    trace_items.append(
        with_verbose_details(
            {
                "agent": "Eval Agent",
                "status": result.get("eval_status", eval_evaluation.get("status", "unknown")),
                "summary": result.get("eval_notes", "Checked grounding, safety, and response quality."),
                "model": result.get("eval_llm_model"),
                "confidence": eval_evaluation.get("confidence"),
            },
            {
                "thinking_summary": (
                    "The agent checked whether the response plan is grounded in available sources, "
                    "avoids unsupported claims, stays safe for brand/tone, and has a clear status."
                ),
                "tools_used": [
                    "Hallucination Check Tool",
                    "Source Grounding Tool",
                    "Confidentiality Check Tool",
                    "Sales Claim Validation Tool",
                    "Tone and Brand Safety Tool",
                    "Completeness Check Tool",
                    "Final Approval Tool",
                ],
                "inputs_considered": ["user_query", "internal_context", "qdrant_sources"],
                "outputs_created": ["eval_status", "eval_notes"],
            },
            verbose,
        )
    )

    trace_items.append(
        with_verbose_details(
            {
                "agent": "Output Guardrail",
                "status": result.get("final_response", {}).get("fallback_status", result.get("fallback_status", "not_used")),
                "summary": "Prepared UI-safe output with sources, evaluation results, fallback status, and trace ID only.",
            },
            {
                "thinking_summary": (
                    "The output guardrail ensures the UI receives only answer text, safe summaries, "
                    "sources, evaluation results, fallback status, model labels, and trace ID."
                ),
                "blocked_from_ui": [
                    "hidden chain-of-thought",
                    "system prompts",
                    "developer instructions",
                    "credentials",
                    "raw unfiltered state",
                ],
            },
            verbose,
        )
    )

    return trace_items


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Predikly Sales Helper</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f5;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #64707d;
      --line: #d8dde3;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --ok: #067647;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 28px auto;
    }

    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      margin-bottom: 18px;
    }

    h1 {
      font-size: 24px;
      margin: 0;
      letter-spacing: 0;
    }

    .status {
      color: var(--muted);
      font-size: 13px;
    }

    .workspace {
      display: grid;
      grid-template-columns: 390px 1fr;
      gap: 16px;
      align-items: start;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }

    label {
      display: block;
      font-weight: 650;
      margin-bottom: 8px;
    }

    textarea {
      width: 100%;
      min-height: 170px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 12px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }

    textarea:focus {
      outline: 2px solid rgba(15, 118, 110, 0.22);
      border-color: var(--accent);
    }

    .controls {
      display: grid;
      gap: 12px;
      margin-top: 12px;
    }

    .options {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      align-items: start;
    }

    .check {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.25;
      margin: 0;
    }

    .model-run {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(120px, 160px) 112px;
      gap: 10px;
      align-items: end;
    }

    .field {
      display: grid;
      gap: 6px;
    }

    .field label {
      color: var(--muted);
      font-size: 12px;
      margin: 0;
    }

    select {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 34px 0 11px;
      color: var(--ink);
      background: #fff;
      font: inherit;
    }

    select:focus {
      outline: 2px solid rgba(15, 118, 110, 0.22);
      border-color: var(--accent);
    }

    button {
      appearance: none;
      border: 0;
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
      min-width: 112px;
    }

    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: .55; cursor: wait; }

    .examples {
      margin-top: 14px;
      display: grid;
      gap: 8px;
    }

    .example {
      text-align: left;
      background: #eef5f4;
      color: #134e4a;
      border: 1px solid #c9ddda;
      min-width: 0;
      width: 100%;
      font-weight: 600;
    }

    .output {
      display: grid;
      gap: 12px;
    }

    .architecture {
      display: grid;
      gap: 10px;
    }

    .arch-row {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      align-items: stretch;
    }

    .arch-row.center {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      width: 76%;
      margin: 0 auto;
    }

    .arch-node {
      min-height: 58px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafb;
      padding: 10px;
      display: grid;
      align-content: center;
      gap: 3px;
      transition: border-color .18s ease, background .18s ease, box-shadow .18s ease;
    }

    .arch-node strong {
      font-size: 12px;
      line-height: 1.2;
    }

    .arch-node span {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.25;
    }

    .arch-node.active {
      border-color: var(--accent);
      background: #e8f4f2;
      box-shadow: 0 0 0 3px rgba(15, 118, 110, .12);
    }

    .arch-node.done {
      border-color: #6bb7ad;
      background: #f0faf8;
    }

    .arch-node.blocked {
      border-color: #f2a69b;
      background: #fff2ef;
    }

    .arch-node.pulse {
      animation: pulseNode 1s ease-in-out infinite;
    }

    .arch-arrow {
      color: var(--muted);
      text-align: center;
      font-size: 18px;
      line-height: 1;
      user-select: none;
    }

    @keyframes pulseNode {
      0%, 100% { box-shadow: 0 0 0 2px rgba(15, 118, 110, .10); }
      50% { box-shadow: 0 0 0 6px rgba(15, 118, 110, .18); }
    }

    .block h2 {
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 7px;
      letter-spacing: .04em;
    }

    .answer {
      white-space: pre-wrap;
      font-size: 15px;
    }

    .runtime-grid {
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 12px;
      align-items: start;
    }

    .timer {
      min-height: 58px;
      display: grid;
      align-content: center;
      justify-items: start;
      gap: 2px;
      padding: 10px 12px;
      background: #f6fbfa;
      border: 1px solid #c9e3df;
      border-radius: 6px;
    }

    .timer strong {
      font-size: 24px;
      line-height: 1;
      font-variant-numeric: tabular-nums;
    }

    .timer span {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }

    .runtime-log {
      min-height: 58px;
      max-height: 120px;
      padding: 10px 12px;
      background: #f1f3f5;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: auto;
      font-size: 12px;
      color: #344054;
      white-space: pre-wrap;
    }

    pre {
      margin: 0;
      padding: 12px;
      background: #f1f3f5;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: auto;
      max-height: 280px;
      font-size: 12px;
    }

    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 700;
      background: #eef5f4;
      color: var(--accent-dark);
    }

    .pill.bad {
      background: #fff0ed;
      color: var(--danger);
    }

    @media (max-width: 860px) {
      .workspace { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
      .arch-row, .arch-row.center { grid-template-columns: 1fr; width: 100%; }
      .runtime-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 520px) {
      .options { grid-template-columns: 1fr; }
      .model-run { grid-template-columns: 1fr; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Predikly Sales Helper</h1>
      <div class="status" id="status">Ready</div>
    </header>

    <div class="workspace">
      <section>
        <label for="query">Query</label>
        <textarea id="query">How many cases of Bill Gosling has Predikly worked on before?</textarea>
        <div class="controls">
          <div class="options">
            <label class="check">
              <input id="dbSearch" type="checkbox" />
              Search in DB
            </label>
            <label class="check">
              <input id="verbose" type="checkbox" checked />
              Verbose agent summaries
            </label>
          </div>
          <div class="model-run">
            <div class="field">
              <label for="llmMode">LLM model</label>
              <select id="llmMode">
                <option value="gemini_ollama" selected>Gemini, fallback to Ollama</option>
                <option value="gemini_only">Gemini only</option>
                <option value="ollama_only">Ollama only</option>
              </select>
            </div>
            <div class="field">
              <label for="answerStyle">Answer style</label>
              <select id="answerStyle">
                <option value="short" selected>Short</option>
                <option value="detailed">Detailed</option>
              </select>
            </div>
            <button id="run" type="button">Run</button>
          </div>
        </div>
        <div class="examples">
          <button class="example" data-query="How many cases of Bill Gosling has Predikly worked on before?">Bill Gosling count</button>
          <button class="example" data-query="Find relevant case studies for invoice reconciliation.">Find case studies</button>
          <button class="example" data-query="Draft a short sales email for invoice reconciliation automation.">Draft sales email</button>
        </div>
      </section>

      <section class="output">
        <div class="block">
          <h2>Answer</h2>
          <div class="answer" id="answer">No query run yet.</div>
        </div>
        <div class="block">
          <h2>Status</h2>
          <span class="pill" id="fallback">not_started</span>
        </div>
        <div class="block">
          <h2>Runtime</h2>
          <div class="runtime-grid">
            <div class="timer">
              <span>Elapsed</span>
              <strong id="elapsed">0.0s</strong>
            </div>
            <div class="runtime-log" id="runtimeLog">No query run yet.</div>
          </div>
        </div>
        <div class="block">
          <h2>Live Agent Architecture</h2>
          <div class="architecture" id="architecture">
            <div class="arch-row center">
              <div class="arch-node" data-agent-box="Input Guardrail">
                <strong>Input Guardrail</strong>
                <span>safety and scope check</span>
              </div>
              <div class="arch-node" data-agent-box="Main Orchestrator Agent">
                <strong>Main Orchestrator</strong>
                <span>intent and routing</span>
              </div>
              <div class="arch-node" data-agent-box="Fallback Handler">
                <strong>Fallback Handler</strong>
                <span>safe recovery path</span>
              </div>
            </div>
            <div class="arch-arrow">↓</div>
            <div class="arch-row center">
              <div class="arch-node" data-agent-box="Knowledge Retrieval Agent">
                <strong>Knowledge Retrieval</strong>
                <span>Qdrant context</span>
              </div>
              <div class="arch-node" data-agent-box="Eval Agent">
                <strong>Eval Agent</strong>
                <span>grounding and quality</span>
              </div>
              <div class="arch-node" data-agent-box="Output Guardrail">
                <strong>Output Guardrail</strong>
                <span>safe UI package</span>
              </div>
            </div>
            <div class="arch-arrow">↓</div>
            <div class="arch-row center">
              <div class="arch-node" data-agent-box="Final Response">
                <strong>Final Response</strong>
                <span>answer, sources, trace</span>
              </div>
            </div>
          </div>
        </div>
        <div class="block">
          <h2>Trace ID</h2>
          <pre id="trace">-</pre>
        </div>
        <div class="block">
          <h2>Agent Trace</h2>
          <pre id="agentTrace">[]</pre>
        </div>
        <div class="block">
          <h2>Sources</h2>
          <pre id="sources">[]</pre>
        </div>
        <div class="block">
          <h2>Evaluation</h2>
          <pre id="evaluation">[]</pre>
        </div>
        <div class="block">
          <h2>LLM Models</h2>
          <pre id="models">{}</pre>
        </div>
      </section>
    </div>
  </main>

  <script>
    const query = document.getElementById("query");
    const dbSearch = document.getElementById("dbSearch");
    const verbose = document.getElementById("verbose");
    const llmMode = document.getElementById("llmMode");
    const answerStyle = document.getElementById("answerStyle");
    const run = document.getElementById("run");
    const status = document.getElementById("status");
    const answer = document.getElementById("answer");
    const fallback = document.getElementById("fallback");
    const elapsed = document.getElementById("elapsed");
    const runtimeLog = document.getElementById("runtimeLog");
    const trace = document.getElementById("trace");
    const agentTrace = document.getElementById("agentTrace");
    const sources = document.getElementById("sources");
    const evaluation = document.getElementById("evaluation");
    const models = document.getElementById("models");
    const agentBoxes = Array.from(document.querySelectorAll("[data-agent-box]"));

    function pretty(value) {
      return JSON.stringify(value ?? null, null, 2);
    }

    function formatElapsed(ms) {
      return `${(ms / 1000).toFixed(1)}s`;
    }

    function logRuntime(message) {
      const timestamp = new Date().toLocaleTimeString();
      const line = `[${timestamp}] ${message}`;
      runtimeLog.textContent = runtimeLog.textContent === "No query run yet."
        ? line
        : `${runtimeLog.textContent}\n${line}`;
      runtimeLog.scrollTop = runtimeLog.scrollHeight;
    }

    function selectedLlmPreferences() {
      const mode = llmMode.value;
      return {
        label: {
          gemini_ollama: "Gemini with Ollama fallback",
          gemini_only: "Gemini only",
          ollama_only: "Ollama only"
        }[mode] || "Gemini with Ollama fallback",
        useGemini: mode !== "ollama_only",
        useLocal: mode !== "gemini_only"
      };
    }

    function resetArchitecture() {
      agentBoxes.forEach((box) => {
        box.classList.remove("active", "done", "blocked", "pulse");
      });
    }

    function markAgent(agentName, className) {
      const box = agentBoxes.find((item) => item.dataset.agentBox === agentName);
      if (box) box.classList.add(className);
    }

    function showRunningArchitecture() {
      resetArchitecture();
      markAgent("Input Guardrail", "active");
      markAgent("Input Guardrail", "pulse");
      markAgent("Main Orchestrator Agent", "active");
    }

    function renderArchitecture(agentTraceItems, fallbackStatus) {
      resetArchitecture();

      (agentTraceItems || []).forEach((item) => {
        const statusValue = String(item.status || "").toLowerCase();
        const className = statusValue === "blocked" || statusValue === "unsafe"
          ? "blocked"
          : "done";
        markAgent(item.agent, className);
      });

      if (fallbackStatus && fallbackStatus !== "not_used") {
        markAgent("Fallback Handler", fallbackStatus === "blocked" ? "blocked" : "done");
      }

      markAgent("Final Response", fallbackStatus === "blocked" ? "blocked" : "done");
    }

    async function submitQuery() {
      const startedAt = performance.now();
      let timerId = null;

      run.disabled = true;
      status.textContent = "Running";
      answer.textContent = "";
      fallback.textContent = "running";
      fallback.classList.remove("bad");
      elapsed.textContent = "0.0s";
      runtimeLog.textContent = "";
      const llmPreferences = selectedLlmPreferences();
      logRuntime(`Started query (${dbSearch.checked ? "DB on" : "DB off"}, ${llmPreferences.label}, ${answerStyle.value} answer).`);
      timerId = window.setInterval(() => {
        elapsed.textContent = formatElapsed(performance.now() - startedAt);
      }, 100);
      showRunningArchitecture();

      try {
        const response = await fetch("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query.value,
            force_db_search: dbSearch.checked,
            use_gemini_llm: llmPreferences.useGemini,
            use_local_llm: llmPreferences.useLocal,
            verbose: verbose.checked,
            answer_style: answerStyle.value
          })
        });
        const data = await response.json();

        answer.textContent = data.answer || data.message || "";
        fallback.textContent = data.fallback_status || "not_used";
        fallback.classList.toggle("bad", fallback.textContent !== "not_used");
        trace.textContent = data.trace_id || "-";
        agentTrace.textContent = pretty(data.agent_trace || []);
        sources.textContent = pretty(data.sources || []);
        evaluation.textContent = pretty(data.evaluation || []);
        models.textContent = pretty(data.llm_models || {});
        renderArchitecture(data.agent_trace || [], data.fallback_status || "not_used");
        status.textContent = "Complete";
        elapsed.textContent = formatElapsed(performance.now() - startedAt);
        logRuntime(`Completed in ${elapsed.textContent}. Cache: ${data.cache_status || "unknown"}. Trace: ${data.trace_id || "-"}.`);
        (data.workflow_timings || []).forEach((item) => {
          const seconds = Number(item.duration_seconds ?? 0).toFixed(3);
          logRuntime(`${item.step}: ${seconds}s (${item.status || "completed"})`);
        });
      } catch (error) {
        answer.textContent = String(error);
        fallback.textContent = "error";
        fallback.classList.add("bad");
        resetArchitecture();
        markAgent("Fallback Handler", "blocked");
        markAgent("Final Response", "blocked");
        status.textContent = "Error";
        elapsed.textContent = formatElapsed(performance.now() - startedAt);
        logRuntime(`Failed after ${elapsed.textContent}: ${String(error)}`);
      } finally {
        if (timerId) window.clearInterval(timerId);
        run.disabled = false;
      }
    }

    run.addEventListener("click", submitQuery);
    document.querySelectorAll(".example").forEach((button) => {
      button.addEventListener("click", () => {
        query.value = button.dataset.query;
      });
    });
  </script>
</body>
</html>
"""


@app.post("/query")
def query_agent(request: QueryRequest) -> dict[str, Any]:
    request_started_at = time.perf_counter()
    request_token = set_request_started_at(request_started_at)
    llm_token = set_llm_preferences(
        use_gemini=request.use_gemini_llm,
        use_local=request.use_local_llm,
    )
    seed = seeded_internal_context(request.query)
    try:
        result = graph.invoke(
            {
                "user_query": request.query,
                "force_db_search": request.force_db_search,
                "use_gemini_llm": request.use_gemini_llm,
                "use_local_llm": request.use_local_llm,
                "answer_style": request.answer_style,
                "workflow_started_at": request_started_at,
                **seed,
            }
        )
    finally:
        reset_request_started_at(request_token)
        reset_llm_preferences(llm_token)

    final_response = result.get("final_response", {})
    total_duration_ms = round((time.perf_counter() - request_started_at) * 1000, 2)
    workflow_timings = [
        *result.get("workflow_timings", []),
        {
            "step": "total_request",
            "status": "completed",
            "duration_ms": total_duration_ms,
            "duration_seconds": round(total_duration_ms / 1000, 3),
        },
    ]

    if final_response:
        response_payload = {
            **final_response,
            "cache_status": result.get("retrieval_cache_status", "not_used"),
            "workflow_timings": workflow_timings,
            "agent_trace": build_agent_trace(result, verbose=request.verbose),
        }
        return response_payload

    response_payload = {
        "message": "No final response was produced.",
        "sources": [],
        "evaluation": result.get("evaluations", []),
        "fallback_status": result.get("fallback_status", "unknown"),
        "trace_id": result.get("trace_id"),
        "cache_status": result.get("retrieval_cache_status", "not_used"),
        "workflow_timings": workflow_timings,
        "agent_trace": build_agent_trace(result, verbose=request.verbose),
    }
    return response_payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/cache/status")
def cache_status() -> dict[str, Any]:
    return retrieval_cache_status()


@app.delete("/cache")
def clear_cache() -> dict[str, Any]:
    return clear_retrieval_cache()
