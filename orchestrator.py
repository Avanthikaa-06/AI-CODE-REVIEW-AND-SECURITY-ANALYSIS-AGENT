import asyncio
import operator
import os
import re
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    END = START = StateGraph = None

try:
    from agents import codeanalysis, securityagent
except ImportError:
    import codeanalysis, securityagent


SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}


class AnalysisState(TypedDict, total=False):
    code: str
    filepath: str
    language: str
    api_key: str
    use_llm: bool
    code_findings: Annotated[list, operator.add]
    security_findings: Annotated[list, operator.add]
    tool_statuses: Annotated[list, operator.add]
    agent_status: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
    findings: list
    summary: dict
    success: bool


def _as_dict_findings(items: list) -> list:
    return [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in (items or [])]


def _normalize_finding(finding: dict) -> dict:
    line = int(finding.get("line_start") or finding.get("line") or 1)
    severity = str(finding.get("severity", "Medium")).strip().capitalize()
    if severity not in SEVERITY_ORDER:
        severity = "Medium"
    agent = "security" if "security" in str(finding.get("agent", "")).lower() else "code_analysis"
    title = finding.get("title") or finding.get("category") or "Finding"
    description = finding.get("description") or finding.get("message") or ""
    cwe = finding.get("cwe_id") or finding.get("cwe")
    owasp = finding.get("owasp_category") or finding.get("owasp")
    return {
        **finding,
        "agent": agent,
        "agent_source": "security_vulnerability" if agent == "security" else "code_analysis",
        "line": line,
        "line_start": line,
        "line_end": int(finding.get("line_end") or line),
        "severity": severity,
        "title": title,
        "category": title,
        "description": description,
        "message": description,
        "cwe": cwe,
        "cwe_id": cwe,
        "owasp": owasp,
        "owasp_category": owasp,
        "recommendation": finding.get("recommendation") or "Review and remediate this finding before release.",
        "tool": finding.get("tool") or "unknown",
    }


def _dedupe(findings: list) -> list:
    seen, deduped = set(), []
    for raw in findings:
        finding = _normalize_finding(raw)
        cwe_raw = str(finding.get("cwe") or "")
        cwe_match = re.search(r"(CWE-\d+)", cwe_raw, re.IGNORECASE)
        cwe_id = cwe_match.group(1).upper() if cwe_match else ""
        key = (
            finding.get("file"),
            finding.get("line"),
            cwe_id,
            re.sub(r"\s+", " ", finding.get("title", "").lower()).strip(),
            finding.get("tool"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _summarize(findings: list) -> dict:
    by_severity = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0, "Info": 0}
    by_agent = {"code_analysis": 0, "security": 0}
    for finding in findings:
        by_severity[finding["severity"]] += 1
        by_agent[finding["agent"]] += 1

    # Balanced security score formula using lighter penalties to avoid score dropping to 0 prematurely:
    # Penalties: Critical=10, High=5, Medium=2, Low=1, Info=0. Bounded between 0 and 100.
    penalty = sum(
        by_severity[severity] * weight
        for severity, weight in {
            "Critical": 10,
            "High": 5,
            "Medium": 2,
            "Low": 1,
            "Info": 0,
        }.items()
    )
    security_score = max(0, min(100, 100 - penalty))

    return {
        "total": len(findings),
        "by_severity": by_severity,
        "by_agent": by_agent,
        "security_score": security_score,
    }


def _code_analysis_node(state: AnalysisState) -> AnalysisState:
    try:
        result = codeanalysis.analyze(
            state["code"],
            state["filepath"],
            state["language"],
            state.get("api_key", ""),
            state.get("use_llm", True),
        )
        return {
            "code_findings": _as_dict_findings(result.findings),
            "tool_statuses": result.tool_statuses,
            "agent_status": [{"code_analysis": result.status}],
            "errors": [result.error] if result.error else [],
        }
    except Exception as exc:
        return {
            "code_findings": [],
            "agent_status": [{"code_analysis": "failed"}],
            "errors": [f"Code analysis crashed: {exc}"],
        }


def _security_node(state: AnalysisState) -> AnalysisState:
    try:
        result = securityagent.analyze(
            state["code"],
            state["filepath"],
            state["language"],
            state.get("api_key", ""),
            state.get("use_llm", True),
        )
        return {
            "security_findings": _as_dict_findings(result.findings),
            "tool_statuses": result.tool_statuses,
            "agent_status": [{"security_vulnerability": result.status}],
            "errors": [result.error] if result.error else [],
        }
    except Exception as exc:
        return {
            "security_findings": [],
            "agent_status": [{"security_vulnerability": "failed"}],
            "errors": [f"Security scan crashed: {exc}"],
        }


def _merge_node(state: AnalysisState) -> AnalysisState:
    findings = _dedupe((state.get("code_findings") or []) + (state.get("security_findings") or []))
    findings.sort(key=lambda item: (SEVERITY_ORDER.get(item["severity"], 2), item["line"], item["tool"]))
    errors = [err for err in state.get("errors", []) if err]
    return {
        "findings": findings,
        "summary": _summarize(findings),
        "success": not errors,
    }


def _merge_agent_status(state: AnalysisState) -> dict:
    status = {"code_analysis": "success", "security_vulnerability": "success"}
    fragments = state.get("agent_status")
    if isinstance(fragments, list):
        for fragment in fragments:
            status.update(fragment)
    elif isinstance(fragments, dict):
        status.update(fragments)
    return status


def _build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("code_analysis_agent", _code_analysis_node)
    graph.add_node("security_agent", _security_node)
    graph.add_node("merge_findings", _merge_node)
    graph.add_edge(START, "code_analysis_agent")
    graph.add_edge(START, "security_agent")
    graph.add_edge(["code_analysis_agent", "security_agent"], "merge_findings")
    graph.add_edge("merge_findings", END)
    return graph.compile()


async def _run_langgraph_pipeline(state: AnalysisState) -> AnalysisState:
    if StateGraph is None:
        code_result, sec_result = await asyncio.gather(
            asyncio.to_thread(_code_analysis_node, state),
            asyncio.to_thread(_security_node, state),
        )
        merged_state = {
            **state,
            "code_findings": code_result.get("code_findings", []),
            "security_findings": sec_result.get("security_findings", []),
            "tool_statuses": code_result.get("tool_statuses", []) + sec_result.get("tool_statuses", []),
            "agent_status": code_result.get("agent_status", []) + sec_result.get("agent_status", []),
            "errors": code_result.get("errors", []) + sec_result.get("errors", []),
        }
        return {**merged_state, **_merge_node(merged_state)}
    graph = _build_graph()
    return await asyncio.to_thread(graph.invoke, state)


async def run_pipeline_async(code: str, filepath: str, language: str, api_key: str = "", use_llm: bool = True) -> dict:
    lang = (language or "").strip().lower()
    if lang not in ("python", "java"):
        return {
            "success": False,
            "error": f"Unsupported or undetected language: {language}",
            "file": filepath,
            "language": language,
            "findings": [],
            "tool_statuses": [],
            "agent_status": {"code_analysis": "skipped", "security_vulnerability": "skipped"},
            "summary": _summarize([]),
            "mode": "No analysis run.",
            "llm_remediation": "No analysis run.",
        }

    initial_state: AnalysisState = {
        "code": code,
        "filepath": filepath,
        "language": lang,
        "api_key": api_key or "",
        "use_llm": use_llm,
        "code_findings": [],
        "security_findings": [],
        "tool_statuses": [],
        "agent_status": [],
        "errors": [],
    }
    final_state = await _run_langgraph_pipeline(initial_state)
    errors = [err for err in final_state.get("errors", []) if err]
    mode_label = (
        "LangGraph parallel local analysis with optional LangChain/Gemini review."
        if api_key and api_key.strip() and use_llm
        else "LangGraph parallel local analysis."
    )
    return {
        "success": not errors,
        "error": "; ".join(errors) if errors else None,
        "file": filepath,
        "language": lang,
        "findings": final_state.get("findings", []),
        "tool_statuses": final_state.get("tool_statuses", []),
        "agent_status": _merge_agent_status(final_state),
        "summary": final_state.get("summary", _summarize([])),
        "mode": mode_label,
        "llm_remediation": mode_label,
    }


def run_analysis(code: str, language: str, api_key: str = None, use_llm: bool = True, filepath: str = None) -> dict:
    if not code or not code.strip():
        return {"success": False, "error": "No code provided.", "findings": [], "summary": _summarize([])}
    target = filepath or f"submitted{'.py' if (language or '').lower() == 'python' else '.java'}"
    return asyncio.run(run_pipeline_async(code, target, language, api_key or "", use_llm))


async def run_pipeline_async_public(code: str, filepath: str, language: str, api_key: str = "", use_llm: bool = True) -> dict:
    return await run_pipeline_async(code, filepath, language, api_key, use_llm)
