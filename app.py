import html
import requests
import streamlit as st

from agents.orchestrator import run_analysis
from backend import (
    clone_and_extract,
    download_file,
    process_pasted_code,
    process_uploaded_file,
    process_url_file,
)
from rag_engine import RAGPipeline
from report_generator import generate_pdf_report


# ==============================================================================
# GEMINI STATUS & VALIDATION CONFIG
# ==============================================================================
GEMINI_MODEL = "gemini-3.5-flash-lite"
_MODELS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


def validate_gemini_api_key(api_key: str, timeout: int = 8) -> dict:
    """
    Validation calls Google's models-list endpoint, which is a cheap,
    read-only request — it confirms the key is real and authorized without
    spending tokens on an actual generation call.
    """
    key = (api_key or "").strip()
    if not key:
        return {"valid": False, "reason": "No API key provided."}

    try:
        response = requests.get(_MODELS_ENDPOINT, params={"key": key}, timeout=timeout)
    except requests.RequestException as exc:
        return {"valid": False, "reason": f"Could not reach the Gemini API: {exc}"}

    if response.status_code == 200:
        return {"valid": True, "reason": None}
    if response.status_code in (400, 401, 403):
        return {"valid": False, "reason": "The API key was rejected by Google (invalid or unauthorized)."}
    return {"valid": False, "reason": f"Unexpected response from the Gemini API (HTTP {response.status_code})."}


# ==============================================================================
# STREAMLIT PAGE CONFIG & STYLES
# ==============================================================================
st.set_page_config(
    page_title="AI Code Review & Security Analysis Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton, section[data-testid="stSidebar"] { display: none !important; }
.stApp { background: radial-gradient(circle at 20% 0%, #101B36 0%, #0B1120 55%); color: #F8FAFC; }
.hero-title { text-align: center; font-size: 2.4rem; font-weight: 700; background: linear-gradient(90deg, #0EA5E9, #2563EB); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-subtitle { text-align: center; color: #94A3B8; font-size: 1.05rem; margin-bottom: 2.2rem; }
.premium-card { background: rgba(23, 37, 84, 0.55); backdrop-filter: blur(12px); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 12px; padding: 1.4rem 1.6rem; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.28); margin-bottom: 1.2rem; }
.status-chip { display: inline-block; padding: 0.35rem 0.85rem; border-radius: 999px; font-weight: 600; font-size: 0.9rem; }
.status-valid { background: rgba(34, 197, 94, 0.12); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.35); }
.status-invalid { background: rgba(244, 63, 94, 0.12); color: #F43F5E; border: 1px solid rgba(244, 63, 94, 0.35); }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 1rem; margin-bottom: 1.25rem; }
.metric-chip { background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 1rem 1.1rem; text-align: center; }
.metric-label { color: #94A3B8; font-size: 0.78rem; text-transform: uppercase; margin-bottom: 0.3rem; }
.metric-value { color: #F8FAFC; font-size: 1.1rem; font-weight: 600; overflow-wrap: anywhere; }
.bar-wrap { display:grid; gap:0.55rem; }
.bar-row { display:grid; grid-template-columns: 92px 1fr 38px; gap:0.7rem; align-items:center; color:#CBD5E1; font-size:0.9rem; }
.bar-track { height:12px; background:rgba(148,163,184,0.16); border-radius:999px; overflow:hidden; }
.bar-fill { height:12px; border-radius:999px; }
.tool-row { display:grid; grid-template-columns: 1.3fr 1fr 1fr 0.6fr 0.7fr; gap:0.75rem; color:#CBD5E1; padding:0.45rem 0; border-bottom:1px solid rgba(148,163,184,0.12); font-size:0.9rem; }
.stButton>button { border-radius: 999px !important; font-weight: 500 !important; }
.stButton>button[kind="primary"] { background: linear-gradient(90deg, #2563EB, #0EA5E9) !important; border: none !important; color: white !important; }
.stTextArea textarea, .stTextInput input { background-color: rgba(15, 23, 42, 0.6) !important; color: #F8FAFC !important; border-radius: 12px !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; justify-content: center; }
.stTabs [aria-selected="true"] { background-color: rgba(37, 99, 235, 0.25) !important; color: #F8FAFC !important; }
@media (max-width: 760px) {
  .metric-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
  .tool-row { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
st.session_state.setdefault("active_view", "code_submission")
st.session_state.setdefault("gemini_api_key", "")
st.session_state.setdefault("rag_pipeline", RAGPipeline())
st.session_state.setdefault("rag_status", None)
st.session_state.setdefault("github_files_cached", None)
st.session_state.setdefault("gemini_key_validated_for", None)
st.session_state.setdefault("gemini_key_valid", None)
st.session_state.setdefault("gemini_key_reason", None)

for key in ["paste", "upload", "github", "url"]:
    st.session_state.setdefault(f"result_{key}", None)
    st.session_state.setdefault(f"deep_{key}", None)


# ==============================================================================
# HELPER & RENDER FUNCTIONS
# ==============================================================================
def _safe(value) -> str:
    return html.escape(str(value or ""))


def planned_tools(language: str) -> list[str]:
    if language == "python":
        return ["Pylint", "Radon", "Python AST", "Semgrep", "Bandit"]
    if language == "java":
        return ["PMD", "javalang AST", "Semgrep"]
    return []


def run_deep_analysis(result: dict, state_key: str) -> None:
    deep_key = f"deep_{state_key}"
    key = st.session_state.gemini_api_key.strip()
    # Only ever call Gemini if a key was provided AND it passed
    # validation. An unvalidated or invalid key silently falls back to
    # local-tools-only instead of attempting (and failing) an LLM call.
    use_llm = bool(key) and st.session_state.gemini_key_valid is True
    progress = st.progress(0, text="Preparing analysis")
    status = st.empty()
    tools = planned_tools(result["language"])
    if tools:
        status.info("Running tools: " + ", ".join(tools))
    progress.progress(20, text="Language detected")
    with st.spinner("Running parallel Code Analysis and Security agents..."):
        st.session_state[deep_key] = run_analysis(
            result["code"],
            result["language"],
            api_key=key if use_llm else None,
            use_llm=use_llm,
            filepath=result.get("file_name"),
        )
    progress.progress(100, text="Analysis complete")


def render_severity_chart(summary: dict) -> None:
    by_severity = summary.get("by_severity", {})
    colors = {
        "Critical": "#F43F5E",
        "High": "#F97316",
        "Medium": "#F59E0B",
        "Low": "#0EA5E9",
        "Info": "#94A3B8",
    }
    max_value = max(by_severity.values() or [1]) or 1
    rows = []
    for severity in ["Critical", "High", "Medium", "Low", "Info"]:
        value = by_severity.get(severity, 0)
        width = max(2, int((value / max_value) * 100)) if value else 0
        rows.append(
            f"<div class='bar-row'><div>{severity}</div><div class='bar-track'>"
            f"<div class='bar-fill' style='width:{width}%; background:{colors[severity]};'></div>"
            f"</div><div>{value}</div></div>"
        )
    st.markdown(f"<div class='premium-card'><div class='bar-wrap'>{''.join(rows)}</div></div>", unsafe_allow_html=True)


def render_tool_statuses(tool_statuses: list) -> None:
    if not tool_statuses:
        return
    rows = ["<div class='tool-row' style='font-weight:600; color:#F8FAFC;'><div>Tool</div><div>Agent</div><div>Status</div><div>Findings</div><div>Duration</div></div>"]
    for status in tool_statuses:
        duration = status.get("duration_seconds")
        duration_label = f"{duration:.3f}s" if isinstance(duration, (int, float)) else "N/A"
        status_label = status.get("status")
        if status.get("message"):
            status_label = f"{status_label}: {status.get('message')}"
        rows.append(
            "<div class='tool-row'>"
            f"<div>{_safe(status.get('tool'))}</div>"
            f"<div>{_safe(status.get('agent'))}</div>"
            f"<div>{_safe(status_label)}</div>"
            f"<div>{_safe(status.get('findings_count'))}</div>"
            f"<div>{_safe(duration_label)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='premium-card'>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_gemini_status() -> None:
    """
    Shows Gemini's status independently from the local static-analysis
    tool list. Four states: not provided / verifying / verified / invalid.
    Uses Streamlit's built-in status components (with their own built-in
    icons) instead of manual emoji.
    """
    key = st.session_state.gemini_api_key.strip()
    st.markdown("**Gemini AI Status**")

    if not key:
        st.info(
            "Gemini API Key: Not Provided  \n"
            "LLM Analysis: Disabled  \n"
            "Local Analysis: Enabled"
        )
        st.caption(
            "Running local static analysis only. Gemini-powered explanations "
            "are disabled because no API key was provided."
        )
        return

    if key != st.session_state.gemini_key_validated_for:
        # Validation runs in the settings expander as soon as the key
        # changes; if we land here it means this render happened before
        # that validation pass reached this key yet.
        st.warning(
            "Gemini API Key: Detected  \n"
            "Verifying API key..."
        )
        return

    if st.session_state.gemini_key_valid:
        st.success(
            "Gemini API Key: Verified  \n"
            f"Gemini Model: {GEMINI_MODEL}  \n"
            "LLM Analysis: Enabled"
        )
    else:
        st.error(
            "Invalid Gemini API Key  \n"
            "LLM Analysis: Disabled"
        )
        if st.session_state.gemini_key_reason:
            st.caption(st.session_state.gemini_key_reason)
        st.caption("Continuing with local analysis tools only.")


def render_findings(findings: list, summary: dict, mode: str = "") -> None:
    if mode:
        st.markdown(f"<div style='color:#94A3B8; font-size:0.85rem; margin-bottom:1rem;'>{_safe(mode)}</div>", unsafe_allow_html=True)
    by_s = summary.get("by_severity", {})
    score = summary.get("security_score", 100)
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-chip"><div class="metric-label">Total</div><div class="metric-value">{summary.get("total", 0)}</div></div>
        <div class="metric-chip"><div class="metric-label">Critical</div><div class="metric-value" style="color:#F43F5E">{by_s.get("Critical", 0)}</div></div>
        <div class="metric-chip"><div class="metric-label">High</div><div class="metric-value" style="color:#F97316">{by_s.get("High", 0)}</div></div>
        <div class="metric-chip"><div class="metric-label">Security Score</div><div class="metric-value">{score}/100</div></div>
    </div>""", unsafe_allow_html=True)
    render_severity_chart(summary)

    if not findings:
        st.markdown('<div class="premium-card" style="text-align:center; color:#94A3B8;">No issues found.</div>', unsafe_allow_html=True)
        return

    chips = {
        "Critical": "background:rgba(244,63,94,0.15); color:#F43F5E; border:1px solid rgba(244,63,94,0.4);",
        "High": "background:rgba(249,115,22,0.12); color:#F97316; border:1px solid rgba(249,115,22,0.35);",
        "Medium": "background:rgba(245,158,11,0.12); color:#F59E0B; border:1px solid rgba(245,158,11,0.35);",
        "Low": "background:rgba(14,165,233,0.12); color:#0EA5E9; border:1px solid rgba(14,165,233,0.35);",
        "Info": "background:rgba(148,163,184,0.1); color:#94A3B8; border:1px solid rgba(148,163,184,0.25);",
    }
    for finding in findings:
        line = finding.get("line") or finding.get("line_start") or "N/A"
        cwe = finding.get("cwe") or "N/A"
        owasp = finding.get("owasp") or finding.get("owasp_category") or "N/A"
        snippet = finding.get("code_snippet")
        st.markdown(f"""
        <div class="premium-card">
            <span class="status-chip" style="{chips.get(finding.get("severity", "Medium"), chips["Medium"])}">{_safe(finding.get("severity", "Medium"))}</span>
            <span style="color:#94A3B8; font-size:0.82rem; margin-left:0.6rem;">Line {_safe(line)} | <code style="color:#0EA5E9">{_safe(finding.get("tool"))}</code> | CWE: {_safe(cwe)} | OWASP: {_safe(owasp)}</span>
            <div style="margin-top:0.55rem; font-weight:600;">{_safe(finding.get("title") or finding.get("category"))}</div>
            <div style="margin-top:0.35rem; color:#F8FAFC; opacity:0.9; font-size:0.92rem;">{_safe(finding.get("description") or finding.get("message"))}</div>
            <div style="margin-top:0.55rem; color:#CBD5E1; font-size:0.9rem;"><b>Recommendation:</b> {_safe(finding.get("recommendation"))}</div>
        </div>""", unsafe_allow_html=True)
        if snippet:
            with st.expander(f"Code snippet - line {line}"):
                st.code(snippet, language=finding.get("language") or "text")


def render_results_dashboard(result: dict, state_key: str = None) -> None:
    if not result:
        st.markdown('<div class="premium-card" style="text-align:center; color:#94A3B8;">Submit code above to see analysis results.</div>', unsafe_allow_html=True)
        return

    confidence = f"{int((result.get('language_confidence') or 0) * 100)}%"
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-chip"><div class="metric-label">Detected Language</div><div class="metric-value">{_safe(result.get("language"))}</div></div>
        <div class="metric-chip"><div class="metric-label">Confidence</div><div class="metric-value">{confidence}</div></div>
        <div class="metric-chip"><div class="metric-label">Lines</div><div class="metric-value">{result.get("lines", 0)}</div></div>
        <div class="metric-chip"><div class="metric-label">Characters</div><div class="metric-value">{result.get("characters", 0)}</div></div>
    </div>""", unsafe_allow_html=True)
    st.caption(result.get("language_reason") or "")

    valid = result.get("valid") and result.get("language") in ("python", "java")
    st.markdown(f'<div class="premium-card"><span class="status-chip status-{"valid" if valid else "invalid"}">Syntax {"Valid" if valid else "Invalid"}</span></div>', unsafe_allow_html=True)
    if not valid:
        st.error(f"**{result.get('error_type', 'Validation Error')}**: {result.get('message', 'Unable to validate code.')} (Line: {result.get('error_line') or 'N/A'})")

    if result.get("code", "").strip():
        with st.expander("Code Preview"):
            st.code(result["code"], language=result["language"] if result["language"] in ("python", "java") else "text", line_numbers=True)

    if valid and state_key:
        st.info("Detected analysis stack: " + ", ".join(planned_tools(result["language"])))
        render_gemini_status()
        if st.button("Run Deep Analysis", key=f"run_deep_{state_key}", type="primary"):
            st.session_state[f"deep_{state_key}"] = None
            run_deep_analysis(result, state_key)

        deep_result = st.session_state.get(f"deep_{state_key}")
        if deep_result:
            if deep_result.get("error"):
                st.warning(deep_result["error"])
            st.markdown("### Tool Status")
            render_tool_statuses(deep_result.get("tool_statuses", []))
            st.markdown("### Deep Analysis Findings")
            render_findings(
                deep_result.get("findings", []),
                deep_result.get("summary", {"total": 0, "by_severity": {}, "by_agent": {}, "security_score": 100}),
                mode=deep_result.get("mode", ""),
            )
            try:
                pdf_bytes = generate_pdf_report(deep_result, result.get("code", ""))
                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{result.get('file_name', 'analysis')}_security_report.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{state_key}"
                )
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")


# ==============================================================================
# NAVIGATION & SETTINGS
# ==============================================================================
_, nav_col, _ = st.columns([1, 1, 1])
with nav_col:
    c1, c2 = st.columns(2)
    for col, label, view in [(c1, "Code Submission", "code_submission"), (c2, "Knowledge Base", "knowledge_base")]:
        if col.button(label, type="primary" if st.session_state.active_view == view else "secondary", use_container_width=True):
            st.session_state.active_view = view
            st.rerun()

st.markdown('<hr style="border:none; border-top:1px solid rgba(148,163,184,0.15); margin:2rem 0;">', unsafe_allow_html=True)

with st.expander("Deep Analysis Settings (Gemini API Key)"):
    st.session_state.gemini_api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="YOUR_GEMINI_API_KEY",
        label_visibility="collapsed",
    )

    _current_key = st.session_state.gemini_api_key.strip()
    if _current_key != (st.session_state.gemini_key_validated_for or ""):
        if _current_key:
            _verifying = st.empty()
            _verifying.warning("Gemini API Key: Detected  \n Verifying API key...")
            _result = validate_gemini_api_key(_current_key)
            _verifying.empty()
            st.session_state.gemini_key_valid = _result["valid"]
            st.session_state.gemini_key_reason = _result.get("reason")
        else:
            st.session_state.gemini_key_valid = None
            st.session_state.gemini_key_reason = None
        st.session_state.gemini_key_validated_for = _current_key

    render_gemini_status()


# ==============================================================================
# MAIN VIEWS (CODE SUBMISSION / KNOWLEDGE BASE)
# ==============================================================================
if st.session_state.active_view == "code_submission":
    st.markdown('<div class="hero-title">Code Submission Module</div><div class="hero-subtitle">Submit and analyze Python or Java source code</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.tabs(["Paste Code", "Upload File", "GitHub Repository", "Raw URL"])

    with t1:
        code = st.text_area("Paste code", height=250, placeholder='print("Hello") or public class Main {}', label_visibility="collapsed")
        if st.button("Analyze Code", key="run_paste", type="primary"):
            st.session_state.result_paste = process_pasted_code(code)
            st.session_state.deep_paste = None
        render_results_dashboard(st.session_state.result_paste, "paste")

    with t2:
        upload = st.file_uploader("Upload a Python or Java file", type=["py", "java"])
        if st.button("Analyze Code", key="run_upload", type="primary") and upload:
            st.session_state.result_upload = process_uploaded_file(upload.read().decode("utf-8", errors="ignore"), file_name=upload.name)
            st.session_state.deep_upload = None
        render_results_dashboard(st.session_state.result_upload, "upload")

    with t3:
        repo_url = st.text_input("GitHub URL", placeholder="https://github.com/user/repo")
        if st.button("Fetch Repository Files", type="primary") and repo_url.strip():
            with st.spinner("Cloning and detecting languages..."):
                repo_result = clone_and_extract(repo_url)
            if repo_result["success"]:
                st.session_state.github_files_cached = repo_result["files"]
            else:
                st.error(repo_result["error"])

        if st.session_state.github_files_cached:
            labels = [
                f"{item['file_name']} [{item['language']} - {int(item.get('language_confidence', 0) * 100)}%]"
                for item in st.session_state.github_files_cached
            ]
            selected = st.selectbox("Select a detected file", labels)
            if st.button("Analyze Selected File", type="secondary"):
                index = labels.index(selected)
                item = st.session_state.github_files_cached[index]
                st.session_state.result_github = process_uploaded_file(item["code"], file_name=item["file_name"])
                st.session_state.deep_github = None
        render_results_dashboard(st.session_state.result_github, "github")

    with t4:
        raw_url = st.text_input("Raw URL", placeholder="https://raw.githubusercontent.com/...")
        if st.button("Analyze Code", key="run_url", type="primary") and raw_url.strip():
            with st.spinner("Downloading and detecting language..."):
                url_result = download_file(raw_url)
            if url_result["success"]:
                st.session_state.result_url = process_url_file(url_result["code"], file_name=url_result["file_name"])
                st.session_state.deep_url = None
            else:
                st.error(url_result["error"])
        render_results_dashboard(st.session_state.result_url, "url")
else:
    st.markdown('<div class="hero-title">Secure Coding Knowledge Base</div><div class="hero-subtitle">Search security guidelines and OWASP standards</div>', unsafe_allow_html=True)
    pipeline = st.session_state.rag_pipeline
    if not st.session_state.rag_status:
        with st.spinner("Index parsing..."):
            st.session_state.rag_status = pipeline.build_index(use_cache=True, api_key=st.session_state.gemini_api_key)

    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        status = st.session_state.rag_status or {}
        st.write(f"Status: {'Ready' if bool(status.get('ready')) else 'Not built'}")
        if status.get("error"):
            st.warning(status["error"])
        if st.button("Build Index"):
            with st.spinner("Building index..."):
                st.session_state.rag_status = pipeline.build_index(use_cache=False, api_key=st.session_state.gemini_api_key)
            st.rerun()
        question = st.text_input("Search query", placeholder="e.g., What is SQL Injection?", label_visibility="collapsed")
        top_k = st.slider("Results count", 1, 10, 3)
        searched = st.button("Search Knowledge Base", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if searched and question.strip():
            with st.spinner("Searching..."):
                output = pipeline.query(question, top_k=top_k)
            if output.get("success") and output.get("results"):
                for chunk in output["results"]:
                    st.markdown(
                        f"<div class='premium-card'><div style='line-height:1.6;'>{_safe(chunk['text'])}</div>"
                        f"<div style='margin-top:0.8rem; color:#0EA5E9; font-size:0.85rem;'>Source: {_safe(chunk['source'])}</div></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.error(output.get("error", "No relevant content found."))
