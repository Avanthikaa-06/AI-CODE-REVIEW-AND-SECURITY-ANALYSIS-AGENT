
import streamlit as st

from backend import (
       process_pasted_code,
       process_uploaded_file,
       process_url_file,
       clone_and_extract,
       download_file,
   )
from rag_engine import RAGPipeline

st.set_page_config(
    page_title="AI Code Review & Security Analysis Agent",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0B1120;
    --card: #172554;
    --primary: #2563EB;
    --secondary: #0EA5E9;
    --accent: #22C55E;
    --danger: #F43F5E;
    --text: #F8FAFC;
    --muted: #94A3B8;
}

/* Hide all Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
.stDeployButton,
section[data-testid="stSidebar"],
div[data-testid="stSidebarNav"] {
    display: none !important;
    visibility: hidden !important;
}

/* Global background and typography */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
.stApp {
    background: radial-gradient(circle at 20% 0%, #101B36 0%, var(--bg) 55%);
    color: var(--text);
}

/* Centered, max-width content */
.block-container {
    max-width: 1300px;
    margin: 0 auto;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3, h4 { color: var(--text) !important; font-weight: 600 !important; }

.hero-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, var(--secondary), var(--primary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    text-align: center;
    color: var(--muted);
    font-size: 1.05rem;
    margin-bottom: 2.2rem;
}

/* Glassmorphism cards */
.premium-card {
    background: rgba(23, 37, 84, 0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 20px;
    padding: 1.75rem 2rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 1.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.premium-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 40px rgba(37, 99, 235, 0.25);
}

/* Status chips */
.status-chip {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
}
.status-valid {
    background: rgba(34, 197, 94, 0.12);
    color: var(--accent);
    box-shadow: 0 0 18px rgba(34, 197, 94, 0.25);
    border: 1px solid rgba(34, 197, 94, 0.35);
}
.status-invalid {
    background: rgba(244, 63, 94, 0.12);
    color: var(--danger);
    box-shadow: 0 0 18px rgba(244, 63, 94, 0.25);
    border: 1px solid rgba(244, 63, 94, 0.35);
}

/* Metric mini-cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.25rem;
}
.metric-chip {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    text-align: center;
}
.metric-label {
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.metric-value {
    color: var(--text);
    font-size: 1.15rem;
    font-weight: 600;
    word-break: break-word;
}

/* Buttons (pill style) */
.stButton>button {
    border-radius: 999px !important;
    padding: 0.55rem 1.6rem !important;
    font-weight: 500 !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton>button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.3);
}
.stButton>button[kind="primary"] {
    background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
    border: none !important;
    color: white !important;
}

/* Text inputs / text areas / file uploader */
.stTextInput>div>div>input,
.stTextArea textarea {
    background-color: rgba(15, 23, 42, 0.6) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(15, 23, 42, 0.5) !important;
    border-radius: 14px !important;
    border: 1px dashed rgba(148, 163, 184, 0.3) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(23, 37, 84, 0.4);
    border-radius: 12px 12px 0 0;
    padding: 0.6rem 1.4rem;
    color: var(--muted);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(37, 99, 235, 0.25) !important;
    color: var(--text) !important;
}

/* Radio (language selector) rendered as segmented control */
div[role="radiogroup"] {
    justify-content: center;
    gap: 0.5rem;
}

/* Section divider */
.section-divider {
    border: none;
    border-top: 1px solid rgba(148, 163, 184, 0.15);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

if "active_view" not in st.session_state:
    st.session_state.active_view = "code_submission"

for _key in ("result_paste", "result_upload", "result_github", "result_url"):
    if _key not in st.session_state:
        st.session_state[_key] = None

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()
if "rag_status" not in st.session_state:
    st.session_state.rag_status = None
nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])
with nav_col2:
    inner_col1, inner_col2 = st.columns(2)
    with inner_col1:
        if st.button(
            "Code Submission",
            type="primary" if st.session_state.active_view == "code_submission" else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_view = "code_submission"
            st.rerun()
    with inner_col2:
        if st.button(
            "Knowledge Base",
            type="primary" if st.session_state.active_view == "knowledge_base" else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_view = "knowledge_base"
            st.rerun()

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

def render_results_dashboard(result: dict) -> None:
    st.markdown("### Results")

    if result is None:
        st.markdown(
            '<div class="premium-card" style="text-align:center; color:var(--muted);">'
            "Submit code above to see analysis results here."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    lang_display = "Python" if result["language"] == "python" else "Java"

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-chip">
                <div class="metric-label">Language</div>
                <div class="metric-value">{lang_display}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-label">File Name</div>
                <div class="metric-value">{result['file_name']}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-label">Lines</div>
                <div class="metric-value">{result['lines']}</div>
            </div>
            <div class="metric-chip">
                <div class="metric-label">Characters</div>
                <div class="metric-value">{result['characters']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)

        if result["valid"]:
            st.markdown('<span class="status-chip status-valid">Syntax Valid</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-chip status-invalid">Syntax Invalid</span>', unsafe_allow_html=True)
            st.write("")
            st.write(f"**Error Type:** {result['error_type']}")
            st.write("**Message:**")
            st.code(result["message"] or "No additional details.", language=None)
            st.write(f"**Line Number:** {result['error_line'] if result['error_line'] is not None else 'N/A'}")

        st.markdown('</div>', unsafe_allow_html=True)

    if result["code"].strip():
        with st.expander("Code Preview", expanded=False):
            lang_for_highlight = "python" if result["language"] == "python" else "java"
            st.code(result["code"], language=lang_for_highlight, line_numbers=True)
def render_code_submission_view() -> None:
    st.markdown('<div class="hero-title">Code Submission Module</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Submit, validate, and analyze source code</div>',
        unsafe_allow_html=True,
    )

    language_choice = st.radio(
        "Language",
        options=["Python", "Java"],
        horizontal=True,
        label_visibility="collapsed",
    )
    language = "python" if language_choice == "Python" else "java"

    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs(["Paste Code", "Upload File", "GitHub Repository", "Raw URL"])
    with tab1:
        placeholder = 'print("Hello World")' if language == "python" else (
            'public class Main {\n    public static void main(String[] args) {\n        System.out.println("Hello");\n    }\n}'
        )
        pasted_code = st.text_area(
            "Paste your code here",
            height=320,
            placeholder=placeholder,
            key="paste_area",
            label_visibility="collapsed",
        )
        if st.button("Analyze Code", key="run_paste", type="primary"):
            st.session_state.result_paste = None
            st.session_state.result_paste = process_pasted_code(pasted_code, language)

        render_results_dashboard(st.session_state.result_paste)
    with tab2:
        accepted_ext = ["py"] if language == "python" else ["java"]
        uploaded_file = st.file_uploader(
            f"Upload a .{accepted_ext[0]} file",
            type=accepted_ext,
            key="file_uploader",
        )
        if st.button("Analyze Code", key="run_upload", type="primary"):
            st.session_state.result_upload = None
            if uploaded_file is None:
                st.error("No file uploaded. Please choose a file first.")
            else:
                try:
                    content = uploaded_file.read().decode("utf-8", errors="ignore")
                    st.session_state.result_upload = process_uploaded_file(content, language, uploaded_file.name)
                except Exception as e:
                    st.error(f"Unsupported file type or unreadable file: {e}")

        render_results_dashboard(st.session_state.result_upload)
    with tab3:
        repo_url = st.text_input("GitHub repository URL", placeholder="https://github.com/user/repo")
        if st.button("Analyze Code", key="run_github", type="primary"):
            st.session_state.result_github = None
            if not repo_url.strip():
                st.error("Invalid Repository URL")
            else:
                with st.spinner("Cloning repository and scanning files..."):
                    repo_result = clone_and_extract(repo_url)
                if not repo_result["success"]:
                    st.error(repo_result["error"])
                else:
                    files = repo_result["files"]
                    matching = [f for f in files if f["language"] == language]
                    target_list = matching if matching else files
                    names = [f["file_name"] for f in target_list]
                    chosen_name = st.selectbox("Select a file to analyze", names, key="github_file_select")
                    chosen = next(f for f in target_list if f["file_name"] == chosen_name)
                    st.session_state.result_github = process_uploaded_file(
                        chosen["code"], chosen["language"], chosen["file_name"]
                    )

        render_results_dashboard(st.session_state.result_github)
    with tab4:
        raw_url = st.text_input("Raw file URL", placeholder="https://github.com/username/repo/blob/main/File.py")
        if st.button("Analyze Code", key="run_url", type="primary"):
            st.session_state.result_url = None
            if not raw_url.strip():
                st.error("Unable to download file")
            else:
                with st.spinner("Downloading file..."):
                    url_result = download_file(raw_url)
                if not url_result["success"]:
                    st.error(url_result["error"])
                else:
                    detected_lang = url_result["language"]
                    final_lang = detected_lang if detected_lang in ("python", "java") else language
                    st.session_state.result_url = process_url_file(
                        url_result["code"], final_lang, url_result["file_name"]
                    )

        render_results_dashboard(st.session_state.result_url)

def render_knowledge_base_view() -> None:
    st.markdown('<div class="hero-title">Secure Coding Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Search security guidelines, OWASP practices, and coding standards</div>',
        unsafe_allow_html=True,
    )

    pipeline = st.session_state.rag_pipeline

    # Ensure the index is available without exposing any technical details.
    if st.session_state.rag_status is None:
        with st.spinner("Preparing knowledge base..."):
            st.session_state.rag_status = pipeline.build_index(use_cache=True)

    status = st.session_state.rag_status
    is_ready = bool(status and status.get("ready"))

    _, center_col, _ = st.columns([1, 3, 1])
    with center_col:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)

        top_left, top_right = st.columns([3, 1])
        with top_left:
            st.markdown(
                f"<span style='color: var(--muted); font-size: 0.9rem;'>"
                f"{'Ready' if is_ready else 'Not built yet'}</span>",
                unsafe_allow_html=True,
            )
        with top_right:
            if st.button("Build Index", key="build_index_btn"):
                with st.spinner("Building knowledge base..."):
                    st.session_state.rag_status = pipeline.build_index(use_cache=False)
                st.rerun()

        query = st.text_input(
            "Search",
            placeholder="e.g. What is SQL Injection?",
            label_visibility="collapsed",
        )

        top_k = st.slider(
            "Number of results to retrieve",
            min_value=1,
            max_value=10,
            value=st.session_state.get("kb_top_k", 3),
            key="kb_top_k",
        )

        search_clicked = st.button("Search Knowledge Base", key="kb_search_btn", type="primary")

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Index Details"):
            if is_ready:
                st.write(f"Documents indexed: {status.get('num_documents', 0)}")
                st.write(f"Chunks indexed: {status.get('num_chunks', 0)}")
            else:
                st.write("The index hasn't been built yet.")
                if status and status.get("error"):
                    st.caption(status["error"])

    if search_clicked:
        _, center_col2, _ = st.columns([1, 3, 1])
        with center_col2:
            if not is_ready:
                st.markdown(
                    '<div class="premium-card" style="text-align:center; color:var(--muted);">'
                    "The knowledge base isn't ready yet. Please build it first."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Searching..."):
                    response = pipeline.query(query, top_k=top_k)

                if not response["success"]:
                    st.markdown(
                        f'<div class="premium-card" style="text-align:center; color:var(--muted);">'
                        f"{response['error']}</div>",
                        unsafe_allow_html=True,
                    )
                elif not response["results"]:
                    st.markdown(
                        '<div class="premium-card" style="text-align:center; color:var(--muted);">'
                        "No relevant content found for that question."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("### Results")
                    for chunk in response["results"]:
                        st.markdown(
                            f"""
                            <div class="premium-card">
                                <div style="color:var(--text); line-height:1.6;">{chunk['text']}</div>
                                <div style="margin-top:0.8rem; color:var(--secondary); font-size:0.85rem;">
                                    Source: {chunk['source']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
if st.session_state.active_view == "code_submission":
    render_code_submission_view()
else:
    render_knowledge_base_view()



