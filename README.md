# AI Code Review & Security Analysis Agent

## 1. Overview

AI Code Review & Security Analysis Agent is a Streamlit-based multi-agent application that performs automated static analysis of **Python** and **Java** source code. The system combines multiple static analysis tools with LangGraph orchestration to identify code quality issues, security vulnerabilities, complexity problems, and design issues. An optional Gemini AI review provides intelligent explanations and remediation suggestions.

---

## 2. Features

- Automatic Python and Java language detection
- Multiple code submission methods
  - Paste Code
  - Upload File
  - GitHub Repository
  - Raw GitHub File URL
- Parallel execution using LangGraph
- Code Quality Analysis
- Security Vulnerability Analysis
- OWASP & CWE Mapping
- Security Score Calculation
- Interactive Streamlit Dashboard
- PDF Report Generation
- Knowledge Base (RAG) Integration
- Optional Gemini AI Review
- Tool Execution Status Monitoring

---

## 3. Technologies Used

| Category | Technologies |
|-----------|--------------|
| Frontend | Streamlit |
| Programming Languages | Python, Java |
| Agent Framework | LangGraph |
| LLM Framework | LangChain |
| AI Model | Google Gemini (Optional) |
| Python Analysis | Pylint, Radon, Python AST |
| Java Analysis | PMD, Javalang AST |
| Security Analysis | Bandit, Semgrep |
| Knowledge Base | FAISS, Sentence Transformers |
| PDF Report | ReportLab |
| Networking | Requests |

---

## 4. Project Structure

```text
AI-Code-Review-Agent/
│
├── app.py
├── backend.py
├── language_detector.py
├── rag_engine.py
├── report_generator.py
├── schemas.py
├── requirements.txt
│
├── agents/
│   ├── orchestrator.py
│   ├── codeanalysis.py
│   └── securityagent.py
│
├── knowledge_base/
│
├── rules/
│
├── samples/
│
├── tests/
│
└── README.md
```

---

## 5. Prerequisites

Before running the project, ensure the following software is installed:

- Python 3.10 or later
- Git
- Java JDK 17 or later (recommended for Java validation)
- Semgrep
- PMD (Optional)
- Google Gemini API Key (Optional)

---

## 6. Installation

### Clone the Repository

```bash
git clone <repository-url>
cd AI-Code-Review-Agent
```

### Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Semgrep

```bash
pip install semgrep
```

### Verify Java Installation

```bash
javac -version
```

### Configure PMD (Optional)

**Windows**

```text
PMD_CMD=C:\path\to\pmd\bin\pmd.bat
```

**Linux/macOS**

```bash
export PMD_CMD=/path/to/pmd/bin/pmd
```

---

## 7. Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

If a valid Gemini API key is provided, the application enables AI-powered analysis. Otherwise, all analysis is performed using local static analysis tools.

---

## 8. Output

The application generates:

- Code Quality Analysis Report
- Security Vulnerability Report
- Complexity Analysis
- OWASP Mapping
- CWE Mapping
- Security Score
- Tool Execution Summary
- Downloadable PDF Security Report
