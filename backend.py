
import os
import re
import shutil
import subprocess
import tempfile
import ast

import requests

def validate_python(code: str) -> dict:
    
    if code is None or not code.strip():
        return {
            "valid": False,
            "error_type": "Empty Submission",
            "message": "No code was provided.",
            "line": None,
        }

    try:
        ast.parse(code)
        return {"valid": True, "error_type": None, "message": None, "line": None}
    except SyntaxError as e:
        return {
            "valid": False,
            "error_type": "Syntax Error",
            "message": e.msg or "Invalid Python syntax",
            "line": e.lineno,
        }
    except (ValueError, IndentationError, TabError) as e:
        return {
            "valid": False,
            "error_type": "Indentation Error",
            "message": str(e),
            "line": getattr(e, "lineno", None),
        }
    except Exception as e:  # final safety net — never let this crash the app
        return {
            "valid": False,
            "error_type": "Invalid Python Code",
            "message": str(e),
            "line": None,
        }


def extract_public_class_name(code: str) -> str:
    
    if not code:
        return "TempClass"

    match = re.search(
        r'public\s+(?:final\s+|abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)',
        code,
    )
    if match:
        return match.group(1)
    return "TempClass"


def validate_java(code: str) -> dict:
    
    if code is None or not code.strip():
        return {
            "valid": False,
            "error_type": "Empty Submission",
            "message": "No code was provided.",
            "line": None,
        }

    if shutil.which("javac") is None:
        return {
            "valid": False,
            "error_type": "Environment Error",
            "message": "javac was not found on this system. Install a JDK to enable Java validation.",
            "line": None,
        }

    class_name = extract_public_class_name(code)
    temp_dir = tempfile.mkdtemp(prefix="java_validate_")
    file_path = os.path.join(temp_dir, f"{class_name}.java")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            ["javac", file_path],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=20,
        )

        if result.returncode == 0:
            return {"valid": True, "error_type": None, "message": None, "line": None}

        stderr = result.stderr or "Unknown compilation error"
        line_match = re.search(r'\.java:(\d+):', stderr)
        line_number = int(line_match.group(1)) if line_match else None

        error_lines = [l for l in stderr.splitlines() if l.strip()]
        message = error_lines[0] if error_lines else stderr
        message = re.sub(r'^.*\.java:\d+:\s*(error:)?\s*', '', message).strip()

        return {
            "valid": False,
            "error_type": "Compilation Error",
            "message": message or "Compilation failed",
            "line": line_number,
        }

    except subprocess.TimeoutExpired:
        return {
            "valid": False,
            "error_type": "Timeout Error",
            "message": "Compilation took too long and was aborted.",
            "line": None,
        }
    except Exception as e:
        return {
            "valid": False,
            "error_type": "Compilation Error",
            "message": str(e),
            "line": None,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def validate_code(code: str, language: str) -> dict:
    
    if language == "python":
        return validate_python(code)
    elif language == "java":
        return validate_java(code)
    else:
        return {
            "valid": False,
            "error_type": "Unsupported Language",
            "message": f"Validation not supported for '{language}'.",
            "line": None,
        }

GITHUB_URL_PATTERN = re.compile(
    r'^https?://github\.com/[\w.-]+/[\w.-]+(\.git)?/?$'
)


def is_valid_github_url(url: str) -> bool:
    if not url or not url.strip():
        return False
    return bool(GITHUB_URL_PATTERN.match(url.strip()))


def clone_and_extract(repo_url: str, max_files: int = 25) -> dict:
   
    repo_url = (repo_url or "").strip()

    if not is_valid_github_url(repo_url):
        return {
            "success": False,
            "error": "Invalid Repository URL",
            "files": [],
        }

    if shutil.which("git") is None:
        return {
            "success": False,
            "error": "'git' is not installed on this system.",
            "files": [],
        }

    temp_dir = tempfile.mkdtemp(prefix="repo_clone_")

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": "Invalid Repository URL or repository is inaccessible.",
                "files": [],
            }

        collected = []
        for root, _dirs, files in os.walk(temp_dir):
            if ".git" in root:
                continue
            for fname in files:
                if fname.endswith(".py") or fname.endswith(".java"):
                    full_path = os.path.join(root, fname)
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        language = "python" if fname.endswith(".py") else "java"
                        collected.append({
                            "file_name": fname,
                            "language": language,
                            "code": content,
                        })
                    except Exception:
                        continue

                if len(collected) >= max_files:
                    break
            if len(collected) >= max_files:
                break

        if not collected:
            return {
                "success": False,
                "error": "No supported Python or Java files found in this repository.",
                "files": [],
            }

        return {"success": True, "error": None, "files": collected}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Repository clone timed out.", "files": []}
    except Exception as e:
        return {"success": False, "error": f"Unable to process repository: {e}", "files": []}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


GITHUB_BLOB_PATTERN = re.compile(
    r'^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$'
)


def convert_github_blob_url(url: str) -> str:
    
    if not url:
        return url

    match = GITHUB_BLOB_PATTERN.match(url.strip())
    if not match:
        return url

    user, repo, branch, file_path = match.groups()
    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"


def is_supported_extension(url: str) -> bool:
    url_lower = (url or "").lower()
    return url_lower.endswith(".py") or url_lower.endswith(".java")


def detect_language_from_url(url: str) -> str:
    url_lower = (url or "").lower()
    if url_lower.endswith(".py"):
        return "python"
    if url_lower.endswith(".java"):
        return "java"
    return "unknown"


def download_file(url: str) -> dict:
  
    url = (url or "").strip()

    if not url:
        return {"success": False, "error": "Unable to download file", "code": None, "file_name": None, "language": None}

    try:
        url = convert_github_blob_url(url)
    except Exception:
        return {
            "success": False,
            "error": "Invalid GitHub file URL. Please check the link and try again.",
            "code": None,
            "file_name": None,
            "language": None,
        }

    if not is_supported_extension(url):
        return {
            "success": False,
            "error": "Unsupported file type. Only .py and .java URLs are supported.",
            "code": None,
            "file_name": None,
            "language": None,
        }

    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            return {
                "success": False,
                "error": "Unable to download file",
                "code": None,
                "file_name": None,
                "language": None,
            }

        file_name = url.split("/")[-1] or "downloaded_file"
        language = detect_language_from_url(url)

        return {
            "success": True,
            "error": None,
            "code": response.text,
            "file_name": file_name,
            "language": language,
        }

    except requests.exceptions.RequestException:
        return {
            "success": False,
            "error": "Unable to download file",
            "code": None,
            "file_name": None,
            "language": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unable to download file: {e}",
            "code": None,
            "file_name": None,
            "language": None,
        }
def _build_result(code: str, language: str, file_name: str) -> dict:
    code = code or ""

    if language not in ("python", "java"):
        language = "python"

    validation = validate_code(code, language)

    if not file_name:
        file_name = "submission.py" if language == "python" else "Main.java"

    return {
        "language": language,
        "file_name": file_name,
        "lines": len(code.splitlines()) if code.strip() else 0,
        "characters": len(code),
        "valid": bool(validation.get("valid", False)),
        "error_type": validation.get("error_type"),
        "message": validation.get("message"),
        "error_line": validation.get("line"),
        "code": code,
    }


def process_pasted_code(code: str, language: str) -> dict:
    """Handle code submitted via the 'Paste Code' tab."""
    default_name = "submission.py" if language == "python" else "Main.java"
    return _build_result(code, language, default_name)


def process_uploaded_file(code: str, language: str, file_name: str) -> dict:
    """Handle code submitted via the 'Upload File' tab (or a GitHub file pick)."""
    return _build_result(code, language, file_name)


def process_url_file(code: str, language: str, file_name: str) -> dict:
    """Handle code downloaded via the 'Raw URL' tab."""
    return _build_result(code, language, file_name)
