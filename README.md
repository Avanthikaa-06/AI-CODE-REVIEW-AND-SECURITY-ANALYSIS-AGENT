# AI Code Review & Security Analysis Agent

## Project Overview

This project was developed as part of the Infosys Springboard Internship.

The application consists of two modules:

1. Code Submission Module
2. Secure Coding Knowledge Base

---

## Features

### Code Submission Module

Users can submit code through:

- Paste Code
- Upload File
- GitHub Repository
- Raw URL

Supported Languages:

- Python
- Java

The system validates code and displays:

- Language
- File Name
- Number of Lines
- Number of Characters
- Validation Status
- Error Details

---

### Secure Coding Knowledge Base

The knowledge base contains:

- OWASP Top 10 2025
- OWASP Secure Coding Practices
- Java Secure Coding Practices
- Python Secure Coding Practices
- Security Best Practices
- Frequently Asked Questions

Users can search security-related topics and retrieve relevant information from the indexed documents.

---

## Technologies Used

- Python
- Streamlit
- FAISS
- Sentence Transformers
- GitHub Integration

---

## Project Structure

```text
app.py
backend.py
rag_engine.py
requirements.txt

Best_practices.txt
FAQ.txt
java_securecoding_practices.txt
OWASP_CODING_PRACTISES.txt
OWASP_Top10_2025.txt
python_securecoding_practices.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
py -m streamlit run app.py
```

The application will open in your browser.

---

## Author

Avanthikaa S G

Infosys Springboard Internship Project