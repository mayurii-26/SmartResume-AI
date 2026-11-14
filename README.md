# SmartResume AI – Automated Resume Information Extraction and Analysis

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8+-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![GroqCloud](https://img.shields.io/badge/Groq-API%20LPU-F55036)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-45%20Passed-10B981)](#12-testing-and-verification)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **SmartResume AI** is an enterprise-grade, evidence-grounded resume processing and candidate intelligence platform. It seamlessly unifies deterministic Natural Language Processing (tokenization, lemmatization, BoW, and TF-IDF), hybrid Named Entity Recognition (NER), multi-stage canonical entity linking, Latent Dirichlet Allocation (LDA) topic modeling, VADER sentiment analysis, and conversational LLM reasoning powered by **GroqCloud LPU Inference**.

---

## 📑 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Technology Stack](#3-technology-stack)
4. [System Architecture](#4-system-architecture)
5. [Project Folder Structure](#5-project-folder-structure)
6. [Installation and Setup (Windows)](#6-installation-and-setup-windows)
7. [Environment Variables](#7-environment-variables)
8. [Running the Application](#8-running-the-application)
9. [How to Use the Application](#9-how-to-use-the-application)
10. [AI Resume Assistant (GroqCloud)](#10-ai-resume-assistant-groqcloud)
11. [PDF Report Generation](#11-pdf-report-generation)
12. [Testing and Verification](#12-testing-and-verification)
13. [Database and Data Governance](#13-database-and-data-governance)
14. [Security and Privacy](#14-security-and-privacy)
15. [Future Enhancements](#15-future-enhancements)
16. [Author and Academic Credits](#16-author-and-academic-credits)

---

## 1. Project Overview

Modern recruitment workflows require evaluating hundreds of unstructured resumes formatted in diverse layouts, styling structures, and file encodings (PDF, DOCX, TXT). Traditional Applicant Tracking Systems (ATS) rely primarily on rigid keyword matching, regex parsers, or opaque scoring algorithms that fail to disambiguate synonyms (e.g., treating `JS`, `JavaScript`, and `ES6` as unrelated terms), fail on complex document structures, and introduce algorithmic bias.

**SmartResume AI** provides an interpretable, transparent, and evidence-grounded alternative:
* **Deterministic NLP First:** Core parsing, entity extraction, ontology linking, topic modeling, and sentiment analysis run entirely locally in your Python environment without requiring external API calls.
* **Zero Hallucination Guarantee:** Information extraction preserves the complete sentences, authentic dates, and exact wording from the candidate's document.
* **Grounded AI Intelligence:** When cloud intelligence is requested, queries are routed through the official **GroqCloud API** under a strict system prompt that constrains answers strictly to facts explicitly stated in the uploaded resume.
* **Publication-Ready Reporting:** Generates styled, multi-page PDF evaluation reports in memory with zero residual disk storage.

---

## 2. Key Features

### 📄 Multi-Format Document Ingestion & Text Extraction
* Supports **PDF**, **DOCX**, and **TXT** resume formats.
* Ingests native text using **PyMuPDF** (`pymupdf`) and **python-docx**.
* Automatic scanned document detection with integrated **Tesseract OCR** fallback for image-based PDFs.
* Sentence continuity restoration: merges wrapped PDF continuation lines to preserve complete bullet points and full sentences without arbitrary truncation.

### 🔬 Deep Linguistic Preprocessing & Vectorization
* Non-destructive text normalization (standardizing bullet glyphs, smart quotes, dashes, and whitespace).
* Sentence segmentation and word tokenization using **NLTK**.
* Part-of-Speech tagging and WordNet lemmatization with configurable stopword elimination.
* Generates **Bag-of-Words (BoW)** count matrices and **TF-IDF** (Term Frequency–Inverse Document Frequency) matrices with customizable n-gram ranges.

### 🏷️ Hybrid Named Entity Recognition (NER)
* Custom transition-based NER component combined with **spaCy**'s `EntityRuler`.
* Includes **301+ curated domain patterns** covering programming languages, frameworks, cloud platforms, job titles, and academic degrees.
* Identifies 10+ entity classes: `PERSON`, `EMAIL`, `PHONE`, `ORGANIZATION`, `LOCATION`, `DATE`, `DEGREE`, `UNIVERSITY`, `SKILL`, `JOB_TITLE`, `CERTIFICATION`.
* Regex post-processing enforces clean character spans and validates contact boundaries.

### 🧩 4-Stage Canonical Entity Linking
* Resolves raw, unstructured skill mentions to canonical entities using a standardized Knowledge Base ontology (`data/skill_knowledge_base.json`).
* Implements a strict hierarchical resolution cascade:
  1. **Exact Matching:** Case-sensitive equivalence ($1.0$ confidence).
  2. **Case-Insensitive Matching:** Lowercase equivalence ($1.0$ confidence).
  3. **Curated Alias Matching:** Maps acronyms and synonyms (e.g., `k8s` $\to$ `Kubernetes`, `py` $\to$ `Python`).
  4. **Fuzzy Levenshtein Distance:** High-speed approximate matching via **RapidFuzz** ($>82\%$ threshold).
* Cross-references standardized entries with external **Wikidata IDs** and category taxonomies.

### 👤 Candidate Information Extraction & Profile Completeness
* Automated semantic section detection (`Summary`, `Experience`, `Education`, `Skills`, `Projects`, `Certifications`) with context-aware heading classification.
* Constructs structured relational candidate profiles capturing contact identity, education timelines, work achievements, and technical competencies.
* Computes an objective, mathematical **Profile Completeness Score (0–100%)** based on weighted section presence.

### ⚖️ Multi-Candidate Comparison
* Side-by-side comparative analysis of two or more uploaded candidate resumes.
* Interactive skill alignment matrix highlighting shared overlapping skills versus unique individual proficiencies.
* Structured comparison tables contrasting education credentials, cumulative experience, and completeness metrics.

### 🌐 Career Domain Topic Modeling (LDA)
* Unsupervised career cluster discovery across the resume corpus using Scikit-Learn **Latent Dirichlet Allocation (LDA)**.
* Adaptive CountVectorizer configuration that dynamically scales `min_df` and `max_df` according to corpus size, preventing empty vocabulary errors.
* Extracts salient keywords per topic, assigns human-readable domain labels (Software Engineering, Data Science & AI, Cloud & DevOps, Database Systems, Project Management), and computes document-topic distribution percentages.

### 💬 Descriptive Sentiment & Tone Analysis
* Multi-section linguistic tone analysis using **NLTK VADER**.
* Analyzes emotional valence across narrative sections (`Summary`, `Experience`, `Projects`), categorizing tone into *Enthusiastic*, *Professional/Positive*, *Neutral*, or *Critical*.
* Prominent ethical disclaimer emphasizing that sentiment scores measure descriptive writing style only and must not be used for hiring decisions.

### 🌍 Multilingual Detection & Translation
* Automatic language identification using `langdetect`.
* Integrated resume translation into professional English powered by GroqCloud inference, preserving proper names, technical keywords, and job designations.

### 🤖 Grounded AI Resume Assistant (GroqCloud)
* Interactive conversational interface powered by the official **GroqCloud API** (`https://api.groq.com/openai/v1`).
* Dynamic model discovery: queries active models on the user's Groq account (supporting `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `allam-2-7b`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`).
* Grounded prompt constraints: strictly replies with *"Not mentioned in the uploaded resume"* whenever requested facts are not in the source text.
* Four specialized modes:
  1. **Interactive Q&A:** Quick question chips and freeform user inquiries.
  2. **AI Executive Summary:** Synthesizes structured candidate overviews.
  3. **Optimization & ATS Critique:** Reviews clarity, metric quantification, missing standard sections, and ATS keyword relevance.
  4. **Structured Resume Insights:** Generates clean JSON capturing summary, skills, education, experience, projects, and suggestions.

### 📥 In-Memory PDF Report Downloads
* Adds dedicated **Download as PDF** functionality to:
  * **AI Executive Summary** (`Resume_Executive_Summary.pdf`)
  * **Optimization & Critique Suggestions** (`Resume_Optimization_Critique.pdf`)
  * **Structured Resume Insights** (`Resume_Structured_Insights.pdf`)
* Built completely in-memory using **PyMuPDF** (`Story.write_with_links`) without writing permanent files to disk.
* Professional typography, candidate metadata header box, running header, running footer with page numbers (`Page X of Y`), and automatic multi-page flow.

### 🛡️ Resume Management & GDPR-Compliant Deletion
* Individual resume deletion: permanently purges an uploaded resume document, its physical storage file in `data/storage/`, and all cascading SQLite records (`candidate_profiles`, `extracted_entities`, `entity_links`, `sentiment_scores`, `resume_topics`, `ai_interactions`).
* Shared skill knowledge bases and other candidates' records remain completely untouched.

---

## 3. Technology Stack

| Category | Technologies / Libraries |
| :--- | :--- |
| **Core Language & Runtime** | Python 3.11+ |
| **Web Application & UI** | Streamlit 1.40+, Custom CSS (Linear/Stripe SaaS Theme), Plotly Express |
| **Natural Language Processing** | spaCy 3.8+, NLTK 3.10+ (VADER, WordNet, Punkt), RapidFuzz 3.14+, langdetect |
| **Machine Learning** | Scikit-Learn 1.4+ (CountVectorizer, TfidfVectorizer, LatentDirichletAllocation) |
| **Document Processing** | PyMuPDF (fitz) 2.1+, python-docx 1.1+, Pillow, pytesseract (OCR) |
| **PDF Generation** | PyMuPDF (Story layout engine), markdown-it-py 4.3+ |
| **LLM Inference Provider** | GroqCloud API (`https://api.groq.com/openai/v1`), Requests, python-dotenv |
| **Database & Persistence** | SQLite3 (`data/smartresume.db`), JSON, pathlib |
| **Testing & Quality Assurance** | Pytest 9.2+, AnyIO |

---

## 4. System Architecture

```
                                  ┌──────────────────────────────┐
                                  │   Document Ingestion Pipeline│
                                  │   (PDF / DOCX / TXT / OCR)   │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │    Linguistic Preprocessing  │
                                  │ (Tokenize, Lemma, BoW, TFIDF)│
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │          Hybrid NER          │
                                  │   spaCy + 301 Entity Rules   │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   4-Stage Entity Linking     │
                                  │ (Exact -> Alias -> Levenshtein)│
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │    Structured Profile &      │
                                  │  Sentence Integrity Extraction│
                                  └──────────────┬───────────────┘
                                                 │
                         ┌───────────────────────┼───────────────────────┐
                         ▼                       ▼                       ▼
            ┌────────────────────────┐┌─────────────────────┐┌────────────────────────┐
            │   Topic Modeling (LDA) ││  Sentiment Analysis ││  Grounded Groq AI LPU  │
            │ Career Domain Clusters ││ NLTK VADER Polarity ││  Q&A, Summary, Critique│
            └────────────┬───────────┘└──────────┬──────────┘└───────────┬────────────┘
                         │                       │                       │
                         └───────────────────────┼───────────────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   SQLite Local Persistence   │
                                  │   & In-Memory PDF Exporter   │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │  Streamlit Multi-Page SaaS   │
                                  │ (Linear/Stripe Light Theme)  │
                                  └──────────────────────────────┘
```

---

## 5. Project Folder Structure

```
SmartResume-AI/
├── app.py                     # Streamlit application entrypoint & multi-page router
├── train.py                   # NER model training and evaluation script
├── download_dataset.py        # Dataset downloader for HuggingFace & Dotin benchmarks
├── requirements.txt           # Production Python dependencies
├── .env.example               # Environment variable configuration template
├── README.md                  # Comprehensive technical documentation
├── ACADEMIC_POSTER.md         # Academic poster summary and project abstract
│
├── app_views/                 # Streamlit UI page controllers
│   ├── dashboard.py           # Overview KPI dashboard and recent uploads
│   ├── upload.py              # Single and bulk resume upload interface
│   ├── analyzer.py            # Linguistic preprocessing, BoW, and TF-IDF view
│   ├── ner_view.py            # Named entity visualizer and label distributions
│   ├── extraction.py          # Candidate profile view and section explorer
│   ├── entity_linking.py      # Entity linking ontology resolution viewer
│   ├── comparison.py          # Multi-candidate side-by-side skill matrix
│   ├── topics.py              # LDA topic modeling and domain probability matrix
│   ├── sentiment.py           # VADER sentiment and narrative tone analysis
│   ├── ai_assistant.py        # Groq-powered AI Assistant and PDF report exports
│   ├── evaluation.py          # Precision, Recall, and F1 model evaluation reports
│   ├── reports.py             # Analytics export reports
│   └── settings_view.py       # API configuration, dynamic models, and GDPR purge
│
├── config/
│   └── settings.py            # Paths, directories, UI theme tokens, and environment configs
│
├── data/
│   ├── smartresume.db         # Persistent SQLite database (created on first run)
│   ├── skill_knowledge_base.json  # Canonical skill taxonomy, aliases, and Wikidata IDs
│   ├── dataset_metadata.json  # Benchmark dataset partition metadata
│   └── storage/               # Uploaded resume file storage directory
│
├── database/
│   └── db_manager.py          # SQLite DatabaseManager with CRUD and cascade deletion
│
├── models/
│   ├── ner_model/             # Trained spaCy transition-based NER model
│   └── evaluation_results.json# Gold benchmark precision, recall, and F1 metrics
│
├── modules/                   # Core business logic and NLP processing engines
│   ├── text_extraction.py     # PDF/DOCX/TXT parser with OCR support
│   ├── preprocessing.py       # Tokenization, lemmatization, and stopword cleaning
│   ├── vectorization.py       # Bag-of-Words and TF-IDF feature matrix generators
│   ├── ner.py                 # Hybrid NER pipeline with rule-based patterns
│   ├── entity_linking.py      # 4-stage ontology resolver and fuzzy matcher
│   ├── information_extraction.py # Semantic section parser and profile extractor
│   ├── candidate_comparison.py# Cross-candidate skill matrix and alignment engine
│   ├── topic_modeling.py      # Scikit-Learn LDA career domain modeler
│   ├── sentiment_analysis.py  # NLTK VADER sentiment analyzer
│   ├── genai_assistant.py     # GroqCloud client with dynamic model discovery
│   ├── multilingual.py        # Language detection and Groq translation
│   ├── pdf_generator.py       # In-memory publication-grade PDF report engine
│   └── report_generator.py    # Analytical summary formatting
│
├── tests/                     # Automated Pytest test suite (45 tests)
│   ├── test_candidate_comparison.py
│   ├── test_database.py
│   ├── test_entity_linking.py
│   ├── test_extraction.py
│   ├── test_genai_assistant.py
│   ├── test_information_extraction.py
│   ├── test_ner.py
│   ├── test_pdf_generator.py
│   ├── test_preprocessing.py
│   └── test_sentiment_and_topics.py
│
└── ui/
    └── styles.py              # Custom CSS styles, theme tokens, and Plotly configs
```

---

## 6. Installation and Setup (Windows)

Follow these step-by-step instructions to set up and run SmartResume AI on Windows:

### Step 1: Clone the Repository
Open PowerShell or Command Prompt and clone the repository:
```powershell
git clone https://github.com/your-username/SmartResume-AI.git
cd SmartResume-AI
```

### Step 2: Create a Virtual Environment
Using Python 3.11+:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
In PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```
*(If you encounter execution policy restrictions, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first).*

In Command Prompt:
```cmd
venv\Scripts\activate.bat
```

### Step 4: Install Dependencies
Install all required packages from `requirements.txt`:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Download Required NLTK and spaCy Models
Download the standard linguistic models:
```powershell
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

### Step 6: Configure Environment Variables
Copy `.env.example` to create your local `.env` file:
```powershell
Copy-Item .env.example .env
```
Open `.env` in an editor and add your **Groq API Key**:
```ini
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_API_BASE=https://api.groq.com/openai/v1
```
*(You can obtain a free API key at [GroqCloud Console](https://console.groq.com/keys)).*

---

## 7. Environment Variables

SmartResume AI uses `python-dotenv` to load local configurations securely.

### Safe Template (`.env.example`)
```ini
# SmartResume AI Environment Configuration

# GroqCloud API Key (Required for AI Resume Assistant)
GROQ_API_KEY=your_groq_api_key_here
# Supported models: qwen/qwen3.8-27b, openai/gpt-oss-120b, openai/gpt-oss-20b, allam-2-7b, llama-3.3-70b-versatile, llama-3.1-8b-instant
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_API_BASE=https://api.groq.com/openai/v1

# Optional: AWS Credentials (if using AWS Bedrock/Comprehend)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# Application Settings
APP_ENV=development
DB_PATH=data/smartresume.db
LOG_LEVEL=INFO
ENABLE_OCR=true
TESSERACT_CMD=tesseract
```

> ⚠️ **Security Warning:** Never commit your `.env` file to version control. The `.gitignore` file is pre-configured to ignore `.env`, local `.db` databases, and cache folders.

---

## 8. Running the Application

Launch the Streamlit web application from your activated virtual environment:

```powershell
streamlit run app.py
```

The application will initialize and open automatically in your browser at:
```
http://localhost:8501
```

---

## 9. How to Use the Application

1. **Dashboard (Overview):** Review global corpus statistics, total parsed candidates, unique skills, and average profile completeness score.
2. **Upload Resumes:** Navigate to **Upload resume** and drag & drop one or multiple PDF, DOCX, or TXT documents. Toggle OCR if uploading scanned documents.
3. **Resume Analyzer:** Inspect raw tokens, WordNet lemmas, stopword removals, Bag-of-Words distributions, and TF-IDF feature weights.
4. **Named Entity Recognition:** Visualize highlighted entity spans across 10+ categories and review extracted label counts.
5. **Information Extraction:** View the parsed candidate profile, section breakdown (Summary, Experience, Education, Skills, Projects), and completeness score.
6. **Entity Linking:** See how raw, messy resume skills are resolved to canonical ontology records with similarity scores and Wikidata identifiers.
7. **Candidate Comparison:** Select two or more uploaded resumes to generate a side-by-side skill comparison matrix and review unique candidate strengths.
8. **Topic Modeling:** Discover latent career domain clusters and inspect document-topic probability distributions across the corpus.
9. **Sentiment Analysis:** Review descriptive linguistic tone and emotional valence across narrative sections.
10. **AI Resume Assistant:** Select a candidate to ask custom questions, run preset question chips, generate executive summaries, obtain ATS critique, and generate structured insights.
11. **PDF Downloads:** Click **Download as PDF** in the AI Assistant to export publication-grade reports.
12. **Settings & Data Privacy:** Update your Groq API credentials, test the live API connection, or permanently erase candidate records.

---

## 10. AI Resume Assistant (GroqCloud)

The AI Resume Assistant provides fast, evidence-grounded generative intelligence powered by **GroqCloud LPU Inference**:

```
                                  ┌─────────────────────────────┐
                                  │   Target Resume Content     │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │    Grounded System Prompt   │
                                  │  "Never invent credentials; │
                                  │  answer only from resume"   │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │   GroqCloud LPU Inference   │
                                  │  (https://api.groq.com/...) │
                                  └──────────────┬──────────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   ▼                             ▼                             ▼
       ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
       │   Interactive Q&A    │      │ Executive Summary &  │      │  Structured Insights │
       │ Chips & Custom Qs    │      │ Optimization Critique│      │   JSON Extraction    │
       └──────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

* **Official Endpoint:** `https://api.groq.com/openai/v1`
* **Dynamic Model Retrieval:** The application automatically queries active chat models available on your GroqCloud account and configures a working default (`qwen/qwen3.8-27b`).
* **Available Models:** `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `allam-2-7b`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`.
* **Zero Hallucination Policy:** If a requested qualification, degree, or employer is absent from the resume, the assistant explicitly states: *"Not mentioned in the uploaded resume"*.
* **Interactive Question Chips:** One-click queries for technical skills, employment history, and educational credentials.

---

## 11. PDF Report Generation

SmartResume AI features a built-in, in-memory PDF generation engine ([modules/pdf_generator.py](modules/pdf_generator.py)) built on **PyMuPDF**:

| Report Type | Download Button | Exported Filename | Contents |
| :--- | :--- | :--- | :--- |
| **AI Executive Summary** | `📥 Download as PDF` | `Resume_Executive_Summary.pdf` | Complete executive summary, professional overview, skills, education, and career highlights. |
| **Optimization & Critique** | `📥 Download as PDF` | `Resume_Optimization_Critique.pdf` | Comprehensive ATS review, missing information analysis, formatting advice, and keyword recommendations. |
| **Structured Insights** | `📥 Download as PDF` | `Resume_Structured_Insights.pdf` | Human-readable report converting structured JSON into styled summary callouts, skill badges, education lists, and project tables. |

### Technical Characteristics of Generated PDFs:
* **In-Memory Rendering:** Generated via `BytesIO` and delivered through Streamlit's native `st.download_button`. Zero temporary files are written to disk.
* **No Redundant API Calls:** Generates the PDF using the already computed results stored in `st.session_state`.
* **Multi-Page Layout:** Automatic content flow with running header rules, footer confidentiality statements, and exact page numbers (`Page X of Y`).
* **Candidate Metadata Header:** Includes candidate name, source resume filename, report type, and generation timestamp.

---

## 12. Testing and Verification

The project includes an extensive automated test suite covering all core modules, database operations, entity linking, vectorization, topic modeling, Groq integration, and PDF generation.

### Run All Tests
Execute pytest from your activated virtual environment:
```powershell
pytest -v
```

### Current Test Execution Summary (45 Passed)
```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.2.0, pluggy-1.7.0
rootdir: SmartResume-AI
collected 45 items

tests/test_candidate_comparison.py ........ [ 5 passed ]
tests/test_database.py .................... [ 3 passed ]
tests/test_entity_linking.py .............. [ 5 passed ]
tests/test_extraction.py .................. [ 4 passed ]
tests/test_genai_assistant.py ............. [ 9 passed ]
tests/test_information_extraction.py ...... [ 2 passed ]
tests/test_ner.py ......................... [ 2 passed ]
tests/test_pdf_generator.py ............... [ 6 passed ]
tests/test_preprocessing.py ............... [ 4 passed ]
tests/test_sentiment_and_topics.py ........ [ 5 passed ]

============================= 45 passed in 11.66s =============================
```

### Test Coverage Highlights:
* **Candidate Comparison:** Validates empty, overlapping, distinct, and 3+ candidate comparisons.
* **Database & Cascade Purge:** Verifies that deleting an individual resume completely removes its storage file and relational rows while leaving other candidates intact.
* **Entity Linking:** Tests Exact, Alias, Case-Insensitive, and Fuzzy Levenshtein resolution.
* **GenAI Assistant Contract:** Verifies offline mock behavior, model discovery, sanitized error handling (ensuring API keys are never leaked in logs), and grounded prompts.
* **PDF Generator:** Validates valid byte headers (`%PDF-`), multi-page pagination, and structured JSON formatting.

---

## 13. Database and Data Governance

SmartResume AI uses a local **SQLite** database (`data/smartresume.db`) with foreign key constraints and cascade deletions:

```sql
resumes (id, filename, file_type, upload_timestamp, file_size, is_scanned, page_count, original_text, cleaned_text, status)
  ├── candidate_profiles (id, resume_id, candidate_name, email, phone, location, summary, education_json, skills_json, completeness_score)
  ├── extracted_entities (id, resume_id, entity_text, entity_label, start_char, end_char, confidence)
  ├── entity_links (id, resume_id, extracted_term, canonical_name, category, match_method, similarity_score)
  ├── sentiment_scores (id, resume_id, section_name, pos, neu, neg, compound, tone_label)
  ├── resume_topics (id, resume_id, dominant_topic_id, dominant_topic_label, topic_distribution_json)
  └── ai_interactions (id, resume_id, timestamp, question, answer, source_section)
```

---

## 14. Security and Privacy

* **Local Storage First:** All parsed candidate resumes, profiles, and entity records are stored strictly on your local machine in `data/smartresume.db`.
* **Zero PII Telemetry:** Candidate information is never broadcast to external telemetry, trackers, or third parties.
* **GDPR Right-to-Erasure:** Users can permanently delete individual resumes at any time from the **Dashboard**, **Information Extraction**, or **Settings** views. Deleting a record removes the physical file from `data/storage/` and deletes all database rows.
* **Safe Credential Handling:** API keys are stored solely in your local `.env` file or environment variables. Error messages sanitize and strip all sensitive credentials to prevent log leakage.

---

## 15. Future Enhancements

The following extensions are planned as potential future enhancements:
* **Distributed Vector Database Integration:** Optional indexing into vector stores (e.g., ChromaDB, Qdrant, Milvus) for semantic resume semantic search across 10,000+ documents.
* **Automated Technical Interview Question Generator:** Dynamically generating behavioral and technical interview questions tailored to specific candidate projects and skill gaps.
* **Asynchronous Celery/Redis Processing Queue:** Dedicated background workers for high-volume enterprise ingestion batches of 1,000+ resumes simultaneously.
* **Docker & Kubernetes Deployment:** Pre-configured Dockerfile and Helm charts for zero-configuration containerized deployment on cloud platforms.

---

## 16. Deployed Link

* **Project Name:** SmartResume AI – Automated Resume Information Extraction and Analysis
* **Link:** https://smartresume-ai-wf.streamlit.app/


---

*SmartResume AI was engineered to bridge deterministic Natural Language Processing with evidence-grounded generative intelligence for ethical, transparent, and accurate candidate analysis.*
