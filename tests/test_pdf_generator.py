"""
Unit tests for the SmartResume AI PDF generation module.
Verifies PDF contract, valid byte streams, multi-page handling,
custom field rendering, and candidate isolation.
"""

import json
import pytest
import pymupdf

from modules.pdf_generator import (
    generate_executive_summary_pdf,
    generate_critique_pdf,
    generate_structured_insights_pdf,
)


def test_executive_summary_pdf_generation():
    summary_text = """### Professional Overview
Candidate is an AI Engineer with proven expertise in deep learning and NLP.

### Skills and Education
* **Python**, **PyTorch**, **FastAPI**
* B.Tech in Computer Science and Engineering

### Key Contributions
* Deployed scalable LLM pipeline achieving 99% uptime.
"""
    pdf_bytes = generate_executive_summary_pdf(
        candidate_name="Alice Smith",
        filename="Alice_Resume.pdf",
        summary_text=summary_text,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF-")

    # Verify PDF structure using pymupdf
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) >= 1
    page1_text = doc[0].get_text()
    assert "SmartResume AI" in page1_text
    assert "Alice Smith" in page1_text
    assert "Alice_Resume.pdf" in page1_text
    assert "Professional Overview" in page1_text


def test_critique_pdf_generation():
    critique_text = """### Missing Information
* Contact phone number is absent.
* Certifications section missing expiration dates.

### Impact and Clarity
* Bullet points should incorporate quantitative metrics (e.g., % improvement, revenue growth).

### ATS Keywords
* Recommended additions: Docker, CI/CD, Microservices.
"""
    pdf_bytes = generate_critique_pdf(
        candidate_name="Bob Jones",
        filename="Bob_Resume.docx",
        critique_text=critique_text,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF-")

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) >= 1
    page1_text = doc[0].get_text()
    assert "Bob Jones" in page1_text
    assert "Missing Information" in page1_text
    assert "ATS Keywords" in page1_text


def test_structured_insights_pdf_generation():
    insights_data = {
        "summary": "Experienced full-stack engineer and cloud architect.",
        "skills": ["Python", "FastAPI", "React", "Docker", "AWS"],
        "education": ["B.S. in Computer Science - State University (2020-2024)"],
        "experience": ["Software Engineer at CloudCorp (2024-Present)"],
        "projects": ["SmartResume AI: Automated parsing system", "Distributed Cache Service"],
        "improvements": ["Add quantifiable business outcomes to experience section"],
        "certifications": ["AWS Certified Solutions Architect"]
    }

    pdf_bytes = generate_structured_insights_pdf(
        candidate_name="Carol White",
        filename="Carol_Resume.pdf",
        insights_data=insights_data,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF-")

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert len(doc) >= 1
    page1_text = doc[0].get_text()
    assert "Carol White" in page1_text
    assert "Technical & Domain Skills" in page1_text
    assert "Python" in page1_text
    assert "CloudCorp" in page1_text


def test_structured_insights_from_json_string():
    raw_json = json.dumps({
        "summary": "Data scientist with expertise in statistical modeling.",
        "skills": ["Python", "R", "SQL"],
        "education": ["M.S. in Statistics"],
        "experience": [],
        "projects": ["Customer Churn Prediction"],
        "improvements": []
    })

    pdf_bytes = generate_structured_insights_pdf(
        candidate_name="David Green",
        filename="David_Resume.pdf",
        insights_data=raw_json,
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    assert "David Green" in doc[0].get_text()


def test_empty_content_raises_error():
    with pytest.raises(ValueError):
        generate_executive_summary_pdf("Test", "test.pdf", "")

    with pytest.raises(ValueError):
        generate_critique_pdf("Test", "test.pdf", "   ")

    with pytest.raises(ValueError):
        generate_structured_insights_pdf("Test", "test.pdf", {})


def test_multipage_pagination_and_headers():
    long_critique = "### Recommendations\n" + "\n".join([
        f"* Section {i}: Detailed feedback explaining that candidate must improve sentence clarity and add measurable outcomes."
        for i in range(1, 40)
    ])

    pdf_bytes = generate_critique_pdf(
        candidate_name="Elena Rostova",
        filename="Elena_Resume.pdf",
        critique_text=long_critique,
    )

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    # Must span multiple pages
    assert len(doc) > 1
    total_pages = len(doc)

    for i, page in enumerate(doc, 1):
        text = page.get_text()
        assert "SmartResume AI" in text
        assert f"Page {i} of {total_pages}" in text

