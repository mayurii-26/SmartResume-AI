"""
SmartResume AI - Professional PDF Report Generation Module
Converts AI Executive Summaries, Resume Critiques, and Structured Resume Insights
into beautifully formatted, multi-page, publication-grade PDF documents using PyMuPDF.
All generation is performed entirely in-memory using BytesIO.
"""

import datetime
import html
import io
import json
import logging
from typing import Any, Dict, List, Optional, Union
import markdown_it
import pymupdf

logger = logging.getLogger(__name__)

# Initialize Markdown-to-HTML parser
_md_parser = markdown_it.MarkdownIt()

# Professional typographic stylesheet optimized for academic & enterprise reporting
PDF_BASE_CSS = """
body {
    font-family: sans-serif;
    font-size: 9.8pt;
    color: #1e293b;
    line-height: 1.55;
    margin: 0;
    padding: 0;
}
h1 {
    font-size: 16pt;
    color: #1e40af;
    border-bottom: 2px solid #2563eb;
    padding-bottom: 4px;
    margin-top: 0;
    margin-bottom: 10px;
}
h2 {
    font-size: 12pt;
    color: #0f172a;
    margin-top: 14px;
    margin-bottom: 6px;
    border-left: 3.5px solid #2563eb;
    padding-left: 7px;
}
h3 {
    font-size: 10.5pt;
    color: #1e40af;
    margin-top: 10px;
    margin-bottom: 4px;
}
h4 {
    font-size: 9.8pt;
    color: #334155;
    margin-top: 8px;
    margin-bottom: 3px;
}
p {
    margin-top: 4px;
    margin-bottom: 7px;
}
ul, ol {
    margin-top: 4px;
    margin-bottom: 8px;
    padding-left: 18px;
}
li {
    margin-bottom: 3.5px;
    line-height: 1.5;
}
strong, b {
    color: #0f172a;
    font-weight: bold;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    margin-bottom: 12px;
}
th, td {
    border: 1px solid #cbd5e1;
    padding: 6px 8px;
    text-align: left;
    font-size: 9pt;
    vertical-align: top;
}
th {
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: bold;
}
.meta-box {
    width: 100%;
    margin-bottom: 12px;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
}
.meta-box td {
    border: none;
    padding: 4px 8px;
    font-size: 8.5pt;
    color: #475569;
}
.badge-pill {
    background-color: #e0e7ff;
    color: #3730a3;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 8.5pt;
    font-weight: bold;
    display: inline-block;
    margin-right: 4px;
    margin-bottom: 4px;
}
.quote-box {
    background: #f8fafc;
    border-left: 3px solid #2563eb;
    padding: 8px 12px;
    font-style: italic;
    margin-top: 4px;
    margin-bottom: 10px;
    color: #334155;
}
.empty-notice {
    color: #64748B;
    font-style: italic;
}
"""


def _build_pdf_document(
    report_title: str,
    candidate_name: str,
    filename: str,
    report_type: str,
    body_html: str,
) -> bytes:
    """Builds a complete, styled multi-page PDF document in memory.
    
    Includes professional running headers, divider lines, and footer page numbering (Page X of Y).
    """
    date_str = datetime.datetime.now().strftime("%B %d, %Y")

    display_name = candidate_name.strip() if candidate_name and candidate_name.strip() else "Candidate Profile"
    display_file = filename.strip() if filename and filename.strip() else "Uploaded Resume"

    full_html = f"""
    <style>{PDF_BASE_CSS}</style>
    <div>
        <h1>SmartResume AI &bull; {html.escape(report_title)}</h1>
        <table class="meta-box">
            <tr>
                <td><b>Candidate:</b> {html.escape(display_name)}</td>
                <td><b>Resume File:</b> {html.escape(display_file)}</td>
            </tr>
            <tr>
                <td><b>Report Type:</b> {html.escape(report_type)}</td>
                <td><b>Date Generated:</b> {date_str}</td>
            </tr>
        </table>
        {body_html}
    </div>
    """

    mediabox = pymupdf.paper_rect("letter")
    margin_x = 45
    margin_top = 48
    margin_bottom = 48
    content_rect = pymupdf.Rect(
        margin_x,
        margin_top,
        mediabox.width - margin_x,
        mediabox.height - margin_bottom,
    )

    def rectfn(rect_num, filled):
        return mediabox, content_rect, None

    story = pymupdf.Story(html=full_html)
    doc = story.write_with_links(rectfn)

    total_pages = len(doc)
    for idx, page in enumerate(doc, 1):
        # Running top header
        page.insert_text(
            pymupdf.Point(margin_x, 30),
            f"SmartResume AI  •  {report_title}",
            fontsize=8,
            color=(0.4, 0.45, 0.55),
        )
        page.draw_line(
            pymupdf.Point(margin_x, 36),
            pymupdf.Point(mediabox.width - margin_x, 36),
            color=(0.85, 0.88, 0.92),
            width=0.75,
        )

        # Running bottom footer
        page.draw_line(
            pymupdf.Point(margin_x, mediabox.height - 32),
            pymupdf.Point(mediabox.width - margin_x, mediabox.height - 32),
            color=(0.85, 0.88, 0.92),
            width=0.75,
        )
        page.insert_text(
            pymupdf.Point(margin_x, mediabox.height - 20),
            "Confidential  •  Generated by SmartResume AI (Evidence-Grounded Inference)",
            fontsize=7.5,
            color=(0.45, 0.5, 0.6),
        )
        page_num_str = f"Page {idx} of {total_pages}"
        text_w = pymupdf.get_text_length(page_num_str, fontsize=8)
        page.insert_text(
            pymupdf.Point(mediabox.width - margin_x - text_w, mediabox.height - 20),
            page_num_str,
            fontsize=8,
            color=(0.45, 0.5, 0.6),
        )

    return doc.tobytes()


def generate_executive_summary_pdf(
    candidate_name: str,
    filename: str,
    summary_text: str,
) -> bytes:
    """Generates a formatted PDF of the AI Executive Summary.
    
    Converts markdown sections, bullet points, and paragraphs into styled HTML.
    """
    if not summary_text or not summary_text.strip():
        raise ValueError("Summary content is empty.")

    body_html = _md_parser.render(summary_text.strip())
    return _build_pdf_document(
        report_title="AI Executive Candidate Summary",
        candidate_name=candidate_name,
        filename=filename,
        report_type="AI Executive Summary",
        body_html=body_html,
    )


def generate_critique_pdf(
    candidate_name: str,
    filename: str,
    critique_text: str,
) -> bytes:
    """Generates a formatted PDF of the Resume Improvement Critique & ATS Guidance.
    
    Preserves all suggestions, missing information, formatting recommendations, and keywords.
    """
    if not critique_text or not critique_text.strip():
        raise ValueError("Critique content is empty.")

    body_html = _md_parser.render(critique_text.strip())
    return _build_pdf_document(
        report_title="Resume Optimization & ATS Critique",
        candidate_name=candidate_name,
        filename=filename,
        report_type="Optimization & ATS Critique Report",
        body_html=body_html,
    )


def _format_insights_to_html(insights: Dict[str, Any]) -> str:
    """Transforms a structured insights dictionary into semantic, human-readable HTML."""
    sections = []

    # 1. Professional Summary
    summary = insights.get("summary")
    if summary and str(summary).strip() and str(summary).strip() != "Not mentioned in the uploaded resume":
        sections.append(f"""
        <h2>Professional Summary</h2>
        <div class="quote-box">
            {html.escape(str(summary).strip())}
        </div>
        """)

    # 2. Skills
    skills = insights.get("skills")
    if skills:
        sections.append("<h2>Technical & Domain Skills</h2>")
        if isinstance(skills, list):
            valid_skills = [s for s in skills if s and str(s).strip()]
            if valid_skills:
                badges = "".join([f'<span class="badge-pill">{html.escape(str(s).strip())}</span>' for s in valid_skills])
                sections.append(f'<div style="margin-bottom: 8px;">{badges}</div>')
            else:
                sections.append('<p class="empty-notice">Not mentioned in the uploaded resume</p>')
        elif isinstance(skills, dict):
            for cat, items in skills.items():
                cat_title = html.escape(str(cat).replace("_", " ").title())
                if isinstance(items, list):
                    badges = "".join([f'<span class="badge-pill">{html.escape(str(s).strip())}</span>' for s in items if s])
                    sections.append(f'<p><b>{cat_title}:</b><br/>{badges}</p>')
                else:
                    sections.append(f'<p><b>{cat_title}:</b> {html.escape(str(items))}</p>')
        else:
            sections.append(f'<p>{html.escape(str(skills))}</p>')

    # 3. Education
    education = insights.get("education")
    if education is not None:
        sections.append("<h2>Academic Background & Degrees</h2>")
        if isinstance(education, list):
            valid_edu = [e for e in education if e and str(e).strip()]
            if valid_edu:
                items = "".join([f'<li>{html.escape(str(e).strip())}</li>' for e in valid_edu])
                sections.append(f'<ul>{items}</ul>')
            else:
                sections.append('<p class="empty-notice">Not mentioned in the uploaded resume</p>')
        else:
            sections.append(f'<p>{html.escape(str(education))}</p>')

    # 4. Experience
    experience = insights.get("experience")
    if experience is not None:
        sections.append("<h2>Professional Work Experience</h2>")
        if isinstance(experience, list):
            valid_exp = [e for e in experience if e and str(e).strip()]
            if valid_exp:
                items = "".join([f'<li>{html.escape(str(e).strip())}</li>' for e in valid_exp])
                sections.append(f'<ul>{items}</ul>')
            else:
                sections.append('<p class="empty-notice">Not mentioned in the uploaded resume</p>')
        else:
            sections.append(f'<p>{html.escape(str(experience))}</p>')

    # 5. Projects
    projects = insights.get("projects")
    if projects is not None:
        sections.append("<h2>Key Projects & Initiatives</h2>")
        if isinstance(projects, list):
            valid_proj = [p for p in projects if p and str(p).strip()]
            if valid_proj:
                items = "".join([f'<li>{html.escape(str(p).strip())}</li>' for p in valid_proj])
                sections.append(f'<ul>{items}</ul>')
            else:
                sections.append('<p class="empty-notice">Not mentioned in the uploaded resume</p>')
        else:
            sections.append(f'<p>{html.escape(str(projects))}</p>')

    # 6. Certifications (if present)
    certifications = insights.get("certifications") or insights.get("certificates")
    if certifications is not None:
        sections.append("<h2>Certifications & Licenses</h2>")
        if isinstance(certifications, list):
            valid_certs = [c for c in certifications if c and str(c).strip()]
            if valid_certs:
                items = "".join([f'<li>{html.escape(str(c).strip())}</li>' for c in valid_certs])
                sections.append(f'<ul>{items}</ul>')
            else:
                sections.append('<p class="empty-notice">Not mentioned in the uploaded resume</p>')
        else:
            sections.append(f'<p>{html.escape(str(certifications))}</p>')

    # 7. Recommendations / Improvements
    improvements = insights.get("improvements") or insights.get("suggestions")
    if improvements is not None:
        sections.append("<h2>Actionable Resume Improvements</h2>")
        if isinstance(improvements, list):
            valid_imp = [i for i in improvements if i and str(i).strip()]
            if valid_imp:
                items = "".join([f'<li>{html.escape(str(i).strip())}</li>' for i in valid_imp])
                sections.append(f'<ul>{items}</ul>')
            else:
                sections.append('<p class="empty-notice">None noted</p>')
        else:
            sections.append(f'<p>{html.escape(str(improvements))}</p>')

    # 8. Any additional custom fields returned by Groq
    standard_keys = {
        "summary", "skills", "education", "experience", "projects",
        "improvements", "suggestions", "certifications", "certificates"
    }
    for key, val in insights.items():
        if key.lower() not in standard_keys and val:
            title = html.escape(str(key).replace("_", " ").title())
            sections.append(f"<h2>{title}</h2>")
            if isinstance(val, list):
                valid_items = [v for v in val if v and str(v).strip()]
                if valid_items:
                    items = "".join([f'<li>{html.escape(str(v).strip())}</li>' for v in valid_items])
                    sections.append(f'<ul>{items}</ul>')
                else:
                    sections.append('<p class="empty-notice">Not mentioned in the uploaded resume</p>')
            elif isinstance(val, dict):
                rows = "".join([
                    f"<tr><td><b>{html.escape(str(k).replace('_', ' ').title())}</b></td><td>{html.escape(str(v))}</td></tr>"
                    for k, v in val.items()
                ])
                sections.append(f'<table>{rows}</table>')
            else:
                sections.append(f'<p>{html.escape(str(val))}</p>')

    return "\n".join(sections)


def generate_structured_insights_pdf(
    candidate_name: str,
    filename: str,
    insights_data: Union[Dict[str, Any], str],
) -> bytes:
    """Generates a structured, human-readable PDF report from the JSON insights data.
    
    Converts JSON structures into formatted sections, badge lists, bullet points, and tables.
    """
    if isinstance(insights_data, str):
        try:
            insights_dict = json.loads(insights_data)
        except Exception as e:
            logger.warning(f"Failed to parse insights JSON string: {e}")
            insights_dict = {"summary": insights_data}
    elif isinstance(insights_data, dict):
        insights_dict = insights_data
    else:
        raise ValueError("Invalid insights data format.")

    if not insights_dict:
        raise ValueError("Insights data is empty.")

    body_html = _format_insights_to_html(insights_dict)
    return _build_pdf_document(
        report_title="Structured Resume Insights Report",
        candidate_name=candidate_name,
        filename=filename,
        report_type="Structured Resume Insights",
        body_html=body_html,
    )

