"""
SmartResume AI - Information Extraction Module
Converts unstructured resume text into a structured, validated candidate profile:
- Personal Information (Name, Email, Phone, Location, LinkedIn, Portfolio)
- Education (Degree, Field, Institution, Year)
- Categorized Skills (Languages, Frameworks, Databases, Cloud, Tools, Soft Skills)
- Experience (Job Title, Company, Dates, Responsibilities)
- Projects (Name, Description, Technologies)
- Certifications (Name, Issuer, Date)
- Completeness Scoring & Structured JSON Export
"""

import re
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from modules.ner import ResumeNER
from config.settings import SKILL_KB_PATH


class InformationExtractor:
    """Extracts structured candidate profile from resume text using rules, NER, and section detection."""

    def __init__(self, ner_engine: Optional[ResumeNER] = None):
        self.ner = ner_engine or ResumeNER()
        self.skill_kb = self._load_skill_kb()

    def _load_skill_kb(self) -> Dict[str, Any]:
        """Loads canonical skills and aliases with categories."""
        if SKILL_KB_PATH.exists():
            try:
                with open(SKILL_KB_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"skills": []}

    @staticmethod
    def _stitch_bullets(lines: List[str]) -> List[str]:
        """
        Merges continuation lines belonging to the same bullet point or narrative item,
        preserving original wording and complete sentences without fragmenting on line wraps.
        """
        bullet_marker = re.compile(r'^\s*(?:[•\*\-\u2022\u25e6\u25aa\u25ab]|\d+\.|\([0-9a-zA-Z]\))\s*')
        stitched = []
        current = ""

        for line in lines:
            l = line.strip()
            if not l:
                continue
            if bullet_marker.match(l):
                if current:
                    stitched.append(current)
                current = bullet_marker.sub('', l).strip()
            elif current:
                # Continuation line: stitch with hyphen handling
                if current.endswith('-') and not current.endswith(' -'):
                    current = current[:-1] + l
                else:
                    current = current + " " + l
            else:
                current = l

        if current:
            stitched.append(current)

        return stitched

    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Segments resume text into standard semantic sections:
        summary, experience, education, skills, projects, certifications.
        Uses anchored pattern matching with corporate/contact exclusions to prevent
        company names (e.g., 'Sagveek Technologies Pvt. Ltd.') from triggering skills headers.
        """
        section_patterns = {
            "summary": r'^(?:professional\s+summary|executive\s+summary|career\s+objective|career\s+summary|profile|about\s+me|about|summary|professional\s+profile|personal\s+profile|overview|biography|bio)$',
            "experience": r'^(?:work\s+experience|professional\s+experience|employment\s+history|employment|experience|career\s+history|work\s+history|internships?|practical\s+experience|professional\s+background)$',
            "education": r'^(?:education|academic\s+background|educational\s+background|educational\s+qualifications?|academic\s+qualifications?|academics?|degrees?|qualifications?)$',
            "skills": r'^(?:technical\s+skills|skills\s+and\s+competencies|core\s+competencies|skills\s*&?\s*expertise|technologies\s*&?\s*tools|technical\s+expertise|key\s+skills|areas\s+of\s+expertise|technical\s+proficiencies|skills|expertise)$',
            "projects": r'^(?:projects?|project\s+work|key\s+projects?|academic\s+projects?|personal\s+projects?|technical\s+projects?|selected\s+projects?)$',
            "certifications": r'^(?:certifications?|certificates?|licenses?|courses\s*(?:and|&)\s*certifications?|awards?\s*(?:and|&)\s*certificates?|awards?\s*(?:and|&)\s*certifications?|credentials?|honors?\s*(?:and|&)\s*awards?|strengths?\s*(?:and|&)\s*certifications?)$'
        }

        lines = text.split("\n")
        current_section = "header"
        sections = {
            "header": [],
            "summary": [],
            "experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "certifications": []
        }

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            clean_line = re.sub(r'^[#•\*\-\s]+', '', line_str).rstrip(':').strip()
            is_header = False

            # Exclude corporate indicators from ever matching as a section header
            is_org = bool(re.search(
                r'\b(?:pvt|ltd|limited|inc|corp|llc|gmbh|technologies\s+pvt|technologies\s+ltd)\b',
                clean_line,
                re.IGNORECASE
            ))
            # Exclude lines with contact information
            is_contact = bool('@' in clean_line or 'github.com' in clean_line or 'linkedin.com' in clean_line)

            if not is_org and not is_contact and len(clean_line.split()) <= 5:
                for sec, pattern in section_patterns.items():
                    if re.search(pattern, clean_line, re.IGNORECASE):
                        current_section = sec
                        is_header = True
                        break

            if not is_header:
                if current_section in sections:
                    sections[current_section].append(line_str)
                else:
                    sections["header"].append(line_str)

        return {k: "\n".join(v) for k, v in sections.items()}

    def extract_personal_info(self, text: str, header_text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts contact and personal identity details."""
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', text)
        email = email_match.group() if email_match else ""

        # Phone
        phone_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
        phone = phone_match.group() if phone_match else ""

        # LinkedIn
        linkedin_match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+', text, re.IGNORECASE)
        linkedin_url = linkedin_match.group() if linkedin_match else ""

        # Portfolio / GitHub
        portfolio_match = re.search(
            r'(?:https?://)?(?:www\.)?(?:github\.com/[A-Za-z0-9_-]+|[A-Za-z0-9_-]+\.github\.io(?:/[A-Za-z0-9_-]+)?|[A-Za-z0-9_-]+\.dev|[A-Za-z0-9_-]+\.me)',
            text,
            re.IGNORECASE
        )
        portfolio_url = portfolio_match.group() if portfolio_match else ""

        # Location
        location = ""
        loc_search_text = header_text if header_text.strip() else text[:800]
        loc_match = re.search(r'\b([A-Z][a-zA-Z ]{2,30},\s*(?:[A-Z]{2}|[A-Z][a-zA-Z ]{2,20})(?:,\s*[A-Z][a-zA-Z ]{2,20})?)\b', loc_search_text)
        if loc_match:
            candidate_loc = loc_match.group(1).strip()
            non_loc_kws = ('university', 'college', 'school', 'academy', 'skills', 'education', 'experience', 'python', 'javascript', 'sql', 'java', 'react')
            if not any(kw in candidate_loc.lower() for kw in non_loc_kws):
                location = candidate_loc

        if not location:
            for ent in entities:
                if ent["label"] == "LOCATION":
                    loc_c = ent["text"].strip(" ,.-|•\t\n")
                    non_loc_kws = ('university', 'college', 'school', 'academy', 'python', 'javascript', 'sql')
                    if len(loc_c) > 2 and not any(kw in loc_c.lower() for kw in non_loc_kws):
                        location = loc_c
                        break

        # Candidate Name: Look in header first or use top PERSON entity
        candidate_name = ""
        header_lines = [l.strip() for l in header_text.split("\n") if l.strip()]
        for l in header_lines[:3]:
            # Clean possible bullets or symbols
            clean_l = re.sub(r'^[#•\*\-\s]+', '', l).strip()
            # Candidate name is typically title-cased or uppercase, 2-4 words, no digits or contact markers
            if re.match(r'^[A-Za-z][A-Za-z\s\.\'-]{2,40}$', clean_l) and len(clean_l.split()) in [2, 3, 4]:
                if not any(token in clean_l.lower() for token in ('resume', 'curriculum', 'developer', 'engineer', 'analyst', 'designer')):
                    candidate_name = clean_l
                    break

        if not candidate_name:
            for ent in entities:
                if ent["label"] == "PERSON" and len(ent["text"].split()) in [2, 3, 4]:
                    candidate_name = ent["text"].strip(" ,.-|•\t\n")
                    break

        return {
            "candidate_name": candidate_name,
            "email": email,
            "phone": phone,
            "location": location,
            "linkedin_url": linkedin_url,
            "portfolio_url": portfolio_url
        }

    def extract_education(self, edu_text: str, full_text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts structured degree, institution, field of study, and graduation year without arbitrary truncation."""
        records = []
        source = edu_text.strip() if edu_text.strip() else full_text.strip()

        deg_regex = re.compile(
            r'\b(Bachelor(?:\'s)?(?:\s+of\s+[A-Za-z]+)?|B\.?Tech|BTech|B\.?S\.?|B\.?A\.?|B\.?E\.?|B\.?Sc|BCA|BBA|'
            r'Master(?:\'s)?(?:\s+of\s+[A-Za-z]+)?|M\.?Tech|MTech|M\.?S\.?|M\.?A\.?|M\.?E\.?|M\.?Sc|MCA|MBA|'
            r'Doctor(?:\s+of\s+Philosophy)?|Ph\.?D\.?|Associate(?:\'s)?|HSC|SSC|12th|10th|Intermediate|High\s+School)\b',
            re.IGNORECASE
        )

        year_regex = re.compile(r'\b(19\d{2}|20\d{2}|Present|Current)\b', re.IGNORECASE)
        inst_regex = re.compile(r'\b([A-Z][A-Za-z0-9\s&,\.\(\)\'-]+?(?:University|College|Institute|Polytechnic|School|Academy(?:\s+of\s+[A-Za-z]+)?|Campus))\b', re.IGNORECASE)

        lines = source.split("\n")
        current_entry = {}

        for line in lines:
            d_match = deg_regex.search(line)
            y_match = year_regex.search(line)
            inst_match = inst_regex.search(line)

            # A new education record starts only when the current entry already has both institution and degree,
            # or a new institution is found while the current entry already has an institution.
            if (d_match or inst_match) and current_entry.get("degree") and current_entry.get("institution"):
                records.append(current_entry)
                current_entry = {}
            elif inst_match and current_entry.get("institution") and (current_entry.get("degree") or current_entry.get("graduation_year")):
                records.append(current_entry)
                current_entry = {}

            if inst_match and not current_entry.get("institution"):
                current_entry["institution"] = inst_match.group(1).strip()
            if d_match and not current_entry.get("degree"):
                current_entry["degree"] = d_match.group(1).strip()
                f_match = re.search(r'(?:in|—|-)\s+([A-Za-z0-9\s&/\(\)]+?)(?:,|\n|from|at|\.|\d{4}|$)', line, re.IGNORECASE)
                if f_match:
                    cand_field = f_match.group(1).strip(" —-|•,")
                    if cand_field and not any(kw in cand_field.lower() for kw in ('university', 'college', 'school', 'academy')):
                        current_entry["field_of_study"] = cand_field
            if y_match and not current_entry.get("graduation_year"):
                current_entry["graduation_year"] = y_match.group(1)

        if current_entry and (current_entry.get("degree") or current_entry.get("institution")):
            records.append(current_entry)

        # Fallback to DEGREE entities if none found by parser
        if not records:
            for ent in entities:
                if ent["label"] == "DEGREE":
                    records.append({
                        "degree": ent["text"],
                        "field_of_study": "",
                        "institution": "",
                        "graduation_year": ""
                    })

        return records

    def extract_categorized_skills(self, text: str, entities: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[str]]:
        """Extracts and groups skills into categorized taxonomy without dropping valid mentions."""
        categories = {
            "Programming Languages": set(),
            "Web Frameworks & Libraries": set(),
            "AI & Data Science": set(),
            "Databases & Storage": set(),
            "Cloud & DevOps": set(),
            "Developer Tools": set(),
            "Big Data": set(),
            "Methodologies & Soft Skills": set()
        }

        # 1. Match skills against Knowledge Base
        for skill in self.skill_kb.get("skills", []):
            canon = skill["canonical_name"]
            cat = skill.get("category", "Developer Tools")
            if cat not in categories:
                categories[cat] = set()

            all_variants = [canon] + skill.get("aliases", [])
            for variant in all_variants:
                pattern = r'(?<![A-Za-z0-9])' + re.escape(variant) + r'(?![A-Za-z0-9])'
                if re.search(pattern, text, re.IGNORECASE):
                    categories[cat].add(canon)
                    break

        # 2. Add verified SKILL entities if provided
        if entities:
            for ent in entities:
                if ent["label"] == "SKILL":
                    s_text = ent["text"].strip()
                    if s_text and len(s_text) > 1:
                        # Check if already present in any category
                        already_present = any(s_text.lower() == s.lower() for cat_set in categories.values() for s in cat_set)
                        if not already_present:
                            categories.setdefault("Developer Tools", set()).add(s_text)

        return {k: sorted(list(v)) for k, v in categories.items() if v}

    def extract_experience(self, exp_text: str, full_text: str) -> List[Dict[str, Any]]:
        """
        Extracts work experience records.
        CRITICAL: If exp_text is empty, returns [] and DOES NOT fall back to full_text,
        preventing candidate personal info and education from being misparsed as jobs.
        """
        if not exp_text or not exp_text.strip():
            return []

        records = []
        date_range_regex = re.compile(
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{2}/\d{4}|\d{4})\s*(?:-|to|–|—)\s*'
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{2}/\d{4}|\d{4}|Present|Current)',
            re.IGNORECASE
        )
        bullet_marker = re.compile(r'^\s*(?:[•\*\-\u2022\u25e6\u25aa\u25ab]|\d+\.)\s*')
        company_indicator = re.compile(r'\b(?:pvt|ltd|limited|inc|corp|llc|gmbh|technologies|solutions|services|consulting|labs|studio|systems)\b', re.IGNORECASE)
        title_indicator = re.compile(r'\b(?:intern|developer|engineer|manager|lead|architect|analyst|specialist|designer|consultant|associate|programmer|scientist)\b', re.IGNORECASE)

        lines = [l.strip() for l in exp_text.split('\n') if l.strip()]
        if not lines:
            return []

        dates_match = date_range_regex.search(exp_text)
        start_date = dates_match.group(1) if dates_match else ""
        end_date = dates_match.group(2) if dates_match else ""

        header_lines = []
        bullet_lines = []

        for l in lines:
            if bullet_marker.match(l):
                bullet_lines.append(l)
            elif bullet_lines:
                bullet_lines.append(l)
            else:
                cleaned_no_date = date_range_regex.sub('', l).strip(' -|•,')
                if cleaned_no_date:
                    header_lines.append(cleaned_no_date)

        # Determine company and title
        company = ""
        title = ""
        for hl in header_lines:
            if not company and company_indicator.search(hl):
                company = hl
            elif not title and title_indicator.search(hl):
                title = hl
            elif not title:
                title = hl
            elif not company:
                company = hl

        # Stitch multi-line continuation bullets into complete sentences
        stitched_bullets = self._stitch_bullets(bullet_lines)

        record = {
            "job_title": title if title else "Professional Role",
            "company": company if company else "Organization",
            "start_date": start_date,
            "end_date": end_date,
            "responsibilities": stitched_bullets  # Complete sentences preserved without arbitrary truncation
        }
        records.append(record)

        return records

    def extract_projects(self, proj_text: str) -> List[Dict[str, Any]]:
        """
        Extracts projects, complete descriptions, and technology tags.
        Supports both inline bullet projects (e.g., '• Project Name: Description')
        and block-style projects, preserving complete sentences without arbitrary truncation.
        """
        if not proj_text or not proj_text.strip():
            return []

        bullet_marker = re.compile(r'^\s*(?:[•\*\-\u2022\u25e6\u25aa\u25ab]|\d+\.)\s*')
        lines = [l.strip() for l in proj_text.split('\n') if l.strip()]
        projects = []

        # Check if project items start with bullets that contain colons/separators
        bullet_header_count = sum(1 for l in lines if bullet_marker.match(l) and (':' in l or ' – ' in l or ' — ' in l or ' | ' in l))

        if bullet_header_count >= 1 and bullet_header_count >= len([l for l in lines if bullet_marker.match(l)]) * 0.3:
            # Inline bullet format (e.g., Resume 6)
            cur_proj = None
            for l in lines:
                if bullet_marker.match(l) and (':' in l or ' – ' in l or ' — ' in l or ' | ' in l):
                    if cur_proj:
                        projects.append(cur_proj)
                    raw = bullet_marker.sub('', l).strip()
                    if ':' in raw and len(raw.split(':')[0].split()) <= 12:
                        parts = raw.split(':', 1)
                        name = parts[0].strip()
                        desc = parts[1].strip()
                    else:
                        sep = re.search(r'^(.*?)(?:\s+[–—]\s+|\s+\|\s+)(.*)$', raw)
                        if sep and len(sep.group(1).split()) <= 10:
                            name = sep.group(1).strip()
                            desc = sep.group(2).strip()
                        else:
                            name = raw.split('.')[0]
                            desc = raw
                    cur_proj = {'name': name, 'description_parts': [desc]}
                elif cur_proj:
                    clean_l = bullet_marker.sub('', l).strip()
                    if cur_proj['description_parts'] and cur_proj['description_parts'][-1].endswith('-') and not cur_proj['description_parts'][-1].endswith(' -'):
                        cur_proj['description_parts'][-1] = cur_proj['description_parts'][-1][:-1] + clean_l
                    else:
                        cur_proj['description_parts'].append(clean_l)
            if cur_proj:
                projects.append(cur_proj)

            for p in projects:
                p['description'] = ' '.join(p.pop('description_parts', []))
                techs = []
                for s in self.skill_kb.get('skills', []):
                    for v in [s['canonical_name']] + s.get('aliases', []):
                        if re.search(r'\b' + re.escape(v) + r'\b', p['description'], re.IGNORECASE):
                            techs.append(s['canonical_name'])
                            break
                p['technologies'] = list(dict.fromkeys(techs))
        else:
            # Block-style format (e.g., Resume 5)
            cur_proj = None
            for l in lines:
                is_bullet = bool(bullet_marker.match(l))
                is_url = bool(re.search(r'(?:https?://|www\.|\.vercel\.app|\.onrender\.com|\.github\.io|github\.com)', l))
                is_tech_line = bool('•' in l or ' | ' in l)

                # A project title line is non-bullet, non-url, non-tech-separated line <= 8 words
                if not is_bullet and not is_url and not is_tech_line and len(l.split()) <= 8 and not l.endswith(':'):
                    if cur_proj:
                        projects.append(cur_proj)
                    cur_proj = {'name': l.strip(' #*•-'), 'description_parts': []}
                elif cur_proj:
                    clean_l = bullet_marker.sub('', l).strip()
                    if not is_url:
                        if cur_proj['description_parts'] and cur_proj['description_parts'][-1].endswith('-') and not cur_proj['description_parts'][-1].endswith(' -'):
                            cur_proj['description_parts'][-1] = cur_proj['description_parts'][-1][:-1] + clean_l
                        else:
                            cur_proj['description_parts'].append(clean_l)

            if cur_proj:
                projects.append(cur_proj)

            for p in projects:
                p['description'] = ' '.join(p.pop('description_parts', []))
                techs = []
                for s in self.skill_kb.get('skills', []):
                    for v in [s['canonical_name']] + s.get('aliases', []):
                        if re.search(r'\b' + re.escape(v) + r'\b', p['description'], re.IGNORECASE):
                            techs.append(s['canonical_name'])
                            break
                p['technologies'] = list(dict.fromkeys(techs))

        return projects

    def extract_certifications(self, cert_text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts certifications and credentials, stitching wrapped lines without truncation."""
        certs = []
        source_lines = cert_text.split("\n") if cert_text else []
        stitched_lines = self._stitch_bullets(source_lines)

        for line in stitched_lines:
            l = line.strip(" -•*")
            if len(l) > 3 and not l.isupper():
                certs.append({"name": l, "issuer": "", "date": ""})

        # Also add CERTIFICATION entities if found
        for ent in entities:
            if ent["label"] == "CERTIFICATION" and ent["text"] not in [c["name"] for c in certs]:
                certs.append({"name": ent["text"], "issuer": "", "date": ""})

        return certs

    def calculate_completeness(self, profile: Dict[str, Any]) -> float:
        """Calculates information completeness score (0-100%) based on populated core fields."""
        weights = {
            "candidate_name": 15,
            "email": 15,
            "phone": 10,
            "location": 5,
            "summary": 10,
            "education": 15,
            "skills": 15,
            "experience": 15
        }

        total = 0.0
        if profile.get("candidate_name"): total += weights["candidate_name"]
        if profile.get("email"): total += weights["email"]
        if profile.get("phone"): total += weights["phone"]
        if profile.get("location"): total += weights["location"]
        if profile.get("summary"): total += weights["summary"]
        if profile.get("education") and len(profile["education"]) > 0: total += weights["education"]
        if profile.get("skills") and sum(len(v) for v in profile["skills"].values()) > 0: total += weights["skills"]
        if profile.get("experience") and len(profile["experience"]) > 0:
            total += weights["experience"]
        elif profile.get("projects") and len(profile["projects"]) > 0:
            # Credit projects if candidate is a student/graduate without formal employment
            total += weights["experience"]

        return min(100.0, round(total, 1))

    def extract_full_profile(self, text: str) -> Dict[str, Any]:
        """Orchestrates end-to-end structured profile extraction."""
        sections = self.detect_sections(text)
        entities = self.ner.extract_entities(text)

        personal = self.extract_personal_info(text, sections.get("header", ""), entities)
        education = self.extract_education(sections.get("education", ""), text, entities)
        skills = self.extract_categorized_skills(text, entities=entities)
        experience = self.extract_experience(sections.get("experience", ""), text)
        projects = self.extract_projects(sections.get("projects", ""))
        certifications = self.extract_certifications(sections.get("certifications", ""), entities)

        profile = {
            **personal,
            "summary": sections.get("summary", "").strip(),  # Complete summary preserved without arbitrary truncation
            "education": education,
            "skills": skills,
            "experience": experience,
            "projects": projects,
            "certifications": certifications,
            "entities": entities
        }

        profile["completeness_score"] = self.calculate_completeness(profile)
        return profile
