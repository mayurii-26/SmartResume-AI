"""
SmartResume AI - Information Extraction Page
Displays structured candidate profiles organized by semantic cards,
completeness gauges, manual correction capability, and JSON export.
"""

import streamlit as st
import json
import pandas as pd
from database.db_manager import db
from modules.information_extraction import InformationExtractor


def render_extraction():
    """Renders the structured Information Extraction profile view."""
    st.markdown('<div class="brand-badge">CORE NLP FEATURE • INFORMATION EXTRACTION</div>', unsafe_allow_html=True)
    st.title("Structured Candidate Profile")
    st.caption("Converts unstructured resume text into a verified, relational candidate profile without fabricating missing details.")

    resumes = db.get_all_resumes()
    if not resumes:
        st.info("Upload a resume to get started.")
        return

    resume_options = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in resumes}
    col_sel, col_del = st.columns([3, 1])
    with col_sel:
        selected_label = st.selectbox("Select Candidate Resume:", list(resume_options.keys()))
        selected_id = resume_options[selected_label]
    with col_del:
        st.write("")
        st.write("")
        btn_del_candidate = st.button("🗑️ Delete Resume", key="btn_del_candidate_top", type="secondary")

    profile = db.get_candidate_profile(selected_id)
    if not profile:
        extractor = InformationExtractor()
        resume = db.get_resume(selected_id)
        profile = extractor.extract_full_profile(resume["original_text"])
        db.save_candidate_profile(selected_id, profile)

    if st.session_state.get(f"show_profile_delete_{selected_id}") or btn_del_candidate:
        st.session_state[f"show_profile_delete_{selected_id}"] = True
        curr_resume = next((r for r in resumes if r["id"] == selected_id), None)
        curr_name = curr_resume["filename"] if curr_resume else f"RES-{selected_id:03d}"
        with st.container(border=True):
            st.error(f"### ⚠️ Confirm Permanent Deletion of RES-{selected_id:03d}")
            st.markdown(
                f"Are you sure you want to delete **{curr_name}** and all data for candidate **{profile.get('candidate_name') or 'Candidate'}**?\n\n"
                "**This action will permanently delete:**\n"
                "- Original uploaded document from storage\n"
                "- Structured profile, education, experience, and projects\n"
                "- All extracted NER entities and skill links\n"
                "- Sentiment metrics and AI conversation history\n\n"
                "*Note: Shared skill knowledge-base data and all other candidates remain completely intact.*"
            )
            ack_delete = st.checkbox(
                f"I understand this is permanent and confirm deletion of {curr_name}",
                key=f"ack_delete_check_{selected_id}"
            )
            col_b1, col_b2 = st.columns([1, 4])
            with col_b1:
                if st.button("Confirm Delete", type="primary", disabled=not ack_delete, key=f"btn_confirm_delete_profile_{selected_id}"):
                    ok = db.delete_resume(selected_id)
                    if ok:
                        if st.session_state.get("active_resume_id") == selected_id:
                            st.session_state.pop("active_resume_id", None)
                        st.session_state.pop(f"show_profile_delete_{selected_id}", None)
                        st.success(f"✓ Resume '{curr_name}' (ID: RES-{selected_id:03d}) was permanently deleted.")
                        st.rerun()
                    else:
                        st.error("Failed to delete candidate record.")
            with col_b2:
                if st.button("Cancel", key=f"btn_cancel_delete_profile_{selected_id}"):
                    st.session_state.pop(f"show_profile_delete_{selected_id}", None)
                    st.rerun()

    # Top Candidate Header Card
    completeness = profile.get("completeness_score", 0.0)
    col_hdr1, col_hdr2 = st.columns([3, 1])

    with col_hdr1:
        st.markdown(f"""
        <div class="saas-card" style="margin-bottom: 12px;">
            <h2 style="margin: 0; font-size: 1.5rem; color: #0F172A;">{profile.get('candidate_name') or 'Candidate Profile'}</h2>
            <p style="color: #64748B; margin: 4px 0 0 0; font-size: 0.9rem;">
                📍 {profile.get('location') or 'Location not specified'} &nbsp;|&nbsp; 
                ✉️ {profile.get('email') or 'No email'} &nbsp;|&nbsp; 
                📞 {profile.get('phone') or 'No phone'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_hdr2:
        st.markdown(f"""
        <div class="kpi-container" style="text-align: center;">
            <span class="kpi-title">Completeness</span>
            <span class="kpi-value" style="color: #2563EB;">{completeness}%</span>
            <span class="kpi-sub">Populated fields</span>
        </div>
        """, unsafe_allow_html=True)

    # Tabs for profile sections
    tabs = st.tabs([
        "👤 Personal Information",
        "🎓 Education",
        "🛠️ Categorized Skills",
        "💼 Experience",
        "🚀 Projects",
        "📜 Certifications",
        "📝 Summary",
        "💾 JSON Export & Manual Edit"
    ])

    # Tab 1: Personal Info
    with tabs[0]:
        st.subheader("Contact & Personal Information")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown(f"**Full Name:** `{profile.get('candidate_name') or 'Not Detected'}`")
            st.markdown(f"**Email Address:** `{profile.get('email') or 'Not Detected'}`")
            st.markdown(f"**Phone Number:** `{profile.get('phone') or 'Not Detected'}`")
        with p_c2:
            st.markdown(f"**Location:** `{profile.get('location') or 'Not Detected'}`")
            st.markdown(f"**LinkedIn URL:** `{profile.get('linkedin_url') or 'Not Detected'}`")
            st.markdown(f"**Portfolio / GitHub:** `{profile.get('portfolio_url') or 'Not Detected'}`")

    # Tab 2: Education
    with tabs[1]:
        st.subheader("Educational Background")
        edu_list = profile.get("education", [])
        if edu_list:
            for i, edu in enumerate(edu_list):
                st.markdown(f"""
                <div class="saas-card" style="padding: 14px 18px; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #1E40AF;">{edu.get('degree', 'Degree')}</h4>
                    <p style="margin: 2px 0 0 0; color: #475569; font-size: 0.9rem;">
                        <strong>Institution:</strong> {edu.get('institution') or 'Not specified'} &nbsp;|&nbsp; 
                        <strong>Field:</strong> {edu.get('field_of_study') or 'General'} &nbsp;|&nbsp; 
                        <strong>Year:</strong> {edu.get('graduation_year') or 'N/A'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No education credentials explicitly detected.")

    # Tab 3: Skills
    with tabs[2]:
        st.subheader("Categorized Technical & Domain Skills")
        skills_dict = profile.get("skills", {})
        if isinstance(skills_dict, dict) and any(skills_dict.values()):
            for cat, s_list in skills_dict.items():
                if s_list:
                    st.markdown(f"**{cat}** ({len(s_list)})")
                    pills = " ".join([
                        f'<span style="background-color: #EFF6FF; color: #1D4ED8; padding: 4px 10px; '
                        f'border-radius: 12px; font-size: 0.8rem; font-weight: 500; margin: 2px; display: inline-block; border: 1px solid #DBEAFE;">'
                        f'{s}</span>'
                        for s in s_list
                    ])
                    st.markdown(pills, unsafe_allow_html=True)
                    st.write("")
        else:
            st.info("No categorized skills detected.")

    # Tab 4: Experience
    with tabs[3]:
        st.subheader("Professional Work Experience")
        exp_list = profile.get("experience", [])
        if exp_list:
            for exp in exp_list:
                st.markdown(f"""
                <div class="saas-card" style="margin-bottom: 12px;">
                    <h4 style="margin: 0; color: #0F172A;">{exp.get('job_title', 'Role')}</h4>
                    <p style="margin: 2px 0 8px 0; color: #2563EB; font-weight: 500; font-size: 0.88rem;">
                        🏢 {exp.get('company', 'Company')} &nbsp;•&nbsp; 
                        📅 {exp.get('start_date', '')} – {exp.get('end_date', 'Present')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if exp.get("responsibilities"):
                    for r in exp["responsibilities"]:
                        st.markdown(f"- {r}")
        else:
            st.info("No work experience blocks detected.")

    # Tab 5: Projects
    with tabs[4]:
        st.subheader("Key Projects")
        proj_list = profile.get("projects", [])
        if proj_list:
            for p in proj_list:
                st.markdown(f"""
                <div class="saas-card" style="margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #0F172A;">{p.get('name', 'Project')}</h4>
                    <p style="margin: 4px 0; color: #475569; font-size: 0.88rem;">{p.get('description', '')}</p>
                </div>
                """, unsafe_allow_html=True)
                if p.get("technologies"):
                    st.markdown(f"**Technologies:** {', '.join(p['technologies'])}")
        else:
            st.info("No project sections detected.")

    # Tab 6: Certifications
    with tabs[5]:
        st.subheader("Licenses & Certifications")
        certs = profile.get("certifications", [])
        if certs:
            for c in certs:
                st.markdown(f"• **{c.get('name')}**")
        else:
            st.info("No certifications detected.")

    # Tab 7: Summary
    with tabs[6]:
        st.subheader("Extracted Summary / Career Objective")
        summary_text = profile.get("summary")
        if summary_text:
            st.info(summary_text)
        else:
            st.info("No professional summary section present in this resume.")

    # Tab 8: JSON Export & Edit
    with tabs[7]:
        st.subheader("Structured Profile JSON & Manual Verification")
        json_str = json.dumps(profile, indent=2)

        st.download_button(
            "⬇️ Download Structured Profile (JSON)",
            data=json_str,
            file_name=f"candidate_profile_RES-{selected_id:03d}.json",
            mime="application/json"
        )

        st.write("")
        st.markdown("**Manual Correction & Override:**")
        edited_name = st.text_input("Candidate Name:", value=profile.get("candidate_name", ""))
        edited_email = st.text_input("Email:", value=profile.get("email", ""))
        edited_phone = st.text_input("Phone:", value=profile.get("phone", ""))
        edited_location = st.text_input("Location:", value=profile.get("location", ""))

        if st.button("💾 Save Manual Corrections"):
            profile["candidate_name"] = edited_name
            profile["email"] = edited_email
            profile["phone"] = edited_phone
            profile["location"] = edited_location
            extractor = InformationExtractor()
            profile["completeness_score"] = extractor.calculate_completeness(profile)
            db.save_candidate_profile(selected_id, profile)
            st.success("Candidate profile updated successfully!")
            st.rerun()

        st.divider()
        st.subheader("⚠️ Danger Zone: Delete Candidate Record")
        st.caption("Permanently purge this candidate profile, original document from storage, and all associated extracted data.")
        del_tab_confirm = st.checkbox(
            f"I confirm permanent deletion of candidate RES-{selected_id:03d}",
            key=f"del_tab_chk_{selected_id}"
        )
        if st.button("🗑️ Permanently Delete Candidate Record", type="primary", disabled=not del_tab_confirm, key=f"del_tab_btn_{selected_id}"):
            ok = db.delete_resume(selected_id)
            if ok:
                if st.session_state.get("active_resume_id") == selected_id:
                    st.session_state.pop("active_resume_id", None)
                st.success(f"✓ Candidate profile RES-{selected_id:03d} successfully erased.")
                st.rerun()
            else:
                st.error("Failed to delete candidate record.")

