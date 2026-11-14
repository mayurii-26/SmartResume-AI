"""
SmartResume AI - Generative AI Resume Assistant Page
Evidence-grounded conversational Q&A, executive summaries, and resume critique
powered by the official GroqCloud API (https://api.groq.com/openai/v1).
"""

import json
import streamlit as st
from database.db_manager import db
from modules.genai_assistant import GenAIAssistant
from modules.pdf_generator import (
    generate_executive_summary_pdf,
    generate_critique_pdf,
    generate_structured_insights_pdf,
)


def render_ai_assistant():
    """Renders the GroqCloud-powered AI Resume Assistant view."""
    st.markdown('<div class="brand-badge">GENERATIVE AI • GROQCLOUD</div>', unsafe_allow_html=True)
    st.title("AI Resume Assistant")
    st.caption("Ask questions, generate grounded executive summaries, and receive actionable resume optimization critiques powered by Groq LPU inference.")

    assistant = GenAIAssistant()
    is_configured = assistant.is_configured()

    if not is_configured:
        st.warning(
            f"🔑 **Groq API Key Required:** {assistant.configuration_message()} "
            "All other NLP modules (Extraction, NER, Linking, Sentiment, Topics, Comparison) remain 100% operational offline without an API key."
        )
    elif st.button("Test Groq connection", icon=":material/wifi:"):
        with st.spinner("Connecting to GroqCloud API (https://api.groq.com/openai/v1)..."):
            status = assistant.test_connection()
        if status["success"]:
            st.session_state["groq_connection_verified"] = True
            st.success(f"✓ Groq connection verified successfully with model: `{status['model']}`.")
        else:
            st.session_state["groq_connection_verified"] = False
            st.error(status["error"])

    resumes = db.get_all_resumes()
    if not resumes:
        st.info("Upload a resume to get started.")
        return

    resume_options = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in resumes}
    selected_label = st.selectbox("Select Target Resume for AI Analysis:", list(resume_options.keys()))
    selected_id = resume_options[selected_label]

    resume = db.get_resume(selected_id)
    raw_text = resume["original_text"]

    candidate_profile = db.get_candidate_profile(selected_id)
    candidate_name = (
        candidate_profile.get("name")
        if (candidate_profile and candidate_profile.get("name"))
        else resume["filename"].rsplit(".", 1)[0]
    )

    tabs = st.tabs([
        "💬 Interactive Resume Q&A",
        "📋 AI Executive Summary",
        "💡 Optimization & Critique Suggestions",
        "🧩 Structured Resume Insights",
    ])

    # Tab 1: Q&A
    with tabs[0]:
        st.subheader("Grounded Resume Question Answering")
        st.caption("All answers are strictly constrained to facts explicitly present in the candidate resume.")

        # Preset Question Chips - Clicking any button directly executes the question
        st.markdown("**Quick Suggested Questions:**")
        col_q1, col_q2, col_q3 = st.columns(3)

        preset_to_run = None
        with col_q1:
            if st.button("🔍 Technical skills?", key=f"btn_chip_skills_{selected_id}", use_container_width=True):
                preset_to_run = "What are the candidate's strongest technical skills mentioned in the resume?"
        with col_q2:
            if st.button("💼 Employment history?", key=f"btn_chip_jobs_{selected_id}", use_container_width=True):
                preset_to_run = "What companies has the candidate worked for, and what were their primary roles?"
        with col_q3:
            if st.button("🎓 Education & degrees?", key=f"btn_chip_edu_{selected_id}", use_container_width=True):
                preset_to_run = "What is the candidate's educational background, degrees, and academic institutions?"

        # If a preset was clicked, execute it immediately
        if preset_to_run:
            if not is_configured:
                st.error("Please configure GROQ_API_KEY in Settings to use the generative assistant.")
            else:
                with st.spinner("SmartResume AI (Groq) is analyzing the resume..."):
                    res = assistant.answer_question(raw_text, preset_to_run)
                    if res["success"]:
                        db.save_ai_interaction(selected_id, preset_to_run, res["answer"])
                        st.session_state[f"last_qa_{selected_id}"] = {
                            "question": preset_to_run,
                            "answer": res["answer"]
                        }
                        st.rerun()
                    else:
                        st.error(res["error"])

        # Manual question input form
        st.write("")
        with st.form(key=f"qa_form_{selected_id}", clear_on_submit=True):
            user_question = st.text_input(
                "Ask any question about this resume:",
                placeholder="e.g., Does the candidate have experience with Kubernetes or microservices?",
                key=f"input_box_{selected_id}"
            )
            submit_btn = st.form_submit_button("Send Question", type="primary")

        if submit_btn and user_question.strip():
            q_text = user_question.strip()
            if not is_configured:
                st.error("Please configure GROQ_API_KEY in Settings to use the generative assistant.")
            else:
                with st.spinner("SmartResume AI (Groq) is analyzing the resume..."):
                    res = assistant.answer_question(raw_text, q_text)
                    if res["success"]:
                        db.save_ai_interaction(selected_id, q_text, res["answer"])
                        st.session_state[f"last_qa_{selected_id}"] = {
                            "question": q_text,
                            "answer": res["answer"]
                        }
                        st.rerun()
                    else:
                        st.error(res["error"])

        # Display Latest Answer Card if available
        last_qa = st.session_state.get(f"last_qa_{selected_id}")
        if last_qa:
            st.markdown(f"""
            <div class="saas-card" style="background: #FFFFFF; border-left: 4px solid #2563EB; margin-top: 15px; margin-bottom: 20px;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #64748B; margin-bottom: 4px;">QUESTION:</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-bottom: 12px;">{last_qa['question']}</div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #2563EB; margin-bottom: 4px;">AI GROUNDED RESPONSE (GROQ):</div>
                <div style="font-size: 0.93rem; line-height: 1.65; color: #1E293B; white-space: pre-wrap;">
{last_qa['answer']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Full Conversation History Thread
        history = db.get_ai_interactions(selected_id)
        # Filter for actual Q&A interactions (skip insights JSON)
        qa_history = [
            h for h in history
            if h.get("question") != "Generate structured resume insights"
        ]

        if qa_history:
            st.markdown(f"#### Conversation History ({len(qa_history)})")
            for h in reversed(qa_history):
                with st.container(border=True):
                    st.markdown(f"**Q:** {h['question']}")
                    st.markdown(f"**A:** {h['answer']}")
                    st.caption(f"Timestamp: {h['timestamp'][:19].replace('T', ' ')}")

    # Tab 2: AI Summary
    with tabs[1]:
        st.subheader("Automated Executive Candidate Summary")
        st.caption("Generates a structured executive overview synthesized strictly from the resume content.")

        cached_summary = st.session_state.get(f"ai_summary_{selected_id}")

        btn_label = "🔄 Regenerate Summary" if cached_summary else "⚡ Generate AI Summary"
        if st.button(btn_label, key=f"btn_gen_summary_{selected_id}", type="primary" if not cached_summary else "secondary"):
            if not is_configured:
                st.error("Please configure GROQ_API_KEY in Settings to generate AI summaries.")
            else:
                with st.spinner("Synthesizing executive summary with Groq..."):
                    res = assistant.generate_summary(raw_text)
                    if res["success"]:
                        st.session_state[f"ai_summary_{selected_id}"] = res["summary"]
                        cached_summary = res["summary"]
                        st.rerun()
                    else:
                        st.error(res["error"])

        if cached_summary:
            st.markdown(f"""
            <div class="saas-card" style="line-height: 1.7; font-size: 0.93rem; margin-top: 15px; margin-bottom: 12px; white-space: pre-wrap;">
{cached_summary}
            </div>
            """, unsafe_allow_html=True)

            try:
                summary_pdf_data = generate_executive_summary_pdf(
                    candidate_name=candidate_name,
                    filename=resume["filename"],
                    summary_text=cached_summary,
                )
                st.download_button(
                    label="📥 Download as PDF",
                    data=summary_pdf_data,
                    file_name="Resume_Executive_Summary.pdf",
                    mime="application/pdf",
                    key=f"dl_summary_pdf_{selected_id}",
                    type="primary",
                )
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")
        else:
            st.info("ℹ️ Generate the executive summary to view and download it as a PDF report.")

    # Tab 3: Critique & Suggestions
    with tabs[2]:
        st.subheader("Resume Improvement Critique & ATS Guidance")
        st.caption("Analyzes clarity, metric quantification, missing standard sections, and ATS keyword relevance.")

        cached_critique = st.session_state.get(f"ai_critique_{selected_id}")

        btn_critique_label = "🔄 Re-analyze Improvements" if cached_critique else "🔍 Analyze for Improvements"
        if st.button(btn_critique_label, key=f"btn_critique_{selected_id}", type="primary" if not cached_critique else "secondary"):
            if not is_configured:
                st.error("Please configure GROQ_API_KEY in Settings to analyze improvements.")
            else:
                with st.spinner("Reviewing resume structure and wording with Groq..."):
                    res = assistant.suggest_improvements(raw_text)
                    if res["success"]:
                        st.session_state[f"ai_critique_{selected_id}"] = res["critique"]
                        cached_critique = res["critique"]
                        st.rerun()
                    else:
                        st.error(res["error"])

        if cached_critique:
            st.markdown(f"""
            <div class="saas-card" style="line-height: 1.7; font-size: 0.93rem; margin-top: 15px; margin-bottom: 12px; white-space: pre-wrap;">
{cached_critique}
            </div>
            """, unsafe_allow_html=True)

            try:
                critique_pdf_data = generate_critique_pdf(
                    candidate_name=candidate_name,
                    filename=resume["filename"],
                    critique_text=cached_critique,
                )
                st.download_button(
                    label="📥 Download as PDF",
                    data=critique_pdf_data,
                    file_name="Resume_Optimization_Critique.pdf",
                    mime="application/pdf",
                    key=f"dl_critique_pdf_{selected_id}",
                    type="primary",
                )
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")
        else:
            st.info("ℹ️ Run the improvement analysis to view recommendations and download them as a PDF report.")

    # Tab 4: Structured Insights
    with tabs[3]:
        st.subheader("Structured Resume Insights")
        st.caption("JSON grounded only in the selected uploaded resume. Missing information is marked explicitly.")

        cached_insights = st.session_state.get(f"ai_insights_{selected_id}")

        btn_insights_label = "🔄 Re-extract Structured Insights" if cached_insights else "Generate Structured Insights"
        if st.button(btn_insights_label, key=f"btn_structured_{selected_id}", type="primary" if not cached_insights else "secondary"):
            if not is_configured:
                st.error(assistant.configuration_message())
            else:
                with st.spinner("Creating grounded structured insights with Groq..."):
                    result = assistant.generate_structured_insights(raw_text)
                if result["success"]:
                    st.session_state[f"ai_insights_{selected_id}"] = result["data"]
                    cached_insights = result["data"]
                    db.save_ai_interaction(selected_id, "Generate structured resume insights", json.dumps(result["data"]))
                    st.rerun()
                else:
                    st.error(result["error"])

        if cached_insights:
            st.json(cached_insights, expanded=2)
            st.write("")
            try:
                insights_pdf_data = generate_structured_insights_pdf(
                    candidate_name=candidate_name,
                    filename=resume["filename"],
                    insights_data=cached_insights,
                )
                st.download_button(
                    label="📥 Download as PDF",
                    data=insights_pdf_data,
                    file_name="Resume_Structured_Insights.pdf",
                    mime="application/pdf",
                    key=f"dl_insights_pdf_{selected_id}",
                    type="primary",
                )
            except Exception as e:
                st.error(f"Could not generate PDF: {e}")
        else:
            st.info("ℹ️ Generate structured insights to view the JSON output and download it as a formatted PDF report.")
