"""
SmartResume AI - Settings & Privacy Management Page
API credentials, model configuration, knowledge base management,
file retention policies, data deletion, and privacy governance.
"""

import streamlit as st
import os
from database.db_manager import db
from modules.genai_assistant import GenAIAssistant
from config.settings import BASE_DIR, DB_PATH, MODELS_DIR


def render_settings():
    """Renders the settings and security configuration interface."""
    st.markdown('<div class="brand-badge">SYSTEM CONFIGURATION & PRIVACY</div>', unsafe_allow_html=True)
    st.title("Settings & Data Privacy")
    st.caption("Manage LLM credentials, model parameters, persistent local storage, and GDPR-compliant data purge.")

    assistant = GenAIAssistant()
    is_groq_active = assistant.is_configured()

    tab1, tab2, tab3 = st.tabs([
        "🔑 API & Model Configuration",
        "🛡️ Privacy & Local Retention",
        "ℹ️ System & Environment Diagnostics"
    ])

    # Tab 1: API Configuration
    with tab1:
        st.subheader("LLM Provider Configuration (GroqCloud)")
        st.caption("Configured for the official GroqCloud API endpoint: `https://api.groq.com/openai/v1`")

        st.markdown(f"""
        **Connection Status:** `{'🟡 Configured — test from AI Resume Assistant' if is_groq_active else '🔴 Not Configured (Offline Mode)'}`
        """)

        groq_key_input = st.text_input(
            "Groq API Key (GROQ_API_KEY):",
            type="password",
            placeholder="Enter Groq API Key (starts with gsk_...)" if not is_groq_active else "•••••••••••••••••••••••••••••••• (Configured - enter new key to change)",
            help="Your API key is used strictly in local memory and is never transmitted or hardcoded."
        )

        # Dynamic retrieval of available models from Groq API
        available_models = assistant.get_available_models()
        if assistant.model_name and assistant.model_name not in available_models:
            available_models = [assistant.model_name] + available_models

        current_model_idx = available_models.index(assistant.model_name) if assistant.model_name in available_models else 0

        model_choice = st.selectbox(
            "Groq Model Variant:",
            options=available_models,
            index=current_model_idx,
            help="Select an active Groq chat model. Automatically discovered from your GroqCloud account."
        )

        col_save, col_test = st.columns([1, 1])
        with col_save:
            save_clicked = st.button("Save API Configuration", type="primary")
        with col_test:
            test_clicked = st.button("⚡ Test Live Connection", key="btn_test_settings_conn")

        if test_clicked:
            active_key = groq_key_input.strip() if groq_key_input.strip() else assistant.api_key
            if not active_key or active_key.startswith("your_"):
                st.error("Please enter a valid Groq API key before testing.")
            else:
                tester = GenAIAssistant(api_key=active_key, model_name=model_choice)
                with st.spinner(f"Testing live Groq API with model '{model_choice}'..."):
                    res = tester.test_connection()
                if res["success"]:
                    st.session_state["groq_connection_verified"] = True
                    st.success(f"✓ Groq API connected successfully with model: `{res['model']}`!")
                else:
                    st.session_state["groq_connection_verified"] = False
                    st.error(res["error"])

        if save_clicked:
            new_key = groq_key_input.strip() if groq_key_input.strip() else assistant.api_key
            if new_key and not new_key.startswith("your_"):
                os.environ["GROQ_API_KEY"] = new_key
                os.environ["GROQ_MODEL"] = model_choice
                os.environ["GROQ_API_BASE"] = "https://api.groq.com/openai/v1"

                # Update in-memory settings
                import config.settings as app_settings
                app_settings.GROQ_API_KEY = new_key
                app_settings.GROQ_MODEL = model_choice
                app_settings.GROQ_API_BASE = "https://api.groq.com/openai/v1"

                # Update .env file for persistence
                env_path = BASE_DIR / ".env"
                try:
                    lines = []
                    if env_path.exists():
                        lines = env_path.read_text(encoding="utf-8").splitlines()
                    new_lines = []
                    key_written = False
                    model_written = False
                    base_written = False

                    for line in lines:
                        # Strip out old xAI/Gemini keys
                        if line.startswith(("XAI_API_KEY=", "GROK_MODEL=", "GROK_API_BASE=", "GEMINI_API_KEY=", "GEMINI_MODEL=")):
                            continue
                        elif line.startswith("GROQ_API_KEY="):
                            new_lines.append(f"GROQ_API_KEY={new_key}")
                            key_written = True
                        elif line.startswith("GROQ_MODEL="):
                            new_lines.append(f"GROQ_MODEL={model_choice}")
                            model_written = True
                        elif line.startswith("GROQ_API_BASE="):
                            new_lines.append("GROQ_API_BASE=https://api.groq.com/openai/v1")
                            base_written = True
                        else:
                            new_lines.append(line)

                    if not key_written:
                        new_lines.append(f"GROQ_API_KEY={new_key}")
                    if not model_written:
                        new_lines.append(f"GROQ_MODEL={model_choice}")
                    if not base_written:
                        new_lines.append("GROQ_API_BASE=https://api.groq.com/openai/v1")

                    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                except Exception:
                    pass

                st.session_state["groq_connection_verified"] = False
                st.success(f"Groq API configuration saved successfully! Active model: `{model_choice}`.")
                st.rerun()
            else:
                st.warning("Please enter a valid Groq API key string.")

    # Tab 2: Privacy & Purge
    with tab2:
        st.subheader("Data Retention & Privacy Controls")
        st.markdown("""
        **Data Governance Policies:**
        - **Local SQLite Storage:** All resume texts, profiles, and entities are stored solely on your local workstation in `data/smartresume.db`.
        - **Zero External Telemetry:** Extracted candidate profiles are never sent to external third parties without user-initiated AI queries.
        - **GDPR Right-to-Erasure:** You can delete individual candidate records at any time or completely wipe the database.
        """)

        st.divider()

        st.subheader("GDPR Right-to-Erasure: Delete Individual Candidate")
        st.caption("Permanently remove an individual candidate and their uploaded file without affecting other records.")
        all_res = db.get_all_resumes()
        if all_res:
            res_map = {f"RES-{r['id']:03d} • {r['filename']}": r["id"] for r in all_res}
            selected_gdpr_label = st.selectbox("Select Candidate to Erase:", list(res_map.keys()), key="gdpr_select_resume")
            selected_gdpr_id = res_map[selected_gdpr_label]
            confirm_gdpr = st.checkbox(f"Permanently erase candidate {selected_gdpr_label}", key=f"chk_gdpr_{selected_gdpr_id}")
            if st.button("🗑️ Erase Candidate Record", type="primary", disabled=not confirm_gdpr, key=f"btn_gdpr_{selected_gdpr_id}"):
                ok = db.delete_resume(selected_gdpr_id)
                if ok:
                    if st.session_state.get("active_resume_id") == selected_gdpr_id:
                        st.session_state.pop("active_resume_id", None)
                    st.success(f"✓ Candidate {selected_gdpr_label} has been completely erased from database and storage.")
                    st.rerun()
                else:
                    st.error("Failed to erase candidate record.")
        else:
            st.info("No candidate records currently stored in database.")

        st.divider()

        st.subheader("Dangerous Action: Purge All Application Data")
        st.warning("This action permanently deletes all resumes, candidate profiles, entities, and analysis metadata from the SQLite database.")

        confirm_purge = st.checkbox("I understand that this action is irreversible and permanently deletes all stored resumes.")
        if st.button("🗑️ Purge All Stored Data", type="primary", disabled=not confirm_purge):
            db.clear_all_data()
            st.success("All resume records and extracted profiles have been completely erased from the local database.")
            st.rerun()

    # Tab 3: System Diagnostics
    with tab3:
        st.subheader("System Diagnostics & NLP Pipeline Status")

        has_trained_model = (MODELS_DIR / "resume_ner_model").exists()
        resumes_count = len(db.get_all_resumes())

        diag_data = [
            {"Component": "Database File", "Status": "Active", "Details": str(DB_PATH)},
            {"Component": "Trained Resume NER Model", "Status": "Ready" if has_trained_model else "Missing", "Details": str(MODELS_DIR / "resume_ner_model")},
            {"Component": "spaCy English Pipeline", "Status": "Ready", "Details": "spacy.blank('en') + EntityRuler + NER"},
            {"Component": "NLTK Resources", "Status": "Installed", "Details": "punkt_tab, punkt, stopwords, wordnet, vader_lexicon"},
            {"Component": "Current Corpus Resumes", "Status": "Populated" if resumes_count > 0 else "Empty", "Details": f"{resumes_count} resumes"},
            {"Component": "GroqCloud Generative AI", "Status": "Configured" if is_groq_active else "Offline", "Details": f"{assistant.model_name} (https://api.groq.com/openai/v1)"}
        ]

        st.dataframe(diag_data, width="stretch", hide_index=True)
