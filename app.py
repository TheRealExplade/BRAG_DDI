import streamlit as st
from pipeline.main_pipeline import run_pipeline
from pipeline.feedback import save_feedback
from pipeline.clinical_formatter import format_for_pharmacist

st.set_page_config(page_title="DDI Clinical Assistant", layout="wide")

st.title("💊 DDI Clinical Assistant")

# 🔹 Inputs
drug1 = st.text_input("Drug A", placeholder="e.g. Warfarin")
drug2 = st.text_input("Drug B", placeholder="e.g. Aspirin")


# 🔹 Analyze Button
if st.button("Analyze"):
    # clear previous state
    st.session_state.pop("error", None)
    st.session_state.pop("result", None)

    if not drug1 or not drug2:
        st.warning("Please enter both drugs.")
    else:
        with st.spinner("Analyzing interaction..."):
            try:
                result = run_pipeline(drug1, drug2)
                st.session_state["result"] = result
            except Exception as e:
                st.session_state["error"] = str(e)


# 🔹 Show errors if any
if "error" in st.session_state:
    st.error(f"Error: {st.session_state['error']}")


# 🔹 Show result (persistent)
if "result" in st.session_state:
    result = st.session_state["result"]

    # Handle pipeline error output
    if isinstance(result, dict) and "error" in result:
        st.error(result.get("reason", "Unknown error"))
    else:
        patient_context = {
            "age": 65,
            "conditions": ["Hypertension"],
            "medications": ["Aspirin"]
        }

        report = format_for_pharmacist(result, patient_context)

        st.markdown("### 📋 Clinical Report")

        # 🔥 better rendering than st.text
        st.code(report, language="markdown")

        # 🔹 Optional debug view
        with st.expander("🔍 View Raw Structured Output"):
            st.json(result)

        st.markdown("---")

        # 🔹 Feedback Section
        st.subheader("🧑‍⚕️ Pharmacist Feedback")

        decision = st.radio(
            "Decision",
            ["Approve", "Approve with changes", "Reject"],
            key="decision"
        )

        notes = st.text_area(
            "Corrections / Notes",
            key="notes"
        )

        confidence = st.selectbox(
            "Confidence in AI Output",
            ["Low", "Medium", "High"],
            key="confidence"
        )

        # 🔹 Submit Feedback
        if st.button("Submit Feedback"):
            if not result:
                st.error("No result to save feedback for.")
            else:
                feedback = {
                    "decision": decision,
                    "notes": notes,
                    "confidence": confidence
                }

                try:
                    save_feedback(result, feedback)
                    st.success("✅ Feedback saved!")
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")