import streamlit as st
from pipeline.main_pipeline import run_pipeline
from pipeline.feedback import save_feedback
from pipeline.clinical_formatter import format_for_pharmacist

st.title("💊 DDI Clinical Assistant")

drug1 = st.text_input("Drug A")
drug2 = st.text_input("Drug B")

# 🔹 Step 1: Button only sets state
if st.button("Analyze"):
    st.session_state["result"] = run_pipeline(drug1, drug2)

# 🔹 Step 2: Always display if result exists
if "result" in st.session_state:
    result = st.session_state["result"]

    patient_context = {
        "age": 65,
        "conditions": ["Hypertension"],
        "medications": ["Aspirin"]
    }

    report = format_for_pharmacist(result, patient_context)
    st.text(report)

    st.markdown("---")
    st.subheader("Pharmacist Feedback")

    decision = st.radio(
        "Decision",
        ["Approve", "Approve with changes", "Reject"]
    )

    notes = st.text_area("Corrections / Notes")

    confidence = st.selectbox(
        "Confidence in AI Output",
        ["Low", "Medium", "High"]
    )

    # 🔹 Step 3: Submit feedback
    if st.button("Submit Feedback"):
        feedback = {
            "decision": decision,
            "notes": notes,
            "confidence": confidence
        }

        save_feedback(result, feedback)
        st.success("Feedback saved!")