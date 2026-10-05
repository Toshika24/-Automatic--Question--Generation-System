import streamlit as st
from question_generator import generate_questions

st.set_page_config(
    page_title="Automatic Question Generation",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Automatic Question Generation System")
st.write("Generate questions automatically from your study material using NLP.")

st.divider()

study_material = st.text_area(
    "Enter your study material:",
    height=220,
    placeholder=(
        "Example:\n"
        "Machine Learning is a branch of Artificial Intelligence. "
        "It allows computers to learn from data without being explicitly programmed. "
        "Machine Learning is used for prediction and classification."
    )
)

num_questions = st.slider(
    "Number of questions:",
    min_value=3,
    max_value=10,
    value=5
)

if st.button("✨ Generate Questions", use_container_width=True):
    if not study_material.strip():
        st.warning("Please enter some study material first.")
    else:
        with st.spinner("Generating questions..."):
            questions = generate_questions(study_material, num_questions)

        if questions:
            st.success(f"{len(questions)} questions generated successfully!")

            st.subheader("📝 Generated Questions")

            for i, question in enumerate(questions, start=1):
                st.write(f"**Q{i}. {question}**")
        else:
            st.info(
                "Could not generate enough questions. "
                "Please enter a longer paragraph with complete sentences."
            )

st.divider()
st.caption("Mini Project | NLP | Rule-Based Question Generation")
