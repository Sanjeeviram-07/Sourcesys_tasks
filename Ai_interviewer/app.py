import streamlit as st
from llm import get_llm_response
import prompts
from utils import parse_questions, extract_score

st.set_page_config(page_title="AI Interview Simulator", layout="centered")

st.title("AI Interview Simulator")

# Inputs
role = st.text_input("Enter Job Role (e.g., Cloud Engineer)")
level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])

# Initialize session state
if "questions" not in st.session_state:
    st.session_state.questions = []
    st.session_state.current_q = 0
    st.session_state.scores = []
    st.session_state.started = False

# 🚀 Start Interview
if st.button("Start Interview"):

    if not role.strip():
        st.warning("⚠️ Please enter a job role")
    else:
        with st.spinner("Generating questions..."):
            prompt_q = prompts.generate_questions(role, level)
            questions_text = get_llm_response(prompt_q)

        questions = parse_questions(questions_text)

        # 🔥 Safety check
        if len(questions) < 3:
            st.error("❌ Failed to generate proper questions. Try again.")
        else:
            st.session_state.questions = questions
            st.session_state.current_q = 0
            st.session_state.scores = []
            st.session_state.started = True

# 🎯 Interview Flow
if st.session_state.started and len(st.session_state.questions) > 0:

    q_index = st.session_state.current_q
    total_q = len(st.session_state.questions)

    if q_index < total_q:

        question = st.session_state.questions[q_index]

        st.subheader(f"Question {q_index + 1} / {total_q}")
        st.write(question)

        answer = st.text_area("Your Answer", key=f"ans_{q_index}")

        if st.button("Submit Answer", key=f"btn_{q_index}"):

            if not answer.strip():
                st.warning("⚠️ Please enter your answer")
            else:
                with st.spinner("Evaluating answer..."):

                    # Step 1: Evaluate
                    eval_prompt = prompts.evaluate_answer(role, question, answer)
                    feedback = get_llm_response(eval_prompt)

                    score = extract_score(feedback)
                    st.session_state.scores.append(score)

                    # Step 2: Improve
                    improve_prompt = prompts.improve_answer(role, answer, feedback)
                    improved = get_llm_response(improve_prompt)

                st.markdown("### 📊 Feedback")
                st.write(feedback)

                st.markdown("### 🚀 Improved Answer")
                st.write(improved)

                # Next question
                st.session_state.current_q += 1

                st.info("👉 Click 'Submit Answer' again for next question")

    else:
        st.success("🎉 Interview Completed!")

        # Final Report
        with st.spinner("Generating final report..."):
            report_prompt = prompts.final_report(role, st.session_state.scores)
            report = get_llm_response(report_prompt)

        st.markdown("## 📄 Final Report")
        st.write(report)

        # Reset option
        if st.button("🔄 Restart Interview"):
            st.session_state.questions = []
            st.session_state.current_q = 0
            st.session_state.scores = []
            st.session_state.started = False