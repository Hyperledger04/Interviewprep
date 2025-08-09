# app.py
import os
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
import requests
import re

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("❌ No API key found in .env. Please add OPENROUTER_API_KEY.")
    st.stop()

# API details
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_ID = "deepseek/deepseek-r1-0528:free"

# Function to read PDF text
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to call DeepSeek R1
def ask_deepseek(prompt, temperature=0.7):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"❌ API Error: {response.text}"
    data = response.json()
    return data["choices"][0]["message"]["content"]

# Function to clean and format extracted skills
def extract_skills(text):
    skills = re.findall(r'\b[A-Za-z][A-Za-z\s\+\#0-9\.-]{1,}\b', text)
    skills = list(set([s.strip() for s in skills if len(s.strip()) > 1]))
    return skills

# Streamlit UI
st.set_page_config(page_title="AI Interview Coach", page_icon="🤖", layout="wide")
st.title("🤖 AI Interview Preparation & Mock Interview Tool")
st.markdown("Upload your CV & JD, get tailored interview questions, notes, and practice live.")

# Step 1 – Upload files
col1, col2 = st.columns(2)
with col1:
    cv_file = st.file_uploader("📄 Upload your CV (PDF)", type=["pdf"])
with col2:
    jd_file = st.file_uploader("📝 Upload Job Description (PDF)", type=["pdf"])

if cv_file and jd_file:
    cv_text = extract_text_from_pdf(cv_file)
    jd_text = extract_text_from_pdf(jd_file)

    # Step 2 – Skill extraction
    cv_skills = extract_skills(cv_text)
    jd_skills = extract_skills(jd_text)

    matched_skills = list(set(cv_skills) & set(jd_skills))
    missing_skills = list(set(jd_skills) - set(cv_skills))

    st.subheader("📊 Skills Analysis")
    st.markdown(f"**Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
    st.markdown(f"**Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")

    # Step 3 – AI research & question generation
    with st.spinner("🤔 Analyzing and generating interview questions..."):
        research_prompt = f"""
        You are an expert career coach.
        Based on the following CV and Job Description, generate:
        1. A brief analysis of the candidate's fit.
        2. A list of missing technical and soft skills with explanations.
        3. 10 probable interview questions for this role.
        4. Short preparation notes for each question.
        
        CV:
        {cv_text}

        JD:
        {jd_text}
        """
        analysis_output = ask_deepseek(research_prompt, temperature=0.7)

    st.subheader("📄 AI Analysis & Questions")
    st.write(analysis_output)

    # Step 4 – Mock Interview
    st.subheader("🎤 Mock Interview Mode")
    if "mock_qs" not in st.session_state:
        st.session_state.mock_qs = []
        st.session_state.current_q = 0
        st.session_state.mock_started = False

    if st.button("Start Mock Interview"):
        mock_prompt = f"""
        Based on the following CV and Job Description, generate 5 challenging, role-specific interview questions.
        CV: {cv_text}
        JD: {jd_text}
        """
        st.session_state.mock_qs = ask_deepseek(mock_prompt, temperature=0.8).split("\n")
        st.session_state.current_q = 0
        st.session_state.mock_started = True

    if st.session_state.mock_started and st.session_state.mock_qs:
        if st.session_state.current_q < len(st.session_state.mock_qs):
            st.markdown(f"**Q{st.session_state.current_q+1}:** {st.session_state.mock_qs[st.session_state.current_q]}")
            answer = st.text_area("Your Answer", key=f"answer_{st.session_state.current_q}")
            if st.button("Get Feedback", key=f"feedback_{st.session_state.current_q}"):
                feedback_prompt = f"""
                Question: {st.session_state.mock_qs[st.session_state.current_q]}
                Candidate's Answer: {answer}
                Provide constructive, detailed feedback on how to improve the answer.
                """
                feedback = ask_deepseek(feedback_prompt, temperature=0.7)
                st.markdown(f"**Feedback:** {feedback}")
                st.session_state.current_q += 1
        else:
            st.success("✅ Mock interview completed!")

else:
    st.info("Please upload both CV and JD to begin.")
