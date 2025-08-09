# app.py
"""
InterviewPrep — CV-first, JD-optional, line-by-line deep notes + QnA + mock interview (text mode).
Model: deepseek/deepseek-r1-0528:free via OpenRouter
Key: st.secrets["OPENROUTER_API_KEY"] or env OPENROUTER_API_KEY
"""

import os
import json
import time
import textwrap
import requests
from io import BytesIO
from typing import List, Dict, Any

import streamlit as st

# PDF parsing: prefer PyMuPDF, fallback to PyPDF2
try:
    import fitz  # pymupdf
    _HAVE_FITZ = True
except Exception:
    _HAVE_FITZ = False
    try:
        import PyPDF2
    except Exception:
        PyPDF2 = None

# duckduckgo search for quick grounding
try:
    from duckduckgo_search import ddg
except Exception:
    ddg = None

# ---------------------------
# Configuration & secrets
# ---------------------------
st.set_page_config(page_title="InterviewPrep — CV Deep-Dive", layout="wide",
                   initial_sidebar_state="expanded")

# CSS + Fonts (Google font + background)
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Poppins:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{
  --accent1: #6EE7B7;
  --accent2: #60A5FA;
  --card-bg: rgba(255,255,255,0.08);
  --glass: rgba(255,255,255,0.06);
}
html, body, [class*="css"]  {
  font-family: "Inter", "Poppins", sans-serif;
}
.big-hero {
  background: linear-gradient(135deg, rgba(96,165,250,0.12), rgba(110,231,183,0.08));
  padding: 18px;
  border-radius: 14px;
  box-shadow: 0 6px 30px rgba(2,6,23,0.18);
  margin-bottom: 18px;
}
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 12px;
  padding: 14px;
  margin-bottom: 12px;
  border: 1px solid rgba(255,255,255,0.03);
}
.kv {
  color: #e6eef8;
  font-weight:600;
}
.small-muted { color: #cfd8e3; font-size:12px }
.q-card { background: linear-gradient(90deg, rgba(96,165,250,0.06), rgba(110,231,183,0.03)); padding:12px; border-radius:10px; margin-bottom:10px; }
footer { color: #9aa7bb; font-size:12px; margin-top:10px; }
a { color: var(--accent2) }
</style>
""", unsafe_allow_html=True)

# Header area
st.markdown('<div class="big-hero"><h1 style="margin:4px 0">InterviewPrep — CV Deep Dive</h1>'
            '<div class="small-muted">CV-first, JD optional. Line-by-line Q&A, web-grounded notes, and text mock interview.</div></div>',
            unsafe_allow_html=True)

# Load API key from Streamlit secrets or env var
OPENROUTER_API_KEY = None
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.sidebar.error("OpenRouter API key missing. Add OPENROUTER_API_KEY to .streamlit/secrets.toml or env.")
    st.stop()

OPENROUTER_CHAT_URL = "https://api.openrouter.ai/v1/chat/completions"
MODEL_ID = "deepseek/deepseek-r1-0528:free"

# ---------------------------
# Helper functions
# ---------------------------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes with fallback."""
    text = ""
    if _HAVE_FITZ:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text()
            return text
        except Exception:
            pass
    # fallback to PyPDF2
    if PyPDF2:
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception:
            pass
    # fallback: try naive decode
    try:
        return pdf_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def split_lines_and_sections(text: str) -> List[str]:
    """Split text into meaningful lines/bullets preserving paragraphs."""
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # merge short lines that look like sentence fragments into last if necessary
        if len(s) < 40 and lines and not lines[-1].endswith('.'):
            # heuristic: if previous line not heading, merge
            lines[-1] = lines[-1] + " " + s
        else:
            lines.append(s)
    # further clean: remove lines that are purely contact info
    cleaned = []
    for l in lines:
        if any(tok in l.lower() for tok in ["mailto:", "@", "linkedin.com", "github.com", "tel:", "phone", "www."]):
            continue
        cleaned.append(l)
    return cleaned

def call_openrouter_chat(messages: list, max_tokens=1200, temperature=0.0):
    """Call OpenRouter chat completion endpoint with model and messages (OpenAI-compatible)."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        r = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        resp = r.json()
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content, resp
    except Exception as e:
        return f"[ERROR] {e}", None

def ddg_search_snippets(query: str, max_results: int = 4):
    if ddg is None:
        return []
    try:
        results = ddg(query, max_results=max_results)
        return results or []
    except Exception:
        return []

# Prompt templates
def prompt_generate_line_notes(line: str, context_jd: str = "") -> str:
    # structured instruction to produce JSON for a CV line
    return textwrap.dedent(f"""
    You are an expert interview coach and researcher. Given a single line from a candidate's CV, produce a JSON object with keys:
    - 'line': original line
    - 'label': short label/category (e.g., Work Experience, Education, Project, Skill, Certification)
    - 'short_summary': one-paragraph summary explaining the line in simple terms
    - 'domain': detected industry/domain (e.g., law, software, marketing) inferred from the line
    - 'questions': array of question objects with 'q' (question), 'type' (technical/behavioral/scenario/followup), and 'sample_answer' (2-3 paragraph model answer that shows deep understanding)
    - 'study_notes': short web-grounded notes (definitions, key points, steps to revise) suitable for last-minute reading
    - 'search_queries': list of 4 concise web search queries that would help learn more about this line
    Keep JSON compact and machine-parseable. Tailor answers to be profession-agnostic but adapt when domain cues exist. If JD is provided, align answers to JD: {bool(context_jd)}.
    """) + f"\nCV_LINE: {line}\nJD_CONTEXT: {context_jd}"

def prompt_generate_assessment(cv_text: str, jd_text: str) -> str:
    return textwrap.dedent(f"""
    You are an HR + domain expert. Compare the CV and JD and produce a JSON object:
    {{
      "fit_summary": "short paragraph about candidate fit for role",
      "strengths": ["short bullets"],
      "gaps": ["short bullets"],
      "actionable_recommendations": ["what to study or improve, prioritized"]
    }}
    CV:
    {cv_text}

    JD:
    {jd_text}
    """)

def prompt_generate_mock_questions(cv_text: str, jd_text: str, n:int=8) -> str:
    return textwrap.dedent(f"""
    You are an interviewer. From the CV below (and JD if given), generate {n} interview questions across categories:
    - scenario-based (STAR)
    - technical/domain-specific
    - culture-fit & behavioral
    - follow-ups for deeper probing

    For each question, return a JSON array item: {{ "q": "...", "category": "...", "ideal_points": ["bullet points of what a strong answer should include"] }}

    CV:
    {cv_text}

    JD:
    {jd_text}
    """)

# ---------------------------
# App state & sidebar
# ---------------------------
st.sidebar.title("Settings & Keys")
st.sidebar.caption("Model: deepseek/deepseek-r1-0528:free (OpenRouter).")

mode = st.sidebar.selectbox("Mode", ["CV only", "CV + JD (recommended)"])
max_lines = st.sidebar.slider("Max CV lines to analyze", 3, 40, 12)
ddg_results_per_line = st.sidebar.slider("Web snippets per line", 0, 6, 3)
temperature = st.sidebar.slider("Model creativity (temperature)", 0.0, 1.0, 0.0)

# File upload area
st.header("1) Upload CV (PDF) — primary input")
col1, col2 = st.columns([2,1])
with col1:
    uploaded_cv = st.file_uploader("Upload your CV PDF (required)", type=["pdf"])
with col2:
    paste_cv_text = st.text_area("Or paste CV text (optional)", height=120)

st.markdown("---")
st.header("2) Job Description (optional)")
jd_text_area = st.text_area("Paste Job Description text here (optional). If left empty, app works CV-first.", height=140)

process_btn = st.button("Analyze CV & Generate Line-by-Line Notes")

# internal storage for immutable assessment
if "initial_assessment" not in st.session_state:
    st.session_state.initial_assessment = None

# ---------------------------
# Main processing
# ---------------------------
if process_btn:
    # 1) Get CV text
    cv_text = ""
    if uploaded_cv is not None:
        pdf_bytes = uploaded_cv.read()
        cv_text = extract_text_from_pdf_bytes(pdf_bytes)
    if paste_cv_text and not cv_text.strip():
        cv_text = paste_cv_text

    if not cv_text.strip():
        st.error("Please upload a CV (PDF) or paste CV text.")
        st.stop()

    # 2) Parse lines
    lines = split_lines_and_sections(cv_text)
    # heuristic: pick lines containing verbs or bullets; keep top N
    candidate_lines = [l for l in lines if len(l) > 12][:max_lines]

    st.session_state["cv_text"] = cv_text
    st.session_state["candidate_lines"] = candidate_lines
    st.session_state["jd_text"] = jd_text_area

    # 3) Produce initial assessment (if JD provided or if CV+JD mode)
    if mode == "CV + JD (recommended)" and jd_text_area.strip():
        st.info("Generating JD–CV initial assessment (strengths, gaps)...")
        prompt = prompt_generate_assessment(cv_text, jd_text_area)
        messages = [
            {"role":"system", "content": "You are an HR and domain expert. Produce precise JSON."},
            {"role":"user", "content": prompt}
        ]
        out, raw = call_openrouter_chat(messages, max_tokens=900, temperature=0.0)
        # store immutable
        try:
            parsed = json.loads(out)
            st.session_state.initial_assessment = parsed
        except Exception:
            # fallback: keep raw text if JSON parse fails
            st.session_state.initial_assessment = {"raw": out}
        st.success("Initial assessment saved (won't be changed by mock interview).")
    else:
        st.info("No JD provided or CV-only mode: skipping JD–CV assessment.")

    # 4) For each candidate line: generate notes + qna + web snippets
    results = []
    progress = st.progress(0)
    total = len(candidate_lines)
    for i, line in enumerate(candidate_lines, start=1):
        st.markdown(f"<div class='card'><strong>Processing line {i}/{total}:</strong> {line}</div>", unsafe_allow_html=True)
        # 4a: web snippets
        snippets = []
        if ddg_results_per_line > 0:
            queries_for_line = [line]
            # add small variants for searching
            queries_for_line += [
                f"{line} explanation",
                f"{line} tutorial",
                f"{line} examples"
            ]
            for q in queries_for_line[:4]:
                if ddg is None:
                    break
                sr = ddg_search_snippets(q, max_results=ddg_results_per_line)
                for r in sr:
                    snippets.append({"title": r.get("title"), "url": r.get("href") or r.get("url"), "snippet": r.get("body")})
                time.sleep(0.1)

        # 4b: prepare LLM prompt for this line
        prompt = prompt_generate_line_notes(line, context_jd=jd_text_area if jd_text_area.strip() else "")
        messages = [
            {"role":"system", "content": "You are an expert interview coach, researcher and note writer. Provide concise JSON."},
            {"role":"user", "content": prompt},
        ]
        # include snippet context in an extra user message if present to ground answers
        if snippets:
            context_msg = {"role":"user", "content": "Web snippets (for context): " + json.dumps(snippets[:3], indent=2)}
            messages.append(context_msg)

        out, raw = call_openrouter_chat(messages, max_tokens=1400, temperature=temperature)
        # try parse JSON; if fails, return raw wrapped
        try:
            parsed = json.loads(out)
        except Exception:
            parsed = {"line": line, "raw": out, "sources": snippets}

        results.append({"line": line, "notes": parsed, "snippets": snippets})
        progress.progress(int(i/total*100))
        time.sleep(0.5)

    st.session_state["analysis_results"] = results
    st.success("Line-by-line notes generated. Scroll down to review.")


# ---------------------------
# Display results & interaction UI
# ---------------------------
if "analysis_results" in st.session_state:
    results = st.session_state["analysis_results"]
    jd_text = st.session_state.get("jd_text", "")
    cv_text = st.session_state.get("cv_text", "")

    # Top area: immutable assessment (if present)
    st.header("Initial Assessment (JD ↔ CV)")
    if st.session_state.initial_assessment:
        ia = st.session_state.initial_assessment
        if "raw" in ia:
            st.warning("Assessment produced but not in JSON-parsable format. Raw output:")
            st.write(ia["raw"])
        else:
            st.markdown(f"**Fit summary:** {ia.get('fit_summary','-')}")
            st.markdown("**Strengths:**")
            for s in ia.get("strengths", []):
                st.markdown(f"- {s}")
            st.markdown("**Gaps:**")
            for g in ia.get("gaps", []):
                st.markdown(f"- {g}")
            st.markdown("**Recommendations:**")
            for r in ia.get("actionable_recommendations", []):
                st.markdown(f"- {r}")
    else:
        st.info("No JD-based assessment available (CV-only or JD missing).")

    st.markdown("---")
    st.header("Line-by-line Preparation (CV-focused)")
    # show each line card with QnA, sample answers and web notes
    for idx, item in enumerate(results):  # ✅ Added enumerate to create a unique index for each loop
        notes = item["notes"]
        line = item["line"]
    # card
    with st.expander(f"🔎 {line}", expanded=False):
        if isinstance(notes, dict) and notes.get("short_summary"):
            st.markdown(f"**Summary:** {notes.get('short_summary')}")
            st.markdown(f"**Detected domain:** {notes.get('domain','General')}")
        elif isinstance(notes, dict) and notes.get("raw"):
            st.markdown("**Raw model output:**")
            st.write(notes.get("raw"))
        else:
            st.write(notes)

        # study notes & web snippets
        st.markdown("**Study notes (web-grounded)**")
        study = notes.get("study_notes") if isinstance(notes, dict) else None
        if study:
            st.write(study)
        if item.get("snippets"):
            st.markdown("**Quick sources**")
            for s in item["snippets"][:4]:
                if s.get("url"):
                    st.markdown(f"- [{s.get('title')}]({s.get('url')}) — {s.get('snippet')[:180]}")

        # questions
        if isinstance(notes, dict) and notes.get("questions"):
            st.markdown("**Generated Questions & Sample Answers**")
            for qobj in notes.get("questions"):
                q = qobj.get("q")
                typ = qobj.get("type", "general")
                ans = qobj.get("sample_answer", "")
                with st.container():
                    st.markdown(f"<div class='q-card'><strong>Q ({typ}):</strong> {q}</div>", unsafe_allow_html=True)
                    with st.expander("Show sample answer"):
                        st.write(ans)
        else:
            # fallback: offer to generate quick mock questions
            # ✅ FIX: Ensure unique key for each button by adding index
            if st.button(f"Generate quick Qs for this line", key=f"gen_{idx}_{line[:10]}"):
                prompt = prompt_generate_line_notes(line, context_jd=jd_text)
                messages = [{"role": "system", "content": "You are an interview coach."},
                            {"role": "user", "content": prompt}]
                out, _ = call_openrouter_chat(messages, max_tokens=700, temperature=0.0)
                st.write(out)
                st.markdown("---")

"""
CHANGES MADE:
1. Added enumerate(results) to get a unique 'idx' for each loop iteration.
2. Modified st.button key to include 'idx' → `key=f"gen_{idx}_{line[:10]}"` so keys are always unique.
3. This prevents Streamlit's 'DuplicateWidgetID' error when `line[:10]` repeats.
"""

               
    # Mock Interview (text-only)
st.header("Mock Interview (text mode) — optional")
st.markdown("This mock interview uses the same CV-focused knowledge generated above. It will NOT change the initial JD–CV assessment.")
if "mock_questions" not in st.session_state:
    st.session_state.mock_questions = []
    st.session_state.mock_idx = 0
    st.session_state.mock_history = []

if st.button("Generate Mock Interview Questions (8)"):
    prompt = prompt_generate_mock_questions(cv_text, jd_text, n=8)
    messages = [{"role":"system","content":"You are an interviewer."},{"role":"user","content":prompt}]
    out, _ = call_openrouter_chat(messages, max_tokens=900, temperature=0.2)
       
    # try parse array
        try:
            parsed = json.loads(out)
            qs = [p.get("q") if isinstance(p, dict) else str(p) for p in parsed]
        except Exception:
            # fallback: split lines
            qs = [line.strip() for line in out.splitlines() if line.strip()][:8]
        st.session_state.mock_questions = qs
        st.session_state.mock_idx = 0
        st.success("Mock questions generated. Use the panel below to answer and get feedback.")

    if st.session_state.mock_questions:
        idx = st.session_state.mock_idx
        if idx < len(st.session_state.mock_questions):
            st.markdown(f"**Q{idx+1}:** {st.session_state.mock_questions[idx]}")
            user_answer = st.text_area("Your answer (type here)", key=f"mock_ans_{idx}", height=160)
            if st.button("Get feedback on this answer"):
                # feed LLM: provide initial assessment + line notes context for better feedback
                context = {"initial_assessment": st.session_state.get("initial_assessment"), "top_lines": [r["line"] for r in results[:6]]}
                fb_prompt = textwrap.dedent(f"""
                You are an interviewer and feedback coach. Use the candidate context and initial assessment to give constructive feedback.
                Context (JSON): {json.dumps(context, indent=2)}
                Question: {st.session_state.mock_questions[idx]}
                Candidate answer: {user_answer}

                Provide:
                1) Short strengths of this answer (bulleted)
                2) Weaknesses / missing points (bulleted)
                3) How to improve (step-by-step)
                4) A model improved answer (2-3 short paragraphs)
                """)
                messages = [{"role":"system","content":"You are a helpful interviewer who gives actionable feedback."},{"role":"user","content":fb_prompt}]
                fb, _ = call_openrouter_chat(messages, max_tokens=900, temperature=0.0)
                st.markdown("**Feedback:**")
                st.write(fb)
                st.session_state.mock_history.append({"q": st.session_state.mock_questions[idx], "answer": user_answer, "feedback": fb})
                st.session_state.mock_idx += 1
        else:
            st.success("Mock interview complete! You can regenerate or download history.")
            if st.session_state.mock_history:
                st.download_button("Download mock history (JSON)", json.dumps(st.session_state.mock_history, indent=2), file_name="mock_history.json")

    st.markdown("---")
    # Exports
    st.header("Export / Download")
    if st.button("Download all notes (JSON)"):
        st.download_button("Download notes JSON", json.dumps(results, indent=2), file_name="cv_notes.json")
    st.markdown("<footer>Built with ♥ — model: deepseek/deepseek-r1-0528:free via OpenRouter. Keep your API key secure in st.secrets.</footer>", unsafe_allow_html=True)

