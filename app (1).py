import streamlit as st
import PyPDF2
import pandas as pd
from datetime import datetime
import json
from google import genai
from google.genai import types

# ============================================
# CONFIGURATION
# ============================================

GEMINI_API_KEY = "AIzaSyAX2jWBnDu3JNfAHjNhud2Yh6a0a72B5hU"

st.set_page_config(
    page_title="CV Screener",
    page_icon="📄",
    layout="wide"
)

# Force light theme colors that work everywhere
st.markdown("""
<style>
    /* Force all text to be dark/visible */
    html, body, [class*="css"] {
        color: #111111 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #111111 !important;
        font-weight: 500 !important;
    }
    
    /* Labels and text */
    label, .stMarkdown, p, span, div {
        color: #111111 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    [data-testid="stSidebar"] * {
        color: #111111 !important;
    }
    
    /* Main content background */
    .main > div {
        background-color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1a73e8 !important;
        color: white !important;
        font-weight: 500 !important;
        border: none !important;
    }
    
    /* File uploader */
    .stFileUploader {
        background-color: #f8f9fa !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #ccc !important;
    }
    
    /* Input field */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #ccc !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #1a73e8 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: #111111 !important;
    }
    
    /* Dataframe */
    .dataframe {
        color: #111111 !important;
    }
    .dataframe th {
        background-color: #f8f9fa !important;
        color: #111111 !important;
    }
    .dataframe td {
        color: #111111 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNCTIONS
# ============================================

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def ai_extract_and_analyze(cv_text, job_description, job_title=""):
    prompt = f"""
You are an expert HR recruiter. Analyze the CV below for the given job position.

Return ONLY valid JSON. No markdown.

JOB TITLE: {job_title if job_title else "Not specified"}

JOB DESCRIPTION:
{job_description}

CV TEXT:
{cv_text[:8000]}

Return EXACTLY this JSON structure:

{{
  "candidate_info": {{
    "name": "full name or null",
    "email": "email or null",
    "phone": "phone or null"
  }},
  "education": {{
    "degree": "highest degree name",
    "field": "field of study",
    "university": "university name",
    "year": "graduation year or null",
    "score/10": 0,
    "details": "brief description"
  }},
  "experience": {{
    "total_years": 0,
    "score/10": 0,
    "positions": [
      {{
        "title": "job title",
        "company": "company",
        "years": 0,
        "description": "responsibilities"
      }}
    ],
    "summary": "brief assessment"
  }},
  "technical_skills": {{
    "score/10": 0,
    "skills": [
      {{
        "name": "skill name",
        "level": "Expert/Advanced/Intermediate/Beginner",
        "years": "estimated years or null"
      }}
    ],
    "summary": "brief assessment"
  }},
  "languages": {{
    "score/10": 0,
    "items": [
      {{
        "name": "language",
        "level": "Native/Fluent/Professional/Intermediate/Basic",
        "certification": "certification or null"
      }}
    ],
    "summary": "brief assessment"
  }},
  "soft_skills": {{
    "score/10": 0,
    "items": ["skill1", "skill2", "skill3"],
    "summary": "brief assessment"
  }},
  "overall": {{
    "score/100": 0,
    "verdict": "STRONG HIRE/HIRE/CONSIDER/REJECT",
    "confidence": "HIGH/MEDIUM/LOW",
    "strengths": ["strength1", "strength2", "strength3", "strength4", "strength5"],
    "weaknesses": ["weakness1", "weakness2", "weakness3"],
    "justification": "2-3 sentence explanation",
    "interview_questions": ["Q1", "Q2", "Q3", "Q4"]
  }}
}}

SCORING (each /10): education, experience, technical_skills, languages, soft_skills
TOTAL /100 = weighted sum (education 20%, experience 25%, technical 25%, languages 15%, soft skills 15%)
"""

    models_to_try = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-flash-lite-preview"]
    
    for model_name in models_to_try:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=5000)
            )
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            return json.loads(response_text.strip()), model_name
        except Exception:
            continue
    return None, None

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0; border-bottom: 1px solid #e5e5e5; margin-bottom: 2rem;">
        <h1 style="font-size: 1.75rem; font-weight: 500; color: #1a73e8; margin: 0;">CV Screener Professional</h1>
        <p style="color: #5f6368; font-size: 0.875rem; margin-top: 0.5rem;">AI-powered candidate evaluation and scoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Evaluation Criteria")
        st.markdown("Each criterion scored out of 10")
        st.markdown("""
        - Education
        - Professional Experience
        - Technical Skills
        - Languages
        - Soft Skills
        """)
        st.markdown("---")
        st.markdown("### Verdicts")
        st.markdown("""
        - Strong Hire: 80-100
        - Hire: 65-79
        - Consider: 50-64
        - Reject: 0-49
        """)
        st.markdown("---")
        st.markdown("**Powered by**")
        st.markdown("Google Gemini AI")
    
    # Input columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Candidate CV")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            st.write(f"Loaded: {uploaded_file.name}")
    
    with col2:
        st.markdown("### Job Description")
        job_description = st.text_area("Paste description here", height=150)
        job_title = st.text_input("Job Title (optional)")
    
    st.markdown("---")
    analyze = st.button("Start Analysis", use_container_width=True)
    
    if analyze:
        if not uploaded_file:
            st.warning("Please upload a CV")
        elif not job_description:
            st.warning("Please paste a job description")
        else:
            with st.spinner("Analyzing CV..."):
                cv_text = extract_text_from_pdf(uploaded_file)
                
                if cv_text and len(cv_text) > 100:
                    result, model = ai_extract_and_analyze(cv_text, job_description, job_title)
                    
                    if result:
                        st.success(f"Analysis complete • {model}")
                        
                        # Overall score row
                        overall = result.get('overall', {})
                        total = overall.get('score/100', 0)
                        verdict = overall.get('verdict', 'N/A')
                        confidence = overall.get('confidence', 'N/A')
                        
                        # Display scores in columns
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Overall Score", f"{total}/100")
                        with col_b:
                            st.metric("Verdict", verdict)
                        with col_c:
                            st.metric("Confidence", confidence)
                        with col_d:
                            st.metric("Analysis Date", datetime.now().strftime("%d/%m/%Y"))
                        
                        # Progress bar
                        st.progress(total/100)
                        
                        # Criteria scores
                        st.markdown("### Evaluation Criteria")
                        
                        edu_score = result.get('education', {}).get('score/10', 0)
                        exp_score = result.get('experience', {}).get('score/10', 0)
                        tech_score = result.get('technical_skills', {}).get('score/10', 0)
                        lang_score = result.get('languages', {}).get('score/10', 0)
                        soft_score = result.get('soft_skills', {}).get('score/10', 0)
                        
                        # Display criteria in columns
                        c1, c2, c3, c4, c5 = st.columns(5)
                        with c1:
                            st.metric("Education", f"{edu_score}/10")
                        with c2:
                            st.metric("Experience", f"{exp_score}/10")
                        with c3:
                            st.metric("Technical Skills", f"{tech_score}/10")
                        with c4:
                            st.metric("Languages", f"{lang_score}/10")
                        with c5:
                            st.metric("Soft Skills", f"{soft_score}/10")
                        
                        # Candidate info
                        candidate = result.get('candidate_info', {})
                        if candidate.get('name'):
                            st.markdown("### Candidate Information")
                            st.write(f"**Name:** {candidate.get('name')}")
                            if candidate.get('email'):
                                st.write(f"**Email:** {candidate.get('email')}")
                            if candidate.get('phone'):
                                st.write(f"**Phone:** {candidate.get('phone')}")
                        
                        # Education
                        edu = result.get('education', {})
                        if edu.get('degree'):
                            st.markdown("### Education")
                            st.write(f"**Degree:** {edu.get('degree')} in {edu.get('field', '')}")
                            st.write(f"**University:** {edu.get('university', '')}")
                            if edu.get('year'):
                                st.write(f"**Graduation Year:** {edu.get('year')}")
                            if edu.get('details'):
                                st.caption(edu.get('details'))
                        
                        # Experience
                        exp = result.get('experience', {})
                        if exp.get('total_years', 0) > 0 or exp.get('positions'):
                            st.markdown("### Professional Experience")
                            st.write(f"**Total Experience:** {exp.get('total_years', 0)} years")
                            if exp.get('summary'):
                                st.caption(exp.get('summary'))
                            for pos in exp.get('positions', []):
                                with st.expander(f"{pos.get('title', 'Position')} - {pos.get('company', 'Company')} ({pos.get('years', 0)} years)"):
                                    st.write(pos.get('description', 'No description'))
                        
                        # Technical Skills
                        skills = result.get('technical_skills', {})
                        if skills.get('skills'):
                            st.markdown("### Technical Skills")
                            if skills.get('summary'):
                                st.caption(skills.get('summary'))
                            skill_tags = ""
                            for s in skills.get('skills', []):
                                skill_tags += f"`{s.get('name', '')}: {s.get('level', '')}` "
                            st.markdown(skill_tags)
                        
                        # Languages
                        langs = result.get('languages', {})
                        if langs.get('items'):
                            st.markdown("### Languages")
                            if langs.get('summary'):
                                st.caption(langs.get('summary'))
                            lang_data = []
                            for l in langs.get('items', []):
                                lang_data.append({
                                    "Language": l.get('name', ''), 
                                    "Level": l.get('level', ''), 
                                    "Certification": l.get('certification', '-')
                                })
                            st.dataframe(pd.DataFrame(lang_data), use_container_width=True)
                        
                        # Soft Skills
                        soft = result.get('soft_skills', {})
                        if soft.get('items'):
                            st.markdown("### Soft Skills")
                            if soft.get('summary'):
                                st.caption(soft.get('summary'))
                            st.write(", ".join(soft.get('items', [])))
                        
                        # Strengths and Weaknesses
                        st.markdown("### Assessment")
                        
                        col_strength, col_weakness = st.columns(2)
                        with col_strength:
                            st.markdown("**Strengths**")
                            for s in overall.get('strengths', []):
                                st.write(f"+ {s}")
                        with col_weakness:
                            st.markdown("**Areas for Improvement**")
                            for w in overall.get('weaknesses', []):
                                st.write(f"- {w}")
                        
                        # Justification
                        if overall.get('justification'):
                            st.markdown("### Justification")
                            st.info(overall.get('justification'))
                        
                        # Interview Questions
                        questions = overall.get('interview_questions', [])
                        if questions:
                            st.markdown("### Interview Questions")
                            for i, q in enumerate(questions, 1):
                                st.write(f"{i}. {q}")
                        
                        # Export
                        st.markdown("---")
                        col_export1, col_export2 = st.columns(2)
                        
                        export_json = {
                            "candidate": uploaded_file.name,
                            "date": datetime.now().isoformat(),
                            "total_score": total,
                            "verdict": verdict,
                            "scores": {
                                "education": edu_score,
                                "experience": exp_score,
                                "technical_skills": tech_score,
                                "languages": lang_score,
                                "soft_skills": soft_score
                            },
                            "strengths": overall.get('strengths', []),
                            "weaknesses": overall.get('weaknesses', []),
                            "justification": overall.get('justification', ''),
                            "interview_questions": questions
                        }
                        
                        with col_export1:
                            st.download_button(
                                "Export Report (JSON)", 
                                data=json.dumps(export_json, indent=2), 
                                file_name=f"cv_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        text_report = f"""
CV SCREENING REPORT
===================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Candidate: {uploaded_file.name}

OVERALL SCORE: {total}/100
VERDICT: {verdict}

CRITERIA SCORES:
- Education: {edu_score}/10
- Experience: {exp_score}/10
- Technical Skills: {tech_score}/10
- Languages: {lang_score}/10
- Soft Skills: {soft_score}/10

STRENGTHS:
{chr(10).join(['- ' + s for s in overall.get('strengths', [])])}

AREAS FOR IMPROVEMENT:
{chr(10).join(['- ' + w for w in overall.get('weaknesses', [])])}

JUSTIFICATION:
{overall.get('justification', '')}

INTERVIEW QUESTIONS:
{chr(10).join([f'{i+1}. {q}' for i, q in enumerate(questions)])}
"""
                        with col_export2:
                            st.download_button(
                                "Export Report (TXT)", 
                                data=text_report, 
                                file_name=f"cv_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                    else:
                        st.error("Analysis failed. Please check your API key and internet connection.")
                else:
                    st.error("Could not extract text from PDF. Please ensure it is a text-based PDF (not scanned/image).")

if __name__ == "__main__":
    main()