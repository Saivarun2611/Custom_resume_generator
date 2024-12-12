import streamlit as st
import json
import PyPDF2
import faiss
import pandas as pd
import re
from fpdf import FPDF
from io import BytesIO
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# ----------------------- Initialization -----------------------
st.set_page_config(
    page_title="Smart Resume Enhancer",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
    background-color: #f7f7f7;
}
.block-container {
    max-width: 800px;
    padding-top: 2rem;
}
.title {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
}
.subtitle {
    font-size: 1.25rem;
    text-align: center;
    color: #666;
    margin-bottom: 1rem;
}
.upload-area {
    border: 2px dashed #ccc;
    border-radius: 5px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Smart Resume Enhancer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enhance your resume based on a job posting and a skills database</div>", unsafe_allow_html=True)

# ----------------------- App Inputs -----------------------
groq_api_key = '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' # replace with your key
llm = ChatGroq(temperature=0.3, groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

# Upload Resume PDF
st.markdown("### Upload Your Resume (PDF)")
resume_file = st.file_uploader("Choose a PDF file", type=["pdf"], help="Upload your resume in PDF format.")

# Upload CSV File
st.markdown("### Upload CSV of Roles and Skills")
csv_file = st.file_uploader("Choose a CSV file", type=["csv"], help="Upload CSV with Role, Skills, Experience, Projects columns.")

# Job URL Input
st.markdown("### Enter Job Posting URL")
job_url = st.text_input("Job Posting URL", placeholder="https://example.com/job-posting")

# Process Button
process_button = st.button("Enhance Resume")

# ----------------------- Core Functions -----------------------
@st.cache_data
def initialize_faiss(csv_data):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = embedding_model.get_sentence_embedding_dimension()
    faiss_index = faiss.IndexFlatL2(dimension)
    metadata = []

    for _, row in csv_data.iterrows():
        role = row['Role']
        role_embedding = embedding_model.encode(role).reshape(1, -1)
        faiss_index.add(role_embedding)
        metadata.append({
            "role": role,
            "skills": row["Skills"],
            "experience": row["Experience"],
            "projects": row["Projects"]
        })
    return faiss_index, metadata, embedding_model


def query_faiss(role, faiss_index, metadata, embedding_model, top_k=1):
    role_embedding = embedding_model.encode(role).reshape(1, -1)
    distances, indices = faiss_index.search(role_embedding, top_k)
    if indices[0][0] != -1:
        result_index = indices[0][0]
        return metadata[result_index]
    return None

def enhance_job_data(raw_data, job_description):
    if raw_data:
        enhanced_skills = raw_data.get('skills', '')

        exp_list = [exp.strip() for exp in raw_data.get('experience', '').split(",") if exp.strip()]
        proj_list = [proj.strip() for proj in raw_data.get('projects', '').split(",") if proj.strip()]

        exp_list.append("Aligned skill sets and methodologies to the specified role, ensuring strategic contributions to organizational goals.")
        proj_list.append("Implemented data-driven methodologies to enhance system efficiency and performance by over 20%.")

        enhanced_experience = "\n- " + "\n- ".join(exp_list)
        enhanced_projects = "\n- " + "\n- ".join(proj_list)

        return {
            "skills": enhanced_skills,
            "experience": enhanced_experience,
            "projects": enhanced_projects,
            "job_description": job_description
        }
    return None

def extract_job_details(job_url):
    try:
        loader = UnstructuredURLLoader(urls=[job_url])
        docs = loader.load()
        page_data = docs[0].page_content if docs else ""
        if not page_data.strip():
            return None

        prompt_extract = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION TEXT:
            {page_data}

            ### INSTRUCTION:
            Extract the following details:
            - Role
            - Required Skills
            - A short summary (3-4 sentences) of the job description

            Provide the output as a valid JSON object with keys:
            `role`, `skills`, `description`.
            """
        )
        chain_extract = prompt_extract | llm
        res = chain_extract.invoke({"page_data": page_data})
        raw_response = res.content.strip()
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        return json.loads(raw_response)
    except:
        return None

def refine_text_with_llm(experience_text, projects_text, job_description, role):
    prompt = f"""
    The candidate is applying for the role: {role}.
    Job description: {job_description}
    
    Below are two sections from the candidate's resume that need improvement:
    EXPERIENCE SECTION (before improvement):
    {experience_text}

    PROJECTS SECTION (before improvement):
    {projects_text}

    ### INSTRUCTION:
    Please rewrite the EXPERIENCE and PROJECTS sections to be more professional, cohesive, and impactful.
    - Use clear, ATS-friendly formatting (no Markdown or special characters for headings).
    - Add quantifiable achievements wherever possible.
    - Ensure the grammar is polished and the tone is professional.
    - Align responsibilities and achievements with the job description if relevant.
    - Use simple dashes for bullet points.
    - Keep the content relevant and clean.

    Provide the improved text in the following format:

    EXPERIENCE:
    <Your improved bullet points here>

    PROJECTS:
    <Your improved bullet points here>
    """

    refinement_template = PromptTemplate.from_template(prompt)
    chain_refine = refinement_template | llm
    res = chain_refine.invoke({})
    improved_text = res.content.strip()
    return improved_text

def read_resume_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text() + "\n"
        return resume_text.strip()
    except:
        return ""

def extract_resume_sections(resume_text):
    name_line = re.search(r"Name:\s*(.*)", resume_text)
    email_line = re.search(r"Email:\s*(.*)", resume_text)
    phone_line = re.search(r"Phone:\s*(.*)", resume_text)
    location_line = re.search(r"Location:\s*(.*)", resume_text)

    name = name_line.group(1).strip() if name_line else "Unknown"
    email = email_line.group(1).strip() if email_line else "Unknown"
    phone = phone_line.group(1).strip() if phone_line else "Unknown"
    location = location_line.group(1).strip() if location_line else "Unknown"

    edu_match = re.search(r"(Education:.*?)(?=Skills:|Experience:|Projects:|$)", resume_text, flags=re.DOTALL|re.IGNORECASE)
    education = edu_match.group(1).strip() if edu_match else ""

    skills_match = re.search(r"(Skills:.*?)(?=Experience:|Projects:|$)", resume_text, flags=re.DOTALL|re.IGNORECASE)
    original_skills = skills_match.group(1).strip() if skills_match else ""

    exp_match = re.search(r"(Experience:.*?)(?=Projects:|$)", resume_text, flags=re.DOTALL|re.IGNORECASE)
    original_experience = exp_match.group(1).strip() if exp_match else ""

    proj_match = re.search(r"(Projects:.*)", resume_text, flags=re.DOTALL|re.IGNORECASE)
    original_projects = proj_match.group(1).strip() if proj_match else ""

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location,
        "education": education,
        "original_skills": original_skills,
        "original_experience": original_experience,
        "original_projects": original_projects
    }

def extract_role_company(original_experience_text):
    role_line = re.search(r"Role:\s*(.*)", original_experience_text)
    company_line = re.search(r"Company:\s*(.*)", original_experience_text)
    role = role_line.group(1).strip() if role_line else ""
    company = company_line.group(1).strip() if company_line else ""
    return role, company

def create_clean_resume(original_text, personal_info, refined_experience, refined_projects, enhanced_skills, role, company):
    updated_text = original_text

    if enhanced_skills:
        skills_bullets = "\n- " + "\n- ".join([s.strip() for s in enhanced_skills.split(",") if s.strip()])
        updated_text = re.sub(r"(Skills:)(.*?)(?=Experience:|Projects:|$)", f"Skills:{skills_bullets}\n", updated_text, flags=re.DOTALL)

    # Insert role and company
    if role and company:
        exp_section = f"Experience:\nRole: {role}\nCompany: {company}\n{refined_experience}\n"
    else:
        exp_section = f"Experience:\n{refined_experience}\n"

    updated_text = re.sub(r"(Experience:)(.*?)(?=Projects:|$)", exp_section, updated_text, flags=re.DOTALL)

    if "Projects:" in updated_text:
        updated_text = re.sub(r"(Projects:)(.*)", f"Projects:\n{refined_projects}", updated_text, flags=re.DOTALL)
    else:
        updated_text += f"\nProjects:\n{refined_projects}"

    updated_text = re.sub(r"(Name:\s*).*", f"Name: {personal_info['name']}", updated_text, count=1)
    updated_text = re.sub(r"(Email:\s*).*", f"Email: {personal_info['email']}", updated_text, count=1)
    updated_text = re.sub(r"(Phone:\s*).*", f"Phone: {personal_info['phone']}", updated_text, count=1)
    updated_text = re.sub(r"(Location:\s*).*", f"Location: {personal_info['location']}", updated_text, count=1)

    return updated_text.strip()

def final_resume_refinement(resume_text, llm):
    prompt = f"""
    Below is a resume. Please rewrite it in an ATS-friendly plain text format:
    - Use uppercase for main headings: NAME, CONTACT INFORMATION, EDUCATION, SKILLS, EXPERIENCE, PROJECTS.
    - Use simple dashes ("-") for bullet points.
    - Do not remove or alter role and company names under EXPERIENCE.
    - Keep all content as is, just improve formatting (no Markdown, no special characters).
    - Maintain the order and the original data.

    RESUME TEXT:
    {resume_text}

    ### INSTRUCTION:
    Return the improved ATS-friendly resume text.
    """

    refinement_template = PromptTemplate.from_template(prompt)
    chain_refine = refinement_template | llm
    res = chain_refine.invoke({})
    improved_resume = res.content.strip()
    return improved_resume

def generate_pdf(resume_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_text_color(0, 0, 0)
    # Generate PDF content in-memory
    pdf.multi_cell(0, 10, resume_text.encode('latin-1', 'replace').decode('latin-1'))
    # Output the PDF as a bytes object
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    bio = BytesIO(pdf_bytes)
    return bio


# ----------------------- Main Logic -----------------------
if process_button:
    if resume_file is None:
        st.error("Please upload your resume PDF.")
    elif csv_file is None:
        st.error("Please upload the CSV file.")
    elif not job_url.strip():
        st.error("Please provide a job posting URL.")
    else:
        with st.spinner("Processing..."):
            # Read CSV
            csv_data = pd.read_csv(csv_file)
            faiss_index, metadata, embedding_model = initialize_faiss(csv_data)

            # Extract job details
            job_details = extract_job_details(job_url)

            if not job_details:
                st.error("Failed to extract job details from the provided URL.")
            else:
                # Query FAISS
                raw_data = query_faiss(job_details["role"], faiss_index, metadata, embedding_model)
                enhanced_content = enhance_job_data(raw_data, job_details["description"])

                resume_text = read_resume_pdf(resume_file)
                personal_info = extract_resume_sections(resume_text)

                role, company = extract_role_company(personal_info["original_experience"])

                refined_text = refine_text_with_llm(
                    enhanced_content['experience'],
                    enhanced_content['projects'],
                    enhanced_content['job_description'],
                    job_details['role']
                )

                exp_match = re.search(r"EXPERIENCE:\s*(.*?)(?=PROJECTS:|$)", refined_text, flags=re.DOTALL)
                proj_match = re.search(r"PROJECTS:\s*(.*)", refined_text, flags=re.DOTALL)

                refined_experience = exp_match.group(1).strip() if exp_match else enhanced_content['experience']
                refined_projects = proj_match.group(1).strip() if proj_match else enhanced_content['projects']

                updated_resume_content = create_clean_resume(
                    resume_text,
                    personal_info,
                    refined_experience,
                    refined_projects,
                    enhanced_content['skills'],
                    role,
                    company
                )

                final_resume = final_resume_refinement(updated_resume_content, llm)

        # Display the final resume
        st.markdown("### Enhanced Resume Preview")
        st.text(final_resume)

        # Offer a download button for the PDF
        pdf_file = generate_pdf(final_resume)
        st.download_button(
            label="Download Enhanced Resume as PDF",
            data=pdf_file,
            file_name="Enhanced_Resume.pdf",
            mime="application/pdf"
        )

        st.success("Resume enhanced successfully!")
        st.balloons()
