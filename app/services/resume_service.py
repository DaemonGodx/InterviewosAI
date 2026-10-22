import os
from pypdf import PdfReader
import docx
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.models.schemas import CandidateProfile
from app.config import GROQ_API_KEY
import logging

logger = logging.getLogger(__name__)

class ResumeExtractionError(Exception):
    pass

class ResumeService:
    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=GROQ_API_KEY,
            temperature=0.1
        )
        self.structured_llm = self.llm.with_structured_output(CandidateProfile)

    def extract_text_from_file(self, file_path: str) -> str:
        """Extracts text from PDF or DOCX file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume file not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == ".pdf":
                return self._extract_from_pdf(file_path)
            elif ext == ".docx":
                return self._extract_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}. Only PDF and DOCX are supported.")
        except Exception as e:
            logger.error("Failed to extract resume text from the selected file.")
            if isinstance(e, ValueError) and str(e).startswith("Unsupported file format:"):
                raise ResumeExtractionError(str(e)) from e
            raise ResumeExtractionError("Extraction failed. Check that the file is a readable PDF or DOCX.") from e

    def _extract_from_pdf(self, file_path: str) -> str:
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise ResumeExtractionError("PDF appears to be empty or unreadable (scanned images without OCR).")
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        parts = []
        for child in doc.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph_text = Paragraph(child, doc).text.strip()
                if paragraph_text:
                    parts.append(paragraph_text)
            elif isinstance(child, CT_Tbl):
                table = Table(child, doc)
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells)
                    if row_text.strip(" |"):
                        parts.append(row_text)
        text = "\n".join(parts)
        
        if not text.strip():
            raise ResumeExtractionError("DOCX appears to be empty.")
        return text

    def analyze_resume(self, resume_text: str) -> CandidateProfile:
        """Uses LLM to structure the resume text into a CandidateProfile."""
        if not resume_text or len(resume_text.strip()) < 50:
            raise ResumeExtractionError("Extracted text is too short to be a valid resume.")
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert technical recruiter and resume analyzer. "
                       "Analyze the following resume and extract the details into the structured format. "
                       "Do NOT invent or hallucinate information that is not present. "
                       "If information is missing, leave it empty. "
                       "Provide an intelligent summary and identify strengths, weaknesses, and key technologies "
                       "that would be good to question the candidate about during a technical interview. "
                       "Also estimate their professional experience in years (as a float), and recommend one of these experience levels: "
                       "'Student', 'Fresher', '0-2 years', '2-5 years', '5+ years'."),
            ("human", "Resume Text:\n\n{resume_text}")
        ])
        
        chain = prompt | self.structured_llm
        
        try:
            profile = chain.invoke({"resume_text": resume_text})
            return profile
        except Exception as e:
            logger.error("Resume profile analysis failed.")
            raise ResumeExtractionError("Failed to analyze resume with LLM. Check the configured AI service and try again.") from e
