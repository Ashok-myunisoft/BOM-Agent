
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import  List
import pdfplumber
import io
import os
import re
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BOM Agent",
    description="Generalizable BOM extraction using OpenAI (GPT-4o-mini) + pdfplumber",
    version="1.2.0",
    docs_url="/docs"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# === Pydantic Models ===
class Component(BaseModel):
    bom_level: str
    part_no: str
    part_rev: str
    description: str
    material_specification: str = ""
    seq_no: str
    quantity: str

    # Auto convert int/float -> string
    @field_validator(
        "bom_level",
        "part_rev",
        "seq_no",
        "quantity",
        mode="before"
    )
    @classmethod
    def convert_to_string(cls, value):
        if value is None:
            return ""
        return str(value)


class BOMResponse(BaseModel):
    part_number: str
    standard_description: str
    bom_revision_no: str
    eco_number: str
    date: str
    mn_document_revision_no: str
    components: List[Component]

    @field_validator("bom_revision_no", mode="before")
    @classmethod
    def convert_revision(cls, value):
        if value is None:
            return ""
        return str(value)


# === Dependency: OpenAI Client ===
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


# === Load Prompt from File ===
def load_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"Prompt file not found at: {prompt_path}")
        raise HTTPException(status_code=500, detail="prompt.txt not found")
    except Exception as e:
        logger.error(f"Error reading prompt.txt: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load prompt")


# === Extract Text and Tables from PDF ===
def extract_text_and_tables(pdf_content: bytes) -> tuple[str, List[List[str]]]:
    text = ""
    full_table = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                # Extract raw text for metadata
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        cleaned_row = [cell.strip() if cell else "" for cell in row]
                        full_table.append(cleaned_row)
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to read PDF")

    return text.strip(), full_table


# === Safely Extract JSON from AI Response ===
def extract_json_safely(text: str) -> dict:
    try:
        # Remove markdown code blocks
        text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```$', '', text)

        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in AI response")
        cleaned = text[start:end + 1]

        return json.loads(cleaned)
    except Exception as e:
        logger.error(f"JSON parse error: {str(e)} | Raw output: {text}")
        raise HTTPException(status_code=500, detail="AI returned invalid JSON")


# === API Endpoint ===
@app.post("/parse-bom", response_model=BOMResponse, tags=["BOM Processing"])
async def parse_bom(
    file: UploadFile = File(...),
    client: OpenAI = Depends(get_openai_client)
):
    """
    Parse Baker Hughes BOM PDF using OpenAI (GPT-4o-mini) + pdfplumber.
    Extracts metadata and BOM table with exact values.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    logger.info(f"Parsing BOM: {file.filename}")

    # Extract text and tables
    raw_text, table_data = extract_text_and_tables(content)

    # Convert table to string format for LLM
    table_str = "\n".join(["\t".join(row) for row in table_data]) if table_data else "No table found in PDF."

    # Prepare context for LLM
    context = f"""
FULL DOCUMENT TEXT:
{raw_text}

STRUCTURED TABLE DATA (tab-separated):
{table_str}
"""

    # Load prompt from external file
    prompt = load_prompt()

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise BOM parser. Return only valid JSON."},
                {"role": "user", "content": f"{prompt}\n\n{context}"}
            ],
            temperature=0,
            max_tokens=8192,
        )

        ai_output = response.choices[0].message.content.strip()
        logger.info(f"AI Response (first 500 chars): {ai_output[:500]}...")

        # Parse JSON
        data = extract_json_safely(ai_output)

        # Validate components
        if not isinstance(data.get("components"), list):
            data["components"] = []

        # Return response
        return BOMResponse(
            success=True,
            file_name=file.filename,
            **{k: (v if v not in ("", None) else None) for k, v in data.items()},
            message="BOM parsed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parsing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


# === Health Endpoints ===
@app.get("/", tags=["Health"])
async def root():
    return {"message": "BOM Parser API is running"}


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "openai_api": "OK" if os.getenv("OPENAI_API_KEY") else "Not Configured",
        "version": "1.2.0"
    }



if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app , host = "0.0.0.0", port= 8007 )