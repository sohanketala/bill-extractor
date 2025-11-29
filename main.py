import os
import io
import json
import base64
import requests
import uvicorn
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from pdf2image import convert_from_bytes
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
GENAI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from environment variables")

genai.configure(api_key=GENAI_API_KEY)

app = FastAPI(title="Bill Extractor API")

# --- Pydantic Models for Response Schema ---

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PageWiseItems(BaseModel):
    page_no: str
    page_type: str  # Bill Detail | Final Bill | Pharmacy
    bill_items: List[BillItem]

class ResponseData(BaseModel):
    pagewise_line_items: List[PageWiseItems]
    total_item_count: int

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class APIResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: ResponseData

# --- Helper Functions ---

def download_file(url: str) -> bytes:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")

def process_document_to_images(file_bytes: bytes, content_type: str = "application/pdf") -> List[Image.Image]:
    """Converts PDF bytes or Image bytes into a list of PIL Images."""
    images = []
    try:
        # Attempt to detect if it's a PDF based on header or extension logic
        # For this snippet, we assume PDF if conversion works, else treat as image
        try:
            images = convert_from_bytes(file_bytes)
        except Exception:
            # If pdf conversion fails, try opening as a standard image
            image = Image.open(io.BytesIO(file_bytes))
            images = [image]
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Failed to process file format: {str(e)}")
    
    return images

# --- Core Logic ---

@app.post("/extract-bill-data", response_model=APIResponse)
async def extract_bill_data(request: dict = Body(...)):
    document_url = request.get("document")
    if not document_url:
        raise HTTPException(status_code=400, detail="Document URL is required")

    # 1. Download and Process Images
    file_bytes = download_file(document_url)
    images = process_document_to_images(file_bytes)

    # 2. Prepare Prompt
    # We instruct the model to strictly follow the output schema
    system_instruction = """
    You are an expert invoice auditor. Your task is to extract line item details from the provided bill images.
    
    RULES:
    1. EXTRACT individual line items only.
    2. EXCLUDE Sub-totals, Taxes, or Final Totals from the 'bill_items' list to avoid double counting.
    3. If an item has a discount, 'item_amount' should be the Net Amount (post-discount).
    4. 'item_rate' is the unit price. 'item_quantity' is the count.
    5. Categorize each page as 'Bill Detail', 'Final Bill', or 'Pharmacy'.
    6. Output strictly in JSON format.
    
    OUTPUT SCHEMA:
    {
        "pagewise_line_items": [
            {
                "page_no": "1",
                "page_type": "Bill Detail",
                "bill_items": [
                    { "item_name": "Tylenol", "item_amount": 10.50, "item_rate": 5.25, "item_quantity": 2.0 }
                ]
            }
        ]
    }
    """

    # 3. Call Gemini 1.5 Flash
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"response_mime_type": "application/json"}
    )
    
    # Prepare content: Text prompt + List of images
    prompt_content = [system_instruction]
    for img in images:
        prompt_content.append(img)
    
    try:
        response = model.generate_content(prompt_content)
        usage = response.usage_metadata
        
        # Parse the JSON string from Gemini
        extracted_json = json.loads(response.text)
        
        # Calculate total item count
        total_count = 0
        if "pagewise_line_items" in extracted_json:
            for page in extracted_json["pagewise_line_items"]:
                total_count += len(page.get("bill_items", []))
        
        extracted_json["total_item_count"] = total_count

        return APIResponse(
            is_success=True,
            token_usage=TokenUsage(
                total_tokens=usage.total_token_count,
                input_tokens=usage.prompt_token_count,
                output_tokens=usage.candidates_token_count
            ),
            data=extracted_json
        )

    except Exception as e:
        # Fallback for errors
        return APIResponse(
            is_success=False,
            token_usage=TokenUsage(total_tokens=0, input_tokens=0, output_tokens=0),
            data=ResponseData(pagewise_line_items=[], total_item_count=0)
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)