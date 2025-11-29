
# AI Bill Line Item Extractor

A robust, AI-powered API designed to extract granular line item details from complex multi-page invoices and bills. This solution utilizes **Google Gemini 2.0 Flash** (Multimodal Vision) to ensure high accuracy in data extraction and total reconciliation, specifically solving the challenge of double-counting subtotals.

## Live Demo

**Base URL:** [https://bill-extractor-q6o3.onrender.com](https://www.google.com/search?q=https://bill-extractor-q6o3.onrender.com)
**Endpoint:** POST /extract-bill-data

*(Note: The API is deployed on Render using a Docker container.)*

-----

## Approach & Methodology

### 1\. Vision-First Extraction (Multimodal AI)

Instead of relying on fragile traditional OCR + RegEx pipelines, this solution processes invoices as **images**. We utilize **Gemini 2.0 Flash**, which natively understands document layouts, tables, and visual hierarchy.

  * **PDF Conversion:** Multi-page PDFs are converted to high-resolution images using `pdf2image` (Poppler) to preserve spatial context.
  * **Context Window:** The model processes all pages of a bill simultaneously to understand the relationship between headers, line items, and totals.

### 2\. Logic for Accuracy & Reconciliation

To meet the specific evaluation criteria of **Accuracy** and **Bill Total Reconciliation**:

  * **Sub-total Exclusion:** The system prompt explicitly instructs the model to identify and **exclude** intermediate summary rows (e.g., "Subtotal", "Tax", "Total Amount") to prevent double-counting.
  * **Net Amount Focus:** For items with discounts, the model extracts the final net amount per line item.
  * **Schema Enforcement:** The output is strictly constrained to a specific JSON schema, ensuring consistent data types (`float` for amounts, `string` for names).

-----

## Tech Stack

  * **Language:** Python 3.11
  * **Framework:** FastAPI
  * **AI Model:** Google Gemini 2.0 Flash (via `google-generativeai` SDK)
  * **PDF Processing:** `pdf2image` / `poppler-utils`
  * **Containerization:** Docker
  * **Deployment:** Render Cloud

-----

## API Documentation

### Extract Bill Data

**URL:** `/extract-bill-data`
**Method:** `POST`

#### Request Body

```json
{
  "document": "https://slicedinvoices.com/pdf/wordpress-pdf-invoice-plugin-sample.pdf"
}
```

#### Success Response (200 OK)

```json
{
  "is_success": true,
  "token_usage": {
    "total_tokens": 850,
    "input_tokens": 720,
    "output_tokens": 130
  },
  "data": {
    "pagewise_line_items": [
      {
        "page_no": "1",
        "page_type": "Bill Detail",
        "bill_items": [
          {
            "item_name": "Web Design Services",
            "item_amount": 1200.00,
            "item_rate": 600.00,
            "item_quantity": 2.0
          }
        ]
      }
    ],
    "total_item_count": 1
  }
}
```

#### Error Response (Examples)

  * `400 Bad Request`: If the document URL is invalid or the file cannot be downloaded.
  * `422 Validation Error`: If the JSON body is missing the "document" field.

-----

## Local Installation & Setup

Follow these steps to run the API on your local machine.

### Prerequisites

1.  Python 3.10 or higher.
2.  **Poppler** installed on your system (Required for PDF conversion):
      * *Mac:* `brew install poppler`
      * *Ubuntu:* `sudo apt-get install poppler-utils`
      * *Windows:* Download binary and add to PATH.
3.  A Google Gemini API Key.

### Steps

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-link>
    cd bill-extractor
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root folder:

    ```env
    GOOGLE_API_KEY=AIzaSy...YourKeyHere...
    ```

4.  **Run the server:**

    ```bash
    uvicorn main:app --reload --port 8000
    ```

5.  **Test:**
    Open `http://localhost:8000/docs` to see the Swagger UI or use Postman.

-----

## Docker Build

This project includes a `Dockerfile` for easy deployment.

```bash
# Build the image
docker build -t bill-extractor .

# Run the container
docker run -p 8000:8000 -e GOOGLE_API_KEY="your_key_here" bill-extractor
```

-----

## Important Note on Testing Data

The problem statement provided a sample dataset URL (`sample_2.png`) hosted on Azure Blob Storage with a SAS signature. **This link has expired or returns a 403 Forbidden error.**

To verify the API functionality, please use standard public invoice URLs.

**Verified Working Samples:**

1.  *PDF Invoice:* `https://slicedinvoices.com/pdf/wordpress-pdf-invoice-plugin-sample.pdf`
2.  *Receipt Image:* `https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png`