# Enhanced main.py with FIXED synchronization, item management, and remove functionality
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, cv2, re, json
from typing import List, Dict, Any, Optional
import ollama
import numpy as np
from datetime import datetime
import logging
import traceback
import asyncio
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Receipt Processor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = "uploads"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"

for directory in [UPLOAD_DIR, TEMPLATES_DIR, STATIC_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible"""
    try:
        models = ollama.list()
        logger.info(f"Ollama connected. Available models: {len(models.get('models', []))}")
        return True
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        return False

def enhance_image_for_ocr(image_path: str) -> Optional[str]:
    """Enhanced preprocessing for noisy/distorted receipts"""
    try:
        logger.info(f"Enhancing image: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to read image")
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Resize for better OCR
        height, width = enhanced.shape
        scale_factor = max(1.5, 2000 / max(height, width))
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized = cv2.resize(enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Add border
        bordered = cv2.copyMakeBorder(resized, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
        
        # Convert back to 3-channel
        final_img = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        
        out_path = image_path.replace(".jpg", "_enhanced.jpg").replace(".png", "_enhanced.png")
        cv2.imwrite(out_path, final_img)
        logger.info(f"Enhanced image saved to: {out_path}")
        
        return out_path
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        return None

def extract_date_and_store_enhanced(text: str) -> Dict[str, str]:
    """Enhanced date and store extraction with better patterns"""
    logger.info("Extracting store and date information...")
    
    # Enhanced date patterns
    date_patterns = [
        r"(?:Date|DATE|Date:)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(?:Date|DATE|Date:)\s*:?\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2})",
        r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})"
    ]
    
    date_str = "Unknown"
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            logger.info(f"Found date: {date_str}")
            break
    
    # Enhanced store name patterns
    store_patterns = [
        r"(?:\*\*Store Name:\*\*|\*\*Store:\*\*|Store Name:|Store:)\s*(.+?)(?:\n|\*\*|$)",
        r"^([A-Z][A-Za-z\s&]+(?:ELECTRONICS|STORE|MART|SHOP|MARKET|SUPERMARKET|GROCERY|HUB|CENTER))",
        r"([A-Z][A-Za-z\s&]+(?:Electronics|Store|Mart|Shop|Market|Supermarket|Grocery|Hub|Center))",
        r"^\*?\*?([A-Z\s&]+(?:STORE|MART|SHOP|MARKET|SUPERMARKET|GROCERY|ELECTRONICS|HUB))",
        r"^([A-Z][A-Za-z\s]{3,30}(?:Store|Electronics|Mart|Shop))",
        r"^([A-Z][A-Za-z\s]{5,25})"
    ]
    
    store_name = "Unknown Store"
    lines = text.split('\n')
    
    # First try structured extraction
    for pattern in store_patterns[:3]:
        for line in lines[:5]:
            match = re.search(pattern, line.strip(), re.IGNORECASE | re.MULTILINE)
            if match:
                store_name = match.group(1).strip()
                logger.info(f"Found store (pattern): {store_name}")
                break
        if store_name != "Unknown Store":
            break
    
    # If no structured store found, try to find business-like names in first few lines
    if store_name == "Unknown Store":
        for line in lines[:5]:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # Skip lines that are clearly not store names
            if any(skip in line.lower() for skip in ['date', 'time', 'receipt', 'invoice', 'bill', 'tax', 'total']):
                continue
            
            # Look for capitalized words that could be store names
            if re.match(r'^[A-Z][A-Za-z\s&]{4,30}', line):
                # Additional filtering
                words = line.split()
                if len(words) >= 2 and len(words) <= 5:
                    store_name = line
                    logger.info(f"Found store (heuristic): {store_name}")
                    break
    
    # Clean up store name
    if store_name and store_name != "Unknown Store":
        store_name = re.sub(r'\*+', '', store_name)
        store_name = re.sub(r'\s+', ' ', store_name)
        store_name = store_name.strip().title()
    
    logger.info(f"Final extracted - Store: {store_name}, Date: {date_str}")
    
    return {
        "date": date_str,
        "store": store_name
    }

async def extract_receipt_data(image_path: str) -> Dict[str, Any]:
    """Extract all receipt data including date, store, and items"""
    try:
        logger.info(f"Processing receipt: {image_path}")
        
        # Check if Ollama is available
        if not check_ollama_connection():
            return {"error": "Ollama service is not available. Please ensure Ollama is running."}
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        # Enhanced OCR prompt with better instructions
        vision_prompt = """
        You are an advanced OCR system. Extract ALL visible text from this receipt/invoice image with high accuracy.
        
        Pay special attention to:
        1. Store name (usually at the top)
        2. Date information 
        3. All item names, quantities, and prices
        4. Preserve exact formatting and line structure
        
        Return the raw extracted text exactly as it appears, maintaining original structure and formatting.
        Do not interpret or reformat - just extract what you see.
        """

        logger.info("Sending image to vision model...")
        try:
            vision_response = ollama.chat(
                model="llama3.2-vision",
                messages=[{"role": "user", "content": vision_prompt, "images": [image_bytes]}],
                options={"temperature": 0.1}
            )
            raw_text = vision_response['message']['content']
            logger.info("Raw OCR text extracted successfully")
            logger.info(f"Raw text preview: {raw_text[:500]}...")
            
        except Exception as e:
            logger.error(f"Vision model error: {e}")
            return {"error": f"Vision processing failed: {str(e)}"}

        # Extract date and store with enhanced function
        metadata = extract_date_and_store_enhanced(raw_text)
        
        # Enhanced item extraction prompt
        item_extraction_prompt = """
        You are an expert receipt parser. Extract ALL purchased items from this receipt/invoice text.
        
        For each item, find:
        - Item name (clean, readable product name)
        - Quantity (number purchased, default to 1 if not specified)  
        - Unit price (price per item in numerical format only)
        
        PARSING RULES:
        1. Look for structured data with items, quantities, and prices
        2. Handle various formats: receipts, invoices, purchase orders
        3. Skip headers, totals, subtotals, taxes, shipping
        4. For technical descriptions, extract main product name
        5. Handle quantity formats: "1", "01", "1x", "2 @", etc.
        6. Extract only numerical price values (remove ₹, $, commas, etc.)
        7. Convert quantity "01" to "1"
        
        EXAMPLES:
        - "Camera 1 $899.00" → "Camera|1|899.00"
        - "Dell Laptop Intel i5 8GB RAM 512GB SSD 1 52500.00" → "Dell Laptop|1|52500.00"
        - "Fitness Tracker - Qty: 1 - Price: $129" → "Fitness Tracker|1|129.00"
        
        Format each item EXACTLY as: ItemName|Quantity|Price
        Only output the item lines, nothing else.
        """

        logger.info("Extracting items with Mistral...")
        try:
            parsed_response = ollama.chat(
                model="mistral",
                messages=[
                    {"role": "system", "content": item_extraction_prompt},
                    {"role": "user", "content": f"Receipt text:\n{raw_text}"}
                ],
                options={"temperature": 0.1}
            )
            
            parsed_content = parsed_response['message']['content']
            logger.info("Item extraction completed")
            logger.info(f"Parsed response: {parsed_content}")
            
        except Exception as e:
            logger.error(f"Mistral parsing error: {e}")
            return {"error": f"Item parsing failed: {str(e)}"}

        # Parse items from response
        items = parse_items_from_response(parsed_content)
        
        logger.info(f"Final extracted items: {items}")

        return {
            "date": metadata["date"],
            "store": metadata["store"],
            "items": items,
            "raw_text": raw_text,
            "parsed_response": parsed_content
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in extract_receipt_data: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Processing failed: {str(e)}"}

def parse_items_from_response(parsed_content: str) -> List[Dict[str, Any]]:
    """Enhanced parsing for various document types"""
    items = []
    
    def clean_price(price_str):
        """Clean price string to extract numerical value"""
        if not price_str:
            return 0.0
            
        cleaned = str(price_str).strip()
        cleaned = re.sub(r'[₹\$€£Rs\.INR\(\)A-Za-z\s]', '', cleaned)
        cleaned = cleaned.replace(',', '')
        
        if cleaned.count('.') > 1:
            parts = cleaned.split('.')
            cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
        
        try:
            return float(cleaned) if cleaned and cleaned.replace('.', '').replace('-', '').isdigit() else 0.0
        except ValueError:
            logger.warning(f"Could not parse price: '{price_str}' -> '{cleaned}'")
            return 0.0
    
    def clean_item_name(name_str):
        """Enhanced item name cleaning"""
        if not name_str:
            return ""
        
        name = str(name_str).strip()
        name = re.sub(r'^\d+\.\s*', '', name)
        name = re.sub(r'^\*\s*', '', name)
        
        # Handle laptop/computer naming
        if any(term in name.lower() for term in ['laptop', 'dell', 'hp', 'lenovo', 'computer']):
            laptop_match = re.search(r'(Dell|HP|Lenovo|Acer|Asus|Apple)\s*([a-zA-Z0-9\s]*?)(?:\s+[iI][35789]|Core|AMD|Intel|,|\d{4,})', name, re.IGNORECASE)
            if laptop_match:
                brand = laptop_match.group(1)
                model = laptop_match.group(2).strip()[:20]
                return f"{brand} {model} Laptop".strip()
            elif 'laptop' in name.lower():
                return "Laptop"
        
        # Remove technical specifications
        name = re.sub(r'\b[A-Z0-9]{6,}\b', '', name)
        name = re.sub(r'\b\d{4,}\b', '', name)
        name = re.sub(r'\s+', ' ', name)
        
        # Limit length
        words = name.split()
        if len(words) > 4:
            name = ' '.join(words[:4])
        
        return name.title().strip()
    
    def clean_quantity(qty_str):
        """Enhanced quantity cleaning"""
        if not qty_str:
            return 1.0
        
        qty = str(qty_str).strip()
        
        if qty.isdigit() and qty.startswith('0'):
            qty = qty.lstrip('0') or '1'
        
        qty = re.sub(r'[x@A-Za-z\s]', '', qty)
        
        try:
            return float(qty) if qty and qty.replace('.', '').isdigit() else 1.0
        except ValueError:
            return 1.0
    
    logger.info(f"Parsing response content: {parsed_content[:200]}...")
    
    # Parse pipe-separated format
    lines = parsed_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if any(skip_word in line.lower() for skip_word in ['total', 'subtotal', 'tax', 'payment', 'shipping']):
            continue
        
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    name = clean_item_name(parts[0])
                    quantity = clean_quantity(parts[1])
                    price = clean_price(parts[2])
                    
                    if name and quantity > 0 and price > 0:
                        items.append({
                            "name": name,
                            "quantity": quantity,
                            "unit_price": price,
                            "total_price": round(quantity * price, 2)
                        })
                        logger.info(f"Parsed item: {name} x{quantity} @ ₹{price}")
                        
                except Exception as e:
                    logger.warning(f"Failed to parse line: {line} - {e}")
                    continue
    
    # Fallback parsing if no structured items found
    if not items:
        logger.warning("No structured items found, trying fallback parsing...")
        
        for line in lines:
            line = line.strip()
            if len(line) < 5:
                continue
                
            if any(skip in line.lower() for skip in ['address', 'phone', 'email', 'total', 'tax']):
                continue
            
            if re.search(r'[a-zA-Z]{3,}', line) and re.search(r'\d{2,}', line):
                price_matches = re.findall(r'\b(\d{3,}\.?\d*)\b', line)
                if price_matches:
                    try:
                        name = clean_item_name(line)
                        price = clean_price(price_matches[-1])
                        
                        if name and price > 10:  # Reasonable price threshold
                            items.append({
                                "name": name,
                                "quantity": 1.0,
                                "unit_price": price,
                                "total_price": price
                            })
                            logger.info(f"Fallback parsed: {name} @ ₹{price}")
                            
                    except Exception as e:
                        logger.debug(f"Fallback parsing failed: {line} - {e}")
                        continue
    
    logger.info(f"Total items parsed: {len(items)}")
    return items

class ReceiptDatabase:
    """FIXED in-memory database with proper item management and synchronization"""
    def __init__(self):
        self.receipts = []
        self.combined_items = {}
        self.item_id_counter = 0
    
    def add_receipt(self, receipt_data: Dict[str, Any]):
        """Add a new receipt and update combined items - FIXED"""
        receipt_id = len(self.receipts)
        receipt_data["id"] = receipt_id
        self.receipts.append(receipt_data)
        
        logger.info(f"Adding receipt {receipt_id} with {len(receipt_data.get('items', []))} items")
        
        for item in receipt_data.get("items", []):
            item_key = self._normalize_item_name(item["name"])
            
            if item_key in self.combined_items:
                # Update existing item
                existing = self.combined_items[item_key]
                existing["total_quantity"] += item["quantity"]
                existing["prices"].append(item["unit_price"])
                existing["average_price"] = round(sum(existing["prices"]) / len(existing["prices"]), 2)
                existing["receipts"].append(receipt_id)
                logger.info(f"Updated existing item: {existing['name']} - new avg price: {existing['average_price']}")
            else:
                # Create new item
                self.combined_items[item_key] = {
                    "id": self.item_id_counter,
                    "name": item["name"],  # Keep original name formatting
                    "total_quantity": item["quantity"],
                    "prices": [item["unit_price"]],
                    "average_price": item["unit_price"],
                    "receipts": [receipt_id]
                }
                logger.info(f"Added new item: {item['name']} @ ₹{item['unit_price']}")
                self.item_id_counter += 1
        
        logger.info(f"Combined items now: {len(self.combined_items)}")
    
    def _normalize_item_name(self, name: str) -> str:
        """Normalize item name for matching while preserving original"""
        if not name:
            return ""
        return name.lower().strip()
    
    def get_summary(self):
        """Get combined items summary - FIXED with better error handling"""
        items_list = []
        total_cost = 0
        
        logger.info(f"Generating summary from {len(self.combined_items)} combined items")
        
        # FIXED: Ensure all items are included in summary
        for item_key, data in self.combined_items.items():
            try:
                # Ensure all required fields exist
                name = data.get("name", "Unknown Item")
                quantity = data.get("total_quantity", 0)
                avg_price = data.get("average_price", 0)
                
                if not name or not name.strip():
                    name = "Unknown Item"
                
                item_total = round(quantity * avg_price, 2)
                total_cost += item_total
                
                items_list.append({
                    "id": data.get("id", 0),
                    "name": name,
                    "quantity": quantity,
                    "average_price": avg_price,
                    "total_price": item_total,
                    "receipt_count": len(data.get("receipts", []))
                })
                
                logger.info(f"Summary item: {name} x{quantity} @ ₹{avg_price} = ₹{item_total}")
                
            except Exception as e:
                logger.error(f"Error processing item {item_key}: {e}")
                continue
        
        # Sort by name for consistent ordering
        items_list = sorted(items_list, key=lambda x: x["name"])
        
        summary = {
            "items": items_list,
            "total_cost": round(total_cost, 2),
            "receipt_count": len(self.receipts)
        }
        
        logger.info(f"Generated summary: {len(items_list)} items, ₹{total_cost:.2f} total")
        return summary
    
    def update_item_by_name(self, old_name: str, new_name: str, quantity: float, price: float):
        """Update an item's name, quantity and price - FIXED"""
        old_key = self._normalize_item_name(old_name)
        new_key = self._normalize_item_name(new_name)
        
        logger.info(f"Updating item: '{old_name}' -> '{new_name}', qty: {quantity}, price: {price}")
        
        if old_key not in self.combined_items:
            logger.error(f"Item not found: {old_name} (key: {old_key})")
            logger.info(f"Available items: {list(self.combined_items.keys())}")
            return False
        
        item_data = self.combined_items[old_key]
        
        # If name changed, handle key change
        if old_key != new_key:
            # Check if new key already exists
            if new_key in self.combined_items and new_key != old_key:
                # Merge with existing item
                existing_item = self.combined_items[new_key]
                existing_item["total_quantity"] += quantity
                existing_item["receipts"].extend(item_data["receipts"])
                existing_item["prices"].append(price)
                existing_item["average_price"] = round(sum(existing_item["prices"]) / len(existing_item["prices"]), 2)
                # Remove old item
                del self.combined_items[old_key]
                logger.info(f"Merged items: {old_name} -> {new_name}")
            else:
                # Move to new key
                del self.combined_items[old_key]
                self.combined_items[new_key] = item_data
                logger.info(f"Moved item: {old_key} -> {new_key}")
        
        # Update the item data
        item_data["name"] = new_name
        item_data["total_quantity"] = quantity
        item_data["average_price"] = price
        
        logger.info(f"Item updated successfully: {new_name}")
        return True
    
    def remove_item_by_name(self, item_name: str):
        """Remove an item completely - NEW"""
        item_key = self._normalize_item_name(item_name)
        
        logger.info(f"Removing item: '{item_name}' (key: {item_key})")
        
        if item_key not in self.combined_items:
            logger.error(f"Item not found for removal: {item_name} (key: {item_key})")
            logger.info(f"Available items: {list(self.combined_items.keys())}")
            return False
        
        # Remove the item
        removed_item = self.combined_items.pop(item_key)
        logger.info(f"Successfully removed item: {removed_item['name']}")
        return True
    
    def get_receipt_by_id(self, receipt_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific receipt by ID"""
        if 0 <= receipt_id < len(self.receipts):
            return self.receipts[receipt_id]
        return None
    
    def clear(self):
        """Clear all data"""
        logger.info("Clearing all database data")
        self.receipts = []
        self.combined_items = {}
        self.item_id_counter = 0

# Global database instance
db = ReceiptDatabase()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main application page"""
    index_path = Path(TEMPLATES_DIR) / "ind.html"
    if not index_path.exists():
        with open(index_path, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Receipt Processor</title>
</head>
<body>
    <h1>Receipt Processor Backend Running</h1>
    <p>Backend is running successfully. Please use the frontend HTML file.</p>
    <p>API endpoints available:</p>
    <ul>
        <li>POST /upload - Upload receipts</li>
        <li>POST /upload-progressive - Progressive upload</li>
        <li>GET /summary - Get summary</li>
        <li>POST /update-item - Update item</li>
        <li>DELETE /remove-item - Remove item</li>
        <li>DELETE /clear - Clear data</li>
        <li>GET /debug - Debug info</li>
        <li>GET /health - Health check</li>
    </ul>
</body>
</html>""")
    
    return templates.TemplateResponse("ind.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    ollama_status = check_ollama_connection()
    return JSONResponse({
        "status": "healthy",
        "ollama_connected": ollama_status,
        "receipts_count": len(db.receipts),
        "items_count": len(db.combined_items)
    })

@app.get("/debug")
async def debug_endpoint():
    """Debug endpoint to check internal state"""
    try:
        summary = db.get_summary()
        return JSONResponse({
            "ollama_connected": check_ollama_connection(),
            "receipts_count": len(db.receipts),
            "combined_items_count": len(db.combined_items),
            "combined_items": {k: v["name"] for k, v in db.combined_items.items()},
            "summary": summary,
            "receipts": [{"id": r.get("id"), "store": r.get("store"), "date": r.get("date"), "items_count": len(r.get("items", []))} for r in db.receipts]
        })
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-progressive")
async def upload_receipts_progressive(files: List[UploadFile] = File(...)):
    """Process receipts progressively and stream results"""
    try:
        logger.info(f"Processing {len(files)} receipt files progressively")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        file_data_list = []
        for file in files:
            try:
                content = await file.read()
                file_data_list.append({
                    "filename": file.filename,
                    "content": content,
                    "content_type": file.content_type
                })
                logger.info(f"Read file: {file.filename} ({len(content)} bytes)")
            except Exception as e:
                logger.error(f"Failed to read file {file.filename}: {e}")
                file_data_list.append({
                    "filename": file.filename,
                    "content": None,
                    "content_type": file.content_type,
                    "error": str(e)
                })
        
        async def generate_progressive_results():
            results = []
            
            for i, file_data in enumerate(file_data_list):
                try:
                    filename = file_data["filename"]
                    
                    progress_data = {
                        "type": "progress",
                        "current": i + 1,
                        "total": len(file_data_list),
                        "filename": filename,
                        "status": "processing"
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    
                    if file_data.get("error"):
                        result = {
                            "filename": filename,
                            "status": "error",
                            "message": f"Failed to read file: {file_data['error']}"
                        }
                    else:
                        result = await process_file_from_content(file_data)
                    
                    results.append(result)
                    
                    # Get current summary after each successful processing
                    current_summary = db.get_summary()
                    
                    individual_result = {
                        "type": "individual_result",
                        "result": result,
                        "summary": current_summary
                    }
                    yield f"data: {json.dumps(individual_result)}\n\n"
                    
                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    logger.error(traceback.format_exc())
                    error_result = {
                        "filename": filename,
                        "status": "error",
                        "message": str(e)
                    }
                    results.append(error_result)
                    
                    individual_result = {
                        "type": "individual_result",
                        "result": error_result,
                        "summary": db.get_summary()
                    }
                    yield f"data: {json.dumps(individual_result)}\n\n"
            
            final_summary = db.get_summary()
            
            final_result = {
                "type": "complete",
                "processing_results": results,
                "summary": final_summary
            }
            yield f"data: {json.dumps(final_result)}\n\n"
        
        return StreamingResponse(
            generate_progressive_results(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        logger.error(f"Progressive upload endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def process_file_from_content(file_data: Dict[str, Any]):
    """Process a file from pre-read content data"""
    try:
        filename = file_data["filename"]
        content = file_data["content"]
        content_type = file_data["content_type"]
        
        if not filename:
            return {"filename": "unknown", "status": "error", "message": "No filename provided"}
        
        if not content_type or not content_type.startswith('image/'):
            return {"filename": filename, "status": "error", "message": "Invalid file type. Please upload image files only."}
        
        if not content:
            return {"filename": filename, "status": "error", "message": "Empty file"}
        
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"Saved file to disk: {file_path} ({len(content)} bytes)")
        
        enhanced_path = enhance_image_for_ocr(file_path)
        if not enhanced_path:
            return {"filename": filename, "status": "error", "message": "Failed to process image"}
        
        receipt_data = await extract_receipt_data(enhanced_path)
        
        if "error" in receipt_data:
            return {"filename": filename, "status": "error", "message": receipt_data["error"]}
        
        # Add to database
        db.add_receipt(receipt_data)
        
        return {
            "filename": filename,
            "status": "success",
            "items_found": len(receipt_data.get("items", [])),
            "store": receipt_data.get("store", "Unknown"),
            "date": receipt_data.get("date", "Unknown"),
            "items": receipt_data.get("items", [])
        }
        
    except Exception as e:
        filename = file_data.get("filename", "unknown")
        logger.error(f"Error processing {filename}: {e}")
        logger.error(traceback.format_exc())
        return {"filename": filename, "status": "error", "message": str(e)}

@app.post("/upload")
async def upload_receipts(files: List[UploadFile] = File(...)):
    try:
        logger.info(f"Processing {len(files)} receipt files")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        processing_results = []
        for file in files:
            content = await file.read()
            file_data = {
                "filename": file.filename,
                "content": content,
                "content_type": file.content_type
            }
            result = await process_file_from_content(file_data)
            processing_results.append(result)
        
        summary = db.get_summary()
        return JSONResponse({
            "processing_results": processing_results,
            "summary": summary
        })
        
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/summary")
async def get_summary():
    """Get current summary of all receipts"""
    try:
        return JSONResponse(db.get_summary())
    except Exception as e:
        logger.error(f"Summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-item")
async def update_item(request: Request):
    """Update an item's name, quantity and price - FIXED"""
    try:
        data = await request.json()
        logger.info(f"Update item request: {data}")
        
        old_name = data.get("old_name")
        new_name = data.get("new_name")
        quantity = float(data.get("quantity", 0))
        price = float(data.get("price", 0))
        
        if not old_name or not new_name:
            raise HTTPException(status_code=400, detail="Both old_name and new_name are required")
        
        if quantity <= 0 or price <= 0:
            raise HTTPException(status_code=400, detail="Quantity and price must be positive")
        
        success = db.update_item_by_name(old_name, new_name, quantity, price)
        
        if success:
            updated_summary = db.get_summary()
            logger.info(f"Item update successful, returning summary with {len(updated_summary['items'])} items")
            return JSONResponse({
                "status": "success", 
                "message": f"Item '{new_name}' updated successfully",
                "summary": updated_summary
            })
        else:
            raise HTTPException(status_code=404, detail=f"Item '{old_name}' not found")
            
    except ValueError as e:
        logger.error(f"Value error in update_item: {e}")
        raise HTTPException(status_code=400, detail="Invalid quantity or price format")
    except Exception as e:
        logger.error(f"Update item error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/remove-item")
async def remove_item(request: Request):
    """Remove an item completely - NEW"""
    try:
        data = await request.json()
        logger.info(f"Remove item request: {data}")
        
        item_name = data.get("item_name")
        
        if not item_name:
            raise HTTPException(status_code=400, detail="item_name is required")
        
        success = db.remove_item_by_name(item_name)
        
        if success:
            updated_summary = db.get_summary()
            logger.info(f"Item removal successful, returning summary with {len(updated_summary['items'])} items")
            return JSONResponse({
                "status": "success", 
                "message": f"Item '{item_name}' removed successfully",
                "summary": updated_summary
            })
        else:
            raise HTTPException(status_code=404, detail=f"Item '{item_name}' not found")
            
    except Exception as e:
        logger.error(f"Remove item error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_data():
    """Clear all receipt data"""
    try:
        db.clear()
        
        # Also clean up uploaded files
        try:
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            logger.info("Uploaded files cleaned up")
        except Exception as cleanup_error:
            logger.warning(f"File cleanup warning: {cleanup_error}")
        
        return JSONResponse({"status": "success", "message": "All data cleared"})
        
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)