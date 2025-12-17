# api.py
import shutil
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import UPLOAD_DIR, STATIC_DIR, RESULTS_DIR
from src.pipeline import run_pipeline

# Log security events
logger = logging.getLogger(__name__)

# Upload file size and type limits
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI()

# Set up CORS to allow only specific origins and methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow required methods
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return "<h1>Upload index.html to static folder</h1>"

@app.post("/predict")
async def predict(file: UploadFile = File(...), explain: bool = Query(False)):
    """
    Predict anemia from uploaded image.
    
    Query Parameters:
        explain (bool): If true, generate Grad-CAM heatmap (default: false)
    
    Response includes heatmap_url only if explain=true and generation succeeds.
    """
    print(f"[API] Received POST /predict with explain={explain} (type: {type(explain).__name__})")
    
    # Check if filename exists and is a supported format
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not supported. Allowed: jpg, jpeg, png, webp")
    
    # Make sure the file is actually an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Use a random filename to avoid conflicts and security issues
    secure_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / secure_filename
    
    # Check file size before writing to disk
    try:
        total_size = 0
        with file_path.open("wb") as buffer:
            while chunk := await file.read(8192):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    # Delete the incomplete file
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File too large. Maximum size: 5MB")
                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        # Track what went wrong
        logger.error(f"File upload error: {e}", exc_info=True)
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Failed to upload file")

    # Run Pipeline with optional Grad-CAM
    try:
        print(f"[API] Processing image with explain={explain}")
        result = run_pipeline(file_path, explain=explain)
        print(f"[API] Pipeline result keys: {result.keys()}")
        print(f"[API] Has heatmap_path: {'heatmap_path' in result}")
    except ValueError as e:
        # Pipeline found an issue with the image
        logger.warning(f"Pipeline validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image or processing failed")
    except Exception as e:
        # Log the full error but tell user something generic
        logger.error(f"Pipeline processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to process image. Please try again.")
    finally:
        # Clean up uploaded file after processing
        file_path.unlink(missing_ok=True)

    # Build the response for the frontend
    # Convert absolute file path to URL: /static/results/filename.jpg
    filename = Path(result["boxed_image_path"]).name
    
    response = {
        "label": result["label"],
        "confidence": result["confidence"],
        "boxed_image_url": f"/static/results/{filename}",
        "note": result["note"]
    }
    
    # Add heatmap to response if it was generated
    if "heatmap_path" in result:
        heatmap_filename = Path(result["heatmap_path"]).name
        response["heatmap_url"] = f"/static/results/{heatmap_filename}"
        print(f"[API] Heatmap URL added to response: {response['heatmap_url']}")
    else:
        print(f"[API] No heatmap_path in result. Result keys: {result.keys()}")
    
    print(f"[API] Returning response with keys: {response.keys()}")
    return response