# ANEMO  
**Automated Conjunctival Analysis for Anemia Risk Screening**

> Investigational ML-based screening tool for estimating anemia risk using conjunctival images.  
> **Not a diagnostic device.**

---

## Overview

**ANEMO** is a computer vision–based research tool that estimates anemia risk by analyzing images of the lower conjunctiva (inner eyelid).

It combines object detection, image preprocessing, deep learning classification, and visual explainability (Grad-CAM) to provide transparent, non-diagnostic insights.

This system is intended strictly for **screening and research purposes** and does **not** replace clinical blood tests such as CBC or hemoglobin assays.

---

## Key Features

- 📸 Image-based screening via upload or camera
- 🎯 YOLOv8-based lower conjunctiva detection
- 🧠 CNN classification using MobileNetV2
- 🔍 Grad-CAM visual explainability
- 📊 Anemia Risk Index with confidence score
- 🗂️ Session logs with detailed breakdowns
- 📄 Downloadable screening report
- 🔒 Security-hardened API (validation, limits, cleanup)

---

## What ANEMO Is (and Is Not)

### ✅ What it is
- A research and screening tool
- An applied ML medical imaging project
- A transparent system showing model attention

### ❌ What it is not
- A diagnostic device
- A hemoglobin estimator
- A replacement for clinical testing
- A medical decision-making system

---

## System Architecture

### Processing Pipeline

1. Image Input
2. Conjunctiva Detection (YOLOv8)
3. Image Preprocessing (OpenCV)
4. CNN Classification (MobileNetV2)
5. Explainability (Grad-CAM)
6. Result Aggregation & Reporting

---

## Tech Stack

### Backend
- Python
- FastAPI
- TensorFlow / Keras
- YOLOv8
- OpenCV

### Frontend
- HTML / CSS
- Vanilla JavaScript
- Responsive dashboard UI

---

## Explainability (Grad-CAM)

ANEMO uses **Grad-CAM** to visualize which regions of the image most influenced the model’s prediction.

> ⚠️ Grad-CAM highlights **model attention**, not physiological measurements such as blood flow or hemoglobin levels.

- Applied to the classification input image (224×224)
- Rendered as a grayscale-focused overlay for clarity
- Intended purely for transparency and interpretability

---

## Security & Stability

Implemented safeguards include:

- File extension and MIME validation
- Upload size limits
- UUID-based filenames (path traversal protection)
- Automatic cleanup of temporary files
- Restricted CORS configuration
- Generic client-facing errors with detailed server-side logging
- Image dimension clamping to prevent memory exhaustion

---

## Installation

```bash
git clone ""
cd anemo

python -m venv anemia-venv
source anemia-venv/bin/activate

pip install -r requirements.txt

uvicorn src.api:app --reload
