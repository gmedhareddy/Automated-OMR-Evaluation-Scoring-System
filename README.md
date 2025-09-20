# Automated OMR Evaluation & Scoring System

## Table of Contents 

- [Overview](#overview)  
- [Problem Statement](#problem-statement)  
- [Features](#features)  
- [Workflow & Architecture](#workflow--architecture)  
- [Tech Stack](#tech-stack)  
- [Getting Started](#getting-started)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Accuracy & Error Tolerance](#accuracy--error-tolerance)  
- [Future Improvements](#future-improvements)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

The Automated OMR Evaluation & Scoring System is designed to fully automate the evaluation of OMR (Optical Mark Recognition) sheets used in placement readiness assessments. By using computer vision techniques, this system aims to transform a manual, error-prone, and slow evaluation process into a fast, accurate, scalable, and auditable pipeline.  

Students fill OMR sheets during exams; these are then captured via mobile phone camera and processed to produce per-subject scores and total scores, with results made available via a web dashboard for evaluators.

---

## Problem Statement

- Examinations have ~100 questions divided into 5 subjects (20 each).  
- Multiple versions of OMR sheets (2-4 versions) to prevent cheating.  
- Manual grading for ~3000 sheets on exam day is expensive in time, human resources, and error-prone.  
- Delays in releasing exam results delay student feedback and learning.  
- Goal is <0.5% error and much faster turnaround (minutes rather than days).

---

## Features

- Upload images of OMR sheets (from mobile phone or scanner).  
- Preprocessing: correct rotation, skew, perspective distortion, adjust for lighting.  
- Version detection / support (multiple sheet layouts).  
- Bubble detection and classification (marked vs unmarked), with fallback / ML model for ambiguous markings.  
- Comparison with answer key per sheet version.  
- Per subject scoring (0-20 each) + total score out of 100.  
- Flagging ambiguous or low-confidence sheets for manual review.  
- Web application dashboard for uploading, viewing, exporting results.  
- Audit trail: store rectified images, overlays, results in JSON, data exports (CSV/Excel).

---

## Workflow & Architecture

1. **Sheet Capture**  
   - Photos are taken via mobile or scanned, uploaded via web UI.

2. **Preprocessing**  
   - Detect corners / fiducial markers or bounding box.  
   - Rectify perspective / correct distortion, skew; normalize lighting.

3. **Bubble Detection & Classification**  
   - Identify grid of bubbles per subject question.  
   - Use thresholding / contour analysis via OpenCV.  
   - ML model component for ambiguous cases (if marking is faint / smudged).

4. **Answer Key Matching & Scoring**  
   - Matches extracted answers vs correct answers depending on sheet version.  
   - Compute subject-wise and total scores.

5. **Result Storage & Dashboard**  
   - Store results and metadata in database.  
   - Provide evaluator interface for uploads, result viewing, exports.  
   - Visual overlays (to show detected vs correct answers) for transparency.

---

## Tech Stack

| Component | Technologies / Libraries |
|-----------|----------------------------|
| Image Processing | OpenCV, NumPy, SciPy, Pillow |
| ML / Classifiers | Scikit-learn or TensorFlow Lite (for ambiguous bubble detection) |
| PDF / Image Handling | PyMuPDF, pdfplumber |
| Backend | Flask or FastAPI |
| Frontend / MVP | Streamlit (for dashboard, uploading, review) |
| Database | SQLite or PostgreSQL |
| Export Formats | JSON, CSV, Excel |

---

## Getting Started

### Prerequisites

- Python 3.8+  
- pip (or conda)  
- Necessary libraries (listed in requirements.txt)  
- A set of sample OMR answer keys for each sheet version  

### Installation

```bash
# Clone the repo
git clone <your_repo_url>
cd <your_project_directory>

# Create a virtual environment
python -m venv omr_env
# Activate it
# On Windows (PowerShell):
.\omr_env\Scripts\Activate
# On Linux/macOS:
source omr_env/bin/activate

# Install dependencies
pip install -r requirements.txt
Usage
Run the web application:

bash
Copy code
streamlit run omr_app.py
(or via backend commands if using Flask/FastAPI)

In the web UI:

Upload the OMR answer key / sheet layout (if required).

Upload one or multiple sheet images.

Trigger evaluation.

See per-subject scores and total score.

View flagged sheets (if ambiguous).

Download/export results as needed (CSV / Excel).

Review in dashboard:

See aggregate stats (class averages, subject level performance, error rates).

Manual corrections if needed (for flagged cases).

Project Structure
text
Copy code
├── omr_app.py                  # Streamlit UI or main entry point
├── engine.py                   # Core logic: image preprocessing, bubble detection, scoring
├── models/                     # ML models (for ambiguous cases)
├── answer_keys/                # JSON or other format answer keys per version
├── utils/                      # Helper modules (image utils, IO, etc.)
├── data/                       # Sample/ test OMR sheet images
├── requirements.txt            # All Python dependencies
├── exports/                    # Folder for exported results (CSV / Excel)
├── database/                   # DB schema or migration scripts
└── README.md                   # This file
Accuracy & Error Tolerance
Target error rate under 0.5% in grading (i.e. fewer than 5 mistakes in 1000 answers).

Ambiguous or low confidence detections should be flagged for manual review.

Performance tested across multiple sheet versions, captured under diverse conditions (lighting, camera angles, etc.).

Future Improvements
Support scanned PDFs or bulk uploads.

Better handling of very poor image quality, smudges, erasures.

Incorporate more robust ML models / deep learning if needed.

Add version control for answer keys.

Implement user authentication / permissions in the dashboard.

Add analytics (student performance trends, subject-wise difficulty).

Contributing
Fork the repository

Create a feature branch (git checkout -b feature/YourFeature)

Make your changes & test thoroughly

Follow coding standards (style, readability)

Submit a pull request with clear description of changes

License
[MIT License] (or whatever license you choose)

Contact
For questions/suggestions, contact:
Name – <gmedharavireddy@gmail>
Innomatics Research Labs





