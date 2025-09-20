import streamlit as st
import cv2 
import numpy as np
import pandas as pd
import base64
from io import BytesIO

# --- 1. OMR ANSWER KEYS ---
# Hardcoded answer keys from the provided CSV files.
# In a real application, these would be loaded from a database.

ANSWER_KEYS = {
    'Set A': {
        'Python': ['a', 'c', 'c', 'c', 'c', 'a', 'c', 'c', 'b', 'c', 'a', 'a', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c'],
        'EDA': ['a', 'd', 'b', 'a', 'c', 'b', 'a', 'b', 'd', 'c', 'c', 'a', 'b', 'c', 'a', 'a', 'd', 'c', 'b', 'c'],
        'SQL': ['c', 'c', 'c', 'b', 'b', 'a', 'c', 'b', 'd', 'a', 'c', 'b', 'c', 'c', 'a', 'd', 'c', 'd', 'c', 'a'],
        'POWER BI': ['b', 'c', 'a', 'b', 'c', 'b', 'b', 'c', 'c', 'b', 'b', 'b', 'd', 'b', 'a', 'b', 'b', 'b', 'c', 'c'],
        'Statistics': ['a', 'b', 'c', 'b', 'c', 'b', 'b', 'b', 'a', 'b', 'c', 'b', 'c', 'b', 'b', 'c', 'd', 'd', 'b', 'd'],
    },
    'Set B': {
        'Python': ['a', 'b', 'd', 'b', 'b', 'd', 'c', 'c', 'a', 'c', 'a', 'b', 'd', 'c', 'c', 'a', 'a', 'c', 'a', 'd'],
        'EDA': ['a', 'a', 'b', 'a', 'b', 'a', 'b', 'b', 'c', 'c', 'b', 'c', 'b', 'c', 'a', 'c', 'b', 'd', 'a', 'a'],
        'SQL': ['b', 'a', 'd', 'b', 'c', 'b', 'b', 'b', 'b', 'b', 'c', 'a', 'c', 'a', 'c', 'b', 'c', 'a', 'b', 'a'],
        'POWER BI': ['b', 'b', 'b', 'd', 'c', 'b', 'b', 'a', 'b', 'b', 'b', 'c', 'a', 'd', 'b', 'c', 'a', 'b', 'a', 'a'],
        'Statistics': ['b', 'c', 'b', 'a', 'c', 'b', 'b', 'a', 'b', 'd', 'c', 'd', 'b', 'b', 'b', 'b', 'd', 'c', 'a', 'b'],
    }
}

# --- 2. IMAGE PROCESSING FUNCTIONS ---

def find_omr_sheet_corners(image):
    """
    Finds the four corners of the OMR sheet in the image.
    This assumes the OMR sheet is the largest rectangular contour.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area in descending order and keep a maximum of 5
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Iterate through contours to find the largest quadrilateral
    for contour in contours:
        # Approximate the contour with a polygon
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # If the contour has 4 vertices, we assume it's the OMR sheet
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def get_bird_eye_view(image, corners):
    """
    Performs perspective correction to get a top-down view of the OMR sheet.
    """
    if corners is None:
        return None
    
    # Order the corners (top-left, top-right, bottom-right, bottom-left)
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left has the smallest sum
        rect[2] = pts[np.argmax(s)]  # Bottom-right has the largest sum
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right has the smallest difference
        rect[3] = pts[np.argmax(diff)] # Bottom-left has the largest difference
        return rect
        
    ordered_corners = order_points(corners)
    
    # Define the destination image size
    (tl, tr, br, bl) = ordered_corners
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_A), int(height_B))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped
    
def get_marked_bubble(question_roi):
    """
    Identifies the marked bubble within a question's ROI.
    Returns the index of the marked bubble (0 for A, 1 for B, etc.)
    or -1 if no bubble is marked or multiple are marked.
    """
    gray_roi = cv2.cvtColor(question_roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Calculate the threshold for a marked bubble (e.g., 50% of area)
    total_area = thresh_roi.shape[0] * thresh_roi.shape[1]
    min_fill = total_area * 0.45
    
    options = []
    # Assuming 4 options (a, b, c, d) with equal spacing
    option_width = question_roi.shape[1] // 4
    for i in range(4):
        x_start = i * option_width
        x_end = (i + 1) * option_width
        bubble_roi = thresh_roi[:, x_start:x_end]
        filled_pixels = cv2.countNonZero(bubble_roi)
        options.append(filled_pixels)
    
    # Find the bubbles with more filled pixels than the threshold
    marked_bubbles = [i for i, pixels in enumerate(options) if pixels > min_fill]

    # Return the marked bubble index if it's unique, otherwise return -1
    if len(marked_bubbles) == 1:
        return marked_bubbles[0]
    else:
        return -1

def process_omr_sheet(image_array, answer_key):
    """
    Main function to process an OMR sheet image and score it.
    """
    # 1. Image Preprocessing and Perspective Correction
    original_image = cv2.imdecode(np.frombuffer(image_array, np.uint8), 1)
    corners = find_omr_sheet_corners(original_image)
    if corners is None:
        return None, "Error: Could not detect OMR sheet. Please try a clearer image.", None
        
    warped = get_bird_eye_view(original_image, corners)
    
    # Create a copy for drawing overlays
    output_image = warped.copy()
    
    # 2. Extract Student ID (assuming it's in a known location)
    # These coordinates are based on a typical OMR sheet template
    height, width, _ = warped.shape
    id_roi = warped[int(height * 0.1):int(height * 0.2), int(width * 0.05):int(width * 0.3)]
    student_id = ""
    # Simplified student ID detection (first 10 digits)
    # A robust system would use a more complex grid detection
    id_grid_width = id_roi.shape[1] // 10
    for i in range(10):
        digit_roi = id_roi[:, i * id_grid_width: (i + 1) * id_grid_width]
        marked_digit = get_marked_bubble(digit_roi)
        if marked_digit != -1:
            student_id += str(marked_digit)
        else:
            student_id += "?"

    # 3. Extract Answers & Score
    scores = {subject: 0 for subject in answer_key.keys()}
    total_score = 0
    
    # Coordinates of the question bubbles
    # Assuming 5 columns, 20 questions each
    bubble_coords = [
        (int(width*0.35), int(height*0.1), int(width*0.55), int(height*0.9)), # Column 1 (Q1-20)
        (int(width*0.55), int(height*0.1), int(width*0.75), int(height*0.9)), # Column 2 (Q21-40)
        (int(width*0.75), int(height*0.1), int(width*0.95), int(height*0.9)), # Column 3 (Q41-60)
        (int(width*0.05), int(height*0.25), int(width*0.25), int(height*0.95)), # Column 4 (Q61-80)
        (int(width*0.25), int(height*0.25), int(width*0.45), int(height*0.95)), # Column 5 (Q81-100)
    ]
    
    subject_map = {
        'Python': (0, 19),
        'EDA': (20, 39),
        'SQL': (40, 59),
        'POWER BI': (60, 79),
        'Statistics': (80, 99)
    }
    
    question_height = (bubble_coords[0][3] - bubble_coords[0][1]) // 20
    
    all_answers = []
    
    for i in range(100):
        question_number = i + 1
        
        # Determine which column and row the question is in
        col_index = i // 20
        row_index = i % 20
        
        # Adjust indices for the new sheet layout
        if question_number >= 61 and question_number <= 80:
            col_index = 3
            row_index = (question_number - 61)
        elif question_number >= 81 and question_number <= 100:
            col_index = 4
            row_index = (question_number - 81)
        elif question_number >= 1 and question_number <= 20:
            col_index = 0
            row_index = (question_number - 1)
        elif question_number >= 21 and question_number <= 40:
            col_index = 1
            row_index = (question_number - 21)
        elif question_number >= 41 and question_number <= 60:
            col_index = 2
            row_index = (question_number - 41)
            
        x_start, y_start, x_end, y_end = bubble_coords[col_index]
        
        # Extract the ROI for the specific question
        question_y_start = y_start + (row_index * question_height)
        question_y_end = question_y_start + question_height
        
        question_roi = warped[question_y_start:question_y_end, x_start:x_end]
        
        # Identify the marked bubble
        marked_index = get_marked_bubble(question_roi)
        
        # Convert index to option letter
        option_letters = ['a', 'b', 'c', 'd']
        marked_answer = option_letters[marked_index] if marked_index != -1 else 'none'
        all_answers.append(marked_answer)
        
        # Draw a circle on the marked answer on the output image
        if marked_index != -1:
            circle_x = x_start + (marked_index * (question_roi.shape[1] // 4)) + (question_roi.shape[1] // 8)
            circle_y = question_y_start + (question_height // 2)
            cv2.circle(output_image, (circle_x, circle_y), 15, (0, 255, 0), 2)
    
    # Score the answers
    for subject, q_range in subject_map.items():
        start_q, end_q = q_range
        correct_answers = answer_key[subject]
        
        subject_score = 0
        for i in range(end_q - start_q + 1):
            student_answer = all_answers[start_q + i]
            correct_answer = correct_answers[i].strip()
            if student_answer == correct_answer:
                subject_score += 1
                
        scores[subject] = subject_score
        total_score += subject_score

    return student_id, scores, output_image

# --- 3. STREAMLIT WEB APPLICATION UI ---
st.set_page_config(layout="wide")

st.title("Innomatics OMR Evaluation & Scoring System")
st.markdown("Automated OMR evaluation with a simple web interface.")

# Corrected and valid base64 encoded data for a small placeholder image.
SAMPLE_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
sample_image_data = base64.b64decode(SAMPLE_IMAGE_BASE64)

col1, col2 = st.columns(2)

with col1:
    st.header("Upload OMR Sheet")
    uploaded_file = st.file_uploader("Choose an OMR sheet image...", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    st.subheader("Or, Process a Sample Sheet")
    process_sample_button = st.button("Process Sample Sheet")

    key_set = st.selectbox(
        "Select Answer Key Set:",
        ('Set A', 'Set B')
    )
    
with col2:
    st.header("Instructions")
    st.markdown("""
    1.  **Upload** an OMR sheet image (taken with a mobile phone camera) or click "Process Sample Sheet" below.
    2.  **Select** the correct exam set (A or B).
    3.  The system will automatically **evaluate** the sheet.
    4.  **View** the detailed score report and a visual overlay of the marked sheet.
    """)

image_to_process = None

if uploaded_file is not None:
    image_to_process = uploaded_file.read()
elif process_sample_button:
    image_to_process = sample_image_data

if image_to_process is not None:
    # Read the image data from the uploaded file
    image_array = np.asarray(bytearray(image_to_process), dtype=np.uint8)

    # Process the image with the selected answer key
    with st.spinner('Evaluating OMR sheet...'):
        student_id, scores, marked_image = process_omr_sheet(image_array, ANSWER_KEYS[key_set])

    st.success("Evaluation complete!")

    if marked_image is not None:
        st.subheader("Results")
        col_results, col_image = st.columns(2)
        
        with col_results:
            st.markdown(f"**Student ID:** `{student_id}`")
            st.markdown(f"**Answer Key:** `{key_set}`")
            st.markdown("---")
            st.markdown("### Subject-wise Scores")
            for subject, score in scores.items():
                st.info(f"**{subject}**: {score}/20")
            st.markdown("---")
            st.subheader(f"Total Score: {sum(scores.values())}/100")
            
        with col_image:
            st.subheader("Processed OMR Sheet")
            st.image(marked_image, caption="Identified and scored bubbles", use_column_width=True)
            
    else:
        st.error(scores) # Display the error message
        st.image(cv2.imdecode(np.frombuffer(image_to_process, np.uint8), 1), caption="Original Image", use_column_width=True)