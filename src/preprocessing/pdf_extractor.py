"""PDF Data Extraction Module

This module extracts patient data from IVF medical reports in PDF format.
It uses regex pattern matching and text parsing to extract clinical parameters.

Author: [Your Name]
Date: 2024
"""

import re
import os
import hashlib
from datetime import datetime

import parse
import pdfplumber
import pandas as pd
from collections import namedtuple


def extract_j5_e2(text):
    """
    Extract E2 (Estradiol) value from day 5 of IVF monitoring table.
    
    Challenge: E2 values are embedded in monitoring tables with multiple measurements
    across different days. We need to specifically extract the day 5 value.
    
    Args:
        text (str): Full text extracted from PDF
        
    Returns:
        float: E2 value on day 5, or None if not found
        
    Note:
        IVF monitoring tables use "J" (Jour = Day in French) as row markers.
        Day 5 is clinically significant for E2 measurement during ovarian stimulation.
    """
    # Step 1: Find all occurrences of "J" (day markers in monitoring table)
    j_matches = list(re.finditer(r"J", text, re.IGNORECASE))
    
    # Step 2: Validate we have enough days recorded (need at least 6 for day 5)
    if len(j_matches) < 6:
        return None  # Not enough monitoring days recorded
    
    # Step 3: Get the 6th "J" which marks day 5 (index 5 because lists start at 0)
    # Monitoring table structure: J1, J2, J3, J4, J5, J6...
    # We want J5 (6th occurrence) for day 5 E2 measurement
    sixth_j = j_matches[5]
    
    # Step 4: Extract text after the day 5 marker
    # This isolates the row containing day 5 data
    content_after_6th_j = text[sixth_j.end():]
    
    # Step 5: Apply regex pattern to find E2 value
    # Pattern explanation:
    # \d{1,2}/\d{1,2} = Date format (e.g., 6/10)
    # .*? = Any characters (non-greedy)
    # \s = Whitespace
    # (\d{3,4}) = Capture 3-4 digit number (E2 value, typically 100-9999)
    # \s = Whitespace
    target_pattern = r"\d{1,2}/\d{1,2}.*?\s(\d{3,4})\s"
    
    match = re.search(target_pattern, content_after_6th_j)
    
    if match:
        return float(match.group(1))  # Return the captured E2 value
    else:
        return None  # E2 value not found in expected format



def extract_patient_data(pdf_path):
    """
    Extract all relevant patient data from a single PDF file.
    
    Extracts 9 clinical parameters:
    - Name, Birth Date, Age, AMH, Protocol, Cycle Number
    - n_Follicles, E2_day5, AFC, Patient Response
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: Dictionary containing all extracted patient data
    """
    # Extract raw text from all pages of the PDF
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    # === DATA EXTRACTION USING REGEX PATTERNS ===
    
    # 1. NAME - Extract patient name for anonymization
    name_match = re.search(r"Name\s*:\s*(.*)", full_text, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else None
    
    # 2. BIRTH DATE
    
    dob_match = re.search(r"Birth date\s*:\s*(\d{1,2}/\d{1,2}/\d{2,4})", full_text, re.IGNORECASE)
    dob_str = dob_match.group(1).strip() if dob_match else None

    # 3. AMH
    amh_match = re.search(r"AMH\s*[:\s]*([\d\.]+)", full_text, re.IGNORECASE)
    amh = float(amh_match.group(1)) if amh_match else None

    # 4. PROTOCOL
    protocol_match = re.search(r"Protocol\s*:\s*(.*)", full_text, re.IGNORECASE)
    protocol_raw = protocol_match.group(1).strip() if protocol_match else "Unknown"

    # 5. CYCLE NUMBER
    cycle_match = re.search(r"Cycle number\s*:\s*(\d+)", full_text, re.IGNORECASE)
    cycle_num = int(cycle_match.group(1)) if cycle_match else 1

    # 6. FOLLICLES
    follicles_match = re.search(r"Number Of follicles\s*=\s*(\d+)", full_text, re.IGNORECASE)
    n_follicles = int(follicles_match.group(1)) if follicles_match else None

    # 7. RESPONSE
    response_match = re.search(r"(optimal|low|high)-response", full_text, re.IGNORECASE)
    response = response_match.group(1).lower() if response_match else None

    # 8. E2 DAY 5
    e2_value = extract_j5_e2(full_text)

    # 9. AFC
    AFC_match = re.search(r"AFC\s*[:\s]*([\d\.]+)", full_text, re.IGNORECASE)
    AFC = int(AFC_match.group(1)) if AFC_match else None

    # === AGE CALCULATION ===
    # Calculate patient age from birth date
    age = None
    if dob_str:
        try:
            # Parse date string (format: DD/MM/YY)
            dob_date = datetime.strptime(dob_str, "%d/%m/%y")
            
            # Handle 2-digit year ambiguity (95 should be 1995, not 2095)
            if dob_date.year > datetime.now().year:
                dob_date = dob_date.replace(year=dob_date.year - 100)
            
            # Calculate exact age accounting for birth month/day
            today = datetime.now()
            age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
        except ValueError:
            pass  # Silently skip invalid dates to prevent pipeline crash

   

    

    # === CONSTRUCT DATA DICTIONARY ===
    # Create structured output with all extracted fields
    row = {
        "patient_id": name,              # Will be anonymized later (25XXX format)
        "cycle_number": cycle_num,       # IVF attempt number
        "Age": age,                      # Patient age in years
        "Protocol": protocol_raw,        # Ovarian stimulation protocol
        "AMH": amh,                      # Anti-Müllerian Hormone level
        "n_Follicles": n_follicles,     # Number of follicles at last monitoring
        "E2_day5": e2_value,            # Estradiol level on day 5
        "AFC" : AFC,                     # Antral Follicle Count
        "Patient Response": response     # Target: low/optimal/high
    }
    
    return row
def process_pdf_folder(folder_path):
    """
    Batch process all PDF files in a folder.
    
    Scans the specified folder for PDF files, extracts data from each,
    and consolidates into a single DataFrame.
    
    Args:
        folder_path (str): Path to folder containing PDF files
        
    Returns:
        pd.DataFrame: Consolidated patient data from all PDFs
        
    Note:
        Failed extractions are silently skipped to prevent pipeline crash.
        This allows partial success when processing multiple files.
    """
    all_patients_data = []  # Accumulator for all extracted patient records
    
    # 1. Loop through every file in the folder
    for filename in os.listdir(folder_path):
        
        # 2. Check if it is actually a PDF
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            
            try:
                # 3. Call your extraction function
                data = extract_patient_data(full_path)
                
                # 4. Append the result to our list
                if data: # Only append if data was found
                    all_patients_data.append(data)
                    
            except Exception as e:
                # This prevents one bad PDF from crashing the whole script
                pass  # Silently skip failed PDFs

    # 5. Convert list of dicts to DataFrame
    if all_patients_data:
        return pd.DataFrame(all_patients_data)
    else:
        return pd.DataFrame() # Return empty DF if nothing found


if __name__ == "__main__":
    # Test on the file
    #new_patient_data = extract_patient_data("data/raw/pdf_reports/sample2.pdf")
    #print("\n\n--- Processing one PDF File ---\n")
    #print("Extracted Data:", new_patient_data)
    #new_row_df = pd.DataFrame([new_patient_data])
    #print(new_row_df)
   
    new_patients = process_pdf_folder("data/raw/pdf_reports")
    print("All Extracted Data:")
    print(new_patients)
