import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_PATIENTS_CSV, PDF_REPORTS_DIR, CLEAN_DATASET_EXP1_CSV, CLEAN_DATASET_EXP2_CSV
from .pdf_extractor import process_pdf_folder
from .helper_functions import standardize_protocol
from .helper_functions import assign_patient_ids
from .helper_functions import remove_duplicate_rows
from .helper_functions import anonymize
from .helper_functions import fill_missing_age_with_median
from .helper_functions import fill_missing_AMH
from .helper_functions import assign_n_follicules
from .helper_functions import fill_missing_AFC
from .helper_functions import fill_e2_values

def preprocess_and_update_dataset():
    
    raw_df = pd.read_csv(RAW_PATIENTS_CSV)

    # Extract new patient data from PDFs
    new_patient_df = process_pdf_folder(PDF_REPORTS_DIR)

    # Validate PDF extraction succeeded
    if new_patient_df.empty:
        # No new data extracted, return processed existing data only
        full_df = raw_df.copy()
    else:
        # Validate extracted data has required columns
        required_cols = ['patient_id', 'cycle_number', 'Age', 'Protocol', 'AMH', 'n_Follicles', 'E2_day5', 'AFC', 'Patient Response']
        missing_cols = [col for col in required_cols if col not in new_patient_df.columns]
        
        if missing_cols:
            raise ValueError(f"PDF extraction failed: Missing required columns {missing_cols}")
        
        # Merge with existing data
        full_df = pd.concat([raw_df, new_patient_df], ignore_index=True)

    #HERE I WILL BE CALLING THE FUNCTION FROM data_preprocessing.py
    #procerssing the protocols
    full_df['Protocol'] = full_df['Protocol'].apply(standardize_protocol)
    
    #removing duplicates if any
    full_df = remove_duplicate_rows(full_df, subset_cols=None)
    #anonymizing the data without taking into consideration patients with multiple cycles
    full_df_experience2 = anonymize(full_df)
    #anonymizing the data taking into consideration patients with multiple cycles
    full_df_experience1 = assign_patient_ids(full_df)
    #filling missing Age values with median age
    full_df_experience1 = fill_missing_age_with_median(full_df_experience1, column_name="Age")
    full_df_experience2 = fill_missing_age_with_median(full_df_experience2, column_name="Age")

    full_df_experience1['n_Follicles'] = full_df_experience1.apply(assign_n_follicules, axis=1)
    full_df_experience2['n_Follicles'] = full_df_experience2.apply(assign_n_follicules, axis=1)


    full_df_experience1 = fill_missing_AMH(full_df_experience1)
    full_df_experience2 = fill_missing_AMH(full_df_experience2) 

    full_df_experience2 = fill_missing_AFC(full_df_experience2)
    full_df_experience1 = fill_missing_AFC(full_df_experience1) 

    full_df_experience1 = fill_e2_values(full_df_experience1)
    full_df_experience2 = fill_e2_values(full_df_experience2)



    full_df_experience1 = pd.get_dummies(full_df_experience1, columns=['Protocol'], drop_first=True)

    target_map = {'low': 0, 'optimal': 1, 'high': 2}
    full_df_experience1['Patient Response'] = full_df_experience1['Patient Response'].map(target_map)


    
    full_df_experience2 = pd.get_dummies(full_df_experience2, columns=['Protocol'], drop_first=True)

    target_map = {'low': 0, 'optimal': 1, 'high': 2}
    full_df_experience2['Patient Response'] = full_df_experience2['Patient Response'].map(target_map)
    # 5. OVERWRITE: Update the single processed file
    full_df_experience1.to_csv(CLEAN_DATASET_EXP1_CSV, index=False)
    full_df_experience2.to_csv(CLEAN_DATASET_EXP2_CSV, index=False)

    # Only append new data to raw CSV if extraction succeeded
    if not new_patient_df.empty:
        new_patient_df.to_csv(RAW_PATIENTS_CSV, mode='a', header=False, index=False)


    return full_df_experience1, full_df_experience2