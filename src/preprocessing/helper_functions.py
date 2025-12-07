"""Helper Functions for Data Preprocessing

This module contains utility functions for:
- Protocol standardization
- Duplicate removal
- Patient anonymization (two approaches)
- Missing data imputation

Author: [Your Name]
Date: 2024
"""

import pandas as pd
from .prediction_models import predicting_AMH_from_n_follicules

# === PROTOCOL STANDARDIZATION ===
def standardize_protocol(value):
    """
    Standardize protocol names to consistent format.
    
    Handles variations in protocol naming:
    - "flex anta", "Flex Antago" -> "Flexible Antagonist"
    - "fix antag", "Fixed Anta" -> "Fixed Antagonist"
    - "ago", "agoni" -> "Agonist"
    
    Args:
        value: Protocol name (any format)
        
    Returns:
        str: Standardized protocol name
    """
    s = str(value).lower()
    if 'flex' in s: return 'Flexible Antagonist'
    if 'fix' in s:  return 'Fixed Antagonist'
    if 'ago' in s:  return 'Agonist'
    return 'Unknown' 

def remove_duplicate_rows(df, subset_cols=None):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        subset_cols (list, optional): Columns to consider for duplicates.
                                      If None, considers all columns.
    
    Returns:
        pd.DataFrame: DataFrame with duplicates removed
    """
    clean_df = df.drop_duplicates(subset=subset_cols, keep='first')
    return clean_df
# === ANONYMIZATION APPROACH 1: INDEPENDENT ROWS ===
def anonymize(df):
    """
    Anonymize patients treating each row as independent.
    
    Use case: Privacy-focused analysis, cross-sectional studies
    Each treatment cycle is treated as a separate patient.
    
    Args:
        df (pd.DataFrame): Input DataFrame with patient names
        
    Returns:
        pd.DataFrame: DataFrame with sequential patient IDs (25XXX format)
        
    Example:
        Row 0 -> ID: 250
        Row 1 -> ID: 251
    """
    df = df.reset_index(drop=True)
    df["patient_id"] = df.index.map(lambda x: f"25{x}")
    
    return df

# === ANONYMIZATION APPROACH 2: MULTI-CYCLE TRACKING ===
def assign_patient_ids(df):
    """
    Anonymize patients while tracking multiple cycles.
    
    Use case: Longitudinal studies, treatment progression analysis
    Same patient across cycles gets same ID with validation.
    
    Validation Rule: Age must decrease with cycle number
    (older data = higher cycle number)
    
    Args:
        df (pd.DataFrame): Input DataFrame with patient names and cycles
        
    Returns:
        pd.DataFrame: DataFrame with validated patient IDs (25XXX format)
        
    Example:
        Patient "Sarah" Cycle 3, Age 35 -> ID: 25001
        Patient "Sarah" Cycle 2, Age 34 -> ID: 25001 (same person)
        Patient "Sarah" Cycle 1, Age 33 -> ID: 25001 (same person)
    """
    df = df.copy()
    df["row_rank"] = (df.reset_index().index + 1)
    df.sort_values(by=["patient_id", "cycle_number"], ascending=[True, False], inplace=True)

    df["group_key"] = None
    global_group_counter = 0

    for name, group in df.groupby("patient_id"):
        records = list(zip(group.index, group["cycle_number"], group["Age"]))

        records.sort(key=lambda x: -x[1])

        current_group = [records[0][0]]
        last_age = records[0][2]
        last_cycle = records[0][1]

        for idx, cycle, age in records[1:]:
            # Rule A: strict monotonic age decrease A person is valid only if ages follow a perfect decreasing pattern with cycle_number.
            #Otherwise, it’s considered a different person.
            if age < last_age and cycle < last_cycle:
                current_group.append(idx)
                last_age = age
                last_cycle = cycle
            else:
                global_group_counter += 1
                for r in current_group:
                    df.loc[r, "group_key"] = f"group_{global_group_counter}"

                current_group = [idx]
                last_age = age
                last_cycle = cycle
        
        global_group_counter += 1
        for r in current_group:
            df.loc[r, "group_key"] = f"group_{global_group_counter}"
    group_to_rank = df.groupby("group_key")["row_rank"].min().to_dict()
    df["new_patient_id"] = df["group_key"].map(lambda g: f"25{str(group_to_rank[g]).zfill(3)}")
    df.drop(columns=["group_key", "patient_id"], inplace=True)
    df.sort_index(inplace=True)

    return df

def fill_missing_age_with_median(df, column_name="Age"):
    """
    Impute missing age values with median.
    
    Strategy: Median is robust to outliers and represents central tendency.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column_name (str): Name of age column
        
    Returns:
        pd.DataFrame: DataFrame with age imputed
    """
    median_age = df[column_name].median()
    df[column_name].fillna(median_age, inplace=True)
    return df


def fill_missing_AMH(df):
    slope, intercept = predicting_AMH_from_n_follicules(df)
    
    # Fill missing AMH using the linear model and round to 2 decimals
    df['AMH'] = df.apply(
        lambda row: round(slope * row['n_Follicles'] + intercept, 2) if pd.isna(row['AMH']) else row['AMH'],
        axis=1
    )
    return df

n_follicules_mapping = {
    'Low': 10,      # median n_Follicules in Low group
    'Medium': 20,   # median n_Follicules in Medium group
    'High': 30      # median n_Follicules in High group
}

def assign_n_follicules(row):
    # 1. Check if 'follicles' already has a value (is not Null)
    if pd.notna(row['n_Follicles']):
        return row['n_Follicles']

    # 2. If it is Null, calculate based on Age
    age = row['Age']
    if age <= 34:
        group = 'Low'
    elif 35 <= age <= 39:
        group = 'Medium'
    else:
        group = 'High'
    
    return n_follicules_mapping[group]


#AFC based in age groups and n_Follicules
def fill_missing_AFC(df):
    """
    Fill missing AFC values based on Age groups and n_Follicules constraints.
    """
    
    def predict_afc(row):
        if not pd.isna(row['AFC']):
            return row['AFC']  # Keep existing value
        
        age = row['Age']
        n_follicles = row['n_Follicles']
        
        # Define age-based constraints
        if 25 <= age <= 34:
            avg, min_val, max_val = 15, 3, 30
        elif 35 <= age <= 40:
            avg, min_val, max_val = 9, 1, 25
        elif 41 <= age <= 46:
            avg, min_val, max_val = 4, 1, 17
        else:
            # If age outside ranges, just use average
            avg, min_val, max_val = 10, 1, 30
        
        # Adjust based on n_follicules
        if n_follicles >= 25:   # High
            return max_val
        elif n_follicles >= 15:  # Medium
            return avg
        else:  # Low
            return min_val
        
    df['AFC'] = df.apply(predict_afc, axis=1)
    return df


def fill_e2_values(df):
    """
    Fill missing E2_day5 values based on n_Follicles ranges.
    
    Parameters:
        df (pd.DataFrame): Input dataframe containing 'E2_day5' and 'n_Follicles' columns.
    
    Returns:
        pd.DataFrame: DataFrame with missing 'E2_day5' filled.
    """
    
    # Calculate medians for each n_Follicles range
    median_low = df[df['n_Follicles'] <= 18]['E2_day5'].median()
    median_mid = df[(df['n_Follicles'] >= 19) & (df['n_Follicles'] <= 24)]['E2_day5'].median()
    median_high = df[df['n_Follicles'] > 24]['E2_day5'].median()
    
    # Function to apply row-wise
    def fill_row(row):
        if pd.notna(row['E2_day5']):
            return row['E2_day5']
        if row['n_Follicles'] <= 18:
            return median_low
        elif 19 <= row['n_Follicles'] <= 24:
            return median_mid
        else:  # n_Follicles > 24
            return median_high
        # Apply the function
    df['E2_day5'] = df.apply(fill_row, axis=1)
    return df
