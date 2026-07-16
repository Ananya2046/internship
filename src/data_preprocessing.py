import os
import pandas as pd
import numpy as np

def clean_column_name(col):
    # Convert to lowercase and remove spaces and special characters
    col = str(col).lower().strip()
    col = ''.join(c for c in col if c.isalnum() or c in ['_', '/'])
    return col

def load_and_preprocess_raw():
    print("Loading raw CSV files...")
    
    # 1. Load files
    df1 = pd.read_csv('data/datafile (1).csv')
    df2 = pd.read_csv('data/datafile (2).csv')
    df3 = pd.read_csv('data/datafile (3).csv')
    df4 = pd.read_csv('data/datafile.csv')
    df5 = pd.read_csv('data/produce.csv')
    
    # 2. Convert all column names to lowercase and remove spaces
    df1.columns = [clean_column_name(c) for c in df1.columns]
    df2.columns = [clean_column_name(c) for c in df2.columns]
    df3.columns = [clean_column_name(c) for c in df3.columns]
    df4.columns = [clean_column_name(c) for c in df4.columns]
    df5.columns = [clean_column_name(c) for c in df5.columns]
    
    # 3. Rename columns for basic raw concatenation alignment
    # In df5, 'particulars' is the crop column
    if 'particulars' in df5.columns:
        df5 = df5.rename(columns={'particulars': 'crop'})
        
    # In df3, 'recommendedzone' is equivalent to 'state'
    if 'recommendedzone' in df3.columns:
        df3 = df3.rename(columns={'recommendedzone': 'state'})
        
    # Trim crop values in all datasets
    for df in [df1, df2, df3, df4, df5]:
        if 'crop' in df.columns:
            df['crop'] = df['crop'].astype(str).str.strip().str.lower()
            
    print("Raw files loaded and initial formatting applied.")
    
    # 4. Perform raw concatenation (axis=0)
    print("Performing raw concatenation of all datasets...")
    raw_concat = pd.concat([df1, df2, df3, df4, df5], axis=0, ignore_index=True)
    
    # Make sure output directory exists
    os.makedirs('data', exist_ok=True)
    raw_concat.to_csv('data/raw_concatenated_dataset.csv', index=False)
    print(f"Saved raw concatenated dataset with shape {raw_concat.shape} to data/raw_concatenated_dataset.csv")
    
    return df1, df2, df3, df4, df5

def build_clean_merged_dataset(df1, df2, df3, df4, df5):
    print("Building clean merged dataset for ML...")
    
    # Standardize crop names across all datasets to ensure proper joins
    crop_map = {
        'paddy': 'rice',
        'rapeseed and mustard': 'rapeseed & mustard',
        'rapeseed&mustard': 'rapeseed & mustard',
        'cotton': 'cotton(lint)',
        'desi cotton': 'cotton(lint)',
        'chickpea': 'gram',
        'bengal gram': 'gram',
        'mungbean': 'moong',
        'sesame': 'sesamum',
        'vegetables': 'vegetables',
        'vegetables ': 'vegetables',
        'sugarcane ': 'sugarcane'
    }
    
    for df in [df1, df2, df3, df4, df5]:
        if 'crop' in df.columns:
            df['crop'] = df['crop'].replace(crop_map)
            
    # --- Process Dataset 2 (Production, Area, Yield over years 2006-07 to 2010-11) ---
    # Melt df2 wide columns into long format
    years = ['2006-07', '2007-08', '2008-09', '2009-10', '2010-11']
    melted_df2_list = []
    for yr in years:
        temp = pd.DataFrame()
        temp['crop'] = df2['crop']
        temp['year'] = yr
        temp['production'] = pd.to_numeric(df2[f'production{yr.replace("-", "")}'], errors='coerce')
        temp['area'] = pd.to_numeric(df2[f'area{yr.replace("-", "")}'], errors='coerce')
        temp['yield'] = pd.to_numeric(df2[f'yield{yr.replace("-", "")}'], errors='coerce')
        melted_df2_list.append(temp)
    df2_long = pd.concat(melted_df2_list, ignore_index=True)
    
    # --- Process Dataset 1 (Costs & Yield by State and Crop) ---
    df1_clean = df1.copy()
    # Let's clean state names
    df1_clean['state'] = df1_clean['state'].str.strip().str.lower()
    
    # --- Process Dataset 3 (Variety, Season, Recommended Zone) ---
    df3_clean = df3.copy()
    df3_clean = df3_clean.rename(columns={'season/durationindays': 'season'})
    # Drop columns we don't need
    if 'unnamed:4' in df3_clean.columns:
        df3_clean = df3_clean.drop(columns=['unnamed:4'])
        
    df3_clean['state_raw'] = df3_clean['state'].fillna('').str.strip().str.lower()
    df3_clean['season'] = df3_clean['season'].fillna('unknown').str.strip().str.lower()
    df3_clean['variety'] = df3_clean['variety'].fillna('unknown').str.strip().str.lower()
    
    # --- Merging Strategy ---
    # Merge df2_long (production/area/yield) with df1_clean (costs and states) on crop.
    merged = pd.merge(df2_long, df1_clean, on='crop', suffixes=('', '_df1'))
    
    # Map the crop-state row to the varieties of the crop that are recommended for that state.
    matched_rows = []
    for idx, row in merged.iterrows():
        crop = row['crop']
        state = row['state']
        
        # Filter varieties for this crop
        sub_df3 = df3_clean[df3_clean['crop'] == crop]
        if len(sub_df3) == 0:
            new_row = row.to_dict()
            new_row['variety'] = 'unknown'
            new_row['season'] = 'unknown'
            matched_rows.append(new_row)
            continue
            
        # Check which varieties recommend this state
        matches = sub_df3[sub_df3['state_raw'].str.contains(state, regex=False)]
        if len(matches) > 0:
            for _, m_row in matches.iterrows():
                new_row = row.to_dict()
                new_row['variety'] = m_row['variety']
                new_row['season'] = m_row['season']
                matched_rows.append(new_row)
        else:
            # Fallback: take the most common variety/season for this crop
            new_row = row.to_dict()
            new_row['variety'] = sub_df3.iloc[0]['variety']
            new_row['season'] = sub_df3.iloc[0]['season']
            matched_rows.append(new_row)
            
    final_df = pd.DataFrame(matched_rows)
    
    # Clean up column names and values in final_df
    final_df['state'] = final_df['state'].str.title()
    final_df['crop'] = final_df['crop'].str.title()
    final_df['season'] = final_df['season'].str.title()
    final_df['variety'] = final_df['variety'].str.title()
    
    # Rename cost columns for readability
    rename_dict = {
        'costofcultivation/hectarea2fl': 'cost_of_cultivation_a2_fl',
        'costofcultivation/hectarec2': 'cost_of_cultivation_c2',
        'costofproduction/quintalc2': 'cost_of_production_c2',
        'yieldquintal/hectare': 'yield_cultivation'
    }
    final_df = final_df.rename(columns=rename_dict)
    
    # Remove duplicate rows
    initial_shape = final_df.shape
    final_df = final_df.drop_duplicates()
    print(f"Removed {initial_shape[0] - final_df.shape[0]} duplicate rows.")
    
    # Handle missing values
    print("Handling missing values...")
    # Find numerical columns and fill missing values with median
    num_cols = final_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if final_df[col].isnull().any():
            median_val = final_df[col].median()
            final_df[col] = final_df[col].fillna(median_val)
            print(f"Imputed missing values in column '{col}' with median: {median_val}")
            
    # Find categorical columns and fill with 'Unknown'
    cat_cols = final_df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if final_df[col].isnull().any():
            final_df[col] = final_df[col].fillna('Unknown')
            print(f"Imputed missing values in column '{col}' with 'Unknown'")
            
    # Save the cleaned merged dataset
    final_df.to_csv('data/clean_merged_dataset.csv', index=False)
    print(f"Saved clean merged dataset with shape {final_df.shape} to data/clean_merged_dataset.csv")
    
    return final_df

if __name__ == '__main__':
    df1, df2, df3, df4, df5 = load_and_preprocess_raw()
    build_clean_merged_dataset(df1, df2, df3, df4, df5)
