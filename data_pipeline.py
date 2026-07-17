import os
import zipfile
import gdown
import pandas as pd
import numpy as np

DRIVE_ID = "1zfqvs8-mAO6E0JpgvhBdueNx8Th03pUp"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
ZIP_PATH = os.path.join(RAW_DIR, "agriculture_dataset.zip")


def download_dataset():
    """Attempt to programmatically download the zip file using gdown."""
    os.makedirs(RAW_DIR, exist_ok=True)
    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    try:
        print(f"Attempting to download dataset from Google Drive (ID: {DRIVE_ID})...")
        gdown.download(url, ZIP_PATH, quiet=False)
        if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 1000:
            raise Exception("Downloaded file is empty or too small.")
        print("Download successful.")
        return True
    except Exception as e:
        print(f"Google Drive download failed: {e}")
        return False


def extract_zip():
    """Extract downloaded ZIP file and find all CSV files inside."""
    try:
        print(f"Extracting {ZIP_PATH} to {RAW_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print("Extraction completed successfully.")

        # Check for any inner zip files and extract them as well
        for root, dirs, files in os.walk(RAW_DIR):
            for file in files:
                if file.endswith('.zip') and os.path.join(root, file) != ZIP_PATH:
                    inner_zip = os.path.join(root, file)
                    print(f"Found inner zip file: {inner_zip}. Extracting...")
                    with zipfile.ZipFile(inner_zip, 'r') as inner_ref:
                        inner_ref.extractall(RAW_DIR)
                    print(f"Extracted {inner_zip}.")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def generate_mock_data():
    """Generate a high-fidelity synthetic mock CSV file when GDrive download fails."""
    print("Generating high-fidelity synthetic mock data...")
    np.random.seed(42)

    crops = ['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize', 'Gram', 'Groundnut', 'Jowar', 'Bajra', 'Ragi']
    varieties = ['Hybrid-A', 'Sharbati', 'IR-64', 'BT-Cotton', 'Co-86032', 'Ganga-11', 'Pusa-44', 'Local', 'Sonalika', 'CO-4']
    states = ['Uttar Pradesh', 'Punjab', 'Andhra Pradesh', 'Karnataka', 'Maharashtra', 'Madhya Pradesh', 'Gujarat', 'Haryana', 'Tamil Nadu', 'West Bengal']
    seasons = ['Kharif', 'Rabi', 'Summer', 'Whole Year']
    zones = ['Zone-A', 'Northern Plains', 'Deccan Plateau', 'Eastern Hills', 'Western Coastal', 'Central Zone', 'Southern Peninsula']

    data = []
    for i in range(1000):
        crop = np.random.choice(crops)
        variety = np.random.choice(varieties)
        state = np.random.choice(states)
        season = np.random.choice(seasons)

        # Generate realistic quantity and cost
        quantity = np.random.uniform(5.0, 300.0)
        cost = np.random.uniform(8000.0, 50000.0)

        # Production is correlated with quantity, crop and cost
        crop_factor = {
            'Rice': 1.8, 'Wheat': 2.2, 'Cotton': 0.8, 'Sugarcane': 15.0, 'Maize': 1.5,
            'Gram': 0.9, 'Groundnut': 1.1, 'Jowar': 0.7, 'Bajra': 0.6, 'Ragi': 0.8
        }
        base_prod = quantity * crop_factor[crop]
        # Add noise
        production = base_prod * np.random.uniform(0.85, 1.15) + (cost / 5000.0)
        recommended_zone = f"{state} - {np.random.choice(zones)}"

        data.append({
            'crop': crop,
            'variety': variety,
            'state': state,
            'quantity': quantity,
            'production': production,
            'season': season,
            'cost': cost,
            'recommended_zone': recommended_zone
        })

    df_mock = pd.DataFrame(data)

    # Introduce some anomalies/noise to test the cleaning process:
    # 1. Duplicate rows
    dup_idx = np.random.choice(1000, 20, replace=False)
    df_mock = pd.concat([df_mock, df_mock.iloc[dup_idx]], ignore_index=True)

    # 2. Capitalization variations
    case_idx = np.random.choice(len(df_mock), 50, replace=False)
    for idx in case_idx:
        df_mock.loc[idx, 'state'] = df_mock.loc[idx, 'state'].lower()
        df_mock.loc[idx, 'crop'] = df_mock.loc[idx, 'crop'].upper()

    # 3. Missing values
    missing_cols = ['cost', 'production', 'variety', 'season', 'recommended_zone']
    for col in missing_cols:
        null_idx = np.random.choice(len(df_mock), 15, replace=False)
        df_mock.loc[null_idx, col] = np.nan

    mock_path = os.path.join(RAW_DIR, "mock_agriculture_data.csv")
    df_mock.to_csv(mock_path, index=False)
    print(f"Mock data created successfully at {mock_path} (shape: {df_mock.shape})")


def load_and_merge_raw_csvs(csv_files):
    """Load and concatenate or merge all discovered CSV files."""
    mock_file = [f for f in csv_files if "mock_agriculture_data.csv" in f]
    raw_files = [f for f in csv_files if "datafile" in f.lower() or "produce.csv" in f.lower()]

    if len(raw_files) >= 3:
        print("Merging raw agricultural CSV files...")
        df1_file = [f for f in raw_files if "datafile (1)" in f or "datafile_1" in f]
        df2_file = [f for f in raw_files if "datafile (2)" in f or "datafile_2" in f]
        df3_file = [f for f in raw_files if "datafile (3)" in f or "datafile_3" in f]

        if not df1_file or not df2_file or not df3_file:
            print("Specific raw files not found for smart merge. Concatenating all discovered CSVs directly.")
            dfs = []
            for f in raw_files:
                df = pd.read_csv(f)
                df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)

        df1 = pd.read_csv(df1_file[0])
        df2 = pd.read_csv(df2_file[0])
        df3 = pd.read_csv(df3_file[0])

        # Clean column names
        df1.columns = [c.lower().strip().replace(' ', '_') for c in df1.columns]
        df2.columns = [c.lower().strip().replace(' ', '_') for c in df2.columns]
        df3.columns = [c.lower().strip().replace(' ', '_') for c in df3.columns]

        # Standardize crop names
        def standardize_crop(crop_name):
            if not isinstance(crop_name, str):
                return crop_name
            crop_name = crop_name.strip().lower()
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
                'vegetables ': 'vegetables',
                'sugarcane ': 'sugarcane',
                'arhar': 'tur',
                'pigeonpea': 'tur'
            }
            return crop_map.get(crop_name, crop_name)

        df1['crop'] = df1['crop'].map(standardize_crop)
        df2['crop'] = df2[df2.columns[0]].map(standardize_crop)
        df3['crop'] = df3['crop'].map(standardize_crop)

        # Process df2: melt production and area (quantity)
        years = ['2006-07', '2007-08', '2008-09', '2009-10', '2010-11']
        df2_melted = []
        for yr in years:
            temp = pd.DataFrame()
            temp['crop'] = df2['crop']
            prod_col = [c for c in df2.columns if 'production' in c and yr in c]
            area_col = [c for c in df2.columns if 'area' in c and yr in c]
            if prod_col and area_col:
                temp['production'] = pd.to_numeric(df2[prod_col[0]], errors='coerce')
                temp['quantity'] = pd.to_numeric(df2[area_col[0]], errors='coerce')
                temp['year'] = yr
                df2_melted.append(temp)
        if df2_melted:
            df2_long = pd.concat(df2_melted, ignore_index=True)
        else:
            df2_long = pd.DataFrame(columns=['crop', 'production', 'quantity', 'year'])

        # Process df1: extract cost (C2 cost of cultivation) and state
        df1_clean = pd.DataFrame()
        df1_clean['crop'] = df1['crop']
        df1_clean['state'] = df1['state'].str.strip().str.lower()
        c2_cols = [c for c in df1.columns if 'cost_of_cultivation' in c and 'c2' in c]
        if c2_cols:
            df1_clean['cost'] = pd.to_numeric(df1[c2_cols[0]], errors='coerce')
        else:
            df1_clean['cost'] = np.nan

        # Process df3: variety, season, recommended_zone
        df3_clean = df3.copy()
        season_cols = [c for c in df3.columns if 'season' in c]
        if season_cols:
            df3_clean = df3_clean.rename(columns={season_cols[0]: 'season'})
        else:
            df3_clean['season'] = 'unknown'

        df3_clean['state_raw'] = df3_clean['recommended_zone'].fillna('').str.strip().str.lower()
        df3_clean['season'] = df3_clean['season'].fillna('unknown').str.strip().str.lower()
        df3_clean['variety'] = df3_clean['variety'].fillna('unknown').str.strip().str.lower()
        df3_clean['recommended_zone'] = df3_clean['recommended_zone'].fillna('unknown').str.strip().str.lower()

        # Merge df2_long and df1_clean on crop
        merged = pd.merge(df2_long, df1_clean, on='crop')

        # Link varieties/seasons/recommended_zone from df3
        matched_rows = []
        for idx, row in merged.iterrows():
            crop = row['crop']
            state = row['state']

            sub_df3 = df3_clean[df3_clean['crop'] == crop]
            if len(sub_df3) == 0:
                new_row = row.to_dict()
                new_row['variety'] = 'unknown'
                new_row['season'] = 'unknown'
                new_row['recommended_zone'] = 'unknown'
                matched_rows.append(new_row)
                continue

            # Check if any recommended zone contains state name
            matches = sub_df3[sub_df3['state_raw'].str.contains(state, regex=False)]
            if len(matches) > 0:
                for _, m_row in matches.iterrows():
                    new_row = row.to_dict()
                    new_row['variety'] = m_row['variety']
                    new_row['season'] = m_row['season']
                    new_row['recommended_zone'] = m_row['recommended_zone']
                    matched_rows.append(new_row)
            else:
                new_row = row.to_dict()
                new_row['variety'] = sub_df3.iloc[0]['variety']
                new_row['season'] = sub_df3.iloc[0]['season']
                new_row['recommended_zone'] = sub_df3.iloc[0]['recommended_zone']
                matched_rows.append(new_row)

        final_df = pd.DataFrame(matched_rows)
        # Reorder/select schema columns
        schema_cols = ['crop', 'variety', 'state', 'quantity', 'production', 'season', 'cost', 'recommended_zone']
        for col in schema_cols:
            if col not in final_df.columns:
                final_df[col] = np.nan
        return final_df[schema_cols]
    elif mock_file:
        print(f"Loading mock agriculture data from {mock_file[0]}...")
        return pd.read_csv(mock_file[0])
    else:
        raise FileNotFoundError("No CSV files found in data/raw/ directory.")


def clean_and_process_dataset(df):
    """Clean the concatenated/merged dataset, handling types, nulls, and formatting."""
    print("Cleaning and processing dataset...")
    df = df.copy()
    # Convert all column names to lowercase and replace spaces with underscores
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]

    # Drop duplicate rows
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {initial_len - len(df)} duplicate rows.")

    # Clean text columns first: Strip whitespace
    cat_cols = ['crop', 'variety', 'state', 'season', 'recommended_zone']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Handle missing values
    # Numeric columns
    numeric_cols = ['quantity', 'production', 'cost']
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
            print(f"Imputed missing values in '{col}' with median: {median_val}")

    # Categorical columns
    for col in cat_cols:
        if col in df.columns:
            # Exclude representations of nan
            df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})
            # Find mode
            non_null = df[col].dropna()
            if not non_null.empty:
                mode_val = non_null.mode()[0]
            else:
                mode_val = 'Unknown'
            if pd.isna(mode_val) or mode_val == '':
                mode_val = 'Unknown'
            df[col] = df[col].fillna(mode_val)

    # Standardize state/crop/season/variety/recommended_zone names to Title Case
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].str.title()

    # Save output to data/processed/cleaned_crop_data.csv
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DIR, "cleaned_crop_data.csv")
    df.to_csv(processed_path, index=False)
    print(f"Saved cleaned and processed dataset to {processed_path} (shape: {df.shape})")
    return df


def main():
    print("=" * 60)
    print("STARTING DATA PIPELINE")
    print("=" * 60)

    # 1. Download and extract dataset
    download_ok = download_dataset()
    extract_ok = False
    if download_ok:
        extract_ok = extract_zip()

    # 2. Fallback to mock data if download or extraction fails
    if not (download_ok and extract_ok):
        print("\nUsing robust fallback to generate mock dataset...")
        generate_mock_data()

    # 3. Find all CSV files inside data/raw/
    csv_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    print(f"\nDiscovered CSV files for loading: {csv_files}")

    # 4. Load and concatenate/merge
    raw_df = load_and_merge_raw_csvs(csv_files)

    # 5. Clean and preprocess
    clean_and_process_dataset(raw_df)

    print("\n" + "=" * 60)
    print("DATA PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
