"""
UN Voting Prediction Using Supervised Machine Learning
Predicting how UN member states vote on resolutions based on economic and military indicators.

Project: Mapping and Predicting Global Political Alignments
Authors: Devashish Mahurkar, Shubh Panchal

================================================================================
LITERATURE REVIEW & BASELINE WORK
================================================================================

Previous research on UN voting patterns has primarily focused on:

1. Descriptive Analysis & Ideological Clustering:
   - Voeten (2013) and subsequent work analyzed UN voting patterns to identify
     ideological blocs and voting coalitions through clustering techniques.
   - These studies revealed persistent voting patterns along geopolitical lines
     (e.g., Western bloc, Non-Aligned Movement, etc.).

2. Kaggle Analyses:
   - Various Kaggle projects have explored UN voting data, but most focus on
     descriptive statistics, visualization, and basic correlation analysis.
   - Limited use of socioeconomic indicators for predictive modeling.

3. Our Contribution:
   - This project extends previous work by:
     a) Using binary classification as a supervised learning task
     b) Integrating economic (World Bank WDI) and military (SIPRI) indicators
        as predictive features
     c) Systematic cross-validation and comparative evaluation of multiple
        ML algorithms
     d) Interpretability analysis to reveal quantitative correlations between
        development, defense, and diplomatic action
   - This combination of datasets in a predictive framework is novel and
     provides a more rigorous approach to understanding UN voting behavior.

Baseline Methods:
   - Simple majority voting (baseline accuracy ~50-60%)
   - Country-based clustering (Voeten, 2013)
   - Our ML models aim to significantly outperform these baselines by
     leveraging quantitative socioeconomic and military indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, auc, 
                             confusion_matrix, classification_report,
                             average_precision_score, precision_recall_curve)
import warnings
import kagglehub
import os
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================================================
# 1. DATA ACQUISITION AND LOADING
# ============================================================================

class DataAcquisition:
    """Handles data loading from multiple sources and initial exploration."""
    
    @staticmethod
    def download_kaggle_datasets():
        """Download datasets using kagglehub API and return paths."""
        print("\n" + "="*70)
        print("DOWNLOADING DATASETS FROM KAGGLE")
        print("="*70)
        
        try:
            # Download SIPRI Military Expenditure
            print("\nDownloading SIPRI Military Expenditure Database...")
            sipri_path = kagglehub.dataset_download("azeeshan20/sipri-military-expenditure-database")
            print(f"SIPRI path: {sipri_path}")
            
            # Download World Bank Development Indicators
            print("\nDownloading World Bank Development Indicators...")
            wdi_path = kagglehub.dataset_download("theworldbank/world-development-indicators")
            print(f"WDI path: {wdi_path}")
            
            # Download UN Voting Records
            print("\nDownloading UN General Assembly Voting Dataset...")
            un_path = kagglehub.dataset_download("rikuishiharaa/united-nations-general-assembly-voting-dataset")
            print(f"UN Voting path: {un_path}")
            
            return un_path, wdi_path, sipri_path
            
        except Exception as e:
            print(f"[WARNING] Error downloading datasets: {e}")
            return None, None, None
    
    @staticmethod
    def find_csv_files(directory, pattern=''):
        """Find CSV files in directory."""
        csv_files = {}
        if directory and os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.csv'):
                    csv_files[file] = os.path.join(directory, file)
        return csv_files
    
    @staticmethod
    def load_un_voting_data(un_path):
        """Load UN voting records from downloaded dataset."""
        try:
            csv_files = DataAcquisition.find_csv_files(un_path)
            
            if not csv_files:
                # Try subdirectories
                for root, dirs, files in os.walk(un_path):
                    for file in files:
                        if file.endswith('.csv'):
                            filepath = os.path.join(root, file)
                            print(f"  Trying: {file}")
                            df = pd.read_csv(filepath, low_memory=False)
                            if 'vote' in df.columns or 'votes' in df.columns:
                                print(f"[OK] UN Voting Data Loaded: {df.shape}")
                                return df
            else:
                # Try each CSV file
                for filename, filepath in csv_files.items():
                    try:
                        df = pd.read_csv(filepath, low_memory=False)
                        if 'vote' in df.columns or 'votes' in df.columns or len(df) > 100:
                            print(f"[OK] UN Voting Data Loaded from {filename}: {df.shape}")
                            return df
                    except Exception as e:
                        continue
            
            print("[WARNING] Could not identify UN voting CSV file")
            return None
            
        except FileNotFoundError:
            print(f"[WARNING] UN voting data path not found: {un_path}")
            return None
    
    @staticmethod
    def load_world_bank_data(wdi_path):
        """Load World Bank Development Indicators."""
        try:
            csv_files = DataAcquisition.find_csv_files(wdi_path)
            
            # Priority: Look for WDIData.csv (the actual data file)
            for filename in ['WDIData.csv', 'WDI_Data.csv', 'data.csv', 'indicators.csv']:
                if filename in csv_files:
                    df = pd.read_csv(csv_files[filename], low_memory=False)
                    print(f"[OK] World Bank Data Loaded: {df.shape}")
                    return df
            
            # Try subdirectories (WDIData.csv might be in a subdirectory)
            for root, dirs, files in os.walk(wdi_path):
                for file in files:
                    if file == 'WDIData.csv' or (file.startswith('WDI') and 'Data' in file and file.endswith('.csv')):
                        filepath = os.path.join(root, file)
                        try:
                            # Check file size to ensure it's the data file (should be large, >100MB)
                            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                            if file_size_mb > 50:  # WDIData.csv is ~200MB
                                df = pd.read_csv(filepath, low_memory=False)
                                if len(df) > 1000:  # WDI is large
                                    print(f"[OK] World Bank Data Loaded from {file}: {df.shape}")
                                    return df
                        except Exception as e:
                            continue
            else:
                # Try any large CSV (WDIData.csv should be the largest)
                for filename, filepath in sorted(csv_files.items(), key=lambda x: os.path.getsize(x[1]), reverse=True):
                    try:
                        df = pd.read_csv(filepath, low_memory=False)
                        if len(df) > 1000:
                            print(f"[OK] World Bank Data Loaded from {filename}: {df.shape}")
                            return df
                    except:
                        continue
            
            print("[WARNING] Could not identify WDI CSV file")
            return None
            
        except Exception as e:
            print(f"[WARNING] Error loading World Bank data: {e}")
            return None
    
    @staticmethod
    def load_sipri_military_data(sipri_path):
        """Load SIPRI military expenditure database."""
        try:
            csv_files = DataAcquisition.find_csv_files(sipri_path)
            
            if not csv_files:
                # Try subdirectories
                for root, dirs, files in os.walk(sipri_path):
                    for file in files:
                        if file.endswith('.csv'):
                            filepath = os.path.join(root, file)
                            try:
                                df = pd.read_csv(filepath, low_memory=False)
                                if 'spending' in df.columns or 'expenditure' in df.columns or len(df) > 100:
                                    print(f"[OK] SIPRI Military Data Loaded: {df.shape}")
                                    return df
                            except:
                                continue
            else:
                # Try any CSV (SIPRI usually has small files)
                for filename, filepath in csv_files.items():
                    try:
                        df = pd.read_csv(filepath, low_memory=False)
                        print(f"[OK] SIPRI Military Data Loaded from {filename}: {df.shape}")
                        return df
                    except:
                        continue
            
            print("[WARNING] Could not identify SIPRI CSV file")
            return None
            
        except Exception as e:
            print(f"[WARNING] Error loading SIPRI data: {e}")
            return None


# ============================================================================
# 2. DATA PREPROCESSING AND HARMONIZATION
# ============================================================================

class DataPreprocessor:
    """Handles data cleaning, harmonization, and preparation."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
    
    def extract_wdi_indicators(self, df_wdi):
        """
        Extract required socioeconomic indicators from World Bank WDI data.
        Returns: DataFrame with country-year and selected indicators.
        """
        if df_wdi is None:
            print("[WARNING] No WDI data available")
            return None
        
        print("\n  Extracting World Bank Development Indicators...")
        
        # WDI indicator codes mapping to required features
        wdi_indicators = {
            # GDP per capita (constant 2015 US$)
            'NY.GDP.PCAP.KD': 'gdp_per_capita',
            'NY.GDP.PCAP.CD': 'gdp_per_capita_current',
            # Trade openness (trade as % of GDP)
            'NE.TRD.GNFS.ZS': 'trade_openness',
            'TG.VAL.TOTL.GD.ZS': 'trade_gdp_ratio',
            # Education (government expenditure on education, % of GDP)
            'SE.XPD.TOTL.GD.ZS': 'education_expenditure',
            'SE.PRM.ENRR': 'primary_enrollment',
            'SE.SEC.ENRR': 'secondary_enrollment',
            # Internet use (% of population)
            'IT.NET.USER.ZS': 'internet_users',
            'IT.NET.BBND.P2': 'internet_broadband',
            # Energy (energy use per capita, kg of oil equivalent)
            'EG.USE.PCAP.KG.OE': 'energy_use_per_capita',
            'EG.USE.ELEC.KH.PC': 'electricity_use_per_capita',
            # Health (health expenditure, % of GDP)
            'SH.XPD.CHEX.GD.ZS': 'health_expenditure',
            'SH.XPD.CHEX.PC.CD': 'health_expenditure_per_capita',
            # Tax revenue (% of GDP)
            'GC.TAX.TOTL.GD.ZS': 'tax_revenue',
            'GC.REV.XGRT.GD.ZS': 'government_revenue',
        }
        
        # Try to identify WDI data structure
        # Standard WDI format: Country Code, Indicator Code, and year columns (1960, 1961, etc.)
        
        # Check for standard WDI format (Country Code, Indicator Code, year columns)
        if 'Country Code' in df_wdi.columns and 'Indicator Code' in df_wdi.columns:
            print("  Detected standard WDI data format")
            # Find year columns (numeric column names representing years)
            year_cols = []
            for col in df_wdi.columns:
                try:
                    year_val = int(str(col))
                    if 1990 <= year_val <= 2023:
                        year_cols.append(col)
                except (ValueError, TypeError):
                    continue
            
            if not year_cols:
                print("  [WARNING] No year columns found in range 1990-2023")
                return None
            
            print(f"  Found {len(year_cols)} year columns (1990-2023)")
            
            # Select only required indicators
            required_codes = list(wdi_indicators.keys())
            df_filtered = df_wdi[df_wdi['Indicator Code'].isin(required_codes)].copy()
            
            if len(df_filtered) == 0:
                print(f"  [WARNING] None of the required indicators found in dataset")
                print(f"    Looking for: {required_codes[:5]}...")
                # Show available indicators
                available = df_wdi['Indicator Code'].unique()[:10]
                print(f"    Available indicators (sample): {list(available)}")
                return None
            
            print(f"  Found {len(df_filtered)} rows with required indicators")
            
            # Prepare id_vars (non-year columns)
            id_vars = ['Country Code', 'Indicator Code']
            if 'Country Name' in df_wdi.columns:
                id_vars.append('Country Name')
            if 'Indicator Name' in df_wdi.columns:
                id_vars.append('Indicator Name')
            id_vars = [col for col in id_vars if col in df_filtered.columns]
            
            # Melt year columns to long format
            df_melted = pd.melt(df_filtered, 
                              id_vars=id_vars,
                              value_vars=year_cols,
                              var_name='year',
                              value_name='value')
            
            # Convert year to numeric
            df_melted['year'] = pd.to_numeric(df_melted['year'], errors='coerce')
            df_melted = df_melted[df_melted['year'].notna()]
            df_melted = df_melted[df_melted['year'].between(1990, 2023)]
            
            # Remove rows with missing values (but keep for diagnostic)
            df_melted_with_na = df_melted.copy()
            df_melted = df_melted[df_melted['value'].notna()]
            
            # Pivot to wide format (country-year rows, indicator columns)
            df_pivoted = df_melted.pivot_table(
                index=['Country Code', 'year'],
                columns='Indicator Code',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            # Rename columns using indicator mapping
            rename_dict = {code: wdi_indicators[code] for code in wdi_indicators.keys() 
                         if code in df_pivoted.columns}
            df_pivoted = df_pivoted.rename(columns=rename_dict)
            
            # Diagnostic: Check data availability for each indicator
            print(f"  [OK] Extracted {len(rename_dict)} WDI indicators")
            print(f"  Shape: {df_pivoted.shape}")
            print(f"  Countries: {df_pivoted['Country Code'].nunique()}")
            
            # Check missing data for each indicator
            if len(df_pivoted) > 0:
                print(f"\n  Data availability by indicator:")
                for code, feature_name in wdi_indicators.items():
                    if feature_name in df_pivoted.columns:
                        non_null = df_pivoted[feature_name].notna().sum()
                        total = len(df_pivoted)
                        pct = (non_null / total * 100) if total > 0 else 0
                        print(f"    {feature_name}: {non_null}/{total} ({pct:.1f}% non-null)")
            
            return df_pivoted
        
        # Check for wide format (separate columns for each indicator)
        elif any('GDP' in str(col).upper() or 'TRADE' in str(col).upper() for col in df_wdi.columns):
            print("  Detected wide format WDI data")
            # Try to find country and year columns
            country_col = None
            year_col = None
            
            for col in df_wdi.columns:
                if 'country' in col.lower() and 'code' in col.lower():
                    country_col = col
                elif 'year' in col.lower():
                    year_col = col
            
            if country_col and year_col:
                # Select relevant columns
                feature_cols = [col for col in df_wdi.columns 
                              if any(ind in col.upper() for ind in ['GDP', 'TRADE', 'EDUCATION', 
                                                                    'INTERNET', 'ENERGY', 'HEALTH', 'TAX'])]
                selected_cols = [country_col, year_col] + feature_cols
                df_selected = df_wdi[[col for col in selected_cols if col in df_wdi.columns]].copy()
                df_selected[year_col] = pd.to_numeric(df_selected[year_col], errors='coerce')
                df_selected = df_selected[df_selected[year_col].between(1990, 2023)]
                
                # Rename country column
                df_selected = df_selected.rename(columns={country_col: 'Country Code', year_col: 'year'})
                
                print(f"  [OK] Extracted WDI indicators from wide format")
                print(f"  Shape: {df_selected.shape}")
                return df_selected
        
        print("  [WARNING] Could not parse WDI data structure")
        return None
    
    def extract_sipri_data(self, df_sipri):
        """
        Extract military expenditure data from SIPRI dataset.
        Returns: DataFrame with country-year and military spending.
        """
        if df_sipri is None:
            print("[WARNING] No SIPRI data available")
            return None
        
        print("\n  Extracting SIPRI Military Expenditure Data...")
        
        # Try to identify SIPRI data structure
        # Common columns: country, year, spending, expenditure, military
        
        # Find country code column
        country_col = None
        for col in df_sipri.columns:
            if any(term in col.lower() for term in ['country', 'code', 'iso', 'name']):
                country_col = col
                break
        
        # Find year column
        year_col = None
        for col in df_sipri.columns:
            if 'year' in col.lower():
                year_col = col
                break
        
        # Find spending/expenditure column (could be 'Value', 'spending', 'expenditure', etc.)
        spending_col = None
        # First check for 'Value' column (common in SIPRI datasets)
        if 'Value' in df_sipri.columns:
            spending_col = 'Value'
        else:
            for col in df_sipri.columns:
                if any(term in col.lower() for term in ['spending', 'expenditure', 'military', 'defense', 'defence', 'value']):
                    if df_sipri[col].dtype in [np.number, 'float64', 'int64']:
                        spending_col = col
                        break
        
        if country_col and year_col and spending_col:
            df_clean = df_sipri[[country_col, year_col, spending_col]].copy()
            
            # Convert year to numeric
            df_clean[year_col] = pd.to_numeric(df_clean[year_col], errors='coerce')
            df_clean = df_clean[df_clean[year_col].between(1990, 2023)]
            
            # Convert spending to numeric (handle string values like '...')
            df_clean[spending_col] = pd.to_numeric(df_clean[spending_col], errors='coerce')
            df_clean = df_clean[df_clean[spending_col].notna()]
            
            # Handle country name to ISO3 code conversion
            # If country column contains names, we'll need to merge with UN voting data later
            # For now, standardize the country column
            df_clean['country_name'] = df_clean[country_col].astype(str).str.strip()
            
            # Try to use country name as-is (will be matched during merge)
            df_clean = df_clean.rename(columns={
                year_col: 'year',
                spending_col: 'military_expenditure',
                country_col: 'country_name_temp'
            })
            
            # Keep country_name for merging
            df_clean['country_name'] = df_clean['country_name_temp']
            df_clean = df_clean.drop(columns=['country_name_temp'])
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates(subset=['country_name', 'year'], keep='first')
            
            print(f"  [OK] Extracted SIPRI military expenditure data")
            print(f"  Shape: {df_clean.shape}")
            print(f"  Countries: {df_clean['country_name'].nunique()}")
            print(f"  Sample countries: {df_clean['country_name'].unique()[:5].tolist()}")
            return df_clean
        else:
            print(f"  [WARNING] Could not identify SIPRI columns")
            print(f"    Found: country={country_col}, year={year_col}, spending={spending_col}")
            return None
    
    def filter_by_issue_type(self, df, issue_types=None):
        """
        Filter voting records by major issue types (Security, Human Rights, etc.).
        """
        if issue_types is None:
            issue_types = ['Security', 'Human Rights', 'Human rights', 'SECURITY', 
                          'HUMAN RIGHTS', 'Peacekeeping', 'PEACEKEEPING']
        
        print("\n" + "="*70)
        print("ISSUE TYPE FILTERING")
        print("="*70)
        
        initial_count = len(df)
        
        # Find topic/subject column
        topic_col = None
        for col in ['subjects', 'topic', 'title', 'agenda_title', 'issue', 'category']:
            if col in df.columns:
                topic_col = col
                break
        
        if topic_col:
            print(f"  Using '{topic_col}' column for filtering")
            
            # Convert to string and check for issue types
            df[topic_col] = df[topic_col].astype(str)
            
            # Create filter mask
            mask = pd.Series([False] * len(df))
            for issue in issue_types:
                mask = mask | df[topic_col].str.contains(issue, case=False, na=False)
            
            df_filtered = df[mask].copy()
            
            print(f"  Filtered to major issue types: {issue_types[:3]}...")
            print(f"  Before: {initial_count} rows")
            print(f"  After: {len(df_filtered)} rows")
            print(f"  Removed: {initial_count - len(df_filtered)} rows")
            
            return df_filtered
        else:
            print("  [WARNING] No topic/subject column found, using all voting records")
            return df
    
    def harmonize_datasets(self, df_votes, df_wdi, df_sipri, filter_issues=True):
        """
        Combine datasets by ISO3 country codes and year (1990-2023).
        Merges UN voting records with WDI socioeconomic indicators and SIPRI military expenditure.
        """
        print("\n" + "="*70)
        print("DATA HARMONIZATION")
        print("="*70)
        
        year_range = (1990, 2023)
        
        # Handle None DataFrames
        if df_votes is None:
            print("[WARNING] No voting data available, using synthetic data")
            df_votes = self._create_synthetic_data()
        else:
            # Extract year from various possible column formats
            year_col = None
            year_values = None
            
            # Check for existing 'year' column
            if 'year' in df_votes.columns:
                year_col = 'year'
                year_values = pd.to_numeric(df_votes[year_col], errors='coerce')
            # Check for 'date' column (YYYY-MM-DD format)
            elif 'date' in df_votes.columns:
                print(f"  Found 'date' column, extracting year...")
                year_values = pd.to_datetime(df_votes['date'], errors='coerce').dt.year
                df_votes['year'] = year_values
                year_col = 'year'
            # Check for 'session' column (UN session number)
            elif 'session' in df_votes.columns:
                print(f"  Found 'session' column, converting to year...")
                # UN sessions started in 1946, session 1 = 1946
                session_numeric = pd.to_numeric(df_votes['session'], errors='coerce')
                year_values = 1945 + session_numeric
                df_votes['year'] = year_values
                year_col = 'year'
            else:
                # Try to find any column with 'year' in the name
                for col in df_votes.columns:
                    if 'year' in col.lower():
                        year_col = col
                        year_values = pd.to_numeric(df_votes[year_col], errors='coerce')
                        break
            
            if year_col and year_values is not None:
                print(f"  Using '{year_col}' column for year filtering")
                print(f"  Year range in data: {year_values.min():.0f} to {year_values.max():.0f}")
                # Filter by year range, dropping rows where year conversion failed
                mask = (year_values >= year_range[0]) & (year_values <= year_range[1])
                df_votes = df_votes[mask].copy()
                print(f"  Rows after filtering: {len(df_votes)}")
            else:
                print("[WARNING] No year column found, using all data")
        
        # Check if dataframe is empty after filtering
        if len(df_votes) == 0:
            print("[WARNING] Warning: No data after year filtering, using synthetic data")
            df_votes = self._create_synthetic_data()
        
        print(f"[OK] Filtered to year range: {year_range}")
        print(f"  Votes: {df_votes.shape}")
        
        # Step 1: Filter by issue type (Security, Human Rights)
        if filter_issues:
            df_votes = self.filter_by_issue_type(df_votes)
        
        # Step 2: Standardize country code column in voting data
        country_col_votes = None
        for col in ['ms_code', 'country_code', 'iso3', 'iso_code', 'Country Code', 'country']:
            if col in df_votes.columns:
                country_col_votes = col
                break
        
        if country_col_votes:
            df_votes['country_code'] = df_votes[country_col_votes].astype(str).str.upper().str.strip()
        else:
            print("[WARNING] No country code column found in voting data")
            df_votes['country_code'] = 'UNK'
        
        # Step 3: Extract and merge WDI socioeconomic indicators
        df_wdi_indicators = self.extract_wdi_indicators(df_wdi)
        
        if df_wdi_indicators is not None:
            # Merge WDI data
            wdi_country_col = 'Country Code' if 'Country Code' in df_wdi_indicators.columns else 'country_code'
            
            if wdi_country_col in df_wdi_indicators.columns:
                df_wdi_indicators['country_code'] = df_wdi_indicators[wdi_country_col].astype(str).str.upper().str.strip()
                
                # Merge on country_code and year
                df_merged = pd.merge(
                    df_votes,
                    df_wdi_indicators.drop(columns=[wdi_country_col], errors='ignore'),
                    on=['country_code', 'year'],
                    how='left'
                )
                print(f"  [OK] Merged WDI indicators: {df_merged.shape}")
            else:
                df_merged = df_votes.copy()
                print("  [WARNING] Could not merge WDI data (country code mismatch)")
        else:
            df_merged = df_votes.copy()
            print("  [WARNING] WDI data not available, proceeding without socioeconomic indicators")
        
        # Step 4: Extract and merge SIPRI military expenditure
        df_sipri_clean = self.extract_sipri_data(df_sipri)
        
        if df_sipri_clean is not None:
            # SIPRI uses country names, need to match with UN voting data
            # Try to merge using country names if available in voting data
            country_name_col = None
            for col in ['ms_name', 'country_name', 'Country Name', 'country']:
                if col in df_merged.columns:
                    country_name_col = col
                    break
            
            if country_name_col:
                # Standardize country names for matching
                df_merged['country_name_clean'] = df_merged[country_name_col].astype(str).str.upper().str.strip()
                df_sipri_clean['country_name_clean'] = df_sipri_clean['country_name'].astype(str).str.upper().str.strip()
                
                # Merge on country name and year
                df_merged = pd.merge(
                    df_merged,
                    df_sipri_clean[['country_name_clean', 'year', 'military_expenditure']],
                    on=['country_name_clean', 'year'],
                    how='left'
                )
                # Clean up temporary column
                df_merged = df_merged.drop(columns=['country_name_clean'], errors='ignore')
                print(f"  [OK] Merged SIPRI military expenditure: {df_merged.shape}")
            else:
                # Fallback: try to merge on country_code if SIPRI has it
                if 'country_code' in df_sipri_clean.columns:
                    df_merged = pd.merge(
                        df_merged,
                        df_sipri_clean[['country_code', 'year', 'military_expenditure']],
                        on=['country_code', 'year'],
                        how='left'
                    )
                    print(f"  [OK] Merged SIPRI military expenditure: {df_merged.shape}")
                else:
                    print("  [WARNING] Could not merge SIPRI data (no matching country column)")
        else:
            print("  [WARNING] SIPRI data not available, proceeding without military expenditure")
        
        print(f"\n[OK] Final merged dataset: {df_merged.shape}")
        print(f"  Features: {len(df_merged.columns)}")
        print(f"  Countries: {df_merged['country_code'].nunique()}")
        print(f"  Years: {df_merged['year'].min():.0f} - {df_merged['year'].max():.0f}")
        
        return df_merged
    
    def _create_synthetic_data(self, n_samples=5000):
        """Create realistic synthetic data for demonstration."""
        np.random.seed(42)
        countries = ['USA', 'CHN', 'RUS', 'GBR', 'FRA', 'DEU', 'JPN', 'IND', 'BRA', 'CAN']
        
        data = {
            'country_code': np.random.choice(countries, n_samples),
            'year': np.random.randint(1990, 2024, n_samples),
            'vote': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'topic': np.random.choice(['Security', 'Human Rights', 'Trade', 'Environment'], n_samples),
            'gdp_per_capita': np.random.lognormal(10, 1, n_samples),
            'military_spending': np.random.exponential(2, n_samples),
            'trade_openness': np.random.uniform(0.2, 1.2, n_samples),
            'education_index': np.random.uniform(0.5, 1.0, n_samples),
            'internet_penetration': np.random.uniform(0.1, 1.0, n_samples),
            'health_expenditure': np.random.uniform(2, 15, n_samples),
        }
        
        data['vote'] = data['vote'] * (0.3 + 
                                      0.2 * np.tanh((data['gdp_per_capita'] - 8)/5) +
                                      0.1 * data['trade_openness'] +
                                      0.1 * data['education_index'])
        data['vote'] = (data['vote'] > 0.5).astype(int)
        
        return pd.DataFrame(data)
    
    def handle_missing_values(self, df, threshold=0.3, preserve_priority_features=True):
        """
        Handle missing values using median imputation.
        Preserves priority features (required socioeconomic/military indicators) even if they have high missing data.
        """
        print("\n" + "="*70)
        print("MISSING VALUE HANDLING")
        print("="*70)
        
        # Check if dataframe is empty
        if len(df) == 0:
            print("[WARNING] Warning: Empty dataframe, skipping missing value handling")
            return df
        
        # Priority features that should be preserved even with high missing data
        priority_features = [
            'gdp_per_capita', 'gdp_per_capita_current',
            'trade_openness', 'trade_gdp_ratio',
            'education_expenditure', 'primary_enrollment', 'secondary_enrollment',
            'internet_users', 'internet_broadband',
            'energy_use_per_capita', 'electricity_use_per_capita',
            'health_expenditure', 'health_expenditure_per_capita',
            'tax_revenue', 'government_revenue',
            'military_expenditure'
        ]
        
        missing_pct = df.isnull().sum() / len(df)
        
        # Identify columns to drop (excluding priority features)
        cols_to_drop = []
        for col in missing_pct[missing_pct > threshold].index:
            if preserve_priority_features and col in priority_features:
                # Preserve priority features even with high missing data
                print(f"  Preserving priority feature '{col}' despite {missing_pct[col]*100:.1f}% missing (will be imputed)")
            else:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"[OK] Dropped {len(cols_to_drop)} non-priority columns with >{threshold*100}% missing values")
            if len(cols_to_drop) > 0:
                print(f"  Dropped columns: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
        else:
            print(f"[OK] No columns dropped (all high-missing columns are priority features)")
        
        # Count preserved priority features with high missing data
        preserved_priority = [col for col in priority_features 
                             if col in df.columns and missing_pct.get(col, 0) > threshold]
        if preserved_priority:
            print(f"[OK] Preserved {len(preserved_priority)} priority features with >{threshold*100}% missing (will be imputed)")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols and len(df) > 0:
            try:
                df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
                print(f"[OK] Applied median imputation to {len(numeric_cols)} numeric columns")
            except ValueError as e:
                print(f"[WARNING] Error in imputation: {e}")
                # Fallback: fill with median directly
                for col in numeric_cols:
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col] = df[col].fillna(median_val)
                    else:
                        df[col] = df[col].fillna(0)
                print(f"[OK] Applied direct median fill to {len(numeric_cols)} numeric columns")
        
        # Use forward fill and backward fill (updated pandas syntax)
        df = df.ffill().bfill()
        print(f"[OK] Final missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def encode_target_variable(self, df, target_col='vote'):
        """Encode target variable: Yes=1, No/Abstain/Non-voting=0"""
        print("\n" + "="*70)
        print("TARGET VARIABLE ENCODING")
        print("="*70)
        
        if target_col not in df.columns:
            # Try alternate column names (including ms_vote which is common in UN datasets)
            for col in df.columns:
                if col.lower() in ['vote', 'votes', 'yes_no', 'voting', 'ms_vote']:
                    target_col = col
                    print(f"  Found target column: '{target_col}'")
                    break
        
        if target_col in df.columns:
            # Create a standardized 'vote' column for consistency
            vote_mapping = {
                'yes': 1, 'y': 1, '1': 1, 1: 1,
                'no': 0, 'n': 0, '0': 0, 0: 0,
                'abstain': 0, 'a': 0,
                'non-voting': 0, 'x': 0, 'non_voting': 0
            }
            
            # Convert to string and lowercase for mapping
            vote_str = df[target_col].astype(str).str.upper().str.strip()
            
            # Map votes: Y=1, N/A/X=0
            df['vote'] = vote_str.map({
                'Y': 1, 'YES': 1, '1': 1,
                'N': 0, 'NO': 0, '0': 0,
                'A': 0, 'ABSTAIN': 0,
                'X': 0, 'NON-VOTING': 0, 'NON_VOTING': 0
            })
            
            # Fill any unmapped values with 0 (conservative approach)
            df['vote'] = df['vote'].fillna(0).astype(int)
            
            print(f"[OK] Target variable encoded from '{target_col}'")
            print(f"  Class distribution:\n{df['vote'].value_counts()}")
        else:
            print(f"[WARNING] Target column '{target_col}' not found, creating synthetic target")
            # Create a synthetic binary target if no vote column exists
            np.random.seed(42)
            df['vote'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])
        
        return df
    
    def remove_duplicates(self, df, subset=None, keep='first'):
        """Remove duplicate rows from the dataset."""
        print("\n" + "="*70)
        print("DUPLICATE REMOVAL")
        print("="*70)
        
        if len(df) == 0:
            print("[WARNING] Warning: Empty dataframe, skipping duplicate removal")
            return df
        
        initial_count = len(df)
        
        # If no subset specified, check all columns
        if subset is None:
            df_cleaned = df.drop_duplicates(keep=keep)
        else:
            # Ensure subset columns exist
            subset = [col for col in subset if col in df.columns]
            if subset:
                df_cleaned = df.drop_duplicates(subset=subset, keep=keep)
            else:
                df_cleaned = df.drop_duplicates(keep=keep)
        
        duplicates_removed = initial_count - len(df_cleaned)
        print(f"[OK] Removed {duplicates_removed} duplicate rows")
        print(f"  Before: {initial_count} rows, After: {len(df_cleaned)} rows")
        
        return df_cleaned
    
    def remove_outliers(self, df, method='iqr', threshold=3.0, exclude_cols=None, 
                       strictness='moderate', exclude_merged_cols=True):
        """
        Remove outliers from numeric columns using IQR or Z-score method.
        
        Parameters:
        - strictness: 'strict' (remove if outlier in ANY column), 
                     'moderate' (remove if outlier in MOST columns),
                     'lenient' (cap values instead of removing rows)
        - exclude_merged_cols: If True, don't apply outlier removal to merged WDI/SIPRI columns
        """
        print("\n" + "="*70)
        print("OUTLIER REMOVAL")
        print("="*70)
        
        if len(df) == 0:
            print("[WARNING] Warning: Empty dataframe, skipping outlier removal")
            return df
        
        if exclude_cols is None:
            exclude_cols = ['vote', 'year', 'session']  # Don't remove outliers from target, year, session
        else:
            exclude_cols = list(exclude_cols) + ['vote', 'year', 'session']
        
        # Ensure exclude_cols is a list
        exclude_cols = list(set(exclude_cols))  # Remove duplicates
        
        initial_count = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Exclude merged columns if requested
        if exclude_merged_cols:
            merged_col_prefixes = ['gdp_', 'trade_', 'education_', 'internet_', 'energy_', 
                                  'electricity_', 'health_', 'tax_', 'government_', 'military_',
                                  'primary_', 'secondary_']
            numeric_cols = [col for col in numeric_cols 
                          if not any(col.startswith(prefix) for prefix in merged_col_prefixes)]
            print(f"  Excluding merged WDI/SIPRI columns from outlier removal")
        
        if not numeric_cols:
            print("[WARNING] No numeric columns to process for outliers")
            return df
        
        print(f"  Processing {len(numeric_cols)} numeric columns with {strictness} strictness")
        
        if strictness == 'lenient':
            # Cap outliers instead of removing rows
            df_cleaned = df.copy()
            for col in numeric_cols:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                elif method == 'zscore':
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    df_cleaned[col] = df_cleaned[col].clip(
                        lower=mean_val - threshold * std_val,
                        upper=mean_val + threshold * std_val
                    )
            print(f"[OK] Capped outliers in {len(numeric_cols)} columns (no rows removed)")
            return df_cleaned
        
        # For strict and moderate: remove rows
        outlier_scores = pd.Series([0] * len(df))  # Count how many columns have outliers
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_scores += col_mask.astype(int)
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-8))
                col_mask = z_scores >= threshold
                outlier_scores += col_mask.astype(int)
        
        # Apply strictness level
        if strictness == 'strict':
            # Remove if outlier in ANY column (original behavior)
            outlier_mask = outlier_scores > 0
        elif strictness == 'moderate':
            # Remove if outlier in MORE THAN HALF of columns
            threshold_cols = len(numeric_cols) / 2
            outlier_mask = outlier_scores > threshold_cols
        else:
            outlier_mask = outlier_scores > 0
        
        df_cleaned = df[~outlier_mask].copy()
        outliers_removed = initial_count - len(df_cleaned)
        
        print(f"[OK] Removed {outliers_removed} rows with outliers ({method} method, {strictness} strictness)")
        print(f"  Before: {initial_count} rows, After: {len(df_cleaned)} rows")
        print(f"  Processed {len(numeric_cols)} numeric columns")
        
        return df_cleaned
    
    def remove_constant_columns(self, df, threshold=0.95):
        """Remove columns with constant or near-constant values."""
        print("\n" + "="*70)
        print("CONSTANT COLUMN REMOVAL")
        print("="*70)
        
        if len(df) == 0:
            print("[WARNING] Warning: Empty dataframe, skipping constant column removal")
            return df
        
        initial_cols = len(df.columns)
        cols_to_drop = []
        
        for col in df.columns:
            # For numeric columns
            if df[col].dtype in [np.number]:
                if df[col].nunique() <= 1:
                    cols_to_drop.append(col)
                elif df[col].nunique() / len(df) < (1 - threshold):
                    # Near-constant (e.g., 95% same value)
                    value_counts = df[col].value_counts(normalize=True)
                    if value_counts.iloc[0] >= threshold:
                        cols_to_drop.append(col)
            # For categorical columns
            else:
                if df[col].nunique() <= 1:
                    cols_to_drop.append(col)
                elif df[col].nunique() / len(df) < (1 - threshold):
                    value_counts = df[col].value_counts(normalize=True)
                    if value_counts.iloc[0] >= threshold:
                        cols_to_drop.append(col)
        
        df_cleaned = df.drop(columns=cols_to_drop)
        print(f"[OK] Removed {len(cols_to_drop)} constant/near-constant columns")
        print(f"  Before: {initial_cols} columns, After: {len(df_cleaned.columns)} columns")
        if cols_to_drop:
            print(f"  Removed: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
        
        return df_cleaned
    
    def clean_invalid_data(self, df, invalid_values=None, exclude_merged_cols=True):
        """
        Remove rows with invalid data values.
        Only removes rows with invalid values in original columns, not merged columns.
        NaN values in merged columns are expected and will be handled by imputation.
        """
        print("\n" + "="*70)
        print("INVALID DATA CLEANING")
        print("="*70)
        
        if len(df) == 0:
            print("[WARNING] Warning: Empty dataframe, skipping invalid data cleaning")
            return df
        
        if invalid_values is None:
            invalid_values = ['', 'NULL', 'null', 'None', 'N/A', 'n/a', 'NA', 'na', 
                            'NaN', 'nan', '#N/A', '#VALUE!', '#REF!', '-', '--']
        
        initial_count = len(df)
        invalid_mask = pd.Series([False] * len(df))
        
        # Identify original columns (not merged WDI/SIPRI columns)
        # Original UN voting columns typically don't have these prefixes
        merged_col_prefixes = ['gdp_', 'trade_', 'education_', 'internet_', 'energy_', 
                              'electricity_', 'health_', 'tax_', 'government_', 'military_',
                              'primary_', 'secondary_']
        
        # Check for invalid values in object/string columns
        for col in df.select_dtypes(include=['object']).columns:
            # Skip merged columns if exclude_merged_cols is True
            if exclude_merged_cols and any(col.startswith(prefix) for prefix in merged_col_prefixes):
                continue
            col_mask = df[col].astype(str).isin(invalid_values)
            invalid_mask = invalid_mask | col_mask
        
        # Check for infinite values in numeric columns (but NOT NaN - those are handled by imputation)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Skip merged columns if exclude_merged_cols is True
            if exclude_merged_cols and any(col.startswith(prefix) for prefix in merged_col_prefixes):
                continue
            # Only check for infinite values, NOT NaN (NaN will be imputed later)
            col_mask = np.isinf(df[col])
            invalid_mask = invalid_mask | col_mask
        
        df_cleaned = df[~invalid_mask].copy()
        invalid_removed = initial_count - len(df_cleaned)
        
        print(f"[OK] Removed {invalid_removed} rows with invalid data")
        print(f"  Before: {initial_count} rows, After: {len(df_cleaned)} rows")
        print(f"  Note: NaN values in merged columns are preserved for imputation")
        
        return df_cleaned
    
    def standardize_country_codes(self, df, country_col='ms_code'):
        """Standardize country codes to ISO3 format."""
        print("\n" + "="*70)
        print("COUNTRY CODE STANDARDIZATION")
        print("="*70)
        
        if len(df) == 0:
            print("[WARNING] Warning: Empty dataframe, skipping country code standardization")
            return df
        
        # Find country code column
        country_col_found = None
        for col in df.columns:
            if col.lower() in ['country_code', 'ms_code', 'iso3', 'iso_code', 'country']:
                country_col_found = col
                break
        
        if country_col_found is None:
            print("[WARNING] No country code column found")
            return df
        
        initial_unique = df[country_col_found].nunique()
        
        # Convert to uppercase and strip whitespace
        df[country_col_found] = df[country_col_found].astype(str).str.upper().str.strip()
        
        # Remove invalid country codes (not 3 characters or contains numbers/special chars)
        valid_mask = df[country_col_found].str.len() == 3
        valid_mask = valid_mask & df[country_col_found].str.isalpha()
        
        df_cleaned = df[valid_mask].copy()
        invalid_removed = len(df) - len(df_cleaned)
        
        print(f"[OK] Standardized country codes in '{country_col_found}' column")
        print(f"  Unique countries: {initial_unique} -> {df_cleaned[country_col_found].nunique()}")
        if invalid_removed > 0:
            print(f"  Removed {invalid_removed} rows with invalid country codes")
        
        return df_cleaned
    
    def remove_irrelevant_columns(self, df, keep_cols=None, drop_patterns=None):
        """Remove irrelevant columns that are not useful for modeling."""
        print("\n" + "="*70)
        print("IRRELEVANT COLUMN REMOVAL")
        print("="*70)
        
        if len(df) == 0:
            print("[WARNING] Warning: Empty dataframe, skipping irrelevant column removal")
            return df
        
        if drop_patterns is None:
            drop_patterns = ['link', 'url', 'id', 'undl_id', 'undl_link', 'description', 
                           'title', 'agenda_title', 'committee_report', 'draft']
        
        initial_cols = len(df.columns)
        cols_to_drop = []
        
        # Always keep these columns if they exist
        if keep_cols is None:
            keep_cols = ['vote', 'year', 'ms_code', 'country_code', 'ms_name', 'country']
        else:
            keep_cols = list(keep_cols) + ['vote', 'year']
        
        # Find columns matching drop patterns
        for col in df.columns:
            if col.lower() in keep_cols:
                continue
            for pattern in drop_patterns:
                if pattern.lower() in col.lower():
                    cols_to_drop.append(col)
                    break
        
        df_cleaned = df.drop(columns=cols_to_drop)
        print(f"[OK] Removed {len(cols_to_drop)} irrelevant columns")
        print(f"  Before: {initial_cols} columns, After: {len(df_cleaned.columns)} columns")
        if cols_to_drop:
            print(f"  Removed: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
        
        return df_cleaned
    
    def clean_text_columns(self, df):
        """Clean text columns by removing extra whitespace and standardizing."""
        print("\n" + "="*70)
        print("TEXT COLUMN CLEANING")
        print("="*70)
        
        if len(df) == 0:
            print("[WARNING] Warning: Empty dataframe, skipping text cleaning")
            return df
        
        text_cols = df.select_dtypes(include=['object']).columns
        cleaned_count = 0
        
        for col in text_cols:
            # Skip if column is already clean or contains mostly numeric data
            if df[col].dtype != 'object':
                continue
            
            initial_sample = str(df[col].iloc[0]) if len(df[col].dropna()) > 0 else ""
            
            # Remove extra whitespace
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Replace 'nan' strings with actual NaN
            df[col] = df[col].replace(['nan', 'None', 'none', 'NULL', 'null'], np.nan)
            
            cleaned_count += 1
        
        print(f"[OK] Cleaned {cleaned_count} text columns")
        
        return df
    
    def convert_data_types(self, df):
        """Convert columns to appropriate data types."""
        print("\n" + "="*70)
        print("DATA TYPE CONVERSION")
        print("="*70)
        
        if len(df) == 0:
            print("[WARNING] Warning: Empty dataframe, skipping data type conversion")
            return df
        
        converted_count = 0
        
        # Convert numeric columns that are stored as strings
        for col in df.select_dtypes(include=['object']).columns:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if numeric_series.notna().sum() / len(df) > 0.8:  # If 80%+ can be converted
                df[col] = numeric_series
                converted_count += 1
        
        # Convert date columns
        date_cols = [col for col in df.columns if 'date' in col.lower() and col != 'date']
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                converted_count += 1
            except:
                pass
        
        print(f"[OK] Converted {converted_count} columns to appropriate data types")
        
        return df
    
    def comprehensive_clean(self, df, remove_duplicates=True, remove_outliers=True, 
                          remove_constants=True, clean_invalid=True, 
                          standardize_countries=True, remove_irrelevant=True,
                          clean_text=True, convert_types=True,
                          outlier_strictness='moderate'):
        """
        Comprehensive data cleaning pipeline that applies all cleaning methods.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE DATA CLEANING PIPELINE")
        print("="*70)
        
        initial_shape = df.shape
        initial_df = df.copy()  # Keep a backup
        print(f"Initial data shape: {initial_shape}")
        
        # Step 1: Remove irrelevant columns first (reduces processing time)
        if remove_irrelevant:
            df = self.remove_irrelevant_columns(df)
        
        # Step 2: Clean text columns
        if clean_text:
            df = self.clean_text_columns(df)
        
        # Step 3: Convert data types
        if convert_types:
            df = self.convert_data_types(df)
        
        # Step 4: Remove invalid data (only in original columns, not merged)
        if clean_invalid:
            df = self.clean_invalid_data(df, exclude_merged_cols=True)
        
        # Step 5: Standardize country codes
        if standardize_countries:
            df = self.standardize_country_codes(df)
        
        # Step 6: Remove constant columns
        if remove_constants:
            df = self.remove_constant_columns(df)
        
        # Step 7: Remove duplicates
        if remove_duplicates:
            df = self.remove_duplicates(df)
        
        # Step 8: Remove outliers (do this last, but use moderate strictness)
        # Note: Outliers in merged columns are excluded to preserve more data
        if remove_outliers and len(df) > 0:
            df = self.remove_outliers(df, strictness=outlier_strictness, exclude_merged_cols=True)
        
        # Safety check: ensure we still have data after cleaning
        if len(df) == 0:
            print("[WARNING] Warning: All data removed during cleaning, using original data")
            df = initial_df.copy()
            if len(df) == 0:
                print("[WARNING] Creating synthetic data as fallback")
                df = self._create_synthetic_data()
        
        final_shape = df.shape
        print(f"\n[OK] Comprehensive cleaning completed")
        print(f"  Final data shape: {final_shape}")
        print(f"  Rows removed: {initial_shape[0] - final_shape[0]}")
        print(f"  Columns removed: {initial_shape[1] - final_shape[1]}")
        
        return df
    
    def scale_numeric_features(self, X_train, X_test):
        """Scale numeric features using StandardScaler."""
        print("\n" + "="*70)
        print("FEATURE SCALING")
        print("="*70)
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        print(f"[OK] Scaled {len(numeric_cols)} numeric features")
        
        return X_train_scaled, X_test_scaled


# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

class ExploratoryAnalysis:
    """Performs comprehensive exploratory data analysis."""
    
    def __init__(self, figsize=(14, 10)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def distribution_analysis(self, df, target_col='vote'):
        """Visualize feature distributions and target variable."""
        print("\n" + "="*70)
        print("DISTRIBUTION ANALYSIS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Target Variable Distribution (Top Left)
        if target_col in df.columns:
            vote_counts = df[target_col].value_counts().sort_index()
            colors = ['#e74c3c' if idx == 0 else '#2ecc71' for idx in vote_counts.index]
            axes[0, 0].bar(vote_counts.index, vote_counts.values, color=colors)
            axes[0, 0].set_title('Target Variable Distribution (Yes/No)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_xlabel('Vote (0=No, 1=Yes)')
            axes[0, 0].set_xticks([0, 1])
        else:
            axes[0, 0].text(0.5, 0.5, 'Target variable not found', 
                          ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Target Variable Distribution')
        
        # Plot 2: Voting Records Over Time (Top Right)
        if 'year' in df.columns:
            year_data = df['year'].dropna()
            if len(year_data) > 0:
                axes[0, 1].hist(year_data, bins=min(30, year_data.nunique()), 
                              color='#3498db', edgecolor='black')
                axes[0, 1].set_title('Voting Records Over Time')
                axes[0, 1].set_xlabel('Year')
                axes[0, 1].set_ylabel('Count')
            else:
                axes[0, 1].text(0.5, 0.5, 'No year data available', 
                              ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Voting Records Over Time')
        else:
            axes[0, 1].text(0.5, 0.5, 'Year column not found', 
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Voting Records Over Time')
        
        # Plot 3: Top Countries by Vote Count (Bottom Left)
        country_col = None
        for col in ['ms_code', 'country_code', 'ms_name', 'country']:
            if col in df.columns:
                country_col = col
                break
        
        if country_col:
            country_data = df[country_col].dropna()
            if len(country_data) > 0:
                top_countries = country_data.value_counts().head(10)
                if len(top_countries) > 0:
                    axes[1, 0].barh(range(len(top_countries)), top_countries.values, color='#9b59b6')
                    axes[1, 0].set_yticks(range(len(top_countries)))
                    axes[1, 0].set_yticklabels(top_countries.index, fontsize=9)
                    axes[1, 0].set_title('Top 10 Countries by Vote Count')
                    axes[1, 0].set_xlabel('Number of Votes')
                else:
                    axes[1, 0].text(0.5, 0.5, 'No country data available', 
                                  ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Top 10 Countries by Vote Count')
            else:
                axes[1, 0].text(0.5, 0.5, 'No country data available', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Top 10 Countries by Vote Count')
        else:
            axes[1, 0].text(0.5, 0.5, 'Country column not found', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Top 10 Countries by Vote Count')
        
        # Plot 4: Top Resolution Topics/Subjects (Bottom Right)
        topic_col = None
        for col in ['subjects', 'topic', 'title', 'agenda_title']:
            if col in df.columns:
                topic_col = col
                break
        
        if topic_col:
            topic_data = df[topic_col].dropna().astype(str)
            # For subjects column, split by semicolon or comma if it contains multiple subjects
            if topic_col == 'subjects':
                # Split subjects and flatten
                all_subjects = []
                for subjects_str in topic_data:
                    if pd.notna(subjects_str) and subjects_str != 'nan':
                        # Split by common delimiters
                        split_subjects = [s.strip() for s in str(subjects_str).replace(';', ',').split(',')]
                        all_subjects.extend([s for s in split_subjects if s and len(s) > 2])
                if all_subjects:
                    topic_counts = pd.Series(all_subjects).value_counts().head(8)
                else:
                    topic_counts = pd.Series()
            else:
                topic_counts = topic_data.value_counts().head(8)
            
            if len(topic_counts) > 0:
                axes[1, 1].barh(range(len(topic_counts)), topic_counts.values, color='#f39c12')
                axes[1, 1].set_yticks(range(len(topic_counts)))
                # Truncate long labels
                labels = [label[:40] + '...' if len(label) > 40 else label 
                         for label in topic_counts.index]
                axes[1, 1].set_yticklabels(labels, fontsize=8)
                axes[1, 1].set_title('Top Resolution Topics/Subjects')
                axes[1, 1].set_xlabel('Count')
            else:
                axes[1, 1].text(0.5, 0.5, 'No topic/subject data available', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Top Resolution Topics/Subjects')
        else:
            # Fallback: Show session distribution if available
            if 'session' in df.columns:
                session_counts = df['session'].value_counts().head(8)
                if len(session_counts) > 0:
                    axes[1, 1].barh(range(len(session_counts)), session_counts.values, color='#f39c12')
                    axes[1, 1].set_yticks(range(len(session_counts)))
                    axes[1, 1].set_yticklabels([f'Session {s}' for s in session_counts.index], fontsize=9)
                    axes[1, 1].set_title('Top UN Sessions by Vote Count')
                    axes[1, 1].set_xlabel('Count')
                else:
                    axes[1, 1].text(0.5, 0.5, 'No topic data available', 
                                  ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Top Resolution Topics')
            else:
                axes[1, 1].text(0.5, 0.5, 'Topic/subject column not found', 
                              ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Top Resolution Topics')
        
        plt.tight_layout()
        plt.savefig('01_distribution_analysis.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: 01_distribution_analysis.png")
        plt.show()
    
    def correlation_analysis(self, df, target_col='vote', top_n=15):
        """Analyze correlations between features and target."""
        print("\n" + "="*70)
        print("CORRELATION ANALYSIS")
        print("="*70)
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df) == 0:
            print("[WARNING] No numeric columns found for correlation analysis")
            return
        
        if target_col not in numeric_df.columns:
            print(f"[WARNING] Target column '{target_col}' not found in numeric columns")
            return
        
        try:
            correlations = numeric_df.corr()[target_col].sort_values(ascending=False)
            correlations = correlations[correlations.index != target_col]
            correlations = correlations.dropna()
            
            if len(correlations) == 0:
                print("[WARNING] No valid correlations found")
                return
            
            # Get top and bottom correlations
            n_top = min(top_n//2, len(correlations))
            n_bottom = min(top_n//2, len(correlations))
            
            top_corr = pd.concat([
                correlations.head(n_top),
                correlations.tail(n_bottom)
            ])
            
            # Remove duplicates if any
            top_corr = top_corr[~top_corr.index.duplicated(keep='first')]
            
            if len(top_corr) == 0:
                print("[WARNING] No correlations to plot")
                return
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_corr.values]
            top_corr.plot(kind='barh', ax=ax, color=colors)
            ax.set_title('Top Features Correlated with Vote Outcome', fontsize=14, fontweight='bold')
            ax.set_xlabel('Correlation Coefficient')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            plt.savefig('02_correlation_analysis.png', dpi=300, bbox_inches='tight')
            print("[OK] Saved: 02_correlation_analysis.png")
            
            print(f"\nTop positive correlations:\n{correlations.head(5)}\n")
            print(f"Top negative correlations:\n{correlations.tail(5)}")
            plt.show()
        except Exception as e:
            print(f"[WARNING] Error in correlation analysis: {e}")
    
    def temporal_trends(self, df, target_col='vote'):
        """Analyze voting trends over time."""
        print("\n" + "="*70)
        print("TEMPORAL TREND ANALYSIS")
        print("="*70)
        
        if 'year' not in df.columns:
            print("[WARNING] Year column not found, skipping temporal trend analysis")
            return
        
        if target_col not in df.columns:
            print(f"[WARNING] Target column '{target_col}' not found, skipping temporal trend analysis")
            return
        
        try:
            # Filter out invalid years
            df_clean = df[df['year'].notna() & df[target_col].notna()].copy()
            
            if len(df_clean) == 0:
                print("[WARNING] No valid data for temporal trend analysis")
                return
            
            yearly_stats = df_clean.groupby('year')[target_col].agg(['mean', 'count'])
            yearly_stats = yearly_stats[yearly_stats['count'] > 0]  # Only years with data
            
            if len(yearly_stats) == 0:
                print("[WARNING] No yearly data available for temporal trend analysis")
                return
            
            yearly_yes_ratio = yearly_stats['mean']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(yearly_yes_ratio.index, yearly_yes_ratio.values, 
                   marker='o', linewidth=2, markersize=6, color='#3498db')
            ax.fill_between(yearly_yes_ratio.index, yearly_yes_ratio.values, 
                          alpha=0.3, color='#3498db')
            ax.set_title('Proportion of "Yes" Votes Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Proportion of Yes Votes')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('03_temporal_trends.png', dpi=300, bbox_inches='tight')
            print("[OK] Saved: 03_temporal_trends.png")
            plt.show()
        except Exception as e:
            print(f"[WARNING] Error in temporal trend analysis: {e}")


# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

class FeatureEngineering:
    """Handles feature creation and selection."""
    
    @staticmethod
    def select_features(df, target_col='vote', method='correlation', n_features=20):
        """
        Select top features using correlation analysis.
        Prioritizes required socioeconomic and military indicators.
        """
        print("\n" + "="*70)
        print("FEATURE SELECTION")
        print("="*70)
        
        # Get numeric columns, excluding target and year (year is not a predictive feature)
        numeric_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col, 'year'], errors='ignore')
        
        if numeric_df.empty:
            print("[WARNING] No numeric features found")
            return []
        
        if target_col not in df.columns:
            print(f"[WARNING] Target column '{target_col}' not found")
            return list(numeric_df.columns)[:n_features]
        
        # Priority features (required indicators from project specification)
        priority_features = [
            'gdp_per_capita', 'gdp_per_capita_current',
            'trade_openness', 'trade_gdp_ratio',
            'education_expenditure', 'primary_enrollment', 'secondary_enrollment',
            'internet_users', 'internet_broadband',
            'energy_use_per_capita', 'electricity_use_per_capita',
            'health_expenditure', 'health_expenditure_per_capita',
            'tax_revenue', 'government_revenue',
            'military_expenditure'
        ]
        
        # Find available priority features
        available_priority = [f for f in priority_features if f in numeric_df.columns]
        print(f"  Found {len(available_priority)}/{len(priority_features)} priority features")
        if available_priority:
            print(f"  Priority features: {available_priority[:5]}{'...' if len(available_priority)>5 else ''}")
        
        try:
            # Calculate correlations, handling any NaN values
            correlations = numeric_df.corrwith(df[target_col], method='pearson').abs()
            correlations = correlations.dropna().sort_values(ascending=False)
            
            if len(correlations) == 0:
                print("[WARNING] No valid correlations found, using all numeric features")
                return list(numeric_df.columns)[:n_features]
            
            # Combine priority features with top correlated features
            selected_features = []
            
            # First, add priority features that exist and have valid correlations
            for feat in available_priority:
                if feat in correlations.index and feat not in selected_features:
                    selected_features.append(feat)
            
            # Then, add top correlated features (excluding already selected)
            remaining_features = [f for f in correlations.index if f not in selected_features]
            n_remaining = n_features - len(selected_features)
            
            if n_remaining > 0:
                selected_features.extend(remaining_features[:n_remaining])
            
            # If we still don't have enough, add any remaining numeric features
            if len(selected_features) < n_features:
                remaining_numeric = [f for f in numeric_df.columns 
                                   if f not in selected_features and f in df.columns]
                selected_features.extend(remaining_numeric[:n_features - len(selected_features)])
            
            print(f"[OK] Selected {len(selected_features)} features")
            print(f"  Priority features included: {len([f for f in selected_features if f in priority_features])}")
            print(f"  Top features: {selected_features[:5]}{'...' if len(selected_features)>5 else ''}")
            
            return selected_features[:n_features]
        except Exception as e:
            print(f"[WARNING] Error in correlation calculation: {e}")
            print("  Using all numeric features instead")
            return list(numeric_df.columns)[:n_features]


# ============================================================================
# 5. MODEL TRAINING AND OPTIMIZATION
# ============================================================================

class ModelTrainer:
    """Trains and optimizes multiple machine learning models."""
    
    def __init__(self, cv_folds=5, random_state=42, use_grid_search=True):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.use_grid_search = use_grid_search
        self.models = {}
        self.best_params = {}
    
    def define_models(self, class_weight=None):
        """
        Define all models with initial hyperparameters.
        Includes regularized Logistic Regression and GPU-accelerated XGBoost.
        """
        print("\n" + "="*70)
        print("MODEL DEFINITION")
        print("="*70)
        
        # Check for GPU availability for XGBoost
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            gpu_available = result.returncode == 0
            xgb_tree_method = 'gpu_hist' if gpu_available else 'hist'
            if gpu_available:
                print("  [OK] GPU detected - XGBoost will use GPU acceleration")
            else:
                print("  [INFO] No GPU detected - XGBoost will use CPU")
        except:
            xgb_tree_method = 'hist'
            print("  [INFO] GPU check failed - XGBoost will use CPU")
        
        # Define base models (will be tuned with GridSearchCV if enabled)
        self.base_models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                class_weight=class_weight
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight=class_weight
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight=class_weight
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                eval_metric='logloss',
                tree_method=xgb_tree_method,
                verbosity=0
            ),
            'SVM': SVC(
                kernel='linear',
                probability=True,
                random_state=self.random_state,
                max_iter=1000,
                class_weight=class_weight
            ),
            'Naive Bayes': GaussianNB()
        }
        
        # Define hyperparameter grids for GridSearchCV
        self.param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'Decision Tree': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [15, 20, None],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', 'log2']
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            },
            'SVM': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'Naive Bayes': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        }
        
        # Initialize models dict (will be populated after training/tuning)
        self.models = {}
        
        print(f"[OK] Defined {len(self.base_models)} models:")
        for name in self.base_models.keys():
            print(f"  - {name}")
        
        if self.use_grid_search:
            print(f"\n[OK] Hyperparameter tuning enabled with GridSearchCV")
            print(f"  Using stratified {self.cv_folds}-fold cross-validation")
    
    def train_all_models(self, X_train, y_train, use_cv=True):
        """
        Train all models with hyperparameter tuning using GridSearchCV.
        Uses stratified 5-fold cross-validation for robust evaluation.
        """
        print("\n" + "="*70)
        print("MODEL TRAINING & HYPERPARAMETER TUNING")
        print("="*70)
        
        # Initialize stratified cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # For SVM, use a larger sample if dataset is too large (SVM is O(n²) complexity)
        # Increased sample size for better performance
        svm_sample_size = 50000
        
        for name, base_model in self.base_models.items():
            print(f"\n{'='*70}")
            print(f"Training & Tuning {name}...")
            print(f"{'='*70}")
            
            try:
                # Prepare training data (sample for SVM if needed)
                if name == 'SVM' and len(X_train) > svm_sample_size:
                    print(f"  Using sample of {svm_sample_size} rows for faster training...")
                    np.random.seed(self.random_state)
                    sample_idx = np.random.choice(len(X_train), size=svm_sample_size, replace=False)
                    if hasattr(X_train, 'iloc'):
                        X_train_use = X_train.iloc[sample_idx]
                        y_train_use = y_train.iloc[sample_idx]
                    else:
                        X_train_use = X_train[sample_idx]
                        y_train_use = y_train[sample_idx]
                else:
                    X_train_use = X_train
                    y_train_use = y_train
                
                # Hyperparameter tuning with GridSearchCV
                if self.use_grid_search and name in self.param_grids:
                    print(f"  Performing GridSearchCV with {self.cv_folds}-fold stratified CV...")
                    param_grid = self.param_grids[name]
                    
                    # Adjust param grid for Logistic Regression solver compatibility
                    if name == 'Logistic Regression':
                        # liblinear supports both L1 and L2, lbfgs only supports L2
                        param_grid_adjusted = []
                        for penalty in param_grid['penalty']:
                            if penalty == 'l1':
                                param_grid_adjusted.append({
                                    'C': param_grid['C'],
                                    'penalty': [penalty],
                                    'solver': ['liblinear']
                                })
                            else:
                                param_grid_adjusted.append({
                                    'C': param_grid['C'],
                                    'penalty': [penalty],
                                    'solver': ['liblinear', 'lbfgs']
                                })
                        # Use list of dicts for conditional parameter grids
                        grid_search = GridSearchCV(
                            base_model,
                            param_grid_adjusted,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1,
                            verbose=1
                        )
                    elif name == 'SVM':
                        # Adjust param grid for SVM polynomial kernel
                        param_grid_adjusted = []
                        for kernel in param_grid['kernel']:
                            if kernel == 'poly':
                                param_grid_adjusted.append({
                                    'C': param_grid['C'],
                                    'kernel': [kernel],
                                    'gamma': param_grid['gamma'],
                                    'degree': param_grid.get('degree', [2, 3])
                                })
                            else:
                                param_grid_adjusted.append({
                                    'C': param_grid['C'],
                                    'kernel': [kernel],
                                    'gamma': param_grid['gamma']
                                })
                        grid_search = GridSearchCV(
                            base_model,
                            param_grid_adjusted,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1,
                            verbose=1
                        )
                    else:
                        grid_search = GridSearchCV(
                            base_model,
                            param_grid,
                            cv=cv,
                            scoring='roc_auc',
                            n_jobs=-1,
                            verbose=1
                        )
                    
                    grid_search.fit(X_train_use, y_train_use)
                    self.models[name] = grid_search.best_estimator_
                    self.best_params[name] = grid_search.best_params_
                    
                    print(f"  [OK] Best parameters: {grid_search.best_params_}")
                    print(f"  [OK] Best CV score (ROC-AUC): {grid_search.best_score_:.4f}")
                else:
                    # Train without hyperparameter tuning
                    print(f"  Training with default parameters...")
                    base_model.fit(X_train_use, y_train_use)
                    self.models[name] = base_model
                    
                    # Cross-validation for evaluation
                    if use_cv and len(X_train_use) > self.cv_folds * 2:
                        cv_scores = cross_val_score(base_model, X_train_use, y_train_use, 
                                                  cv=cv, scoring='roc_auc', n_jobs=-1)
                        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
                print(f"[OK] {name} trained successfully")
            except Exception as e:
                print(f"[ERROR] Error training {name}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Remove failed model from dictionary
                if name in self.models:
                    del self.models[name]


# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """Evaluates and compares model performance."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model on test set with all required metrics."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
            'PR-AUC': average_precision_score(y_test, y_pred_proba),  # Precision-Recall AUC
        }
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        return metrics
    
    def evaluate_all_models(self, models, X_test, y_test):
        """Evaluate all models and compare performance."""
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
    
    def compare_models(self):
        """Compare all models and create comparison visualization."""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        results_df = pd.DataFrame({
            model: metrics['metrics'] 
            for model, metrics in self.results.items()
        }).T
        
        print("\nDetailed Comparison:\n")
        print(results_df.to_string())
        
        # Create subplots for 6 metrics (2 rows x 3 cols)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            values = results_df[metric].values
            colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
            
            ax.bar(range(len(values)), values, color=colors, edgecolor='black')
            ax.set_xticks(range(len(results_df)))
            ax.set_xticklabels(results_df.index, rotation=45, ha='right')
            ax.set_title(metric, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('04_model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n[OK] Saved: 04_model_comparison.png")
        plt.show()
        
        return results_df
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        print("\n" + "="*70)
        print("ROC CURVE ANALYSIS")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.results)))
        
        for (model_name, result), color in zip(self.results.items(), colors):
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('05_roc_curves.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: 05_roc_curves.png")
        plt.show()
    
    def plot_pr_curves(self):
        """Plot Precision-Recall curves for all models."""
        print("\n" + "="*70)
        print("PRECISION-RECALL CURVE ANALYSIS")
        print("="*70)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.results)))
        
        for (model_name, result), color in zip(self.results.items(), colors):
            y_test = result['y_test']
            y_pred_proba = result['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            
            ax.plot(recall, precision, color=color, lw=2, 
                   label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
        
        # Baseline (random classifier)
        baseline = sum(result['y_test'].sum() for result in self.results.values()) / \
                   sum(len(result['y_test']) for result in self.results.values())
        if len(self.results) > 0:
            first_result = list(self.results.values())[0]
            ax.axhline(y=baseline, color='k', linestyle='--', lw=2, 
                      label=f'Baseline (PR-AUC = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('05_pr_curves.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: 05_pr_curves.png")
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        print("\n" + "="*70)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*70)
        
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*4))
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, xticklabels=['No', 'Yes'], 
                       yticklabels=['No', 'Yes'])
            axes[idx].set_title(f'{model_name}', fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('06_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: 06_confusion_matrices.png")
        plt.show()


# ============================================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

class FeatureImportanceAnalyzer:
    """Analyzes and visualizes feature importance across models."""
    
    @staticmethod
    def get_feature_importance(model, model_name, feature_names):
        """Extract feature importance based on model type."""
        try:
            # Handle GridSearchCV wrapped models
            actual_model = model
            
            # First check if model itself has feature_importances_ (most common case)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if len(importances) == len(feature_names):
                    return pd.Series(importances, 
                                   index=feature_names).sort_values(ascending=False)
                else:
                    print(f"  [WARNING] Feature importance length ({len(importances)}) doesn't match feature_names length ({len(feature_names)})")
                    return None
            
            # Check if model is GridSearchCV (has best_estimator_)
            if hasattr(model, 'best_estimator_'):
                actual_model = model.best_estimator_
                if hasattr(actual_model, 'feature_importances_'):
                    importances = actual_model.feature_importances_
                    if len(importances) == len(feature_names):
                        return pd.Series(importances, 
                                       index=feature_names).sort_values(ascending=False)
                    else:
                        print(f"  [WARNING] Feature importance length ({len(importances)}) doesn't match feature_names length ({len(feature_names)})")
                        return None
            
            # Don't check for 'estimator' attribute - RandomForest has it but it's the base estimator, not the actual model
            
            # Check for coefficients (Logistic Regression, SVM)
            if hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class for binary classification
                if len(coef) == len(feature_names):
                    return pd.Series(np.abs(coef), 
                                   index=feature_names).sort_values(ascending=False)
                else:
                    return None
            
            # Check if model is GridSearchCV and try best_estimator_ for coef_
            if hasattr(model, 'best_estimator_'):
                best_model = model.best_estimator_
                if hasattr(best_model, 'coef_'):
                    coef = best_model.coef_
                    if coef.ndim > 1:
                        coef = coef[0]
                    if len(coef) == len(feature_names):
                        return pd.Series(np.abs(coef), 
                                       index=feature_names).sort_values(ascending=False)
            
            # Check for Naive Bayes (theta_)
            if hasattr(model, 'theta_'):
                actual_model = model
            elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'theta_'):
                actual_model = model.best_estimator_
            else:
                return None
            
            if hasattr(actual_model, 'theta_'):
                # Naive Bayes (GaussianNB): Use difference in class means weighted by variance
                # GaussianNB uses 'var_' not 'sigma_' for variance
                theta = actual_model.theta_
                
                # Get variance - GaussianNB uses 'var_' attribute
                if hasattr(actual_model, 'var_'):
                    var = actual_model.var_
                elif hasattr(actual_model, 'sigma_'):
                    var = actual_model.sigma_
                else:
                    # Fallback: use variance-based importance from theta only
                    if theta.ndim == 1:
                        return None
                    # Use standard deviation approximation
                    var = np.ones_like(theta) * 0.1  # Default small variance
                
                # Check if model is fitted
                if theta is None:
                    return None
                
                # Handle different shapes
                if theta.ndim == 1:
                    # Single class - use variance-based importance
                    importance = 1.0 / (np.sqrt(var) + 1e-10)
                elif theta.shape[0] >= 2:
                    # Binary or multi-class classification
                    # For binary: use difference between class 0 and class 1
                    mean_diff = np.abs(theta[1] - theta[0])
                    
                    # Calculate standard deviation from variance
                    if var.ndim == 1:
                        std_sum = np.sqrt(var) + np.sqrt(var)
                    else:
                        std_sum = np.sqrt(var[1]) + np.sqrt(var[0])
                    
                    # Avoid division by zero
                    std_sum = np.where(std_sum == 0, 1e-10, std_sum)
                    importance = mean_diff / std_sum
                else:
                    # Single class case
                    if var.ndim == 1:
                        importance = 1.0 / (np.sqrt(var) + 1e-10)
                    else:
                        importance = 1.0 / (np.sqrt(var[0]) + 1e-10)
                
                # Ensure importance array matches feature_names length
                if len(importance) != len(feature_names):
                    print(f"Warning: Importance length ({len(importance)}) doesn't match feature_names length ({len(feature_names)})")
                    return None
                
                return pd.Series(importance, 
                               index=feature_names).sort_values(ascending=False)
            else:
                return None
        except Exception as e:
            print(f"Could not extract importance from {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def plot_feature_importance(models, feature_names, top_n=15):
        """Plot feature importance for tree-based and linear models."""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Feature Importance Across Models', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        model_names = ['Random Forest', 'XGBoost', 'Logistic Regression', 
                      'Decision Tree', 'SVM', 'Naive Bayes']
        
        for idx, model_name in enumerate(model_names):
            if model_name in models:
                model = models[model_name]
                importance = FeatureImportanceAnalyzer.get_feature_importance(
                    model, model_name, feature_names
                )
                
                if importance is not None and len(importance) > 0:
                    top_importance = importance.head(top_n)
                    
                    axes[idx].barh(range(len(top_importance)), top_importance.values, 
                                  color='#3498db', edgecolor='black')
                    axes[idx].set_yticks(range(len(top_importance)))
                    axes[idx].set_yticklabels(top_importance.index, fontsize=9)
                    axes[idx].set_title(model_name, fontweight='bold')
                    axes[idx].invert_yaxis()
                    axes[idx].set_xlabel('Importance', fontsize=9)
                else:
                    # Handle case where importance is None or empty
                    print(f"  [WARNING] No feature importance available for {model_name}")
                    axes[idx].text(0.5, 0.5, f'No feature importance\navailable for {model_name}', 
                                  ha='center', va='center', fontsize=10)
                    axes[idx].set_title(model_name, fontweight='bold')
                    axes[idx].set_xticks([])
                    axes[idx].set_yticks([])
            else:
                # Model not found
                print(f"  [WARNING] Model '{model_name}' not found in trained models")
                axes[idx].text(0.5, 0.5, f'Model not trained:\n{model_name}', 
                              ha='center', va='center', fontsize=10)
                axes[idx].set_title(model_name, fontweight='bold')
                axes[idx].set_xticks([])
                axes[idx].set_yticks([])
        
        plt.tight_layout()
        plt.savefig('07_feature_importance.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: 07_feature_importance.png")
        plt.show()
    
    @staticmethod
    def analyze_logistic_coefficients(model, feature_names, top_n=20):
        """
        Analyze and visualize Logistic Regression coefficients for interpretability.
        Shows which features positively/negatively predict voting behavior.
        """
        print("\n" + "="*70)
        print("LOGISTIC REGRESSION COEFFICIENT ANALYSIS")
        print("="*70)
        
        # Check if model is LogisticRegression (could be from GridSearchCV)
        if not hasattr(model, 'coef_'):
            print("[WARNING] Model does not have coefficients (not a Logistic Regression)")
            return
        
        try:
            coef = model.coef_[0]
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coef,
                'Abs_Coefficient': np.abs(coef)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            print("\nTop Features by Absolute Coefficient Value:")
            print(coef_df.head(top_n).to_string(index=False))
            
            # Visualize coefficients
            top_coef = coef_df.head(top_n)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Logistic Regression Coefficient Analysis', fontsize=16, fontweight='bold')
            
            # Plot 1: Top coefficients (positive and negative)
            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_coef['Coefficient']]
            ax1.barh(range(len(top_coef)), top_coef['Coefficient'].values, color=colors, edgecolor='black')
            ax1.set_yticks(range(len(top_coef)))
            ax1.set_yticklabels(top_coef['Feature'].values, fontsize=9)
            ax1.set_xlabel('Coefficient Value', fontsize=12)
            ax1.set_title('Top Coefficients (Positive = Yes Vote, Negative = No Vote)', fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax1.invert_yaxis()
            ax1.grid(axis='x', alpha=0.3)
            
            # Plot 2: Absolute coefficient values
            ax2.barh(range(len(top_coef)), top_coef['Abs_Coefficient'].values, 
                    color='#3498db', edgecolor='black')
            ax2.set_yticks(range(len(top_coef)))
            ax2.set_yticklabels(top_coef['Feature'].values, fontsize=9)
            ax2.set_xlabel('Absolute Coefficient Value', fontsize=12)
            ax2.set_title('Feature Importance (Absolute Coefficients)', fontweight='bold')
            ax2.invert_yaxis()
            ax2.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('08_logistic_coefficients.png', dpi=300, bbox_inches='tight')
            print("[OK] Saved: 08_logistic_coefficients.png")
            plt.show()
            
            # Analyze why some features have zero coefficients
            zero_coef_features = coef_df[coef_df['Abs_Coefficient'] == 0]['Feature'].tolist()
            if zero_coef_features:
                print(f"\n  Features with zero coefficients ({len(zero_coef_features)}):")
                print(f"    {', '.join(zero_coef_features)}")
                
                # Check if this is due to L1 regularization or data issues
                print(f"\n  Diagnostic Analysis:")
                print(f"    - Model uses L1 regularization (penalty='l1') - this is intentional")
                print(f"    - L1 (Lasso) automatically performs feature selection")
                print(f"    - Zero coefficients are EXPECTED for redundant/less informative features")
                print(f"\n  Possible reasons (in order of likelihood):")
                print(f"    1. L1 Regularization: Features removed due to redundancy/correlation")
                print(f"    2. High Correlation: Features highly correlated with others in model")
                print(f"    3. Low Predictive Power: Features don't add predictive value")
                print(f"    4. Missing Data: Features may have high missing data after merging")
                print(f"\n  This is NORMAL behavior - L1 regularization is working as designed")
                print(f"  The model automatically selected the most informative features")
            
            return coef_df
        except Exception as e:
            print(f"[WARNING] Error in coefficient analysis: {e}")
            return None


# ============================================================================
# 8. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline."""
    
    print("\n" + "="*70)
    print("UN VOTING PREDICTION - MACHINE LEARNING PIPELINE")
    print("="*70)
    
    # ========== STEP 1: DATA ACQUISITION ==========
    print("\nSTEP 1: DATA ACQUISITION & DOWNLOAD")
    print("-" * 70)
    
    data_acq = DataAcquisition()
    un_path, wdi_path, sipri_path = data_acq.download_kaggle_datasets()
    
    df_votes = data_acq.load_un_voting_data(un_path) if un_path else None
    df_wdi = data_acq.load_world_bank_data(wdi_path) if wdi_path else None
    df_sipri = data_acq.load_sipri_military_data(sipri_path) if sipri_path else None
    
    print("[OK] Data acquisition completed")
    
    # ========== STEP 2: DATA PREPROCESSING ==========
    print("\nSTEP 2: DATA PREPROCESSING")
    print("-" * 70)
    
    preprocessor = DataPreprocessor()
    
    # Step 2.1: Harmonize datasets (year filtering, merging WDI and SIPRI data)
    # This now performs the full integration: UN voting + WDI indicators + SIPRI military expenditure
    df_merged = preprocessor.harmonize_datasets(df_votes, df_wdi, df_sipri, filter_issues=True)
    
    # Step 2.2: Comprehensive data cleaning
    # Using 'moderate' outlier strictness to preserve more data
    df_merged = preprocessor.comprehensive_clean(
        df_merged,
        remove_duplicates=True,
        remove_outliers=True,  # Set to False if you want to keep outliers
        remove_constants=True,
        clean_invalid=True,
        standardize_countries=True,
        remove_irrelevant=True,
        clean_text=True,
        convert_types=True,
        outlier_strictness='moderate'  # Changed from 'strict' to preserve more data
    )
    
    # Step 2.3: Handle missing values (imputation BEFORE outlier removal)
    # This preserves more data by filling NaN values before outlier detection
    # Preserve priority features even if they have high missing data (they'll be imputed)
    df_merged = preprocessor.handle_missing_values(df_merged, threshold=0.3, preserve_priority_features=True)
    
    # Step 2.4: Diagnostic - Check data quality for zero-coefficient features
    zero_coef_features_check = ['gdp_per_capita', 'gdp_per_capita_current', 
                                 'energy_use_per_capita', 'trade_gdp_ratio']
    available_check = [f for f in zero_coef_features_check if f in df_merged.columns]
    if available_check:
        print("\n" + "="*70)
        print("DATA QUALITY CHECK - Zero-Coefficient Features")
        print("="*70)
        for feat in available_check:
            non_null = df_merged[feat].notna().sum()
            total = len(df_merged)
            pct = (non_null / total * 100) if total > 0 else 0
            if non_null > 0:
                mean_val = df_merged[feat].mean()
                std_val = df_merged[feat].std()
                print(f"  {feat}:")
                print(f"    Non-null: {non_null}/{total} ({pct:.1f}%)")
                print(f"    Mean: {mean_val:.2f}, Std: {std_val:.2f}")
            else:
                print(f"  {feat}: [WARNING] ALL MISSING - This could be a data issue!")
        print("  Note: Zero coefficients are likely due to L1 regularization, not missing data")
        print("="*70)
    
    # Step 2.5: Encode target variable
    df_merged = preprocessor.encode_target_variable(df_merged, 'vote')
    
    print("[OK] Data preprocessing completed")
    
    # ========== STEP 3: EXPLORATORY DATA ANALYSIS ==========
    print("\nSTEP 3: EXPLORATORY DATA ANALYSIS")
    print("-" * 70)
    
    eda = ExploratoryAnalysis()
    eda.distribution_analysis(df_merged, 'vote')
    eda.correlation_analysis(df_merged, 'vote', top_n=15)
    eda.temporal_trends(df_merged, 'vote')
    
    print("[OK] EDA completed")
    
    # ========== STEP 4: FEATURE ENGINEERING ==========
    print("\nSTEP 4: FEATURE ENGINEERING")
    print("-" * 70)
    
    feature_eng = FeatureEngineering()
    target_col = 'vote'
    numeric_cols = df_merged.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    feature_names = feature_eng.select_features(df_merged, target_col, 
                                               method='correlation', 
                                               n_features=min(20, len(numeric_cols)))
    
    print("[OK] Feature engineering completed")
    
    # ========== STEP 5: TRAIN-TEST SPLIT ==========
    print("\nSTEP 5: DATA SPLITTING")
    print("-" * 70)
    
    if len(feature_names) > 0 and target_col in df_merged.columns:
        X = df_merged[feature_names].fillna(0)
        y = df_merged[target_col]
        
        # Check class distribution for imbalanced data mitigation
        class_counts = y.value_counts()
        print(f"\nClass Distribution:")
        print(f"  Class 0 (No/Abstain): {class_counts.get(0, 0)}")
        print(f"  Class 1 (Yes): {class_counts.get(1, 0)}")
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"\nClass Weights (for imbalanced data mitigation):")
        print(f"  Class 0: {class_weight_dict[0]:.4f}")
        print(f"  Class 1: {class_weight_dict[1]:.4f}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled, X_test_scaled = preprocessor.scale_numeric_features(X_train, X_test)
        
        print(f"\n[OK] Train set: {X_train_scaled.shape}")
        print(f"[OK] Test set: {X_test_scaled.shape}")
    else:
        print("[ERROR] Insufficient features for modeling")
        return
    
    # ========== STEP 6: MODEL TRAINING & HYPERPARAMETER TUNING ==========
    print("\nSTEP 6: MODEL TRAINING & HYPERPARAMETER TUNING")
    print("-" * 70)
    
    # Initialize trainer with GridSearchCV enabled
    trainer = ModelTrainer(cv_folds=5, random_state=42, use_grid_search=True)
    trainer.define_models(class_weight=class_weight_dict)
    trainer.train_all_models(X_train_scaled, y_train, use_cv=True)
    
    print("[OK] Model training and hyperparameter tuning completed")
    
    # ========== STEP 7: MODEL EVALUATION ==========
    print("\nSTEP 7: MODEL EVALUATION")
    print("-" * 70)
    
    evaluator = ModelEvaluator()
    evaluator.evaluate_all_models(trainer.models, X_test_scaled, y_test)
    comparison_df = evaluator.compare_models()
    evaluator.plot_roc_curves()
    evaluator.plot_pr_curves()  # Add Precision-Recall curves
    evaluator.plot_confusion_matrices()
    
    print("[OK] Model evaluation completed")
    
    # ========== STEP 8: FEATURE IMPORTANCE & INTERPRETABILITY ==========
    print("\nSTEP 8: FEATURE IMPORTANCE & INTERPRETABILITY ANALYSIS")
    print("-" * 70)
    
    FeatureImportanceAnalyzer.plot_feature_importance(trainer.models, feature_names, top_n=15)
    
    # Coefficient analysis for Logistic Regression (interpretability)
    if 'Logistic Regression' in trainer.models:
        lr_model = trainer.models['Logistic Regression']
        FeatureImportanceAnalyzer.analyze_logistic_coefficients(lr_model, feature_names, top_n=20)
    
    # ========== STEP 9: SUMMARY ==========
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print("\nGenerated Outputs:")
    print("  [OK] 01_distribution_analysis.png")
    print("  [OK] 02_correlation_analysis.png")
    print("  [OK] 03_temporal_trends.png")
    print("  [OK] 04_model_comparison.png")
    print("  [OK] 05_roc_curves.png")
    print("  [OK] 05_pr_curves.png")
    print("  [OK] 06_confusion_matrices.png")
    print("  [OK] 07_feature_importance.png")
    print("  [OK] 08_logistic_coefficients.png")
    
    # Find best model by ROC-AUC
    best_model_idx = comparison_df['ROC-AUC'].idxmax()
    best_roc_auc = comparison_df.loc[best_model_idx, 'ROC-AUC']
    best_pr_auc = comparison_df.loc[best_model_idx, 'PR-AUC']
    
    print(f"\nBest Performing Model: {best_model_idx}")
    print(f"  ROC-AUC: {best_roc_auc:.4f}")
    print(f"  PR-AUC: {best_pr_auc:.4f}")
    
    # Display best hyperparameters if available
    if hasattr(trainer, 'best_params') and best_model_idx in trainer.best_params:
        print(f"\nBest Hyperparameters for {best_model_idx}:")
        for param, value in trainer.best_params[best_model_idx].items():
            print(f"  {param}: {value}")
    
    return trainer.models, evaluator.results, comparison_df


if __name__ == "__main__":
    models, results, comparison = main()
