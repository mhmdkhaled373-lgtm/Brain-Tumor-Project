# ============================================================
#       Big Data Mini Project — Brain Tumor Patient Records
#       Course: Big Data | Tool: Python + Simulated Hadoop
# ============================================================
# Dataset : brain_tumor_dataset.csv (20,000 rows, 20 columns)
# Source  : https://www.kaggle.com/datasets/miadul/brain-tumor-dataset
# Note    : Synthetic dataset for academic/research purposes
# ============================================================

# ====================== Install (Colab / first run) ======================
# !pip install pandas numpy matplotlib seaborn

# ====================== Imports ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import random
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────
RAW_PATH   = "brain_tumor_records_raw.csv"
CLEAN_PATH = "brain_tumor_records_clean.csv"
SOURCE     = "brain_tumor_dataset.csv"       # your downloaded file

# ─────────────────────────────────────────────────────────────────────────
#   PART 0 — INTRODUCE MESS  (creates a realistic "dirty" dataset)
#   This simulates real-world data quality issues so we can demonstrate
#   all required preprocessing steps.
# ─────────────────────────────────────────────────────────────────────────
def introduce_mess(source=SOURCE, raw_path=RAW_PATH, seed=42):
    """
    Load the clean source CSV and deliberately introduce:
      • ~600 missing values  (NaN) spread across key columns
      • ~300 duplicate rows
      • Inconsistent category spellings  (e.g. 'male', 'MALE', 'M')
      • Wrong Stage formats             (e.g. 'stage I', 'Stage-II')
      • Mixed-case Treatment_Response   (added column with noise)
      • Out-of-range Age values         (a few set to 0 or 999)
    """
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(source)

    print("=" * 60)
    print("  PART 0 — Creating Raw (Dirty) Dataset")
    print("=" * 60)
    print(f"  Original shape : {df.shape}")

    # ── 1. Add Treatment_Response column (with noise) ─────────────
    responses = ['Improved', 'Worsened', 'Stable']
    df['Treatment_Response'] = np.random.choice(responses, size=len(df))

    # ── 2. Inject missing values ──────────────────────────────────
    miss_cols = ['Age', 'Tumor_Size', 'Survival_Rate',
                 'Gender', 'Histology', 'Treatment_Response',
                 'Tumor_Growth_Rate', 'Location']
    n_missing = 600
    for _ in range(n_missing):
        col = random.choice(miss_cols)
        idx = random.randint(0, len(df) - 1)
        df.at[idx, col] = np.nan

    # ── 3. Add duplicate rows ─────────────────────────────────────
    dup_indices = np.random.choice(df.index, size=300, replace=False)
    duplicates  = df.loc[dup_indices]
    df = pd.concat([df, duplicates], ignore_index=True)

    # ── 4. Inconsistent Gender spellings ─────────────────────────
    bad_gender = ['male', 'MALE', 'M', 'female', 'FEMALE', 'F']
    noise_idx  = np.random.choice(df.index, size=400, replace=False)
    for idx in noise_idx:
        if pd.notna(df.at[idx, 'Gender']):
            original = df.at[idx, 'Gender']
            if original == 'Male':
                df.at[idx, 'Gender'] = random.choice(['male', 'MALE', 'M'])
            else:
                df.at[idx, 'Gender'] = random.choice(['female', 'FEMALE', 'F'])

    # ── 5. Inconsistent Stage formats ────────────────────────────
    bad_stage = {
        'I':   ['stage I', 'Stage-I', 'stage_1', 'i'],
        'II':  ['stage II', 'Stage-II', 'stage_2', 'ii'],
        'III': ['stage III', 'Stage-III', 'stage_3', 'iii'],
        'IV':  ['stage IV', 'Stage-IV', 'stage_4', 'iv'],
    }
    stage_idx = np.random.choice(df.index, size=350, replace=False)
    for idx in stage_idx:
        original = df.at[idx, 'Stage']
        if pd.notna(original) and original in bad_stage:
            df.at[idx, 'Stage'] = random.choice(bad_stage[original])

    # ── 6. Out-of-range Age values ────────────────────────────────
    outlier_idx = np.random.choice(df.index, size=30, replace=False)
    for idx in outlier_idx:
        df.at[idx, 'Age'] = random.choice([0, 999, -1, 200])

    # ── Save raw ──────────────────────────────────────────────────
    df.to_csv(raw_path, index=False)
    print(f"  Dirty shape    : {df.shape}")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Duplicates     : {df.duplicated().sum()}")
    print(f"  ✅ Raw dataset saved → '{raw_path}'\n")
    return df


# ─────────────────────────────────────────────────────────────────────────
#   PART 1 — DATA PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────
def preprocess(raw_path=RAW_PATH, clean_path=CLEAN_PATH):
    """
    Full preprocessing pipeline:
      1. Load & display structure
      2. Identify columns and data types
      3. Remove duplicate rows
      4. Handle missing values
      5. Fix inconsistent values
      6. Fix out-of-range values
      7. Rename columns
      8. Convert data types
      9. Save clean dataset
    """
    print("=" * 60)
    print("  PART 1 — DATA PREPROCESSING")
    print("=" * 60)

    # ── Step 1: Load & display structure ─────────────────────────
    print("\n📂 Step 1: Load & Display Dataset Structure")
    print("-" * 45)
    df = pd.read_csv(raw_path)
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print("\n  First 3 rows:")
    print(df.head(3).to_string())

    # ── Step 2: Identify columns and data types ───────────────────
    print("\n\n🔍 Step 2: Column Names and Data Types")
    print("-" * 45)
    for col in df.columns:
        n_unique = df[col].nunique()
        n_null   = df[col].isnull().sum()
        print(f"  {col:<25} dtype={str(df[col].dtype):<10} "
              f"unique={n_unique:<6} missing={n_null}")

    # ── Step 3: Remove duplicate rows ────────────────────────────
    print("\n\n🗑️  Step 3: Remove Duplicate Rows")
    print("-" * 45)
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    after  = len(df)
    print(f"  Rows before    : {before:,}")
    print(f"  Duplicates     : {before - after:,}")
    print(f"  Rows after     : {after:,}")

    # ── Step 4: Handle missing values ────────────────────────────
    print("\n\n🩹 Step 4: Handle Missing Values")
    print("-" * 45)
    print("  Missing per column (before):")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    for col, count in missing.items():
        print(f"    {col:<25} {count} missing")

    # Numeric → fill with median
    num_cols = ['Age', 'Tumor_Size', 'Survival_Rate', 'Tumor_Growth_Rate']
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  ✅ {col}: filled with median ({median_val:.2f})")

    # Categorical → fill with mode
    cat_cols = ['Gender', 'Histology', 'Treatment_Response', 'Location']
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"  ✅ {col}: filled with mode ('{mode_val}')")

    print(f"\n  Total missing after fix : {df.isnull().sum().sum()}")

    # ── Step 5: Fix inconsistent values ──────────────────────────
    print("\n\n🔧 Step 5: Fix Inconsistent Values")
    print("-" * 45)

    # Gender → standardize to 'Male' / 'Female'
    gender_map = {
        'male': 'Male', 'MALE': 'Male', 'M': 'Male',
        'female': 'Female', 'FEMALE': 'Female', 'F': 'Female',
        'Male': 'Male', 'Female': 'Female'
    }
    before_unique = df['Gender'].unique().tolist()
    df['Gender'] = df['Gender'].map(gender_map).fillna(df['Gender'])
    print(f"  Gender before  : {before_unique}")
    print(f"  Gender after   : {df['Gender'].unique().tolist()}")

    # Stage → standardize to Roman numerals only (I, II, III, IV)
    def fix_stage(val):
        if pd.isna(val):
            return val
        val = str(val).strip().upper()
        val = val.replace('STAGE', '').replace('-', '').replace('_', '').strip()
        mapping = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV',
                   'I': 'I', 'II': 'II', 'III': 'III', 'IV': 'IV'}
        return mapping.get(val, val)

    before_stage = df['Stage'].unique().tolist()
    df['Stage'] = df['Stage'].apply(fix_stage)
    print(f"\n  Stage before   : {sorted(set(str(x) for x in before_stage))}")
    print(f"  Stage after    : {sorted(df['Stage'].unique().tolist())}")

    # Treatment_Response → title case
    if 'Treatment_Response' in df.columns:
        df['Treatment_Response'] = df['Treatment_Response'].str.strip().str.title()
        print(f"\n  Treatment_Response: {df['Treatment_Response'].unique().tolist()}")

    # ── Step 6: Fix out-of-range Age values ──────────────────────
    print("\n\n🚦 Step 6: Fix Out-of-Range Values")
    print("-" * 45)
    bad_age_mask = (df['Age'] < 1) | (df['Age'] > 100)
    print(f"  Out-of-range Age values : {bad_age_mask.sum()}")
    median_age = df.loc[~bad_age_mask, 'Age'].median()
    df.loc[bad_age_mask, 'Age'] = median_age
    print(f"  Replaced with median age: {median_age:.0f}")
    print(f"  Age range now           : {df['Age'].min():.0f} – {df['Age'].max():.0f}")

    # ── Step 7: Rename columns ────────────────────────────────────
    print("\n\n✏️  Step 7: Rename Columns")
    print("-" * 45)
    rename_map = {
        'Tumor_Size'         : 'Tumor_Size_cm',
        'Tumor_Growth_Rate'  : 'Growth_Rate_cm_month',
        'Survival_Rate'      : 'Survival_Rate_pct',
    }
    df.rename(columns=rename_map, inplace=True)
    print(f"  Renamed : {rename_map}")

    # ── Step 8: Convert data types ────────────────────────────────
    print("\n\n🔄 Step 8: Convert Data Types")
    print("-" * 45)

    # Age → replace inf, fill NaN, then convert to integer
    df['Age'] = df['Age'].replace([np.inf, -np.inf], np.nan)
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    df['Age'] = df['Age'].round(0).astype('Int64')
    print("  Age              → Int64")

    # Categorical columns → category dtype
    cat_convert = ['Gender', 'Tumor_Type', 'Location', 'Histology',
                   'Stage', 'Radiation_Treatment', 'Surgery_Performed',
                   'Chemotherapy', 'Family_History', 'MRI_Result',
                   'Follow_Up_Required', 'Treatment_Response']
    for col in cat_convert:
        if col in df.columns:
            df[col] = df[col].astype('category')
    print(f"  Categorical cols → category dtype ({len(cat_convert)} columns)")

    # Survival_Rate_pct → round to 2 decimals
    df['Survival_Rate_pct']      = df['Survival_Rate_pct'].round(2)
    df['Tumor_Size_cm']          = df['Tumor_Size_cm'].round(3)
    df['Growth_Rate_cm_month']   = df['Growth_Rate_cm_month'].round(4)
    print("  Numeric floats   → rounded")

    # ── Step 9: Save clean dataset ────────────────────────────────
    df.to_csv(clean_path, index=False)
    print(f"\n\n💾 Clean dataset saved → '{clean_path}'")
    print(f"  Final shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Missing     : {df.isnull().sum().sum()}")
    print(f"  Duplicates  : {df.duplicated().sum()}")
    print("=" * 60 + "\n")

    return df


# ─────────────────────────────────────────────────────────────────────────
#   PART 2 — DOCKER ENVIRONMENT SETUP
#   (Instructions + Dockerfile content printed for reference)
# ─────────────────────────────────────────────────────────────────────────
def setup_docker():
    """
    Print Docker setup instructions and generate a Dockerfile.
    The Dockerfile sets up a Jupyter Notebook environment.
    """
    print("=" * 60)
    print("  PART 2 — DOCKER ENVIRONMENT SETUP")
    print("=" * 60)

    # ── requirements.txt with exact versions ──────────────
    requirements_content = """# Auto-generated requirements.txt
# Exact library versions used in this project
pandas==3.0.2
numpy==2.2.5
matplotlib==3.10.9
seaborn==0.13.2
jupyter==1.0.0
"""

    # Write requirements.txt
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    print("\n  ✅ requirements.txt created with exact versions:")
    print("     pandas==3.0.2")
    print("     numpy==2.2.5")
    print("     matplotlib==3.10.9")
    print("     seaborn==0.13.2")
    print("     jupyter==1.0.0")

    dockerfile_content = """# ── Dockerfile ─────────────────────────────────────────
# Base image: Python 3.13 slim (matches development environment)
FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Copy project files into container
COPY . /app

# Install exact library versions from requirements.txt
# This guarantees same environment as development machine
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter Notebook port
EXPOSE 8888

# Run Jupyter Notebook on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", \\
     "--no-browser", "--allow-root", \\
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
"""

    # Write Dockerfile
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(dockerfile_content)

    print("\n  ✅ Dockerfile created.")
    print("\n  📋 To run the Docker container:")
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │  Step 1: Build the image                         │")
    print("  │  > docker build -t brain-tumor-bigdata .         │")
    print("  │                                                  │")
    print("  │  Step 2: Run the container                       │")
    print("  │  > docker run -p 8888:8888 brain-tumor-bigdata   │")
    print("  │                                                  │")
    print("  │  Step 3: Open in browser                         │")
    print("  │  > http://localhost:8888                         │")
    print("  └──────────────────────────────────────────────────┘")
    print("\n  📸 Take a screenshot of the running container")
    print("     for your project submission report.\n")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────
#   PART 3 — SIMULATED MAPREDUCE (Hadoop Concept)
# ─────────────────────────────────────────────────────────────────────────
def simulated_mapreduce(df):
    """
    Simulates the Hadoop MapReduce paradigm in pure Python.

    Task 1 — Count patients per Tumor Type
      MAP    : Emit (tumor_type, 1) for each patient row
      REDUCE : Sum all counts per tumor type

    Task 2 — Average Survival Rate per Histology type
      MAP    : Emit (histology, survival_rate) for each row
      REDUCE : Compute mean survival rate per histology
    """
    print("=" * 60)
    print("  PART 3 — SIMULATED MAPREDUCE (Hadoop Concept)")
    print("=" * 60)

    # ── Task 1: Count patients per Tumor Type ─────────────────────
    print("\n🗺️  Task 1: Patient Count per Tumor Type")
    print("   (MapReduce: MAP → emit (tumor_type, 1)  |  REDUCE → sum)")
    print("-" * 45)

    # MAP phase
    mapped_1 = [(str(row['Tumor_Type']), 1) for _, row in df.iterrows()]
    print(f"  MAP    : Emitted {len(mapped_1):,} key-value pairs")

    # SHUFFLE phase (group by key)
    shuffled_1 = {}
    for key, val in mapped_1:
        shuffled_1.setdefault(key, []).append(val)
    print(f"  SHUFFLE: Grouped into {len(shuffled_1)} buckets")

    # REDUCE phase (sum)
    reduced_1 = {key: sum(vals) for key, vals in shuffled_1.items()}
    print(f"  REDUCE : Final counts:")
    for tumor, count in sorted(reduced_1.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 500)
        print(f"    {tumor:<15} {count:>6,}  {bar}")

    # ── Task 2: Avg Survival Rate per Histology ───────────────────
    print("\n🗺️  Task 2: Average Survival Rate per Histology")
    print("   (MapReduce: MAP → emit (histology, survival_rate)  |  REDUCE → mean)")
    print("-" * 45)

    # MAP phase
    mapped_2 = [(str(row['Histology']), row['Survival_Rate_pct'])
                for _, row in df.iterrows()
                if pd.notna(row['Survival_Rate_pct'])]
    print(f"  MAP    : Emitted {len(mapped_2):,} key-value pairs")

    # SHUFFLE phase
    shuffled_2 = {}
    for key, val in mapped_2:
        shuffled_2.setdefault(key, []).append(val)
    print(f"  SHUFFLE: Grouped into {len(shuffled_2)} buckets")

    # REDUCE phase (mean)
    reduced_2 = {key: round(np.mean(vals), 2) for key, vals in shuffled_2.items()}
    print(f"  REDUCE : Average survival rates:")
    for hist, avg in sorted(reduced_2.items(), key=lambda x: -x[1]):
        bar = "█" * int(avg / 5)
        print(f"    {hist:<20} {avg:>6.2f}%  {bar}")

    # ── Task 3: Count Treatment Responses per Stage ───────────────
    print("\n🗺️  Task 3: Treatment Response Count per Stage")
    print("   (MapReduce: MAP → emit ((stage, response), 1)  |  REDUCE → sum)")
    print("-" * 45)

    mapped_3 = [((str(row['Stage']), str(row['Treatment_Response'])), 1)
                for _, row in df.iterrows()]
    shuffled_3 = {}
    for key, val in mapped_3:
        shuffled_3.setdefault(key, []).append(val)
    reduced_3 = {key: sum(vals) for key, vals in shuffled_3.items()}

    print(f"  MAP    : Emitted {len(mapped_3):,} key-value pairs")
    print(f"  REDUCE : Stage × Response combinations: {len(reduced_3)}")
    print("  Sample results:")
    for (stage, response), count in sorted(reduced_3.items())[:8]:
        print(f"    Stage {stage} | {response:<10} → {count:,} patients")

    print("\n  ✅ MapReduce simulation complete.\n")
    print("=" * 60 + "\n")

    return reduced_1, reduced_2, reduced_3


# ─────────────────────────────────────────────────────────────────────────
#   PART 4 — DATA VISUALIZATION (6 Charts)
# ─────────────────────────────────────────────────────────────────────────
def visualize(df):
    """
    Generate 6 meaningful charts:
      1. Bar Chart    — Patient count per Tumor Type
      2. Pie Chart    — Treatment Response distribution
      3. Line Chart   — Average Survival Rate by Age Group
      4. Histogram    — Tumor Size distribution
      5. Scatter Plot — Tumor Size vs Survival Rate (colored by Tumor Type)
      6. Heatmap      — Correlation matrix of numeric columns
    """
    print("=" * 60)
    print("  PART 4 — DATA VISUALIZATION")
    print("=" * 60)

    # ── Color palette ─────────────────────────────────────────────
    palette = {
        'blue'   : '#3498db',
        'red'    : '#e74c3c',
        'green'  : '#2ecc71',
        'orange' : '#f39c12',
        'purple' : '#9b59b6',
        'teal'   : '#1abc9c',
    }

    # ── Chart 1: Bar Chart — Tumor Type Distribution ──────────────
    print("\n📊 Chart 1: Bar Chart — Tumor Type Distribution")
    tumor_counts = df['Tumor_Type'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(tumor_counts.index.astype(str),
                  tumor_counts.values,
                  color=[palette['blue'], palette['red']],
                  alpha=0.85, edgecolor='white', linewidth=1.2)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f"{bar.get_height():,}",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title("Patient Count by Tumor Type", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Tumor Type", fontsize=12)
    ax.set_ylabel("Number of Patients", fontsize=12)
    ax.set_ylim(0, tumor_counts.max() * 1.15)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig("chart1_bar_tumor_type.png", dpi=150)
    plt.show()
    print("  ✅ Saved → chart1_bar_tumor_type.png")

    # ── Chart 2: Pie Chart — Treatment Response ───────────────────
    print("\n🥧 Chart 2: Pie Chart — Treatment Response Distribution")
    response_counts = df['Treatment_Response'].value_counts()
    colors_pie      = [palette['green'], palette['red'], palette['orange']]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        response_counts.values,
        labels=response_counts.index.astype(str),
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=140,
        pctdistance=0.82,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for text in autotexts:
        text.set_fontsize(12)
        text.set_fontweight('bold')

    ax.set_title("Treatment Response Distribution", fontsize=15,
                 fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("chart2_pie_treatment_response.png", dpi=150)
    plt.show()
    print("  ✅ Saved → chart2_pie_treatment_response.png")

    # ── Chart 3: Line Chart — Avg Survival Rate by Age Group ──────
    print("\n📈 Chart 3: Line Chart — Avg Survival Rate by Age Group")
    df['Age_Group'] = pd.cut(df['Age'],
                             bins=[0, 20, 30, 40, 50, 60, 70, 80, 100],
                             labels=['<20', '20-29', '30-39', '40-49',
                                     '50-59', '60-69', '70-79', '80+'])
    age_survival = (df.groupby('Age_Group', observed=True)['Survival_Rate_pct']
                      .mean()
                      .reset_index())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(age_survival['Age_Group'].astype(str),
            age_survival['Survival_Rate_pct'],
            color=palette['blue'], linewidth=2.5,
            marker='o', markersize=8, markerfacecolor='white',
            markeredgewidth=2.5)

    ax.fill_between(range(len(age_survival)),
                    age_survival['Survival_Rate_pct'],
                    alpha=0.12, color=palette['blue'])

    for i, row in age_survival.iterrows():
        ax.annotate(f"{row['Survival_Rate_pct']:.1f}%",
                    (i, row['Survival_Rate_pct']),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color=palette['blue'])

    ax.set_xticks(range(len(age_survival)))
    ax.set_xticklabels(age_survival['Age_Group'].astype(str))
    ax.set_title("Average Survival Rate by Age Group",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Avg Survival Rate (%)", fontsize=12)
    ax.set_ylim(age_survival['Survival_Rate_pct'].min() - 5,
                age_survival['Survival_Rate_pct'].max() + 8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig("chart3_line_survival_by_age.png", dpi=150)
    plt.show()
    print("  ✅ Saved → chart3_line_survival_by_age.png")

    # ── Chart 4: Histogram — Tumor Size Distribution ──────────────
    print("\n📊 Chart 4: Histogram — Tumor Size Distribution")
    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(df['Tumor_Size_cm'].dropna(),
                                bins=40, color=palette['purple'],
                                edgecolor='white', alpha=0.85)

    # Color bars by size range
    for patch, left in zip(patches, bins[:-1]):
        if left < 3:
            patch.set_facecolor(palette['green'])
        elif left < 7:
            patch.set_facecolor(palette['orange'])
        else:
            patch.set_facecolor(palette['red'])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palette['green'],  label='Small  (< 3 cm)'),
        Patch(facecolor=palette['orange'], label='Medium (3–7 cm)'),
        Patch(facecolor=palette['red'],    label='Large  (> 7 cm)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    mean_size = df['Tumor_Size_cm'].mean()
    ax.axvline(mean_size, color='black', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean_size:.2f} cm')

    ax.set_title("Distribution of Tumor Size", fontsize=15,
                 fontweight='bold', pad=15)
    ax.set_xlabel("Tumor Size (cm)", fontsize=12)
    ax.set_ylabel("Number of Patients", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig("chart4_histogram_tumor_size.png", dpi=150)
    plt.show()
    print("  ✅ Saved → chart4_histogram_tumor_size.png")

    # ── Chart 5: Scatter Plot — Tumor Size vs Survival Rate ───────
    print("\n🔵 Chart 5: Scatter — Tumor Size vs Survival Rate")
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    color_map = {'Malignant': palette['red'], 'Benign': palette['blue']}

    fig, ax = plt.subplots(figsize=(9, 6))
    for tumor_type, group in sample.groupby('Tumor_Type', observed=True):
        ax.scatter(group['Tumor_Size_cm'],
                   group['Survival_Rate_pct'],
                   c=color_map.get(str(tumor_type), 'gray'),
                   alpha=0.45, s=25,
                   label=str(tumor_type))

    ax.set_title("Tumor Size vs Survival Rate\n(by Tumor Type)",
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Tumor Size (cm)", fontsize=12)
    ax.set_ylabel("Survival Rate (%)", fontsize=12)
    ax.legend(title="Tumor Type", fontsize=10, title_fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig("chart5_scatter_size_vs_survival.png", dpi=150)
    plt.show()
    print("  ✅ Saved → chart5_scatter_size_vs_survival.png")

    # ── Chart 6: Heatmap — Correlation Matrix ─────────────────────
    print("\n🌡️  Chart 6: Heatmap — Correlation Matrix")
    num_cols = ['Age', 'Tumor_Size_cm', 'Survival_Rate_pct', 'Growth_Rate_cm_month']
    corr     = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, linewidths=0.5, linecolor='white',
                ax=ax, vmin=-1, vmax=1,
                annot_kws={"size": 11, "weight": "bold"})

    ax.set_title("Correlation Matrix — Numeric Features",
                 fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig("chart6_heatmap_correlation.png", dpi=150)
    plt.show()
    print("  ✅ Saved → chart6_heatmap_correlation.png")

    print("\n  ✅ All 6 charts generated and saved.\n")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────
#   PART 5 — KEY INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────
def print_insights(df):
    """Print key findings from the dataset for the project report."""
    print("=" * 60)
    print("  PART 5 — KEY INSIGHTS & FINDINGS")
    print("=" * 60)

    print(f"""
  📌 Dataset Overview
  ───────────────────
  • Total patients      : {len(df):,}
  • Age range           : {df['Age'].min()} – {df['Age'].max()} years
  • Average age         : {df['Age'].mean():.1f} years
  • Avg tumor size      : {df['Tumor_Size_cm'].mean():.2f} cm
  • Avg survival rate   : {df['Survival_Rate_pct'].mean():.1f}%

  📌 Tumor Type
  ─────────────
  • Malignant patients  : {(df['Tumor_Type']=='Malignant').sum():,} ({(df['Tumor_Type']=='Malignant').mean()*100:.1f}%)
  • Benign patients     : {(df['Tumor_Type']=='Benign').sum():,} ({(df['Tumor_Type']=='Benign').mean()*100:.1f}%)

  📌 Survival Rate
  ────────────────
  • Avg (Malignant)     : {df[df['Tumor_Type']=='Malignant']['Survival_Rate_pct'].mean():.1f}%
  • Avg (Benign)        : {df[df['Tumor_Type']=='Benign']['Survival_Rate_pct'].mean():.1f}%
  • Highest avg by loc  : {df.groupby('Location',observed=True)['Survival_Rate_pct'].mean().idxmax()}

  📌 Treatment
  ────────────
  • Received Surgery    : {(df['Surgery_Performed']=='Yes').sum():,}
  • Received Radiation  : {(df['Radiation_Treatment']=='Yes').sum():,}
  • Received Chemo      : {(df['Chemotherapy']=='Yes').sum():,}
  • Treatment Response:
  {df['Treatment_Response'].value_counts().to_string()}
    """)
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────
#   MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  🧠 Big Data Project — Brain Tumor Patient Records")
    print("  Course : Big Data | Student Submission")
    print("=" * 60 + "\n")

    # Part 0 — Create dirty dataset from clean source
    introduce_mess()

    # Part 1 — Preprocess
    df_clean = preprocess()

    # Part 2 — Docker setup
    setup_docker()

    # Part 3 — Simulated MapReduce
    simulated_mapreduce(df_clean)

    # Part 4 — Visualizations
    visualize(df_clean)

    # Part 5 — Insights
    print_insights(df_clean)

    print("🎉 Project pipeline complete!")
    print(f"   Clean dataset : {CLEAN_PATH}")
    print(f"   Dockerfile    : Dockerfile")
    print(f"   Charts        : chart1_*.png … chart6_*.png\n")


if __name__ == "__main__":
    main()
