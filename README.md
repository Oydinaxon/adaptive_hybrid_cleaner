import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from scipy.stats import skew

# =========================================================
# 1️⃣ COLUMN TYPE DETECTION
# =========================================================
def detect_column_types(df, categorical_ratio=0.05, max_unique_for_cat=50):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    categorical_cols = []
    text_cols = []

    for col in object_cols:
        unique_count = df[col].nunique(dropna=True)
        unique_ratio = unique_count / len(df)

        if unique_ratio < categorical_ratio or unique_count < max_unique_for_cat:
            categorical_cols.append(col)
        else:
            text_cols.append(col)

    return numeric_cols, categorical_cols, text_cols

# =========================================================
# 2️⃣ ADAPTIVE HYBRID CLEANER
# =========================================================
def adaptive_hybrid_cleaner(
        df,
        rare_threshold=0.01,
        skew_threshold=0.5,
        outlier_skew_switch=1,
        knn_missing_threshold=0.1,
        random_state=42
):
    df = df.copy()
    n_rows = len(df)
    changes_made = False

    numeric_cols, categorical_cols, text_cols = detect_column_types(df)

    print(f"Dataset size: {n_rows} rows")

    # =====================================================
    # NUMERIC MISSING HANDLING
    # =====================================================
    if numeric_cols:
        total_missing_ratio = df[numeric_cols].isna().mean().mean()

        if n_rows > 1000 and total_missing_ratio > knn_missing_threshold:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            changes_made = True
        else:
            for col in numeric_cols:
                if df[col].isna().sum() == 0:
                    continue

                col_skew = skew(df[col].dropna())
                if np.isnan(col_skew):
                    col_skew = 0

                if abs(col_skew) > skew_threshold:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mean())

                changes_made = True

    # =====================================================
    # CATEGORICAL CLEANING
    # =====================================================
    for col in categorical_cols:
        df[col] = df[col].astype("string")
        df[col] = df[col].str.strip().str.lower()
        df[col] = df[col].replace({"none": np.nan, "nan": np.nan, "": np.nan})

        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
                changes_made = True

        # Rare category grouping
        freq = df[col].value_counts(normalize=True)
        rare_categories = freq[freq < rare_threshold].index
        if len(rare_categories) > 0:
            df.loc[df[col].isin(rare_categories), col] = "other"
            changes_made = True

        # COUNTRY CODE UPPERCASE, others Capitalized
        if col.lower().find("country") != -1 or col.lower().find("code") != -1:
            df[col] = df[col].str.upper()
        else:
            df[col] = df[col].str.capitalize()

    # =====================================================
    # TEXT CLEANING
    # =====================================================
    for col in text_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna("unknown")
            changes_made = True

        df[col] = df[col].astype(str).str.strip()

    # =====================================================
    # UNIVARIATE OUTLIER HANDLING
    # =====================================================
    for col in numeric_cols:
        col_skew = skew(df[col].dropna())
        if np.isnan(col_skew):
            col_skew = 0

        if abs(col_skew) > outlier_skew_switch:
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = (df[col] < lower) | (df[col] > upper)
        else:
            # Z-score
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            z = (df[col] - mean) / std
            outliers = np.abs(z) > 3

        if outliers.sum() > 0:
            df.loc[outliers, col] = df[col].median()
            changes_made = True

    # =====================================================
    # MULTIVARIATE OUTLIER (ISOLATION FOREST)
    # =====================================================
    if len(numeric_cols) > 1 and n_rows > 500:
        contamination = min(0.02, 50 / n_rows)

        iso = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )

        preds = iso.fit_predict(df[numeric_cols])
        mask = preds == 1

        if mask.sum() < len(df):
            df = df[mask]
            changes_made = True

    # =====================================================
    # DUPLICATE REMOVAL
    # =====================================================
    before_dup = len(df)
    df = df.drop_duplicates()
    if len(df) < before_dup:
        changes_made = True

    return df, numeric_cols, categorical_cols, text_cols, changes_made

# =========================================================
# SUMMARY REPORT
# =========================================================
def summary_report(original_df, cleaned_df):
    print("\n--- CLEANING SUMMARY ---")
    print("Rows before:", len(original_df))
    print("Rows after :", len(cleaned_df))
    print("Missing BEFORE:", original_df.isna().sum().sum())
    print("Missing AFTER :", cleaned_df.isna().sum().sum())
    print("Duplicates BEFORE:", original_df.duplicated().sum())
    print("Duplicates AFTER :", cleaned_df.duplicated().sum())

# =========================================================
# RUN CLEANING
# =========================================================
df = pd.read_csv("synthetic_dirty_large_dataset.csv")

clean_df, num_cols, cat_cols, txt_cols, changed = adaptive_hybrid_cleaner(df)
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1️⃣ Missing values comparison
# =========================
missing_before = df.isna().sum()
missing_after = clean_df.isna().sum()

missing_df = pd.DataFrame({
    'Before Cleaning': missing_before,
    'After Cleaning': missing_after
})

missing_df.plot(kind='bar', figsize=(10,5))
plt.title('Missing Values: Before vs After Cleaning')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# =========================
# 2️⃣ Duplicates removed
# =========================
duplicates_before = df.duplicated().sum()
duplicates_after = clean_df.duplicated().sum()

plt.figure(figsize=(6,4))
sns.barplot(x=['Before Cleaning', 'After Cleaning'], y=[duplicates_before, duplicates_after])
plt.title('Duplicates: Before vs After Cleaning')
plt.ylabel('Count')
plt.show()

# =========================
# 3️⃣ Numeric columns distribution (example: Age)
# =========================
plt.figure(figsize=(12,5))
sns.kdeplot(df['Age'].dropna(), label='Before Cleaning', fill=True)
sns.kdeplot(clean_df['Age'].dropna(), label='After Cleaning', fill=True)
plt.title('Distribution of Age Before and After Cleaning')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

# =========================
# 4️⃣ Numeric columns distribution (example: Income)
# =========================
plt.figure(figsize=(12,5))
sns.kdeplot(df['Income'].dropna(), label='Before Cleaning', fill=True)
sns.kdeplot(clean_df['Income'].dropna(), label='After Cleaning', fill=True)
plt.title('Distribution of Income Before and After Cleaning')
plt.xlabel('Income')
plt.ylabel('Density')
plt.legend()
plt.show()


print(summary_report(df, clean_df))
print(clean_df)

