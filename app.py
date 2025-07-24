from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
import re
from collections import OrderedDict
from typing import List

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

selected_features = ["Age", "Support Calls", "Payment Delay", "Total Spend", "Contract Length"]

REQUIRED_COLUMNS = {
    "Age": ["age"],
    "Support Calls": ["support calls", "support_calls"],
    "Payment Delay": ["payment delay", "payment_delay"],
    "Total Spend": ["total spend", "total_spend"],
    "Contract Length": ["contract length", "contract_length"]
}

REQUIRED_DTYPES = {
    "Age": "number",
    "Support Calls": "number",
    "Payment Delay": "number",
    "Total Spend": "number",
    "Contract Length": "categorical"
}

VALID_CATEGORIES = {
    "Contract Length": ["Monthly", "Quarterly", "Annual"]
}

CONTRACT_MAPPING = {
    "Monthly": 0.0,
    "Quarterly": 0.5,
    "Annual": 1.0
}

MISSING_THRESHOLD = 0.05

# ===========================
# Helper Function
# ===========================

def column_mapping(columns_from_user):
    mapped = {}
    ambiguous = []

    columns_stripped = [col.strip() for col in columns_from_user]
    columns_normalized = [col.lower() for col in columns_stripped]

    for std_col, keywords in REQUIRED_COLUMNS.items():
        matches = []
        for keyword in keywords:
            for original, norm in zip(columns_stripped, columns_normalized):
                if re.search(rf'\b{re.escape(keyword)}\b', norm):
                    matches.append(original)

        if len(matches) == 1:
            mapped[std_col] = matches[0]
        elif len(matches) > 1:
            ambiguous.append((std_col, matches))

    return mapped, ambiguous

def check_column_dtypes(df):
    invalid_types = []
    for col, expected_type in REQUIRED_DTYPES.items():
        if col not in df.columns:
            continue
        if expected_type == "number":
            if not pd.api.types.is_numeric_dtype(df[col]):
                invalid_types.append({
                    "column": col,
                    "expected": "numeric",
                    "found": str(df[col].dtype)
                })
        elif expected_type == "categorical":
            if pd.api.types.is_numeric_dtype(df[col]):
                invalid_types.append({
                    "column": col,
                    "expected": "categorical",
                    "found": "numeric"
                })
    return invalid_types

def check_categorical_values(df):
    invalid = []
    for col, valid_values in VALID_CATEGORIES.items():
        if col in df.columns:
            found_values = set(df[col].dropna().unique())
            unexpected = found_values - set(valid_values)
            if unexpected:
                invalid.append({
                    "column": col,
                    "invalid_values": list(unexpected),
                    "expected": valid_values
                })
    return invalid

def clean_missing_values(df):
    for col in df.columns:
        missing_ratio = df[col].isna().mean()
        if missing_ratio == 0:
            continue
        elif missing_ratio < MISSING_THRESHOLD:
            df.dropna(subset=[col], inplace=True)
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_contract_length(df):
    if "Contract Length" in df.columns:
        df["Contract Length"] = df["Contract Length"].map(CONTRACT_MAPPING)
    return df

# ===========================
# Endpoint FastAPI
# ===========================

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Cek nama dan format file
    filename = file.filename
    if not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read CSV")

    filename_base = os.path.splitext(filename)[0]
    mapping, ambiguous = column_mapping(df.columns)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in mapping]

    if missing_columns:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": f"Missing required columns: {', '.join(missing_columns)}",
                "code": "MISSING_COLUMNS",
            },
        )

    if ambiguous:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": "Multiple possible matches found for some required columns.",
                "code": "AMBIGUOUS_COLUMNS",
                "details": [{'expected': col, 'matches': matches} for col, matches in ambiguous]
            },
        )

    df_renamed = df.rename(columns={v: k for k, v in mapping.items()})
    X = df_renamed[selected_features]

    invalid_types = check_column_dtypes(X)
    if invalid_types:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": "Invalid data types detected in input.",
                "code": "INVALID_DATA_TYPES",
                "details": invalid_types,
            },
        )

    invalid_categories = check_categorical_values(X)
    if invalid_categories:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "error": "Some categorical columns contain unknown values.",
                "code": "INVALID_CATEGORY_VALUES",
                "details": invalid_categories,
            },
        )

    X = clean_missing_values(X)
    X = encode_contract_length(X)
    X_scaled = scaler.transform(X)

    try:
        predictions = model.predict(X_scaled)
        df_pred = pd.DataFrame({"Prediction": predictions}, index=X.index)
        df_final = df.join(df_pred)

        # Preview
        cols = df_final.columns.tolist()
        preview_rows = [
            OrderedDict((col, row[col]) for col in cols)
            for _, row in df_final.head().iterrows()
        ]

        counts = df_final["Prediction"].value_counts()
        percentages = df_final["Prediction"].value_counts(normalize=True) * 100
        summary = {
            str(k): {
                "count": int(counts[k]),
                "percent": round(percentages[k], 2)
            } for k in counts.index
        }

        os.makedirs("tmp", exist_ok=True)
        output_filename = f"{filename_base}_prediction_result.csv"
        csv_path = os.path.join("tmp", output_filename)
        df_final.to_csv(csv_path, index=False)

        return {
            "status": "success",
            "data": {
                "filename": filename,
                "preview": preview_rows,
                "summary": summary,
                "total": len(df_final),
                "download_url": f"/download?file={output_filename}",
            },
        }

    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/download")
def download_csv(file: str = Query(...)):
    file_path = os.path.join("tmp", file)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Prediction file not found")

    return FileResponse(
        path=file_path,
        filename=file,
        media_type='text/csv'
    )
