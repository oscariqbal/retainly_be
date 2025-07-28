from flask import Flask, request, jsonify, send_file, Response
import json
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import re
from collections import OrderedDict
from datetime import datetime
import time
import requests
import gdown

MODEL_ID = "1MGQEo0B54S4E8xW7zA15tlR9PFokMP9W"
MODEL_PATH = "model.pkl"

SCALER_PATH = "scaler.pkl"

def download_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)

download_model()

app = Flask(__name__)
CORS(app)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

selected_features = ["Age", "Support Calls", "Payment Delay", "Total Spend", "Contract Length"]

# =========================
# FUNTION UNTUK VALIDATION
# =========================

# Function untuk menangani potensi nama kolom tidak match 100% dan menangani nama kolom ambigu (Percabangan 3 & 4)
REQUIRED_COLUMNS = {
    "Age": ["age"],
    "Support Calls": ["support calls", "support_calls"],
    "Payment Delay": ["payment delay", "payment_delay"],
    "Total Spend": ["total spend", "total_spend"],
    "Contract Length": ["contract length", "contract_length"]
}
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

# Function untuk menangani potensi tipe data tiap kolom tidak sesuai (Percabangan 5)
REQUIRED_DTYPES = {
    "Age": "number",
    "Support Calls": "number",
    "Payment Delay": "number",
    "Total Spend": "number",
    "Contract Length": "categorical"
}

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

# Function untuk menangani potensi kolom kategorik (Contract Length) tidak punya kategori yang sesuai (Percabangan 6)
VALID_CATEGORIES = {
        "Contract Length": ["Monthly", "Quarterly", "Annual"]
    }

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

# ============================
# FUNTION UNTUK PREPROCESSING
# ============================

# Function untuk membersihkan nilai null (Preprocessing 1)
MISSING_THRESHOLD = 0.05

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

# Function untuk encoding fitur contract_length (Preprocessing 2)
CONTRACT_MAPPING = {
    "Monthly": 0.0,
    "Quarterly": 0.5,
    "Annual": 1.0
}

def encode_contract_length(df):
    if "Contract Length" in df.columns:
        df["Contract Length"] = df["Contract Length"].map(CONTRACT_MAPPING)
    return df

def cleanup_tmp(folder="tmp", max_age_minutes=10):
    now = time.time()
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            age_minutes = (now - os.path.getmtime(filepath)) / 60
            if age_minutes > max_age_minutes:
                os.remove(filepath)

# ================================================
# FUNTION UNTUK CREATE OBJEK DATA (RESPONSE JSON)
# ================================================

# Function untuk create objek previews
def generate_preview_rows(df_final):
    cols = df_final.columns.tolist()
    return [
        OrderedDict((col, row[col]) for col in cols)
        for _, row in df_final.head().iterrows()
    ]

# Function untuk create array summary
def generate_summary(df_final):
    counts = df_final["Churn Prediction"].value_counts()
    percentages = df_final["Churn Prediction"].value_counts(normalize=True) * 100
    summary = {}
    for label in [0.0, 1.0]:
        summary[str(label)] = {
            "count": int(counts.get(label, 0)),
            "percent": round(percentages.get(label, 0), 2)
        }
    return summary

# Function untuk create array insight
def generate_churn_insights(df_final):
    df_churn = df_final[df_final["Churn Prediction"] == 1]

    insights = []

    # 1. Age
    mean_age = df_churn["Age"].mean()
    insights.append({
        "feature": "Age",
        "insight": f"Customers predicted to churn have an average age of {mean_age:.1f} years."
    })

    # 2. Support Calls
    mean_calls = df_churn["Support Calls"].mean()
    prop_over_3 = (df_churn["Support Calls"] >= 3).mean() * 100
    insights.append({
        "feature": "Support Calls",
        "insight": f"Churning customers contacted customer support an average of {mean_calls:.1f} times, with {prop_over_3:.0f}% contacting more than 3 times."
    })

    # 3. Payment Delay
    mean_delay = df_churn["Payment Delay"].mean()
    prop_delay_over_5 = (df_churn["Payment Delay"] > 5).mean() * 100
    insights.append({
        "feature": "Payment Delay",
        "insight": f"The average payment delay is {mean_delay:.1f} days, and {prop_delay_over_5:.0f}% had delays longer than 5 days."
    })

    # 4. Total Spend
    avg_spend = df_churn["Total Spend"].mean()
    low_spender_ratio = (df_churn["Total Spend"] > 300).mean() * 100
    insights.append({
        "feature": "Total Spend",
        "insight": f"Churning customers spent an average of {avg_spend:,.0f}, with {low_spender_ratio:.0f}% spending more than 300."
    })

    # 5. Contract Length
    contract_counts = df_churn["Contract Length"].value_counts(normalize=True)
    if 1 in contract_counts:
        short_contract_pct = contract_counts[1] * 100
        insight_text = f"{short_contract_pct:.0f}% of churning customers had a 1-month contract."
    else:
        insight_text = "No churning customers had a 1-month contract."
    
    insights.append({
        "feature": "Contract Length",
        "insight": insight_text
    })

    return insights

# =========================
# FUNTION UNTUK ABOUT PAGE
# =========================

def generate_model_info(model):
    return {
        "model_name": model.__class__.__name__,
        "n_estimators": getattr(model, "n_estimators", None),
        "max_depth": getattr(model, "max_depth", None),
        "random_state": getattr(model, "random_state", None),
        "trained_on": "2025-07-22",
        "train_size": 352666,
        "test_accuracy": 0.9870,
        "cross_val_score": 0.9871,
    }

def get_feature_importance(model, selected_features):
    importances = model.feature_importances_
    importance_data = [
        {"feature": feature, "importance": round(imp, 4)}
        for feature, imp in zip(selected_features, importances)
    ]
    importance_data.sort(key=lambda x: x["importance"], reverse=True)
    return importance_data

@app.route('/')
def home():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    cleanup_tmp("tmp", max_age_minutes=10)
    # Percabangan 1: Cek kondisi apakah file ada di request
    if 'file' not in request.files:
        return jsonify({
            'status': "error",
            'error': "No file uploaded.",
            'code': "MISSING_FILE",
        }), 400


    uploaded_file = request.files['file']

    # Percabangan 2: Cek kondisi apakah .csv dapat di-read
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return jsonify({
            'status': "error",
            'error': "Failed to read CSV",
            'code' : "INVALID_CSV",
        }), 400
    

    original_filename = uploaded_file.filename
    filename_base = os.path.splitext(original_filename)[0]

    mapping, ambiguous = column_mapping(df.columns)

    # Percabangan 3: Apakah .csv memiliki kolom yang dibutuhkan
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in mapping]

    if missing_columns:
        print("User columns:", df.columns.tolist())
        print("Mapping:", mapping)
        return jsonify({
            'status' : "error",
            'error': f"Missing required columns: {', '.join(missing_columns)}",
            'code': "MISSING_COLUMNS",
        }), 400
    
    # Percabangan 4: Apakah .csv memiliki kolom ambigu
    if ambiguous:
        return jsonify({
            'status': "error",
            'error': "Multiple possible matches found for some required columns.",
            'code': "AMBIGUOUS_COLUMNS",
            'details': [
                {'expected': col, 'matches': matches} for col, matches in ambiguous
            ]
        }), 400

    df_renamed = df.rename(columns={v: k for k, v in mapping.items()})
    X = df_renamed[selected_features]

    # Percabangan 5: Apakah tipe data sudah sesuai
    invalid_types = check_column_dtypes(X)
    if invalid_types:
        return jsonify({
            'status': "error",
            'error': "Invalid data types detected in input.",
            'code': "INVALID_DATA_TYPES",
            'details': invalid_types
        }), 400
    
    # Precabangan 6: Apakah kolom kategorik (Contract Length) punya kategori yang sesuai
    invalid_categories = check_categorical_values(X)
    if invalid_categories:
        return jsonify({
            'status': "error",
            'error': "Some categorical columns contain unknown values.",
            'code': "INVALID_CATEGORY_VALUES",
            'details': invalid_categories
        }), 400

    # Preprocessing 1: Data Cleaning
    X = clean_missing_values(X)

    # Preprocessing 2: Data Encoding (Categorical Feature)
    X = encode_contract_length(X)

    # Preprocessing 3: Data Normalization (Scaling by MinMax Scaler)
    X_scaled = scaler.transform(X)

    # Inference
    try:
        # Prediction & Probabilitas
        predictions = model.predict(X_scaled)
        probas = model.predict_proba(X_scaled)[:, 1]

        df_pred = pd.DataFrame({
            "Churn Prediction": predictions,
            "Churn Probability": probas
        }, index=X.index)
        df_final = df.join(df_pred)

        # Generate objek data
        preview_rows = generate_preview_rows(df_final)
        summary = generate_summary(df_final)
        insights = generate_churn_insights(df_final)

        # Generate file .csv
        os.makedirs("tmp", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{filename_base}_{timestamp}_prediction_result.csv"

        csv_path = os.path.join("tmp", output_filename)
        df_final.to_csv(csv_path, index=False)
        
        return Response(
            json.dumps({
                'status': 'success',
                'data': {
                    'original_filename': original_filename,
                    'preview': preview_rows,
                    'summary': summary,
                    'insights': insights,
                    'output_filename': output_filename,
                }
            }),
            mimetype='application/json'
    )
    except Exception as e:
        print("Error saat prediksi:", str(e))
        return jsonify({
            'status': "error",
            'error': "Prediction failed due to some issues",
            'code': "PREDICTION_FAILED",
        }), 500

@app.route("/download", methods=["GET"])
def download_csv():
    filename = request.args.get('file')

    if not filename:
        return jsonify({
            "status": "error",
            "error": "Missing parameter in request",
            'code': "MISSING_PARAMS",
        }), 400

    file_path = os.path.join("tmp", filename)

    if not os.path.exists(file_path):
        return jsonify({
            "status": "error",
            "error": "File not found",
            'code': "FILE_NOT_FOUND",
        }), 404

    return send_file(
        file_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )

@app.route("/about", methods=["GET"])
def about_model():
    try:
        model_info = generate_model_info(model)
        feature_importance = get_feature_importance(model, selected_features)

        return Response(
            json.dumps({
                "status": "success",
                "data": {
                    "model_info": model_info,
                    "feature_importance": feature_importance
                }
            }),
            mimetype='application/json'
        )
    except Exception as e:
        return Response(
            json.dumps({
                "status": "error",
                'error': "Failed to get model's information.",
                'code': "INFO_NOT_FOUND",
            }),
            mimetype='application/json',
            status=500
        )


if __name__ == '__main__':
    app.run(debug=True)
