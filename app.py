from flask import Flask, request, jsonify, send_file, Response
import json
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import re
from collections import OrderedDict

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

selected_features = ["Age", "Support Calls", "Payment Delay", "Total Spend", "Contract Length"]

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


@app.route('/')
def home():
    return "API is running!"

@app.route('/predict', methods=['POST'])
def predict():
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
        # Prediction
        predictions = model.predict(X_scaled)

        # Buat DF baru dari hasil prediction
        df_pred = pd.DataFrame({
            "Prediction": predictions
        }, index=X.index)
        
        print("==== DF Asli ====")
        print(df.head(5))

        print("==== DF Prediction ====")
        print(df_pred.head(5))

        # Join DF baru dg DF ori
        df_final = df.join(df_pred)

        print("==== DF Final (Gabungan) ====")
        print(df_final.head(5))

        # Preview Rows dan Prediction Counts untuk fitur
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

        # DF Final disimpan sebagai file CSV sementara untuk endpoint download
        os.makedirs("tmp", exist_ok=True)
        output_filename = f"{filename_base}_prediction_result.csv"
        csv_path = os.path.join("tmp", output_filename)
        df_final.to_csv(csv_path, index=False)

        print("==== Preview Rows ====")
        for row in preview_rows:
            print(row)

        print(json.dumps(preview_rows, indent=2))
        
        return Response(
            json.dumps({
                'status': 'success',
                'data': {
                    'filename': original_filename,
                    'preview': preview_rows,
                    'summary': summary,
                    'total': len(df_final),
                    'download_url': f"/download?file={output_filename}",
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
            "error": "Missing 'file' parameter in request"
        }), 400

    file_path = os.path.join("tmp", filename)

    if not os.path.exists(file_path):
        return jsonify({
            "status": "error",
            "error": "Prediction file not found"
        }), 404

    return send_file(
        file_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    app.run(debug=True)
