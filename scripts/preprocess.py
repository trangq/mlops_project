"""
Simple preprocessing pipeline script for Titanic data.
- Expects raw data file (`titanic_raw.csv`) as input argument.
- Writes processed CSV to the output path argument.
"""
import sys
import pandas as pd
import numpy as np
import os

# --- Configuration (Based on DVC arguments, not hardcoded paths) ---

def preprocess_titanic(input_file: str, output_file: str):
    """
    Loads raw Titanic data, performs feature engineering, and saves the cleaned data.
    """
    
    print(f"Loading raw data from: {input_file}")
    
    try:
        # Tải dữ liệu thô (đã được DVC kéo về)
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_file}. Cannot proceed.")
        sys.exit(1)

    # --- Logic Tiền xử lý (Feature Engineering & Cleaning) ---

    # 1. Cleaning: Điền giá trị Age thiếu bằng trung bình
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # 2. Tạo Feature mới: FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 3. Mã hóa Categorical: Sex (dùng Label/One-hot Encoding đơn giản)
    df['Sex_male'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)

    # 4. Chọn Features và Target
    features = ['Age', 'Pclass', 'Fare', 'FamilySize', 'Sex_male']
    target = 'Survived'

    df_cleaned = df[features + [target]].dropna()

    # Đổi tên cột Target thành 'target' (chuẩn hóa cho bước training)
    df_cleaned.rename(columns={'Survived': 'target'}, inplace=True) 

    # --- Kết thúc Logic Tiền xử lý ---

    print(f"Data processing complete. Writing processed dataset to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    print(f"Wrote processed dataset to {output_file}")


if __name__ == '__main__':
    # Logic kiểm tra đối số: Phải có 2 đối số (input path và output path)
    if len(sys.argv) != 3:
        # Nếu thiếu đối số, DVC/Airflow sẽ báo lỗi, nhưng ta cung cấp hướng dẫn rõ ràng
        print("Error: Missing input/output paths.")
        print("Usage: python preprocess.py <input_raw_csv_path> <output_cleaned_csv_path>")
        sys.exit(1)
        
    input_path = sys.argv[1] # Đường dẫn Input (e.g., data/raw_data.csv)
    output_path = sys.argv[2] # Đường dẫn Output (e.g., data/cleaned_data.csv)
    
    preprocess_titanic(input_path, output_path)