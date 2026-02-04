import os
import pandas as pd
import tempfile
import sys
from scripts.preprocess import preprocess_titanic


def test_preprocess_titanic_tmpfile():
    # Create a small sample Titanic dataframe
    df = pd.DataFrame({
        'PassengerId': [1,2,3],
        'Pclass': [3,1,2],
        'Name': ['a', 'b', 'c'],
        'Sex': ['male', 'female', 'male'],\
        'Age': [22, None, 35],
        'SibSp': [1, 0, 0],
        'Parch': [0, 1, 0],
        'Fare': [7.25, 71.2833, 8.05],
        'Survived': [0, 1, 1]
    })

    tmpdir = tempfile.mkdtemp()
    input_path = os.path.join(tmpdir, 'titanic.csv')
    out_path = os.path.join(tmpdir, 'processed.csv')
    df.to_csv(input_path, index=False)

    preprocess_titanic(input_path, out_path)

    assert os.path.exists(out_path)
    out_df = pd.read_csv(out_path)
    assert 'target' in out_df.columns
    assert len(out_df) == 3
    assert 'Age' in out_df.columns
