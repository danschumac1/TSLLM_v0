from typing import List
import csv
import numpy as np
import pandas as pd

#endregion ------------------------------------------------------------------------


#endregion   ------------------------------------------------------------------------











def load_csv(file_path:str) -> List[dict]:

    """Load a CSV file and return a list of dictionaries representing each row."""
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            data.append(row)
        
    return data


def read_ecg_references(file_path: str):
    df = pd.read_csv(file_path, header=None, names=["record_id", "label"])
    label_map = df.set_index("record_id")["label"].to_dict()   # {"A00001": "N", ...}

    # Optional: convert to integers
    # cls2idx = {"N": 0, "A": 1, "O": 2, "~": 3}
    # idx2cls = {v: k for k, v in cls2idx.items()}

    return label_map
