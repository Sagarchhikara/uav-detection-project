
import pickle
import numpy as np

file_path = "/home/sagar/uav-detection-project/data/raw/archive/RML2016.10a_dict.pkl"

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:5]} ... (total {len(data)} keys)")
        # Inspect one item
        first_key = list(data.keys())[0]
        sample = data[first_key]
        print(f"Sample shape for key {first_key}: {sample.shape}")
        print(f"Sample data type: {sample.dtype}")
    else:
        print("Data is not a dictionary.")

except Exception as e:
    print(f"Error loading pickle: {e}")
