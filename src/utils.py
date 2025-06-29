import pickle
import os

def write_pickle(data_path, file_name, obj):
    """
    Write a Python object to a pickle file.

    Args:
        data_path (str): Directory path where the file should be saved.
        file_name (str): Name of the pickle file (e.g., 'model.pkl').
        obj (Any): Python object to be serialized and saved.

    Returns:
        None
    """
    full_path = os.path.join(data_path, file_name)
    with open(full_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Pickle file successfully written to: {full_path}")


def read_pickle(data_path, file_name):
    """
    Read a Python object from a pickle file.

    Args:
        data_path (str): Directory path where the pickle file is located.
        file_name (str): Name of the pickle file (e.g., 'model.pkl').

    Returns:
        Any: The deserialized Python object.
    """
    full_path = os.path.join(data_path, file_name)
    with open(full_path, 'rb') as f:
        obj = pickle.load(f)
    print(f"Pickle file successfully loaded from: {full_path}")
    return obj