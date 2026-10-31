import json

def save_predictions(val_result, path):
    """
    Save predictions to a file.

    Args:
        val_result (dict): A dictionary containing the validation results.
        path (str): The path where the predictions will be saved.
    """

    print(type(val_result), val_result.keys(), val_result.values(), path)
    with open(path, 'w') as f:
        json.dump(val_result, f, indent=4)

def load_predictions(path):
    """
    Load predictions from a file.

    Args:
        path (str): The path from where the predictions will be loaded.

    Returns:
        dict: A dictionary containing the loaded predictions.
    """

    with open(path, 'r') as f:
        val_result = json.load(f)
    return val_result