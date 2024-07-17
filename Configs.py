from pathlib import Path

def get_config():
    return {
        "batch_size": 40,
        "num_epochs": 3500,
        "lr": 10**-4,
        "seq_len_tgt": 9,
        "seq_len_src": 15,
        "d_model": 100,
        "datasource": 'MatchCal',
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])