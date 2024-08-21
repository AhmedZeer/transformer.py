from pathlib import Path

def generate_config():
    return {
        "lang_src":"en",
        "lang_trg":"tr",
        "tokenizer_path":f"tokenizer_{0}.json",
        "seq_len":2500,
        "batch_size":4,
        "model_folder":"weihgts",
        "model_basename":"tmodel_",
        "preload": None,
        "lr":1e-4,
        "experiment_name":"runs/tmodel",
        "num_epochs":1
    }

def get_weight_file_path(config, epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}"
    return str(Path('.') / model_folder / model_filename)

