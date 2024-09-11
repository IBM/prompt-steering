import json
import os
import pickle
import yaml


def load_config(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def save_to_pkl(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_json(filename):
    with open(filename) as file:
        data = json.load(file)
    return data


def load_from_pkl(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def prune_persona_dims(persona_dims, num_trials, filepath, model_name):
    dims = []
    trial_sets = {}
    for dim in persona_dims:
        trial_sets[dim] = []
        dim_added = False
        for trial in range(num_trials):
            profiling_file = f"profiling_dataset_{trial}_{dim}_{model_name}.pkl"
            responses_file = f"responses_{trial}_{dim}_{model_name}.json"
            profiling_path = os.path.join(filepath, profiling_file)
            responses_path = os.path.join(filepath, responses_file)
            if not (os.path.exists(profiling_path) and os.path.exists(responses_path)):
                if not dim_added:
                    dims.append(dim)
                    dim_added = True
                trial_sets[dim].append(trial)
    return dims, trial_sets
