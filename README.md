# `steer-bench`

This repo contains the development of a prompt steerability benchmark, termed `steer-bench`.

## Installation

Create and activate a virtual environment (Python 3.10). Run:

    pip install -r requirements.txt  

Install package:

    pip install -e steer_bench

## Running the benchmark

To run persona steering on a given model, navigate to `persona_steerability/benchmark/` and open `config.yaml`. List the models that you wish to benchmark:

    benchmark-config: {
        model_ids: [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "microsoft/Phi-3-medium-4k-instruct"
        ],
        ...
    }

Note that the model names here must match the names as specified in `model_setup/model_config.yaml`. 

Navigate to `persona_steerability/benchmark/` and run `run_persona_steering.py`.
