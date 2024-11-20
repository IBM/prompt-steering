# `prompt-steering`

This repository is a landing page for tools related to prompt steering in language models. 
Our primary focus is to measure and analyze how controllable models are through prompting.

### Current features (last updated 2024-11-15)
- `prompt-steerability`: A benchmark suite for measuring the extent to which models can be prompted to change behavior

## Installation

To install, set up a virtual environment using Python 3.10, navigate to `prompt_steerability/`, and run:

    pip install -e .

## Setting up vLLM

Model access is enabled via HuggingFace and [vLLM](https://docs.vllm.ai/en/latest/). For a simple guide please refer to the [quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) or follow these steps:

1. Request access to models through HuggingFace:
   - visit the model page on HuggingFace
   - click "Access" button and fill out the form if model is gated
   - once approved, get your HuggingFace token from the account settings
   - log in using: `huggingface-cli login`


2. Install vLLM with GPU support in your virtual environment:
```bash
pip install vllm[all]
```

3. Start a vLLM server (note model access uses the OpenAI API format):
```bash
python -m vllm.entrypoints.openai.api_server \
    --model <your-model-name> \
    --host 0.0.0.0 \
    --port 8000 \
    --device cuda 
```

Ensure that `<your-model-name>` exists as one of the `model-id`'s in `model-config` in `config.yaml` (and as named on HuggingFace).

For a complete list of parameter options (including options for instantiating multiple GPUs), please see [the vLLM documentation](https://docs.vllm.ai/en/v0.4.1/serving/openai_compatible_server.html#command-line-arguments-for-the-server). 

## Configuring the benchmark 

Configuration details are specified in `prompt_steerability/config.yaml`. Key parameters are:

- `num-benchmark-trials`: the number of (outer) benchmark trials
- `persona-dimensions`: the persona dimensions you want to benchmark the models over
- `models`: the models you wish to benchmark (ensure these are hosted at the `base-url` locations specified in `model-config`)
- `steering` config:
  - `method`: the steering method (`principles`)
  - `params`:
    - `num-steering-trials`: the number of steering trials
    - `steering-budgets`: dictates the number of steering statements to include
- `profiling`:
  - `method`: the profiling method (`query-principles`)
  - `params`:
    - `num-questions`: how many profiling questions to ask per trial
    - `representation`: how to represent profiles
  - `inference`: how to parse outputs (`log-probs` vs `output-parsing`)


## Running the benchmark

After hosting the desired models, navigate to `persona/benchmark/` and run `run_persona_steerability.py` to run the benchmark.

## Citation

```bash
@article{miehling2024evaluating,
    title={Evaluating the Prompt Steerability of Large Language Models},
    author={Miehling, Erik and Desmond, Michael and Ramamurthy, Karthikeyan Natesan and Daly, Elizabeth M. and Dognin, Pierre and Rios, Jesus and Bouneffouf, Djallel and Liu, Miao},
    journal={arXiv preprint arXiv:2411.12405},
    year={2024}
}
```