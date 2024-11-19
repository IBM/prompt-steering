import prompt_steerability
from prompt_steerability import PromptSteerability
from prompt_steerability.utils.data_utils import load_config
from pathlib import Path

config_path = Path(prompt_steerability.__file__).parent / "config.yaml"
config = load_config(config_path)

# initialize benchmark from config
benchmark = PromptSteerability(config=config, experiment_name="persona-steering")

# run benchmark
results = benchmark.run()
