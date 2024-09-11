from steer_bench import SteerBench
from steer_bench.utils.data_utils import load_config
from pathlib import Path

config_path = Path(__file__).parent / "config.yaml"
config = load_config(config_path)

# initialize benchmark from config
benchmark = SteerBench(config=config)

# run benchmark
results = benchmark.run()
