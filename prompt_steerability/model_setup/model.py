import prompt_steerability
from prompt_steerability.utils.data_utils import load_config
from prompt_steerability.model_setup.model_wrappers.vllm_model import VLLMModel
from pathlib import Path

package_root = Path(prompt_steerability.__file__).parent
config_path = package_root / "config.yaml"
CONFIG = load_config(config_path)

# print(CONFIG)


class Model:
    def __new__(cls, model_id):
        model_config = cls._get_model_config(model_id)
        return VLLMModel(model_id=model_id, api_config=model_config)

    @staticmethod
    def _get_model_config(model_id):
        for model_dict in CONFIG['model-config']:
            if model_dict['model-id'] == model_id:
                return {
                    "api-name": "vllm-api",
                    "parameters": {},
                    **model_dict
                }
