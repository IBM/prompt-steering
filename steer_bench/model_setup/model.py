import steer_bench
from steer_bench.utils.data_utils import load_config
from pathlib import Path

# import appropriate modules from config
package_root = Path(steer_bench.__file__).parent
config_path = package_root / "model_setup" / "model_config.yaml"
CONFIG = load_config(config_path)

# print(CONFIG)

for api_name, api_dict in CONFIG['apis'].items():
    if api_dict['models']:
        if api_name == "vllm-api":
            from steer_bench.model_setup.model_wrappers.vllm_model import VLLMModel

class Model:
    def __new__(cls,
                model_id):
        api_config = cls._get_api_config(model_id)
        
        # print(api_config)

        if api_config["api-name"] == "vllm-api":
            return VLLMModel(model_id=model_id,
                               api_config=api_config)

    @staticmethod
    def _get_api_config(model_id):

        # print(model_id)

        for api_name, api_dict in CONFIG['apis'].items():

            if api_dict['models']:

                if api_name == "vllm-api":
                    # print("API NAME", api_name)
                    # print("API DICT MODELS", api_dict['models'])

                    parameters = api_dict['parameters']
                    for model_dict in api_dict['models']:
                        if model_dict['model-id'] == model_id:
                            return {
                                "api-name": api_name,
                                "parameters": parameters,
                                **model_dict
                            }