from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self,
                 model_id,
                 api_config):
        self.model_id = model_id
        self.api_config = api_config

    @abstractmethod
    def call(self,
             prompt,
             save_dir):
        pass
