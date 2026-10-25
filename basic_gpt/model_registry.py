from transformers import AutoConfig, AutoModelForCausalLM
from basic_gpt import GPT, ModelConfig

# TODO: think about making model registry shared
def register_models() -> None:
    AutoConfig.register(ModelConfig.model_type, ModelConfig)
    AutoModelForCausalLM.register(ModelConfig, GPT)