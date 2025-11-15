# ./src/utils/get_prompter.py

from typing import Tuple, Union
import torch
from global_utils.prompter import (
    GemmaPrompter, HFPrompter, OpenAIPrompter
)

def get_prompter_class_and_kwargs(
    model_type: str,
    device_map,
    show_prompt: bool = False
) -> Tuple[Union[GemmaPrompter, HFPrompter, OpenAIPrompter], dict]:
    # Which Prompter class to use for each model_type
    prompt_class_mapping = {
        "gpt":            OpenAIPrompter,
        "llama":          HFPrompter,
        "mistral":        HFPrompter,
        "gemma":          GemmaPrompter,
        "TimeMQA_llama":  HFPrompter,
        "TimeMQA_qwen":   HFPrompter,
        "TimeMQA_mistral":HFPrompter,
    }

    # Default (full checkpoint) base models
    llm_model_mapping = {
        "gpt":     "gpt-4o-mini",
        "llama":   "meta-llama/Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "gemma":   "google/gemma-3-12b-it",

        # For TimeMQA_* we still set llm_model to the base; adapters are added below
        "TimeMQA_llama":   "meta-llama/Llama-3.1-8B-Instruct",
        "TimeMQA_qwen":    "Qwen/Qwen2.5-7B-Instruct",
        "TimeMQA_mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    }

    if model_type not in prompt_class_mapping:
        raise ValueError(
            f"Invalid model type: {model_type}. "
            f"Choose from {list(prompt_class_mapping.keys())}."
        )

    PrompterClass = prompt_class_mapping[model_type]
    llm_model = llm_model_mapping[model_type]

    # Shared kwargs passed into the Prompter __init__
    shared_kwargs = {
        "llm_model": llm_model,
        "device_map": device_map,
        "show_prompt": show_prompt,
    }

    # ---------- Plain HF checkpoints: eager-load model/tokenizer/processor ----------
    if model_type in ["llama", "mistral"]:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            device_map=normalize_device_map(device_map),
            torch_dtype=torch.float16,
        )
        shared_kwargs.update({"model": model, "tokenizer": tokenizer})

    elif model_type == "gemma":
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(llm_model, use_fast=True)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        model = Gemma3ForConditionalGeneration.from_pretrained(
            llm_model,
            device_map=normalize_device_map(device_map),
            torch_dtype=torch.bfloat16,
        )
        shared_kwargs.update({"model": model, "processor": processor})

    # ---------- TimeMQA adapters: let HFPrompter load base + PEFT adapter ----------
    elif model_type == "TimeMQA_qwen":
        shared_kwargs.update({
            # Tell HFPrompter to load base + adapter
            "base_model_id": "Qwen/Qwen2.5-7B-Instruct",
            "adapter_id": "Time-MQA/Qwen-2.5-7B",
            "trust_remote_code": True,    # Qwen chat template
            "torch_dtype": torch.bfloat16,
        })

    elif model_type == "TimeMQA_llama":
        shared_kwargs.update({
            "base_model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "adapter_id": "Time-MQA/Llama-3-8B",
            "use_fast_tokenizer": True,
            "torch_dtype": torch.bfloat16,
        })

    elif model_type == "TimeMQA_mistral":
        # Only keep this if you actually have this adapter repo
        shared_kwargs.update({
            "base_model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "adapter_id": "Time-MQA/Mistral-7B",
            "torch_dtype": torch.float16,
        })

    return PrompterClass, shared_kwargs
