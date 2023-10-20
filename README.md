# Mistral_test
I tried to fine tune the LLM Mistral model on my personnal thoughts(writings,poems and more personnal thoughts)
# Fine Tuning Mistral on My Thoughts Project

This project involves fine-tuning the Mistral model on a custom dataset using a Jupyter notebook (`Fine_tuning_mistral_on_my_thoughts.ipynb`). The script has been developed and executed in a Google Colab environment with GPU support.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Preparation](#model-preparation)
4. [Training](#training)
5. [Issues Encountered](#issues-encountered)
6. [Additional Information](#additional-information)

## Installation

```bash
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U datasets scipy ipywidgets matplotlib
!pip install FuzzyTM >=0.4.0

## Dataset Preparation

Datasets are loaded from JSON files and prepared for training and evaluation:

```bash
from datasets import load_dataset
train_dataset = load_dataset('json', data_files='train_data.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='val_data.jsonl', split='train')

## Model Preparation

```bash
import bitsandbytes
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)



