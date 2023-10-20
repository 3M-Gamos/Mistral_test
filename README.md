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

