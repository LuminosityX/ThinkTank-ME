#!/bin/sh

# 定义变量
COUNTRY_CODE="country_select_train"

conda activate meta_llama_31

python finetuning.py  --use_wandb --project Polecat_MidEast_LLaMA31 --name H100_35A_${COUNTRY_CODE} --use_peft --peft_method lora --quantization 8bit --use_fp16 --model_name /meta-llama/Llama-3.1-8B-Instruct --output_dir /llama-recipes/src/llama_recipes/peft_model/llama31/35A_${COUNTRY_CODE} --train_split country_select_train_dataset.json --test_split country_select_test_dataset.json
