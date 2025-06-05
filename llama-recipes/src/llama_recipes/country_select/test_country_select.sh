#!/bin/sh

# 定义变量
COUNTRY_CODE="country_select_train"

conda activate meta_llama_31

python -u test_precision.py --use_wandb --use_peft --peft_method lora --quantization 8bit --use_fp16 --from_peft_checkpoint /llama-reciprs/src/llama_recipes/peft_model/llama31/35A_${COUNTRY_CODE}/0 --model_name /meta-llama/Llama-3.1-8B-Instruct --output_dir /llama-recipes/src/llama_recipes/peft_model/llama31/35A_${COUNTRY_CODE} --test_split  country_select_test_dataset.json --test_json_file /llama-reciprs/src/llama_recipes/peft_model/llama31/35A_${COUNTRY_CODE}/country_select_result.json
