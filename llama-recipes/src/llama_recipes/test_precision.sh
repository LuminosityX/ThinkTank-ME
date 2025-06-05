#!/bin/sh

# 定义变量
COUNTRY_CODE="ISR"
COUNTRY_NAME="israel"

conda activate meta_llama_31

python -u test_precision.py --test --test_model_name ${COUNTRY_NAME} --use_wandb --use_peft --peft_method lora --quantization 8bit --use_fp16 --from_peft_checkpoint /llama-recipes/src/llama_recipes/peft_model/llama31/35A_${COUNTRY_CODE}_object/2 --model_name /meta-llama/Llama-3.1-8B-Instruct --output_dir /llama-recipes/src/llama_recipes/peft_model/llama31/35A_${COUNTRY_CODE}_object --test_split  DATA_PATH/mideast_test_object.json --test_json_file /llama-recipes/src/llama_recipes/peft_model/llama31/35A_${COUNTRY_CODE}_object/test_result.json
