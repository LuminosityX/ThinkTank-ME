#!/bin/sh

COUNTRY_CODE="RUS"
COUNTRY_NAME="russia"

conda activate meta_llama_31

python finetuning.py --test_model_name ${COUNTRY_NAME} --use_wandb --project Polecat_MidEast_LLaMA31 --name H100_35A_${COUNTRY_CODE}_object_testall --use_peft --peft_method lora --quantization 8bit --use_fp16 --model_name /meta-llama/Llama-3.1-8B-Instruct --output_dir /llama-recipes/src/llama_recipes/peft_model/llama31/35A_${COUNTRY_CODE}_object --train_split COUNTRY_DATA_PATH/train_object.json --test_split COUNTRY_DATA_PATH/mideast_test_object.json
