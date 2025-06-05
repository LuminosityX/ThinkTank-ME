import torch
import json
import sys
import fire
import time
import random
import numpy as np
from warnings import warn

from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoConfig, AutoTokenizer

sys.path.append("../")

from llama_recipes.configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset, get_custom_data_collator
from llama_recipes.utils.config_utils import (
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)

from test_llama import Trie_LlamaForCausalLM as LlamaForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
import os


def main(**kwargs):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_config = TRAIN_CONFIG()
    update_config((train_config), **kwargs)

    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    # setting quantization configs
    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning,
            )
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError(
                "8bit quantization is not supported with FSDP, please use 4bit quantization"
            )

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    config = AutoConfig.from_pretrained(train_config.model_name)
    if config.model_type == "llama":
        is_vision = False
        model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )
    else:
        raise ValueError(
            f"Model type {config.model_type} is not supported. Please use llama or mllama model."
        )

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name
        if train_config.tokenizer_name is None
        else train_config.tokenizer_name
    )
    # tokenizer.eos_token_id == 128009
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint)
            print(f"load peft weigth in the {train_config.from_peft_checkpoint}.")
            # peft_config = model.peft_config()
    

    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
    )

    custom_data_collator = get_custom_data_collator(tokenizer, dataset_config)
    val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
    val_dl_kwargs["batch_sampler"] = None
    if custom_data_collator:
        val_dl_kwargs["collate_fn"] = custom_data_collator

    eval_dataloader = torch.utils.data.DataLoader(
        dataset_val,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        shuffle=False,
        **val_dl_kwargs,
    )
    if len(eval_dataloader) == 0:
        raise ValueError(
            f"The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. ({len(eval_dataloader)=})"
        )
    else:
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    model.eval()
    
    json_output = {}
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            begin = time.time()
            print()
            print("################################")   
            orignal_example = eval_dataloader.dataset.training_file[str(step)]
            orignal_target = eval_dataloader.dataset.training_file[str(step)]["target"]
            orignal_example["output"] = []
            orignal_example["scores"] = []
            orignal_country_name = eval_dataloader.dataset.training_file[str(step)]["country_name"]
            print(f"The country of Query is {orignal_country_name}")
            

            # "repetition_penalty": 1.2
            generation_config = GenerationConfig(num_beams=2, num_return_sequences=2, return_dict_in_generate=True, output_scores=True, eos_token_id=128009, length_penalty=0.0)
            generate_ids = model.generate(generation_config=generation_config, input_ids=batch["prompt_ids"].cuda(), attention_mask=batch["prompt_attention_mask"].cuda())
            len_prompt = batch["prompt_ids"].shape[-1]
            # odict_keys(['sequences', 'sequences_scores', 'scores', 'beam_indices', 'past_key_values'])
            # print(f"keys of generate_ids: {generate_ids.keys()}")

            # print(batch["prompt_ids"])
            # print(batch["labels_ids"])

            sequences_i = generate_ids["sequences"]
            sequences_scores_i = generate_ids["sequences_scores"]

            print(f"***********generated scores*********")
            print(sequences_scores_i)
            
            for ii, _ in enumerate(sequences_i):
                
                print(f"***********generated target {ii}*********")
                if ii == 0:
                    generate_text = tokenizer.batch_decode(sequences_i[ii:ii+1], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                    print(generate_text)
                    print(f"id: {sequences_i[ii:ii+1, len_prompt-3:]}")
                    print(f"score: {sequences_scores_i[ii]}")
                else:
                    generate_text = tokenizer.batch_decode(sequences_i[ii:ii+1, len_prompt:], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                    print(generate_text)
                    print(f"id: {sequences_i[ii:ii+1, len_prompt-3:]}")
                    print(f"score: {sequences_scores_i[ii]}")

                generate_text = tokenizer.batch_decode(sequences_i[ii:ii+1, len_prompt:], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                orignal_example["output"].append(generate_text)
                orignal_example["scores"].append(sequences_scores_i[ii].item())

            print("***********orignal target*********")
            print(orignal_target)
            print("**********************************")
            json_output[f"{step}"] = orignal_example
            print(f"Time: {time.time()-begin}")
            print("################################")
            print()
            
            #if step == 5000:
            #    with open("/home/e/e0210514/workspace/LLM_Event/llama-recipes/src/llama_recipes/llama-7b-EGY-polecat_two_stage_object_0_5000.json", "w",  encoding='latin-1') as f:
            #        json.dump(json_output, f, indent=4)
            
    
    with open(train_config.test_json_file, "w",  encoding='latin-1') as f:
        json.dump(json_output, f, indent=4)

    
if __name__ == "__main__":
    fire.Fire(main)