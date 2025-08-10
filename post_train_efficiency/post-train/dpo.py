# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Modified from the dpo.py example in trl library.
"""
import torch, sys
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os, json
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from typing import  Optional
from dataclasses import dataclass

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclass
class CustomArguments:
    peft_on_token_embed: bool = False
    custom_chat_template_path: Optional[str] = None
    multigpu: bool = False
    DEBUG: bool = False


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig, CustomArguments))
    script_args, training_args, model_config, custom_args = parser.parse_args_and_config()
    ################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if training_args.gradient_checkpointing and "gemma" in model_config.model_name_or_path:
        model_kwargs["cache_implementation"] = None

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]


    ################   
    # Dataset
    ################  

    if os.path.exists(os.path.join(script_args.dataset_name, "train.json")):
        train_dataset = load_dataset("json", data_files=os.path.join(script_args.dataset_name, "train.json"))
        train_dataset = train_dataset["train"]
        eval_dataset=load_dataset("json", data_files=os.path.join(script_args.dataset_name, "eval.json"))["train"]
    else:
        train_dataset = load_from_disk(os.path.join(script_args.dataset_name, 'train'))
        try:
            eval_dataset = load_from_disk(os.path.join(script_args.dataset_name, 'eval'))
        except:
            eval_dataset=None

    with open(os.path.join(script_args.dataset_name, 'special_tokens.json')) as f:
        new_special_tokens=json.load(f)
    if len(new_special_tokens)>0 and not custom_args.DEBUG:
        for special_tok in new_special_tokens.values():
            assert(special_tok in tokenizer.additional_special_tokens)
            assert(len(tokenizer) == model.config.vocab_size)

        

    if custom_args.custom_chat_template_path:
        with open(custom_args.custom_chat_template_path) as f:
            tokenizer.chat_template = f.read()

    if peft_config and custom_args.peft_on_token_embed:
        peft_config.modules_to_save = ["embed_tokens", "lm_head"]
        print("="*30)
        print("additional modules to save: ", peft_config.modules_to_save)

    if not custom_args.DEBUG:
        #check if training_args.output_dir exists
        if custom_args.multigpu:
            os.makedirs(training_args.output_dir, exist_ok=True)
        else:
            if os.path.exists(training_args.output_dir):
                #a temporary fix for multigpu
                    new_output_dir=None
                    for idx in range(20):
                        if not os.path.exists(training_args.output_dir + f"_{idx}"):
                            new_output_dir=training_args.output_dir + f"_{idx}"
                            break

                    if new_output_dir is not None:
                        training_args.output_dir=new_output_dir
                    else:
                        raise ValueError("Too many output directories of the same name already exist.")
            else:
                os.makedirs(training_args.output_dir)
        #check if the file exists
        if not os.path.exists(os.path.join(training_args.output_dir, "example_train_data.json")):
            with open(os.path.join(training_args.output_dir, "example_train_data.json"), "w") as f:
                json.dump(train_dataset[:10], f)




    ##########
    # Training
    ################

    
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,

    )
    
    # show out_dir to stderr
    print("output_dir: ", training_args.output_dir, file=sys.stderr)
    trainer.train(training_args.resume_from_checkpoint)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)
 