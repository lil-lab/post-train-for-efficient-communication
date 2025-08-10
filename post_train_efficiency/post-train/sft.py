# flake8: noqa
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
Modified from example sft.py in trl
"""

import logging
import os
from contextlib import nullcontext
import json, re, sys
from glob import glob
TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)
from datasets import load_dataset, load_from_disk
import datasets
from dataclasses import dataclass
from trl.commands.cli_utils import init_zero_verbose, ScriptArguments, TrlParser
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler
import warnings
import torch
from datasets import load_dataset, Dataset
import numpy as np
from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, EarlyStoppingCallback

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
    apply_chat_template,
    DataCollatorForCompletionOnlyLM
)

import torch.nn.functional as F
from trl.trainer.utils import (
    disable_dropout_in_model,
    empty_cache,
    pad,
)
from collections import defaultdict


tqdm.pandas()


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


@dataclass
class CustomArguments:
    use_custom_trainer: bool = False
    peft_on_token_embed: bool = False #peft_on_token_embed on all embed and lm_head weights
    train_on_new_embed_only: bool = False
    DEBUG: bool = False
    kl_loss: bool = False
    loss_on_special_token_only: bool = False 
    jsd_weight: float = None
    ref_attn_mask_on_special_token: bool = True
    custom_chat_template_path: str = None
    loss_on_remention_only: bool = False #assumes that the special tokens are for locating the rementions; model does not see the special tokens. 
    untie_lm_head: bool = False
    multigpu: bool = False
    model_family: str = None #used to override the default response template for the customized data collator





class CustomizedDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        padding_free: bool = False,
        our_special_token_ids: List[int] = None,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index
        self.padding_free = padding_free
        self.our_special_token_ids=our_special_token_ids

    @staticmethod
    def process_row_for_remention_labels(example, markers, ignore_index):
        updated_input_ids=[]
        updated_attention_mask=[]
        remention_labels=[]

        in_remention=False

        for i in range(len(example["input_ids"])):

            if example["input_ids"][i]==markers[0]:
                in_remention=True
            elif example["input_ids"][i]==markers[1]:
                in_remention=False
            else:
                updated_input_ids.append(example["input_ids"][i])
                updated_attention_mask.append(example["attention_mask"][i])
                if in_remention or example["input_ids"][i-1]==markers[1]: #include rementions and any token right after it
                    remention_labels.append(example["input_ids"][i])
                else:
                    remention_labels.append(ignore_index)

        updated_example = {
            "input_ids":updated_input_ids,
            "attention_mask": updated_attention_mask,
        }
        return remention_labels, updated_example


                


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        remention_labels=[]
        for i in range(len(examples)):
            remention_label, updated_example=self.process_row_for_remention_labels(examples[i], self.our_special_token_ids, self.ignore_index)
            examples[i]=updated_example
            remention_labels.append(remention_label)
            assert(len(remention_label)==len(examples[i]["input_ids"]))

        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                response_token_ids_idxs = []
                human_token_ids_idxs = []

                for assistant_idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # find the indexes of the start of a response.
                    if (
                        self.response_token_ids
                        == batch["labels"][i][assistant_idx : assistant_idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_idxs.append(assistant_idx + len(self.response_token_ids))

                if len(response_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                human_token_ids = self.instruction_token_ids
                for human_idx in np.where(batch["labels"][i] == human_token_ids[0])[0]:
                    # find the indexes of the start of a human answer.
                    if human_token_ids == batch["labels"][i][human_idx : human_idx + len(human_token_ids)].tolist():
                        human_token_ids_idxs.append(human_idx)

                if len(human_token_ids_idxs) == 0:
                    warnings.warn(
                        f"Could not find instruction key `{self.instruction_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index

                if (
                    len(human_token_ids_idxs) > 0
                    and len(response_token_ids_idxs) > 0
                    and human_token_ids_idxs[0] > response_token_ids_idxs[0]
                ):
                    human_token_ids_idxs = [0] + human_token_ids_idxs

                for idx, (start, end) in enumerate(zip(human_token_ids_idxs, response_token_ids_idxs)):
                    # Make pytorch loss function ignore all non response tokens
                    if idx != 0:
                        batch["labels"][i, start:end] = self.ignore_index
                    else:
                        batch["labels"][i, :end] = self.ignore_index

                if len(response_token_ids_idxs) < len(human_token_ids_idxs):
                    batch["labels"][i, human_token_ids_idxs[-1] :] = self.ignore_index

        remention_labels=[torch.tensor(x) for x in remention_labels]

        if self.padding_free:
            raise NotImplementedError("padding_free not supported for customized data collator")
        
        batch["remention_labels"]=pad(remention_labels, padding_value=self.ignore_index, padding_side="right")
        return batch


class CustomTrainer(SFTTrainer):
    #call the SFTTrainer's __init__ method
    def __init__(
        self,
        model = None,
        args = None,
        data_collator = None,  # type: ignore
        train_dataset = None,
        eval_dataset = None,
        processing_class = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        peft_config= None,
        dataset_text_field = None,
        packing = False,
        formatting_func = None,
        max_seq_length = None,
        infinite = None,
        num_of_sequences = None,
        chars_per_token = None,
        dataset_num_proc = None,
        dataset_batch_size = None,
        neftune_noise_alpha = None,
        model_init_kwargs = None,
        dataset_kwargs = None,
        eval_packing = None,
        custom_args = None,
        ref_model = None,
        our_special_token_ids = None,
    ):  

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            dataset_text_field=dataset_text_field,
            packing=packing,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            infinite=infinite,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            dataset_num_proc=dataset_num_proc,
            dataset_batch_size=dataset_batch_size,
            neftune_noise_alpha=neftune_noise_alpha,
            model_init_kwargs=model_init_kwargs,
            dataset_kwargs=dataset_kwargs,
            eval_packing=eval_packing,
        )
        assert not (custom_args.loss_on_special_token_only and custom_args.loss_on_remention_only), "loss_on_special_token_only and loss_on_remention_only cannot be True at the same time"

        
        self.kl_loss = custom_args.kl_loss
        self.jsd_weight = custom_args.jsd_weight
        self.loss_on_special_token_only = custom_args.loss_on_special_token_only
        self.ref_attn_mask_on_special_token = custom_args.ref_attn_mask_on_special_token
        #disable dropout
        if custom_args.kl_loss:
            disable_dropout_in_model(self.model)

        self.loss_on_remention_only=custom_args.loss_on_remention_only
        self.our_special_token_ids=our_special_token_ids
        if custom_args.loss_on_special_token_only or custom_args.loss_on_remention_only:
            assert self.our_special_token_ids is not None, "our_special_token_ids cannot be None when training only on special tokens"
    
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if custom_args.train_on_new_embed_only:
            if not custom_args.peft_on_token_embed:
                raise ValueError("train_on_new_embed_only currently requires peft_on_token_embed to be True")
            print('-'*30, file=sys.stderr)
            print('hook disable updating original embed and lm_head weights', file=sys.stderr)
            # num_added_tokens=len(tokenizer)-original_vocab_size
            # grad_mask=torch.zeros(model.get_input_embeddings().weight.shape[0], device="cuda")
            # grad_mask=grad_mask.unsqueeze(1)
            # grad_mask[-num_added_tokens:]=1.0
            def freeze_original_embed_fn(grad):
                grad[:original_vocab_size, :]=0.0
                return grad
            
            if "gemma" in model_config.model_name_or_path:
                if custom_args.untie_lm_head: #use the default peft behavior which unties the lm_head
                    self.hook_handle1=self.model.model.model.embed_tokens.modules_to_save["default"].weight.register_hook(freeze_original_embed_fn)
                    self.hook_handle2=self.model.lm_head.modules_to_save["default"].weight.register_hook(freeze_original_embed_fn)
                else: #manually tie the module_to_save lm_head with the module_to_save embed again
                    self.hook_handle1=self.model.model.model.embed_tokens.modules_to_save["default"].weight.register_hook(freeze_original_embed_fn)
                    self.model.lm_head.modules_to_save["default"].weight=self.model.model.model.embed_tokens.modules_to_save["default"].weight
                    print("gemma model, so tie the new lm_head with the new embed by default", file=sys.stderr)
            else:
                self.hook_handle1=self.model.model.model.embed_tokens.modules_to_save["default"].weight.register_hook(freeze_original_embed_fn)
                self.hook_handle2=self.model.lm_head.modules_to_save["default"].weight.register_hook(freeze_original_embed_fn)
   

    

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
            
        del self._stored_metrics[train_eval]
        return super().log(logs)
    
    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    @staticmethod
    def generalized_jsd_loss(
        policy_logits, ref_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
    ):
        """
        Adapted from the gkd_trainer.py
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            policy_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            ref_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            labels: Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing loss
            beta: Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature: Softmax temperature (default: 1.0)
            reduction: Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        policy_logits = policy_logits / temperature
        ref_logits = ref_logits / temperature

        # Compute log probabilities for policy and probabilities for ref
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)

        # Compute the log of the mixture distribution
        # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
        beta = torch.tensor(beta, dtype=policy_log_probs.dtype)
        mixture_log_probs = torch.logsumexp(
            torch.stack([policy_log_probs + torch.log(beta), ref_log_probs + torch.log(1 - beta)]),
            dim=0,
        )

        # Compute KL divergences using F.kl_div
        # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
        kl_ref = F.kl_div(mixture_log_probs, ref_log_probs, reduction="none", log_target=True)
        kl_policy = F.kl_div(mixture_log_probs, policy_log_probs, reduction="none", log_target=True)

        # Compute the Generalized Jensen-Shannon Divergence
        jsd = beta * kl_ref + (1 - beta) * kl_policy

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (jsd.size(0) * jsd.size(1))
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd   
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        original_labels=inputs["labels"].clone()
        special_token_mask=torch.isin(original_labels, torch.tensor(self.our_special_token_ids, device=inputs["labels"].device))
        if self.loss_on_special_token_only:

            labels_for_regularization=original_labels
            inputs["labels"][~special_token_mask]=-100

        elif self.loss_on_remention_only:
            inputs["labels"]=inputs["remention_labels"]
        else:
            labels_for_regularization=original_labels


        if not self.kl_loss:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        

        policy_outputs = model(**inputs)

        with torch.no_grad(), model.disable_adapter():
            if self.ref_attn_mask_on_special_token:
                ref_attention_mask = inputs["attention_mask"].clone()
                ref_attention_mask[special_token_mask] = 0
                ref_output = model(input_ids=inputs["input_ids"], attention_mask=ref_attention_mask, labels=None)
            else:
                ref_output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=None) 

        if self.loss_on_remention_only:
            labels_for_regularization=original_labels
            labels_for_regularization[inputs["remention_labels"]==labels_for_regularization]=-100
        else:
            labels_for_regularization[special_token_mask]=-100


        shifted_policy_logits = policy_outputs.logits[..., :-1, :]
        shifted_ref_logits = ref_output.logits[..., :-1, :]
        shifted_labels = labels_for_regularization[:, 1:]

        jsd_loss = self.generalized_jsd_loss(
            policy_logits=shifted_policy_logits,
            ref_logits=shifted_ref_logits,
            labels=shifted_labels,
        )
        if torch.isnan(jsd_loss):
            if torch.any(torch.all(shifted_labels==-100, dim=1)):
                print("="*30, file=sys.stderr)
                for row in inputs["input_ids"]:
                    print("-"*20, file=sys.stderr)
                    print("jsd set to 0 calculated for instance ", self.processing_class.decode(row, skip_special_tokens=True),file=sys.stderr)
            else:
                print("="*30, file=sys.stderr)
                for row in inputs["input_ids"]:
                    print("-"*20, file=sys.stderr)
                    print("jsd set to 0 calculated for instance ", self.processing_class.decode(row, skip_special_tokens=True),file=sys.stderr)
                raise ValueError("JSD loss is NaN for unknown reason")
            loss=policy_outputs["loss"]
            self.store_metrics({"lm_loss": policy_outputs["loss"].detach().cpu(), "jsd_loss": torch.zeros_like(policy_outputs["loss"]).cpu()}, train_eval="train" if self.model.training else "eval") 
        else:
            jsd_loss=self.jsd_weight*jsd_loss
            loss=policy_outputs["loss"]+jsd_loss 
            self.store_metrics({"lm_loss": policy_outputs["loss"].detach().cpu(), "jsd_loss": jsd_loss.detach().cpu()}, train_eval="train" if self.model.training else "eval") 
        
    
        return (loss, policy_outputs) if return_outputs else loss


    def get_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        return batch_samples, None



if __name__ == "__main__":
 
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, CustomArguments))
    args, training_args, model_config, custom_args = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if training_args.gradient_checkpointing and (model_config.model_name_or_path=="google/gemma-2-9b-it" or custom_args.model_family=="gemma"):
        model_kwargs["cache_implementation"] = None
        model_kwargs["attn_implementation"]="eager"

    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)

    ################
    # Dataset
    ################

    if os.path.exists(os.path.join(args.dataset_name, "train.json")):
        train_dataset = load_dataset("json", data_files=os.path.join(args.dataset_name, "train.json"))
        train_dataset = train_dataset["train"]
        eval_dataset=load_dataset("json", data_files=os.path.join(args.dataset_name, "eval.json"))["train"]
    else:
        train_dataset = load_from_disk(os.path.join(args.dataset_name, 'train'))
        try:
            eval_dataset = load_from_disk(os.path.join(args.dataset_name, 'eval'))
        except:
            eval_dataset=None


    with open(os.path.join(args.dataset_name, 'special_tokens.json')) as f:
        new_special_tokens=json.load(f)
    
    original_vocab_size = len(tokenizer)
    if "llama" in  model_config.model_name_or_path:
        tokenizer.pad_token = "<|reserved_special_token_247|>"
    our_special_token_ids=None
    if len(new_special_tokens)>0:
        print("adding new special tokens: ", new_special_tokens, file=sys.stderr)
        tokenizer.add_special_tokens({"additional_special_tokens": list(new_special_tokens.values())})
        our_special_token_ids=tokenizer.convert_tokens_to_ids(list(new_special_tokens.values()))

    if custom_args.custom_chat_template_path:
        with open(custom_args.custom_chat_template_path) as f:
            tokenizer.chat_template = f.read()

    tokenizer.padding_side="right"

    
    if "llama" in model_config.model_name_or_path:
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_config.model_name_or_path=="google/gemma-2-9b-it":
        response_template = "<start_of_turn>model\n"
    elif custom_args.model_family=="gemma":
        response_template = "<start_of_turn>model\n"
    elif custom_args.model_family=="llama":
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        raise ValueError(f"Need to know the model family for response template")
        
    if custom_args.loss_on_remention_only:
        collator = CustomizedDataCollator(response_template, tokenizer=tokenizer, our_special_token_ids=our_special_token_ids)
    else:
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)



    def formatting_func(examples):
        if isinstance(examples["prompt"], list):
            output_texts = []
            for i in range(len(examples["prompt"])):
                converted_sample = examples["prompt"][i]+examples["completion"][i]
                output_texts.append(tokenizer.apply_chat_template(converted_sample, tokenize=False))
            return output_texts
        else:
            converted_sample = examples["prompt"][i]+examples["completion"][i]
            return tokenizer.apply_chat_template(converted_sample, tokenize=False)

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training  
    ################
    lora_config=get_peft_config(model_config)
    if lora_config and custom_args.peft_on_token_embed:
        if "gemma" in model_config.model_name_or_path:
            lora_config.modules_to_save = ["embed_tokens", "lm_head"]
        else:
            lora_config.modules_to_save = ["embed_tokens", "lm_head"]
        print("="*30)
        print("additional modules to save: ", lora_config.modules_to_save)
    if not custom_args.DEBUG:
     
        if custom_args.multigpu:
            os.makedirs(training_args.output_dir, exist_ok=True)
        else:
            os.makedirs(training_args.output_dir)
        with open(os.path.join(training_args.output_dir, "example_train_data.json"), "w") as f:
            json.dump(train_dataset[:10], f)
    
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    if len(tokenizer) != model.config.vocab_size and not custom_args.loss_on_remention_only: #TODO: currently assumes model does not see markers under loss_on_remention_only
        if custom_args.DEBUG:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        else:
            model.resize_token_embeddings(len(tokenizer))

    training_args.dataset_kwargs={"add_special_tokens":False}



    with init_context:
        if custom_args.use_custom_trainer:
            print('='*60, file=sys.stderr)
            print("Using Customized Trainer!!!", file=sys.stderr)
            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                peft_config=lora_config, 
                data_collator=collator,
                formatting_func=formatting_func,
                custom_args=custom_args,
                our_special_token_ids=our_special_token_ids,
                # callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            )
        else:
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                peft_config=lora_config, 
                data_collator=collator,
                formatting_func=formatting_func,
                # callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            )
    #if report_to wandb
    if "wandb" in training_args.report_to:
        import wandb
        wandb.init(config=vars(custom_args), name=training_args.run_name)
    trainer.train()
    if custom_args.train_on_new_embed_only:
        trainer.hook_handle1.remove()
        if hasattr(trainer, "hook_handle2"):
            trainer.hook_handle2.remove()
    with save_context:
        trainer.save_model(training_args.output_dir)
