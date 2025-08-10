from itertools import chain
import json, os, re
import time, copy, random
import numpy as np
import openai
from openai import OpenAI
from openai import (
    APIError,
    RateLimitError,
    APITimeoutError,
)
import httpx, sys
from abc import ABC, abstractmethod
from anthropic import Anthropic

import torch
from peft import PeftModel


class ModelWrapper(ABC):
    def __init__(self, model_args):
        self.model_args = model_args
        print("model_ckpt:", model_args.model_ckpt)

    @abstractmethod
    def get_spkr_intro(self, context_referents):
        pass

    @abstractmethod
    def get_lsnr_intro(self):
        pass

    def _model_specific_prompt_postprocessing(self, prompt):
        return prompt  # can override this as needed

    def get_spkr_prompt(
        self, intro, t, context_referents, target_name, interaction_args, records=[]
    ):
        label_space = self.model_args.label_space
        history = [entry["spkr_trial_record"] for entry in records[:t]] if t > 0 else []

        if interaction_args.no_history:
            round_name = "Current Round"
        else:
            round_name = f"Round {t+1}"
        target_label = target_name
        target_referent = target_name
        context_referents = context_referents
        trial_prompt = self._get_spkr_prompt_text_game(round_name, target_label, t)

        history = list(chain.from_iterable(history))

        if intro not in history and interaction_args.has_intro:
            history = intro + history

        prompt = history + trial_prompt

        prompt = self._model_specific_prompt_postprocessing(prompt)
        return prompt, trial_prompt, target_referent, target_label, context_referents

    def get_lsnr_prompt(
        self,
        intro,
        t,
        context_referents,
        target_fn,
        msg,
        records=[],
        random_seed=None,
        no_history=False,
        do_shuffle=False,
        misleading=False,
        has_intro=True,
    ):
        if no_history:
            history = []
        else:
            history = (
                [entry["lsnr_trial_record"] for entry in records[:t]] if t > 0 else []
            )

        trial_referents = context_referents.copy()
        if do_shuffle:
            random.seed(random_seed)
            random.shuffle(trial_referents)

        label_space = self.model_args.label_space

        for i in range(4):
            if trial_referents[i] == target_fn:
                target_item = trial_referents[i]
                target_label = label_space[i]
                break

        trial_items_after_1st_shuffle = trial_referents.copy()

        if misleading:
            random.seed(t + 1)
            random.shuffle(trial_referents)

        if no_history:
            round_name = "Current Round"
        else:
            round_name = f"Round {t+1}"

        trial_prompt = self._get_lsnr_prompt(round_name, trial_referents, msg)

        history = list(chain.from_iterable(history))

        if intro not in history and has_intro:
            history = intro + history

        prompt = history + trial_prompt

        prompt = self._model_specific_prompt_postprocessing(prompt)
        return (
            prompt,
            trial_prompt,
            target_item,
            target_label,
            trial_items_after_1st_shuffle,
            trial_referents,
        )

    @abstractmethod
    def _get_spkr_prompt(self, round_name, target_label):
        pass

    @abstractmethod
    def _get_spkr_prompt_text_game(self, round_name, target_label, t):
        pass

    @abstractmethod
    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        pass

    @abstractmethod
    def query(self, query):
        pass

    @abstractmethod
    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        pass

    @abstractmethod
    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        pass

    # how the feedback is presented is model-dependent and can be thought of as a hyperparameter. can try different phrasing/formats.
    def get_spkr_feedback(self, pred_fn, spkr_tgt_img, spkr_trial_imgs):
        return self._get_spkr_feedback(pred_fn, spkr_tgt_img, spkr_trial_imgs)   

    def get_lsnr_feedback(self, pred, target_item, context_items, spkr_msg):
        target_label = self.model_args.label_space[
            context_items.index(target_item)
        ]
        return self._get_lsnr_feedback(pred, target_label, spkr_msg)

    @abstractmethod
    def _get_lsnr_feedback(self, pred, target_label, spkr_msg):
        pass


class GPTModel(ModelWrapper):
    def __init__(self, model_args, organization_id, api_key):
        super().__init__(model_args)
        openai.organization = organization_id
        openai.api_key = api_key
        self.client = OpenAI(
            api_key=openai.api_key,
            organization=openai.organization,
            timeout=httpx.Timeout(15.0, read=5.0, write=10.0, connect=3.0),
        )
        self.spkr_system_msg = {
            "role": "system",
            "content": "You're a helpful assistant.",
        }
        self.lsnr_system_msg = {
            "role": "system",
            "content": "You are an assistant who will play a series of reference games with the user. You will pay close attention to the conversation history as more rounds are played.",
        }
        self.enforce_alternating_roles = True
        self.retry_after_seconds = 15
        self.retry_limit = 10
        self.bad_words=[]
      

    def get_spkr_intro(self, context_referents):
        intro_text = self.model_args.intro_text
        intro = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": intro_text},
                    {"type": "text", "text": f"\nReferential Context: \"{context_referents[0]}\", \"{context_referents[1]}\", \"{context_referents[2]}\", \"{context_referents[3]}\"\n"},
                ],
            }
        ]
        return intro

    def _get_spkr_prompt(self, round_name, target_label):
        raise NotImplementedError("Only text-based games supported.")

    def _get_spkr_prompt_text_game(self, round_name, target_label, t):

        if t==0:
            prompt_content=f"Target: \"{target_label}\""
        else:
            prompt_content = f"\nNext target: \"{target_label}\""

        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_content,
                    }
                ]
            }
        ]

        return prompt
    

    def get_lsnr_intro(self, context_referents):
        if isinstance(context_referents[0], str):
            #no shuffling for text only games, contexts in intro text
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.model_args.intro_text+f"\nReferential Context:\nA. {context_referents[0]}\nB. {context_referents[1]}\nC. {context_referents[2]}\nD. {context_referents[3]}"}
                    ],
                }
            ]
        
        else:
            #for image games, don't put images in the intro message to follow the style of the original ICCA
            return [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self.model_args.intro_text}],
                }
            ]

    def _get_lsnr_prompt(self, round_name, trial_items, msg):
        prompt = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{round_name}\nWhich item is this message referring to: {msg}? Output the label only (a single letter)."}]
            }
        ]
        return prompt

    def query(self, query, attempt=0):
        response = self._gpt_query(query)
        gen_msg = response.choices[0].message.content

        if self.model_args.role == "spkr":
            bad_words_mentioned=[]
            for bad_word in self.bad_words:
                if bad_word.lower() in gen_msg.lower(): 
                    if attempt>9:
                        return "Invalid message"
                    bad_words_mentioned.append(bad_word)
            if len(bad_words_mentioned)>0:
                query.append({"role": "assistant", "content": gen_msg})
                bad_words_string=", ".join(bad_words_mentioned)
                query.append({"role": "user", "content": f"Your response cannot contain any part of Referential Context. You cannot mention \"{bad_words_string}\""})
                print(f"retrying due to bad words: {bad_words_string}, msg: {gen_msg}")
                return self.query(query, attempt=attempt+1)
        return gen_msg

    def _gpt_query(self, query, times_retried=0):
        if times_retried > self.retry_limit:
            raise Exception("retry failed")

        system_msg = (
            self.spkr_system_msg
            if self.model_args.role == "spkr"
            else self.lsnr_system_msg
        )
        messages = [system_msg] + query

        try:
            response = self.client.chat.completions.create(
                model=self.model_args.model_ckpt,
                messages=messages,
                seed=42,
                max_tokens=self.model_args.max_output_tokens,
                temperature=0,
                timeout=60,
            )

        except APIError as e:
            print(e)
            print(f"retrying in {self.retry_after_seconds} seconds")
            time.sleep(self.retry_after_seconds)
            times_retried += 1
            return self._gpt_query(query, times_retried=times_retried)

        except RateLimitError as e:
            print(
                f"Rate limit exceeded. Waiting and retrying in {self.retry_after_seconds} seconds..."
            )
            time.sleep(self.retry_after_seconds)
            times_retried += 1
            return self._gpt_query(query, times_retried=times_retried)
        except APITimeoutError as e:
            print(e)
            print(
                f"Waiting and retrying in {self.retry_after_seconds} seconds...", file=sys.stderr
            )
            time.sleep(self.retry_after_seconds)
            times_retried += 1
            return self._gpt_query(query, times_retried=times_retried)

        except Exception as e:
            print(e)
            print(f"retrying in {self.retry_after_seconds} seconds...", file=sys.stderr)
            time.sleep(self.retry_after_seconds)
            times_retried += 1
            return self._gpt_query(query, times_retried=times_retried)

        return response

    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_pred_formatted = {"role": "assistant", "content": spkr_pred}
        spkr_trial_prompt.append(spkr_pred_formatted)
        return spkr_trial_prompt

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        assistant_pred = {"role": "assistant", "content": lsnr_pred}
        lsnr_trial_prompt.append(assistant_pred)
        return lsnr_trial_prompt

    def _get_spkr_feedback(self, pred_name, spkr_tgt_referent, spkr_trial_referents):
        if pred_name == "invalid":
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The listener didn't give a valid answer.",
                    }
                ],
            }
        else:
            if isinstance(spkr_trial_referents[0], str):
                pred_label = pred_name
                if pred_name == spkr_tgt_referent:
                    feedback = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "The listener answered correctly.",
                            }
                        ],
                    }
                else:
                    feedback = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The listener mistakenly answered \"{pred_label}\".",
                            }
                        ],
                    }
            else:
                for i in range(4):
                    if spkr_trial_referents[i]["filename"] == pred_name:
                        pred_label = self.model_args.label_space[i]
                        break

                if pred_name == spkr_tgt_referent["filename"]:
                    feedback = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The listener correctly answered Image {pred_label}.",
                            }
                        ],
                    }
                else:
                    feedback = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The listener mistakenly answered Image {pred_label}.",
                            }
                        ],
                    }

        return feedback

    def _get_lsnr_feedback(self, pred, target_label, spkr_msg=None):
        if pred == target_label:
            feedback = {
                "role": "user",
                "content": [{"type": "text", "text": "Correct."}],
            }
        elif pred not in self.model_args.label_space:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Invalid answer. Answer must be one of {self.model_args.label_space}.",
                    }
                ],
            }
        else:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Wrong, I'm referring to {target_label}.", 
                    }
                ],
            }

        return feedback

    def _model_specific_prompt_postprocessing(self, prompt):
        if self.enforce_alternating_roles:
            prompt = copy.deepcopy(prompt)
            formatted_prompt = [prompt[0]]
            for i in range(1, len(prompt)):
                if prompt[i]["role"] == formatted_prompt[-1]["role"]:
                    if isinstance(prompt[i]["content"], str):
                        prompt[i]["content"] = "\n" + prompt[i]["content"]
                    else:
                        prompt[i]["content"][0]["text"] = (
                            "\n" + prompt[i]["content"][0]["text"]
                        )
                    formatted_prompt[-1]["content"] = (
                        formatted_prompt[-1]["content"] + prompt[i]["content"]
                    )
                else:
                    formatted_prompt.append(prompt[i])

            return formatted_prompt
        return prompt


class ClaudeModel(ModelWrapper):
    def __init__(self, model_args, api_key):
        super().__init__(model_args)
        self.client = Anthropic(api_key=api_key)

        self.lsnr_system_msg = "You are an assistant who will play a series of reference games with the user. You will pay close attention to the conversation history as more rounds are played."
        self.bad_words=[]

    def get_spkr_intro(self, context_referents):
        intro_text = self.model_args.intro_text
        label_space = self.model_args.label_space

        if isinstance(context_referents[0], str):
            intro = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": intro_text},
                        {"type": "text", "text": f"\nReferential Context: \"{context_referents[0]}\", \"{context_referents[1]}\", \"{context_referents[2]}\", \"{context_referents[3]}\"\n"},
                    ],
                }
            ]
            

        else:
            mode = self.model_args.img_mode
            intro = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": intro_text},
                        {"type": "text", "text": f"Image {label_space[0]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": context_referents[0][mode],
                            },
                        },
                        {"type": "text", "text": f"Image {label_space[1]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": context_referents[1][mode],
                            },
                        },
                        {"type": "text", "text": f"Image {label_space[2]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": context_referents[2][mode],
                            },
                        },
                        {"type": "text", "text": f"Image {label_space[3]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": context_referents[3][mode],
                            },
                        },
                    ],
                }
            ]

        return intro

    def _get_spkr_prompt(self, round_name, target_label):
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{round_name}, the target is Image {target_label}.",
                    }
                ],
            }
        ]

        return prompt


    def _get_spkr_prompt_text_game(self, round_name, target_label, t):

        if t==0:
            prompt_content=f"Target: \"{target_label}\""
        else:
            prompt_content = f"\nNext target: \"{target_label}\""
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_content,
                    }
                ]
            }
        ]

        return prompt

    def get_lsnr_intro(self):
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": self.model_args.intro_text}],
            }
        ]

    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        label_space = self.model_args.label_space
        mode = self.model_args.img_mode
        if omit_img:
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{round_name}, "},
                        {
                            "type": "text",
                            "text": f"which image is this message referring to: {msg}?",
                        },
                    ],
                },
                {"role": "assistant", "content": "Image"},
            ]

        else:
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{round_name}, "},
                        {"type": "text", "text": f"Image {label_space[0]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": trial_imgs[0][mode],
                            },
                        },
                        {"type": "text", "text": f"Image {label_space[1]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": trial_imgs[1][mode],
                            },
                        },
                        {"type": "text", "text": f"Image {label_space[2]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": trial_imgs[2][mode],
                            },
                        },
                        {"type": "text", "text": f"Image {label_space[3]}:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": trial_imgs[3][mode],
                            },
                        },
                        {
                            "type": "text",
                            "text": f"which image is this message referring to: {msg}?",
                        },
                    ],
                },
                {"role": "assistant", "content": "Image"},
            ]

        return prompt

    def query(self, query, attempt=0):
        if attempt>3:
            raise Exception("retry limit exceeded")
        if self.model_args.role == "lsnr":
            response = self.client.messages.create(
                model=self.model_args.model_ckpt,
                max_tokens=self.model_args.max_output_tokens,
                messages=query,
                temperature=0,
                top_k=1,
                system=self.lsnr_system_msg,
            )
        else:
            try:
                response = self.client.messages.create(
                    model=self.model_args.model_ckpt,
                    max_tokens=self.model_args.max_output_tokens,
                    messages=query,
                    temperature=0,
                    system="You're a helpful assistant.",
                    top_k=1,
                )
            except Exception as e:
                print(f"Error: {e}, trying again...")
                time.sleep(60)
                return self.query(query, attempt=attempt+1)
            bad_words_mentioned=[]
            for bad_word in self.bad_words:
                if bad_word.lower() in response.content[0].text.lower():
                    if attempt>4:
                        return "Invalid message"
                    bad_words_mentioned.append(bad_word)
            if len(bad_words_mentioned)>0:
                query.append({"role": "assistant", "content": response.content[0].text})
                bad_words_string=", ".join(bad_words_mentioned)
                query.append({"role": "user", "content": f"Your response cannot contain any part of Referential Context. You cannot mention \"{bad_words_string}\". Respond again with a valid message. Provide the message only and nothing else."})
                print(f"retrying due to bad words: {bad_words_string}, msg: {response.content[0].text}")
                return self.query(query, attempt=attempt+1)

        return response.content[0].text

    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_pred_formatted = {
            "role": "assistant",
            "content": [{"type": "text", "text": spkr_pred}],
        }
        spkr_trial_prompt.append(spkr_pred_formatted)
        return spkr_trial_prompt

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        lsnr_trial_prompt[-1]["content"] = (
            lsnr_trial_prompt[-1]["content"] + " " + lsnr_pred
        )
        return lsnr_trial_prompt

    def _get_spkr_feedback(self, pred_name, spkr_tgt_referent, spkr_trial_referents):
        if pred_name == "invalid":
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The listener didn't give a valid answer.",
                    }
                ],
            }
        else:
            if isinstance(spkr_trial_referents[0], str):
                pred_label = pred_name
                if pred_name == spkr_tgt_referent:
                    feedback = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "The listener answered correctly.",
                            }
                        ],
                    }
                else:
                    feedback = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The listener mistakenly answered \"{pred_label}\".",
                            }
                        ],
                    }


            else:
                for i in range(4):
                    if spkr_trial_referents[i]["filename"] == pred_name:
                        pred_label = self.model_args.label_space[i]
                        break

                if pred_name == spkr_tgt_referent["filename"]:
                    feedback = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The listener correctly answered Image {pred_label}.",
                            }
                        ],
                    }
                else:
                    feedback = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The listener mistakenly answered Image {pred_label}.",
                            }
                        ],
                    }

        return feedback

    def _get_lsnr_feedback(self, pred, target_label, spkr_msg=None):
        if pred == target_label:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Correct, I'm referring to Image {target_label}.",
                    }
                ],
            }
        elif pred not in self.model_args.label_space:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Invalid answer. Answer must be one of {self.model_args.label_space}.",
                    }
                ],
            }
        else:
            feedback = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Wrong, I'm referring to Image {target_label}.",
                    }
                ],
            }

        return feedback

    def _model_specific_prompt_postprocessing(self, prompt):
        prompt = copy.deepcopy(prompt)
        formatted_prompt = [prompt[0]]
        for i in range(1, len(prompt)):
            if prompt[i]["role"] == formatted_prompt[-1]["role"]:
                prompt[i]["content"][0]["text"] = "\n" + prompt[i]["content"][0]["text"]
                formatted_prompt[-1]["content"] = (
                    formatted_prompt[-1]["content"] + prompt[i]["content"]
                )
            else:
                formatted_prompt.append(prompt[i])

        return formatted_prompt



class LlamaModel(ModelWrapper):
    def __init__(self, model_args, isLora=False, loaded_model=None):
        super().__init__(model_args)
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM
        )


        
        self.isLora=isLora

        self.system_msg="You're a helpful assistant."
        self.new_special_tokens=[]
        self.marker_token_id=None
        self.device = "cuda" 

        if model_args.model_ckpt in ["meta-llama/Llama-3.1-8B-Instruct"]:
            self.isLora=False
        if model_args.model_ckpt and self.isLora:
            with open(os.path.join(model_args.model_ckpt, "adapter_config.json"), "r") as f:
                adapter_config=json.load(f)
                model_name_or_path=adapter_config["base_model_name_or_path"]
        else:
            model_name_or_path=model_args.model_ckpt


        print("base model before any adapters: ", model_name_or_path, file=sys.stderr)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_ckpt)

        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        self.terminators = [self.tokenizer.eos_token_id]
        self.max_output_tokens = model_args.max_output_tokens
        self.enforce_alternating_roles = True


        if model_args.model_ckpt:

            if self.isLora:
                self.model=PeftModel.from_pretrained(self.model, model_args.model_ckpt, adapter_name="trained_adapter")
                print("loaded adapter from ", model_args.model_ckpt, file=sys.stderr)
        
        else:
            print("="*60, file=sys.stderr)
            print("no trained model loaded, using base model", file=sys.stderr)
            print("="*60, file=sys.stderr)


        self.referential_context=None
        self.model.eval()
        self.bad_words_ids = []  
        self.bad_words = []
        self.label_tokens=[]
   
        self.additional_forbidden_tokens=[]
    

    def update_bad_words(self, ad_hoc_bad_words_ids, word):
        ids_with_space=self.tokenizer.encode(word, add_special_tokens=False)
        if len(ids_with_space)==1:
            self.additional_forbidden_tokens.append(ids_with_space[0])
        else:
            #if not a single token, then we need to add multiple tokens, some of which may be valid in other message, so we only add them as ad_hoc bad words, which would reset after this query.
            ad_hoc_bad_words_ids.update(ids_with_space)

    def query(self, query, attempt=0, ad_hoc_bad_word_ids=[]):
        if attempt>9:
            return "Invalid message, failed retries"
        msg = self._query(query, ad_hoc_bad_word_ids)
        if self.model_args.role == "spkr":
            ad_hoc_bad_words=[]
            ad_hoc_bad_word_ids=set()
            for bad_word in self.bad_words:
                start_idx=msg.lower().find(bad_word.lower())
                if start_idx!=-1:
                    end_idx=start_idx+len(bad_word)
       

                    left_sep=max(msg.rfind(" ", 0, start_idx),0)
                    if left_sep>=len(msg):
                        raise ValueError("msg too short, left_sep out of range")
                    right_sep=min(msg.find(" ", end_idx), len(msg))
                    enclosing_word=msg[left_sep:right_sep]
                    ad_hoc_bad_words.append(enclosing_word)
                    self.update_bad_words(ad_hoc_bad_word_ids, enclosing_word)
                    
            if len(ad_hoc_bad_words)>0:
                bad_words_string=", ".join(ad_hoc_bad_words)
                print(f"retrying due to bad words: {bad_words_string}, msg: {msg}")
                return self.query(query, attempt=attempt+1, ad_hoc_bad_word_ids=list(ad_hoc_bad_word_ids))
        
        return msg


    def _query(self, query, ad_hoc_bad_word_ids=[]):
        query_copy=copy.deepcopy(query)

        query_tokenized=self.tokenizer.apply_chat_template(query_copy, add_generation_prompt=True, return_tensors="pt").to(self.device)

 


        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids(".")
        ]

    
        additional_bad_words=[]


        suppress_tokens= [token for token in self.additional_forbidden_tokens if token is not None]
    
        suppress_tokens.extend(ad_hoc_bad_word_ids)
        outputs = self.model.generate(
                **query_tokenized,
                max_new_tokens=32,
                eos_token_id=terminators,  
                do_sample=False,
                top_p=None,
                temperature=None,
                pad_token_id=self.tokenizer.eos_token_id,
                begin_suppress_tokens=terminators,
                suppress_tokens=suppress_tokens,
                bad_words_ids=self.bad_words_ids+additional_bad_words+self.label_tokens,
                
            )


        response_simple = outputs[0][query_tokenized.input_ids.shape[-1]:]
        response_text=self.tokenizer.decode(response_simple, skip_special_tokens=False)

        return re.sub(r"\<\|start_header_id\|\>|assistant|\<\|end_header_id\|\>|\<\|eot_id\|\>", "", response_text).replace("\n", " ").strip()

    def get_spkr_intro(self, context_referents):
        intro_text = self.model_args.intro_text
     
        introduce_contexts=f"\nReferential Context: \"{context_referents[0]}\", \"{context_referents[1]}\", \"{context_referents[2]}\", \"{context_referents[3]}\"\n"



        if isinstance(context_referents[0], str):
   
            intro = [
                {"role": "system", "content": self.system_msg},
                {"role": "user", "content": intro_text+"\n"+introduce_contexts},
            ]
        
          
        else:
            raise NotImplementedError("llama is text only model and does not support image inputs")

        return intro 
    
    def _get_spkr_prompt(self, round_name, target_label):
        raise NotImplementedError("llama is text only model and does not support image inputs")

    def _get_spkr_prompt_text_game(self, round_name, target_label, t):

        if t==0:
            prompt = [
                {"role": "user", "content": f"Target: \"{target_label}\""}
            ]
        else:
            prompt=[{"role": "user", "content": f"Next target: \"{target_label}\""}]



        return prompt
    
    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        raise NotImplementedError("llama is text only model and does not support image inputs")
    

    def get_lsnr_intro(self):
        raise NotImplementedError

    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        raise NotImplementedError
    
    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_pred_formatted = {"role": "assistant", "content": spkr_pred}
        spkr_trial_prompt.append(spkr_pred_formatted)
        return spkr_trial_prompt

    def _get_spkr_feedback(self, pred_name, spkr_tgt_referent, spkr_trial_referents):
        if pred_name == "invalid":
            feedback = {
                "role": "user",
                "content": "The listener didn't give a valid answer.",
            }
        
        else:
            if isinstance(spkr_trial_referents[0], str):

                pred_label=pred_name
        
                if pred_name == spkr_tgt_referent:

                    feedback = {
                        "role": "user",
                        "content": "The listener answered correctly.",
                    }
                else:
                    feedback = {
                        "role": "user",
                        "content": f"The listener mistakenly answered \"{pred_label}\".",
                    }
            else:
                raise NotImplementedError("llama is text only model and does not support image inputs")

        return feedback
    

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        raise NotImplementedError
    def _get_lsnr_feedback(self, pred, target_label, spkr_msg):
        raise NotImplementedError
    
    def _model_specific_prompt_postprocessing(self, prompt):


        if self.enforce_alternating_roles:
            prompt = copy.deepcopy(prompt)
            formatted_prompt = [prompt[0]]
            for i in range(1, len(prompt)):
                if prompt[i]["role"] == formatted_prompt[-1]["role"]:
                    prompt[i]["content"] = (
                        "\n" + prompt[i]["content"]
                    )
                    formatted_prompt[-1]["content"] = (
                        formatted_prompt[-1]["content"] + prompt[i]["content"]
                    )
                else:
                    formatted_prompt.append(prompt[i])

            return formatted_prompt
        else:
            return prompt
    

class GemmaModel(ModelWrapper):
    def __init__(self, model_args, isLora=False):
        super().__init__(model_args)
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM
        )



        self.system_msg="You are a helpful assistant."
        self.new_special_tokens=[]
        self.marker_token_id=None
        self.device = "cuda" 
        self.isLora=isLora


        if model_args.model_ckpt=="google/gemma-2-9b-it":
            self.isLora=False
        if model_args.model_ckpt:
            if self.isLora:
                with open(os.path.join(model_args.model_ckpt, "adapter_config.json"), "r") as f:
                    adapter_config=json.load(f)
                    model_name_or_path=adapter_config["base_model_name_or_path"]
            else:
                model_name_or_path=model_args.model_ckpt
        else:
            model_name_or_path="google/gemma-2-9b-it"

        print("base model before any adapters: ", model_name_or_path, file=sys.stderr)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map=self.device)

        if model_args.model_ckpt:
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_ckpt)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

   
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        self.terminators = [self.tokenizer.eos_token_id]
        self.max_output_tokens = model_args.max_output_tokens
        self.enforce_alternating_roles = True
     
        

        if model_args.model_ckpt:
            if self.isLora:
                self.model=PeftModel.from_pretrained(self.model, model_args.model_ckpt, adapter_name="trained_adapter")

        else:
            print("="*60, file=sys.stderr)
            print("no trained model loaded, using base model", file=sys.stderr)
            print("="*60, file=sys.stderr)


        self.referential_context=None
        self.model.eval()
        self.bad_words = []
        self.bad_words_ids = []
        self.additional_forbidden_tokens=[]

        self.label_tokens=[]

    def update_bad_words(self, ad_hoc_bad_words_ids, word):

        ids_with_space=self.tokenizer.encode(word, add_special_tokens=False)

        if len(ids_with_space)==1:
            self.additional_forbidden_tokens.append(ids_with_space[0])
        else:
            ad_hoc_bad_words_ids.update(ids_with_space)


    def query(self, query, attempt=0, ad_hoc_bad_word_ids=[]):
        if attempt>9:
            return "Invalid message, failed retries"
        msg = self._query(query, ad_hoc_bad_word_ids)
        if self.model_args.role == "spkr":
            ad_hoc_bad_words=[]
            ad_hoc_bad_word_ids=set()
            for bad_word in self.bad_words:
                start_idx=msg.lower().find(bad_word.lower())
                if start_idx!=-1:
                    end_idx=start_idx+len(bad_word)

                    left_sep=max(msg.rfind(" ", 0, start_idx),0)
                    if left_sep>=len(msg):
                        raise ValueError("msg too short, left_sep out of range")
                    right_sep=min(msg.find(" ", end_idx), len(msg))
                    enclosing_word=msg[left_sep:right_sep]
                    ad_hoc_bad_words.append(enclosing_word)
                    self.update_bad_words(ad_hoc_bad_word_ids, enclosing_word)
                    
            if len(ad_hoc_bad_words)>0:
                bad_words_string=", ".join(ad_hoc_bad_words)
                print(f"retrying due to bad words: {bad_words_string}, msg: {msg}")
                return self.query(query, attempt=attempt+1, ad_hoc_bad_word_ids=list(ad_hoc_bad_word_ids))
        
        return msg



    def _query(self, query, ad_hoc_bad_word_ids=[]):
        query_copy=copy.deepcopy(query)

    
   
        query_tokenized=self.tokenizer.apply_chat_template(query_copy, add_generation_prompt=True, return_tensors="pt").to(self.device)



        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("."),
        
        ]

        #gemma sometimes generates multiple lines and separate the referring expression into multiple lines. Suppress newlines or anything the model may use in place of a newline.
        suppress_tokens=[
            self.tokenizer.convert_tokens_to_ids("\n"),
            self.tokenizer.convert_tokens_to_ids("\n\n"),
            self.tokenizer.convert_tokens_to_ids("\n\n\n"),
            self.tokenizer.convert_tokens_to_ids("\n\n\n\n"),
            self.tokenizer.convert_tokens_to_ids("\n\n\n\n\n"),
            self.tokenizer.convert_tokens_to_ids('▁**'),
            self.tokenizer.convert_tokens_to_ids('▁▁'),
            self.tokenizer.convert_tokens_to_ids('...'),
            self.tokenizer.convert_tokens_to_ids('**'),
            self.tokenizer.convert_tokens_to_ids("*."),
            self.tokenizer.convert_tokens_to_ids("</td>")
            ]
        

        for token in self.additional_forbidden_tokens:
            if token is not None:
                suppress_tokens.append(token)

        suppress_tokens.extend(ad_hoc_bad_word_ids)

     

        self.model.eval()


        begin_suppress_ids=terminators
        bad_words_ids=self.bad_words_ids+self.label_tokens


        outputs = self.model.generate(
            **query_tokenized,
            max_new_tokens=32,
            eos_token_id=terminators,  
            do_sample=False,
            top_p=None,
            temperature=None,
            pad_token_id=self.tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids,
            begin_suppress_tokens=begin_suppress_ids,
            suppress_tokens=suppress_tokens,
        )


        response_simple = outputs[0][query_tokenized.input_ids.shape[-1]:]
        response_text=self.tokenizer.decode(response_simple, skip_special_tokens=False)
      
        response_text=response_text.replace("<end_of_turn>", "").replace("<eos>", "")
        response_text=re.sub(r'\n+', ' ', response_text)

        return response_text.strip()
        


    def get_spkr_intro(self, context_referents):
        intro_text = self.model_args.intro_text
       
        introduce_contexts=f"\nReferential Context: \"{context_referents[0]}\", \"{context_referents[1]}\", \"{context_referents[2]}\", \"{context_referents[3]}\""


        if isinstance(context_referents[0], str):

           
            intro = [
                {"role": "user", "content": intro_text+introduce_contexts},
            ]
        
        else:
            raise NotImplementedError("llama is text only model and does not support image inputs")

        return intro 
    
    def _get_spkr_prompt(self, round_name, target_label):
        raise NotImplementedError("llama is text only model and does not support image inputs")

    def _get_spkr_prompt_text_game(self, round_name, target_label, t):
     

      

        if t==0:
            prompt = [
                {"role": "user", "content": f"\nTarget: \"{target_label}\""}
            ]
        else:
            prompt = [
                {"role": "user", "content": f"\nNext target: \"{target_label}\""}
            ]
       

        

        return prompt
    
    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        raise NotImplementedError("Gemma is text only model and does not support image inputs")
    

    def get_lsnr_intro(self):
        raise NotImplementedError

    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        raise NotImplementedError
    
    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_pred_formatted = {"role": "model", "content": spkr_pred}
        spkr_trial_prompt.append(spkr_pred_formatted)
        return spkr_trial_prompt

    def _get_spkr_feedback(self, pred_name, spkr_tgt_referent, spkr_trial_referents):
        if pred_name == "invalid":
            feedback = {
                "role": "user",
                "content": "The listener didn't give a valid answer.",
            }
        
        else:
            if isinstance(spkr_trial_referents[0], str):
               
                pred_label=pred_name

             

                if pred_name == spkr_tgt_referent:
                    feedback = {
                        "role": "user",
                        "content": "The listener answered correctly.",
                    }
                else:
                    feedback = {
                        "role": "user",
                        "content": f"The listener mistakenly answered {pred_label}.",
                    }
            else:
                raise NotImplementedError("Gemma is text only model and does not support image inputs")

        return feedback
    

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        raise NotImplementedError
    def _get_lsnr_feedback(self, pred, target_label, spkr_msg):
        raise NotImplementedError
    
    def _model_specific_prompt_postprocessing(self, prompt):


        if self.enforce_alternating_roles:
            prompt = copy.deepcopy(prompt)
            formatted_prompt = [prompt[0]]
            for i in range(1, len(prompt)):
                if prompt[i]["role"] == formatted_prompt[-1]["role"]:
                    prompt[i]["content"] = (
                        "\n" + prompt[i]["content"]
                    )
                    formatted_prompt[-1]["content"] = (
                        formatted_prompt[-1]["content"] + prompt[i]["content"]
                    )
                else:
                    formatted_prompt.append(prompt[i])

            return formatted_prompt
        else:
            return prompt
    


