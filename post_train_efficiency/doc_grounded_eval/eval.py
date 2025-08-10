
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json, os, torch, glob
from tqdm import tqdm
from peft import PeftModel
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import concurrent.futures
import argparse
from datetime import datetime
import copy
import spacy
nlp = spacy.load("en_core_web_sm")
import re
import pickle
import pandas as pd
import random
import numpy as np
import math
from anthropic import Anthropic
with open("00_APIs/anthropic_api.json") as f:
    API_key=json.load(f)['api_key']
claude_client=Anthropic(api_key=API_key)
claude_model_ckpt="claude-3-5-sonnet-20240620"


import openai

from openai import OpenAI
with open('00_APIs/openai_api.json') as f:
    api_info=json.load(f)

openai.api_key=api_info['api_key']
openai.organization=api_info['organization']
openai_client = OpenAI(api_key=openai.api_key, organization=openai.organization)
gpt_model_ckpt="gpt-4o-2024-08-06"

def remove_possessives_determiners(phrase):
        words_to_remove={"the", "a", "an", "my", "your", "his", "her", "its", "their", "our", "that", "this", "these", "those"}
        phrase_words=phrase.split()
        if phrase_words[0] in words_to_remove:
            if len(phrase_words)>1:
                phrase_words=phrase_words[1:]
            
        
        return " ".join(phrase_words)

def find_substring_occurences(substring, long_string):
    substring_len=len(substring)
    occurrences=[(i, i+substring_len) for i in range(len(long_string)) if long_string.startswith(substring, i)] 
    return occurrences


def query_claude(instance):
    try:
        response = claude_client.messages.create(
                model=claude_model_ckpt,
                max_tokens=128,
                system=instance['API_query'][0]['content'],
                messages=instance['API_query'][1:],
                temperature=0,
                top_k=1
                )
        
        instance=copy.deepcopy(instance)
        if len(response.content[0].text.strip())==0:
            instance['claude_output']="No Output"
            instance['error_message']=response
            print(response)
        else:
            instance['claude_output']=response.content[0].text
    except Exception as e:
        instance['claude_output']="No Output"
        instance['error_message']=str(e)

    return instance


def query_gpt(instance):
    try:
        response = openai_client.chat.completions.create(
                model=gpt_model_ckpt,
                messages=instance["API_query"],
                seed=42,
                max_tokens=128,
                temperature=0,
                timeout = 60,
            )
    
        response_text=response.choices[0].message.content
        instance=copy.deepcopy(instance)
        if len(response_text.strip())==0:
            instance['gpt_output']="No Output"
            instance['error_message']=response
            print(response)
        else:
            instance['gpt_output']=response_text
    except Exception as e:
        instance['gpt_output']="No Output"
        instance['error_message']=str(e)

    return instance


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, default="") 
        parser.add_argument("--model_type", type=str, default="llama")
        parser.add_argument("--isLora", action='store_true')
        parser.add_argument("--debug", action='store_true')
        parser.add_argument("--added_marker", type=str, default=None)
        parser.add_argument("--test_data_fp", type=str)
        args = parser.parse_args()
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model_type in ["gemma", "llama"]:
            if args.isLora:
                with open(os.path.join(args.model_path, "adapter_config.json"), "r") as f:
                    adapter_config=json.load(f)
                    model_name=adapter_config["base_model_name_or_path"]
                print("base model: ", model_name)
            else:
                model_name=args.model_path
        

            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='cuda')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if args.isLora:
                model = PeftModel.from_pretrained(model, args.model_path, adapter_name="trained_adapter")
                model.set_adapter("trained_adapter")


 
        source_fps="corpora/doc2dial/doc2dial_dial_*.json"

        reference_fp="corpora/doc2dial/doc2dial_doc.json"
        da_rules="corpora/doc2dial/assistant_da_annotation_rules.json"



        source_fps=glob.glob(source_fps)
        source_data=[] 
        for source_fp in source_fps:
            with open(source_fp) as f:
                source_data.append(json.load(f))

        with open(reference_fp) as f:
            reference_data=json.load(f)

        with open(da_rules) as f:
            da_rules=json.load(f)

        all_convos={}
        ref_docs={}

        for data_split in source_data:
            for domain in data_split['dial_data']:
                for doc_id, convos in data_split['dial_data'][domain].items():
                    for convo in convos:
                        all_convos[convo['dial_id']]=convo
                
        for domain in reference_data['doc_data']:
            for doc_id, doc in reference_data['doc_data'][domain].items():
                ref_docs[doc_id]=doc







        partial_completion_given=True

        # model_type=args.model_type+args.model_path.replace("trl_out/", "_").replace("/checkpoint", "")
        model_type=args.model_type

        prompt_style="chat_template"
        filter_style="dynamic"
        context_style="full"
        skip_user_eval=True
        response_type="model_output"
        save_name=model_type
        # %%
        with open(args.test_data_fp) as f:
            coref_data=json.load(f)

        if args.debug:
            coref_data=coref_data[0:5]
            

        if "gpt" in model_type:
            additional_instruction="\n\nDo not introduce new information that's not supported by the existing utterances or the reference texts. Make your utterance sound natural. Completing the agent's last utterance from where it's left off. Do not generate any utterance for the user."
        else:
            additional_instruction="\n\nDo not introduce new information that's not supported by the existing utterances or the reference texts. Make your utterance sound natural. Generate the agent's next utterance only. Do not generate any utterance for the user."



        evaled_data=[]
        example_added=False
        filtered_due_to_user_turn=[]
        filtered_due_to_ref_span=[]
        easy_ref_instances=[]
        processed_for_eval=[]

        prepped_data=[]

        added_marker=tokenizer.convert_tokens_to_ids(args.added_marker)
     







        if "gemma" in model_type:
            terminators = [
                tokenizer.convert_tokens_to_ids("<eos>"),
                tokenizer.convert_tokens_to_ids("."),
                tokenizer.convert_tokens_to_ids("?"),
                tokenizer.convert_tokens_to_ids("!"),
                tokenizer.convert_tokens_to_ids("<end_of_turn>"),
            ]
             #gemma sometimes generates multiple lines and separate the referring expression into multiple lines. Suppress newlines or anything the model may use in place of a newline.
            suppress_tokens = [       
                tokenizer.convert_tokens_to_ids("\n"),
                tokenizer.convert_tokens_to_ids("\n\n"),
                tokenizer.convert_tokens_to_ids("\n\n\n"),
                tokenizer.convert_tokens_to_ids("\n\n\n\n"),
                tokenizer.convert_tokens_to_ids("\n\n\n\n\n"),
                tokenizer.convert_tokens_to_ids("▁▁"),
                tokenizer.convert_tokens_to_ids("▁▁▁"),
                tokenizer.convert_tokens_to_ids("▁"),
                tokenizer.convert_tokens_to_ids("▁▁▁▁"),
                tokenizer.convert_tokens_to_ids("▁▁▁▁▁"),
                tokenizer.convert_tokens_to_ids("\n\n\n\n\n\n"),
                954, 143, 226, 2722, 144, 2692,114, 148, 38104, 115,146, 147, 145, 184, 149, 150, 152, 44416]


                    
        elif "llama" in model_type:
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            suppress_tokens = []


        if added_marker:
            bad_words_ids=[[added_marker, terminator] for terminator in terminators]
        else:
            bad_words_ids=None



        def enforce_alternate_roles(input_prompt):
            prompt = copy.deepcopy(input_prompt)
            prompt[0]["role"] = "user"
            prompt[0]["content"] +="[conversation begins]:\nuser:"
            prompt[0]["content"] +=prompt[1]["content"]
            formatted_prompt = [prompt[0]]
            for i in range(2, len(prompt)):
                if prompt[i]["role"] == formatted_prompt[-1]["role"]:
                    prompt[i]["content"] = (
                        "\n" + prompt[i]["content"]
                    )
                    formatted_prompt[-1]["content"] = (
                        formatted_prompt[-1]["content"] + prompt[i]["content"]
                    )
                    #strip away newlines and whitespace
                    prompt[i]["content"] = re.sub(r"^\s+|\s+$", "", prompt[i]["content"])
                else:
                    formatted_prompt.append(prompt[i])

            return formatted_prompt

        def enforce_claude_alternating_roles(input_prompt):
            prompt = copy.deepcopy(input_prompt)
            formatted_prompt = [prompt[0]]
            for i in range(1, len(prompt)):
                if prompt[i]["role"] == formatted_prompt[-1]["role"]:
                    prompt[i]["content"] = "\n" + prompt[i]["content"]
                    formatted_prompt[-1]["content"] = (
                        formatted_prompt[-1]["content"] + prompt[i]["content"]
                    )
                else:
                    formatted_prompt.append(prompt[i])

            return formatted_prompt


        for idx, instance in tqdm(enumerate(coref_data)):
            next_turn_speaker=instance['last_utt_pre'].split(':')[0]
   

            if skip_user_eval and next_turn_speaker=="user":
                filtered_due_to_user_turn.append(instance)
                continue
            dial_id=instance['doc_key'].split('_')[0]
            convo=all_convos[dial_id]
            domain_name=convo['domain']
            id_turn_to_complete=len(instance['context_till_last_utt']) 
            turn_metadata=convo['turns'][id_turn_to_complete]
            ref_sents_keys=turn_metadata['references']
            ref_doc=ref_docs[convo['doc_id']]

            ref_sec_sents=[]
            ref_span_sents=[]
            for sent_key in ref_sents_keys:
                if len(ref_sec_sents)==0 or ref_sec_sents[-1]!=ref_doc['spans'][sent_key['sp_id']]['text_sec']:
                    ref_sec_sents.append(ref_doc['spans'][sent_key['sp_id']]['text_sec'])
                ref_span_sents.append(ref_doc['spans'][sent_key['sp_id']]['text_sp'])
            ref_span="".join(ref_span_sents).strip("\n ")
            instance['ref_span']=ref_span
            ref_doc=ref_doc['doc_text'].strip("\n ")

 
            skip_this_instance=False
            try:
                mention1_base=remove_possessives_determiners(instance['mention1'])
                mention2_base=remove_possessives_determiners(instance['mention2']) 
            except:
                mention1_base=instance['mention1']
                mention2_base=instance['mention2']

            
            mention1_occurs=find_substring_occurences(mention1_base, ref_span)
            mention2_occurs=find_substring_occurences(mention2_base, ref_span)
            if len(mention2_occurs)>0:
                easy_ref_instances.append(instance)
                
                for start, end in mention2_occurs:
                    contained_in_mention1=False
                    for start1, end1 in mention1_occurs:
                        if start1<=start and end1>=end:
                            contained_in_mention1=True
                            break
                    if contained_in_mention1:
                        continue
                    else:
                        skip_this_instance=True
                        break
            
            if skip_this_instance:
                filtered_due_to_ref_span.append(instance)
                continue


                   
            generation_prefix=instance['last_utt_pre']
            DA_based_instruction=da_rules[turn_metadata['da']]['description']
                
            system_msg=f"Imagine a conversation between a user and a {domain_name} customer service center agent. You will act as the agent. Complete the last utterance of the dialogue. You are provided with a reference document, which may have relevant information.\n\n[reference document]:\n\"{ref_doc}\"\n\nYou are also given an excerpt of the reference document, i.e., the reference span. Your utterance should {DA_based_instruction}. Do not introduce new information that's not supported by the existing utterances or the reference texts. The user cannot see the reference texts, so do not directly refer to them. Limit your response to one sentence.\n\n[reference span]: \"{ref_span}\"{additional_instruction}"
            context_utterances=[]
            for line in instance['context_till_last_utt']:
                role, utterance=line.split(":", 1)
                role=role.replace("agent", "assistant")
                context_utterances.append({"role": role, "content": utterance.strip()})


            messages = [{"role": "system", "content": system_msg}]+context_utterances
         
            if "reft_old" in model_type:
                messages[-1]["content"]=messages[-1]["content"]+" [natural assistant]:"


            if "gemma" in model_type:
                messages=enforce_alternate_roles(messages)

            if "claude" in model_type or "gpt" in model_type:
                messages.append({"role": "assistant", "content": generation_prefix.replace("agent:", "").strip()})
                messages=enforce_claude_alternating_roles(messages)
                instance['reference_span']=ref_span
                instance['API_query']=messages
                prepped_data.append(instance)   
                continue

            prompt=tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            base_prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            base_prompt_len=base_prompt_tokenized["input_ids"].shape[-1]
            if partial_completion_given:
                # if 'last_utt_pre2' in instance:
                #     prompt+=instance['last_utt_pre2']
                # else:
                prompt+=generation_prefix.replace("agent:", "").strip()
            else:
                raise NotImplementedError("only works in the utter completion format")

                
            prompt_tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)





            processed_for_eval.append(instance)
            # continue
            outputs = model.generate(
                    **prompt_tokenized,
                    max_new_tokens=32,
                    eos_token_id=terminators,  
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                    begin_suppress_tokens=terminators,
                    bad_words_ids=bad_words_ids,
                    suppress_tokens=suppress_tokens,
                    output_scores=True,
                    return_dict_in_generate=True
                )
            response_simple = outputs[0][prompt_tokenized.input_ids.shape[-1]:]
            response_text=tokenizer.decode(response_simple, skip_special_tokens=True)
            # remove extra whitespace and new lines in the beginning and end of the string
            response_text=re.sub(r"^\s+|\s+$", "", response_text)
            instance['reference_span']=ref_span
            instance[response_type]=response_text
            if not example_added:
                instance[f'query_{model_type}']=prompt
                example_added=True

            evaled_data.append(instance)

            if len(evaled_data)<=10:
                print(instance["instance_id"])
                print(response_text, '\n')
                print("[concise_complete]: ", instance['completion_ori'], "\n[verbose_complete]: ", instance['completion_verbose'])
                print("="*50)


        save_fp=f"output_inference/dev_Dec/doc2dial_{timestamp}_{save_name}_.json"
        print(save_fp)
        print("#eval instances", len(evaled_data))


        with open(save_fp, "w") as f:   
            json.dump(evaled_data, f, indent=4)



        if "claude" in model_type or "gpt" in model_type:
            evaled_data=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_prompt = {executor.submit(query_claude, prepped_instance): prepped_instance for prepped_instance in prepped_data}
                
                for future in concurrent.futures.as_completed(future_to_prompt):
                    result = future.result()
                    evaled_data.append(result)



                fn=f"output_inference/dev_Dec/doc2dial_{save_name}.pkl"
                with open(fn, "wb") as f:
                    pickle.dump(evaled_data, f)
                print("saved API query results:", fn)



        def gpt_judge_query(model_ckpt, query):
            response = openai_client.chat.completions.create(
                model=model_ckpt,
                messages=query,
                seed=42,
                max_tokens=512,
                temperature=0,
                timeout = 60, 
                response_format={"type": "json_object"}

            )
            return response.choices[0].message.content

        def eval_inference(data_for_query, key_type, model_completion_name):
            instance=data_for_query['instance']
            is_shuffle=data_for_query['is_shuffle']
            order_dict=data_for_query['order_dict']
       

            context="\n".join(instance['context_till_last_utt']+[instance["last_utt_pre"]])
            model_completion=instance[model_completion_name].strip()
            model_completion.replace(instance["last_utt_pre"], "")
            model_completion=re.sub(r" \[re.*?\] ", " ", model_completion)
            if len(model_completion)==0 or model_completion=="No Output":
                print('invalid completion output, skipping')
                winner="invalid completion output"
                return {"instance_id": instance[key_type], 
                    "winner": winner,
                    "pred": None, 
                    "query": None, 
                    "context": instance['context_till_last_utt']}


            baseline_completion=instance['baseline_output'].strip()
            if is_shuffle==0:
                completion_A=model_completion
                completion_B=baseline_completion
            else:
                completion_A=baseline_completion
                completion_B=model_completion
          

            instruct_p1=f"""[context]:\n{context}\n\nCompletion A:\n{completion_A}\n\nCompletion B:\n{completion_B}\n\nAbove are two candidate completions for the agent's turn. Which completion shows stronger convention formation? Convention formation manifests when the referring expression for an item or a group of items becomes more concise after the item was initially mentioned. For example, an item initially mentioned as "the slim ceramic vase that has floral patterns" may later be re-mentioned as "the floral ceramic vase" or simply "the ceramic vase."\n\nConsider the items in the [context] that are re-mentioned in both completions. The completion that uses more concise phrases (shorter phrases) overall when re-mentioning those items is considered showing stronger convention formation. Remember that a re-mention is the exact noun phrase referring to the item, which includes all the words, phrases, and clauses modifying the noun. These modifiers will also affect the verbosity (conciseness) of the re-mention. Also, the same item may be re-mentioned multiple times in a completion and the two completions may eventually reach the same level of conciseness for that item. In this case, one completion may still show stronger convention formation by using the more concise referring expression more often.\n\nYour answer should follow one of the following formats:\n1) If one completion shows stronger convention formation overall, output its label (A or B) and an example (evidence) where an item is re-mentioned more concisely in that completion than in the other.\n2) If the completions show the same level of convention formation, output C and an example (evidence) where the re-mentions in the two completions have similar conciseness.\n3) If you cannot find any example (evidence) to make a judgment, output D.\n\nReturn a json of the following format: {{"initial mention in the [context]": "<the item in the [context] that's later re-mentioned in both completions>", "re-mention in Completion A": "<how Completion A refers to the initial mention; output the noun phrase or NA>", "re-mention in Completion B": "<how Completion B refers to the initial mention; output the noun phrase or NA>", "stronger convention formation in": "<one of A, B, C, D>"}}. Output NA only if you cannot find any example (evidence) to make a judgment."""
            query=[
                            {
                                "role":"user", 
                                "content": [
                                {"type": "text", "text": instruct_p1},
                                ]
                            }
                        ]
            
            pred_p1=gpt_judge_query(query=query, model_ckpt=gpt_model_ckpt)
            pred_p1=json.loads(pred_p1)

            pred_label=pred_p1["stronger convention formation in"]
            if pred_label not in {'A', 'B', 'C', 'D'}:
                print(f"invalid pred_label at {instance[key_type]}", pred_label)
                winner="invalid judge output"
            elif pred_label=='C':
                winner="tie"
            elif pred_label=='D':
                winner="cannot decide"
            elif order_dict[pred_label]=='baseline':
                winner='baseline'
            elif order_dict[pred_label]=='model':
                winner='model'
            else:
                raise NotImplementedError

            return {"instance_id": instance[key_type], 
                "evaluator_pred": pred_p1, 
                "query_p1": instruct_p1, 
                "mention1": instance['mention1'],
                "mention2": instance['mention2'],
                "Completion_A": completion_A,
                "Completion_B": completion_B,
                "order_dict": order_dict,
                "winner": winner,}


        with open(save_fp) as f:
            model1_data=json.load(f)
      
        model1_completion_name="model_output"

        with open(save_fp) as f:
            model2_data=json.load(f)
       
        model2_completion_name="completion_ori"
        test_size=len(model1_data)


      

        shuffler=[0]*test_size
        shuffler[::2]=[1]*(math.ceil(test_size/2))
        np.random.seed(42)
        shuffler=np.random.permutation(shuffler)
        order_dicts=[]
        for s in shuffler:
            if s==0:
                order_dicts.append({'A':'model', 'B':'baseline'})
            else:
                order_dicts.append({'A':'baseline', 'B':'model'})

        results=[]
        prepped_instances=[]
        # prepped_instances=[{"instance": instance, "is_shuffle": is_shuffle, "order_dict": order_dict} for instance, is_shuffle, order_dict in zip(model_data, shuffler, order_dicts)]

        for i, (model1_instance, model2_instance) in enumerate(zip(model1_data, model2_data)):
            assert(model1_instance['instance_id']==model2_instance['instance_id'])
            model1_instance['baseline_output']=model2_instance[model2_completion_name]
            prepped_instances.append({"instance": model1_instance, "is_shuffle": shuffler[i], "order_dict": order_dicts[i], "key_type": "instance_id"})


        # prepped_instances=prepped_instances[0:20]
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_prompt = {executor.submit(eval_inference, prepped_instance, key_type="instance_id", model_completion_name=model1_completion_name): prepped_instance for prepped_instance in prepped_instances}
            
            for future in concurrent.futures.as_completed(future_to_prompt):
                result = future.result()
                results.append(result)


        results=pd.DataFrame(results)


        results['winner'].value_counts()
        save_fp=f"output_inference/gpt_eval/00_judge_output_{timestamp}_{save_name}_{model2_completion_name}.csv"
        print("gpt judge result saved to: ", save_fp)
        results.to_csv(save_fp, index=False)


if __name__ == "__main__":
    main()