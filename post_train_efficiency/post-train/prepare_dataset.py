import logging
import os, sys
import json, re
from glob import glob
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from nltk.tokenize import word_tokenize 
from copy import deepcopy
import torch
import time
import spacy
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()


train_src="path_to_extracted_convenb/*_tokenized.json"
model_name="google/gemma-2-9b-it" 
tokenizer = AutoTokenizer.from_pretrained(model_name) #to calculate the tokenized length
max_token_size=768

src_concise_token="[re-mention natural]"
src_verbose_token="[re-mention verbose]"
new_concise_token="[remention]"
firstmention_token=""
new_tokens={"firstmention_token": firstmention_token, "concise_token": new_concise_token}


src_concise_token_stripped=src_concise_token.strip()
src_verbose_token_stripped=src_verbose_token.strip()
src_concise_token_len=len(src_concise_token_stripped)

tokenizer = AutoTokenizer.from_pretrained(model_name)



tokenizer.add_special_tokens({"additional_special_tokens": list(new_tokens.values())})
with open("chat_templates/gemma_customized_template.txt") as f:
    tokenizer.chat_template = f.read()

regex_pattern_to_filter='|'.join([re.escape(new_tokens[token]) for token in new_tokens])

def our_detokenize(sent, special_token_old, special_token_new):
    if special_token_old in sent[0]:
        sent[0]=re.sub(r'\s*' + re.escape(special_token_old) + r'\s*', special_token_new, sent[0])

    sent=detokenizer.detokenize(sent).replace(' .', '.').replace(" '", "'")
    sent=re.sub(r'\s*' + re.escape(special_token_old) + r'\s*', special_token_new+" ", sent)
    sent=re.sub(r'\s+', ' ', sent)
    return sent


def remove_special_token(sents, old_special_token, new_special_token):
    sents=deepcopy(sents)
    for i in range(len(sents)):
        sents[i]=re.sub(r'\s*' + re.escape(old_special_token) + r'\s*', new_special_token, sents[i])
    return sents


def isDeterminer(word):
    return word.lower() in {"the", "an", "a", "this", "that", "these", "those", "your", "my", "his", "her", "their", "its"}


nouns_to_filter=["the one", "the thing", "the object", "this one", "that one", "this thing", "that thing", "this object", "that object"]

def is_trivial_remention(instance):
    for nouns_to_filter_i in nouns_to_filter:
        if nouns_to_filter_i in instance[0]['content']:
            return True
    return False

def editDistance(h, r):
    #https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8).reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1]+1
                insert = d[i][j-1]+1 
                delete = d[i-1][j]
                d[i][j] = min(substitute, insert, delete)
    d=d[len(r)][len(h)]

    if len(r)!=0:
        wnr=d/len(r)
    else:
        wnr=1

    return d, wnr

def process_pair(accept, reject, first_mention_neg=False, min_remention_count=3, max_token_size=512, no_marker_neg=False, firstmention_token=None, max_completion_token_size=80):
    assert(accept['special_token']==src_concise_token.strip())
    assert(reject['special_token']==src_verbose_token.strip())

    prompt_ids=[]
    prompts=[]
    accept_responses=[]
    reject_responses=[]
    instance_types=[]


    covered_firstMention_neg_sents=set()
    if first_mention_neg:
        for cluster in accept["clusters"]:
            if len(cluster["re_mentions"])==0:
                continue

            remention_sent_i=cluster["first_mention"]["loc"][0]-accept["segment_offset"]
            this_prompt=accept['span_tokenized'][:remention_sent_i]
            pos_utter=accept['span_tokenized'][remention_sent_i]
            neg_utter=deepcopy(accept['span_tokenized'][remention_sent_i])


            if firstmention_token is not None:
                pos_utter[max(cluster["first_mention"]["loc"][1]-1,0)]+=firstmention_token


            if remention_sent_i in covered_firstMention_neg_sents or (remention_sent_i==0 and cluster["first_mention"]["loc"][1]<20):
                continue

            accepted_tokens=accept['span_tokenized'][remention_sent_i][cluster["first_mention"]["loc"][1]:cluster["first_mention"]["loc"][2]+1]
            rejected_tokens=accept['span_tokenized'][cluster["re_mentions"][-1]["loc"][0]-accept["segment_offset"]][cluster["re_mentions"][-1]["loc"][1]:cluster["re_mentions"][-1]["loc"][2]+1]
            rejected_tokens=[w.lower() for w in rejected_tokens]

            if isDeterminer(accepted_tokens[0]):
                pos_determiner=accepted_tokens[0]
            else:
                pos_determiner=""

            if isDeterminer(rejected_tokens[0]):
                if pos_determiner=="a" or pos_determiner=="an":
                    if rejected_tokens[1][0] in {'a', 'e', 'i', 'o', 'u'}:
                        replacement_det=src_concise_token+" an"
                    else:
                        replacement_det=src_concise_token+" a"
                else:
                    replacement_det=pos_determiner
                rejected_tokens[0]=src_concise_token+" "+replacement_det
            neg_utter[cluster["first_mention"]["loc"][1]:cluster["first_mention"]["loc"][2]+1]=rejected_tokens


            messages=[]
            for utter in this_prompt:
                found=False
                for j, c in enumerate(utter[1:3], start=1):
                    if c==":":
                        if len(utter[j+1:])>0:
                            utter_spkr=utter[:j]
                            messages.append({
                            "role": our_detokenize(utter_spkr, src_concise_token, new_concise_token).title(),
                            "content": our_detokenize(utter[j+1:], src_concise_token, new_concise_token)
                            })
                            found=True
                            break
                if not found:
                    messages.append({
                        "role": "Narrator",
                        "content": our_detokenize(utter, src_concise_token, new_concise_token)
                    })

            last_utt_spkr="Narrator"
            utter_begin_indx=0
            for j, c in enumerate(pos_utter[1:3], start=1):
                if c==":":
                    if len(pos_utter[j+1:])>0:
                        last_utt_spkr=our_detokenize(pos_utter[:j], src_concise_token, new_concise_token).title()
                        utter_begin_indx=j+1
                        break

            this_accepted=[{
                "role": "assistant",
                "content": our_detokenize(pos_utter[utter_begin_indx:], src_concise_token, new_concise_token)
            }]
            this_rejected=[{
                "role": "assistant",
                "content": our_detokenize(neg_utter[utter_begin_indx:], src_concise_token, new_concise_token) 
            }]

            instruction_prompt=f"Write the next line of this excerpt of TV show transcript from where it's left off. You will play the role of {last_utt_spkr}. Transcript begins:"
            instruction_utter=[{"role":"user", "content":instruction_prompt}]

            if "gemma" in model_name:
                prompt_messages=instruction_utter+messages
            else:
                system_prompt="You're a helpful assistant."
                prompt_messages=[{"role":"system", "content":system_prompt}]+instruction_utter+messages

            token_count=len(tokenizer.apply_chat_template(prompt_messages+this_accepted, add_generation_prompt=True, tokenize=True))
            if token_count>max_token_size:
                break


            prompts.append(prompt_messages)
            accept_responses.append(this_accepted)
            reject_responses.append(this_rejected)
            prompt_ids.append(accept['id']+'FirstMentionNeg'+"_"+str(remention_sent_i))
            instance_types.append('FirstMentionNeg')

            if no_marker_neg:
                prompt_ids.append(accept['id']+'FirstMentionNoMarker'+"_"+str(remention_sent_i))
                prompts.append(prompt_messages)
                instance_types.append('FirstMentionNoMarker')
                accept_responses.append(this_accepted) 
                this_rejected_no_marker=deepcopy(this_rejected)
                this_rejected_no_marker[0]["content"]=re.sub(regex_pattern_to_filter, "", this_rejected_no_marker[0]["content"])
                reject_responses.append(this_rejected_no_marker)
                covered_firstMention_neg_sents.add(remention_sent_i)



    clusters_list=accept["clusters"]
    verbose_clusters_list=reject["clusters"]
    accept_span=deepcopy(accept['span_tokenized'])
    reject_span=deepcopy(reject['span_tokenized'])

    sents_w_rementions=set()
    for cluster, verbose_cluster in zip(clusters_list, verbose_clusters_list):
        cluster["re_mentions"].sort(key=lambda x: x["loc"][0]+x["loc"][1]*0.0001)
        verbose_cluster["re_mentions"].sort(key=lambda x: x["loc"][0]+x["loc"][1]*0.0001)


        first_mention_len=len(cluster["first_mention"]["text"])
        prev_remention_len=first_mention_len
        filtered_rementions=[]
        filtered_verbose_rementions=[]
        for remention_idx in range(len(cluster["re_mentions"])):
            this_remention_len=len(cluster["re_mentions"][remention_idx]["text"])
            if this_remention_len>=first_mention_len or this_remention_len>prev_remention_len+2:
                continue
            elif remention_idx>1 and editDistance(cluster["re_mentions"][remention_idx]["text"], cluster["re_mentions"][0]["text"])[1]>0.5 and editDistance(cluster["re_mentions"][remention_idx]["text"], cluster["re_mentions"][1]["text"])[1]>0.5:
                continue
            else:
                filtered_rementions.append(cluster["re_mentions"][remention_idx])
                filtered_verbose_rementions.append(verbose_cluster["re_mentions"][remention_idx])
                prev_remention_len=this_remention_len

        cluster["re_mentions"]=filtered_rementions
        verbose_cluster["re_mentions"]=filtered_verbose_rementions


        for remention, verbose_mention in zip(cluster["re_mentions"], verbose_cluster["re_mentions"]):
            remention_sent_i=remention["loc"][0]-accept["segment_offset"]
            if remention['loc'][1]==0: 
                continue

            if len(cluster["re_mentions"])<min_remention_count: 
                continue
            if remention["loc"][0]==accept["segment_offset"]:
                continue

            if (cluster["first_mention"]['loc'][-1]-cluster["first_mention"]['loc'][-2])-(remention['loc'][-1]-remention['loc'][-2])==0:
                #skip due to limited shortening
                continue

            sents_w_rementions.add(remention_sent_i)

    sents_w_rementions=list(sents_w_rementions)
    sents_w_rementions.sort()

    for rement_sent_i in sents_w_rementions:
        this_prompt=accept_span[:rement_sent_i]
        pos_utter=accept_span[rement_sent_i]
        neg_utter=reject_span[rement_sent_i]

        if len(pos_utter)<3:   #discard instances that are too short
            continue


        messages=[]
        for utter in this_prompt:
            found=False
            for j, c in enumerate(utter[1:3], start=1):
                if c==":":
                    if len(utter[j+1:])>0:
                        utter_spkr=utter[:j]
                        messages.append({
                        "role": our_detokenize(utter_spkr, src_concise_token, new_concise_token).title(),
                        "content": our_detokenize(utter[j+1:], src_concise_token, new_concise_token)
                        })
                        found=True
                        break

            if not found:
                messages.append({
                    "role": "Narrator",
                    "content": our_detokenize(utter, src_concise_token, new_concise_token)
                })

        last_utt_spkr="Narrator"
        last_utt_spkr_tokenized=["Narrator"]
        utter_begin_indx=0
        for j, c in enumerate(pos_utter[1:3], start=1):
            if c==":":
                if len(pos_utter[j+1:])>0:
                    last_utt_spkr=our_detokenize(pos_utter[:j], src_concise_token, new_concise_token).title()
                    last_utt_spkr_tokenized=pos_utter[:j]
                    utter_begin_indx=j+1
                    break

        this_accepted=[{
            "role": "assistant",
            "content": our_detokenize(pos_utter[utter_begin_indx:], src_concise_token, new_concise_token)
        }]

        if is_trivial_remention(this_accepted):
            continue

        if src_concise_token in ' '.join(last_utt_spkr_tokenized): #Skip, because remention is a role.
            continue

        this_rejected=[{
            "role": "assistant",
            "content": our_detokenize(neg_utter[utter_begin_indx:], src_verbose_token, new_concise_token) 
        }]

        system_prompt="You're a helpful assistant."
        instruction_prompt=f"Write the next line of this excerpt of TV show transcript from where it's left off. You will play the role of {last_utt_spkr}. Transcript begins:"
        instruction_utter=[{"role":"user", "content":instruction_prompt}]
        if "gemma" in model_name:
            prompt_messages=instruction_utter+messages
        else:
            prompt_messages=[{"role":"system", "content":system_prompt}]+instruction_utter+messages


        prompt_token_count=len(tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=True))
        completion_token_count=len(tokenizer.apply_chat_template(this_accepted, add_generation_prompt=False, tokenize=True))
        if (prompt_token_count+completion_token_count)>max_token_size or completion_token_count>max_completion_token_size:
            break

        prompt_ids.append(accept['id']+'StandardNeg'+"_"+str(rement_sent_i))
        prompts.append(prompt_messages)
        instance_types.append('StandardNeg')
        accept_responses.append(this_accepted)
        reject_responses.append(this_rejected)
 

        if no_marker_neg:
            prompt_ids.append(accept['id']+'no_marker_neg'+"_"+str(rement_sent_i))
            prompts.append(prompt_messages)
            instance_types.append('NoMarkerNeg')
            accept_responses.append(this_accepted) 
            this_rejected_no_marker=deepcopy(this_rejected)
            this_rejected_no_marker[0]["content"]=re.sub(regex_pattern_to_filter, "", this_rejected_no_marker[0]["content"])
            reject_responses.append(this_rejected_no_marker)


    return prompt_ids, prompts, accept_responses, reject_responses, instance_types



def remove_prompt_markers(instance, markers):
    regex_pattern = r'(' + '|'.join(map(re.escape, markers)) + r')'
    for i in range(len(instance["prompt"])):
        instance["prompt"][i]["content"]=re.sub(regex_pattern, "", instance["prompt"][i]["content"]).strip()
    return instance


def save_datasets(train_dataset, eval_dataset, max_token_size, new_special_tokens, date_time, dataset_prefix="", additional_postfix=""):
    processed_data_dir=f"preference_data/{date_time}_{dataset_prefix}_{additional_postfix}_json"
    train_dataset.to_json(os.path.join(processed_data_dir, "train.json"))
    eval_dataset.to_json(os.path.join(processed_data_dir, "eval.json"))

    with open(os.path.join(processed_data_dir, 'special_tokens.json'), 'w') as f:
        json.dump(new_special_tokens, f)

    dataset_config={"max_token_size":max_token_size}
    with open(os.path.join(processed_data_dir, 'dataset_config.json'), 'w') as f:
        json.dump(dataset_config, f)

def format_paired_dataset(examples, min_remention_count=2, max_token_size=1048, no_marker_neg=False, firstmention_token=None):

    id_list=[]
    prompt_list=[]
    chosen_list=[]
    rejected_list=[]
    instance_types_list=[]

    for i in range(0, len(examples), 2):
        accept=examples[i]
        reject=examples[i+1]

        prompt_ids, prompts, accept_responses, reject_responses, instance_types = process_pair(accept, reject, first_mention_neg=True, min_remention_count=min_remention_count, max_token_size=max_token_size, no_marker_neg=no_marker_neg, firstmention_token=firstmention_token)
        id_list.extend(prompt_ids)
        prompt_list.extend(prompts)
        chosen_list.extend(accept_responses)
        rejected_list.extend(reject_responses)
        instance_types_list.extend(instance_types)


    return {"ID": id_list, "prompt": prompt_list, "chosen":chosen_list, "rejected":rejected_list, "instance_types":instance_types_list}




def main():

    train_raw_data=[]
    all_train_files=glob(train_src)
    all_train_files.sort()

    for file in all_train_files:
        with open(file) as f:
            data = json.load(f)
            train_raw_data+=data


    formatted_train=format_paired_dataset(train_raw_data, min_remention_count=2,  max_token_size=max_token_size, no_marker_neg=True, firstmention_token=firstmention_token)

    #create eval set
    train_dataset=Dataset.from_dict(formatted_train)

    df=pd.DataFrame({"instance_types":train_dataset['instance_types']})
    df['instance_types'].value_counts()
    split_dataset=train_dataset.train_test_split(test_size=500/len(train_dataset), shuffle=False)
    train_dataset=split_dataset['train']
    eval_dataset=split_dataset['test']

    #prepare SFT data
    instance_types_for_sft=["StandardNeg"]
    train_dataset_sft=train_dataset.filter(lambda example: example["instance_types"] in instance_types_for_sft)
    train_dataset_sft=train_dataset_sft.map(lambda x: remove_prompt_markers(x, [new_concise_token]))
    train_dataset_sft=train_dataset_sft.remove_columns("rejected").rename_column("chosen", "completion")


    eval_dataset_sft=eval_dataset.filter(lambda example: example["instance_types"] in instance_types_for_sft)
    eval_dataset_sft=eval_dataset_sft.map(lambda x: remove_prompt_markers(x, [new_concise_token]))
    eval_dataset_sft=eval_dataset_sft.remove_columns("rejected").rename_column("chosen", "completion")

    train_dataset_dpo=train_dataset
    eval_dataset_dpo=eval_dataset

    date_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    save_datasets(train_dataset_sft, eval_dataset_sft, max_token_size, new_tokens, date_time, dataset_prefix="sft", additional_postfix=f"{max_token_size}")
    save_datasets(train_dataset_dpo, eval_dataset_dpo, max_token_size, new_tokens, date_time, dataset_prefix="dpo", additional_postfix=f"{max_token_size}")

if __name__ == "__main__":
    main()
