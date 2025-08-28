
import json, itertools, os, glob
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()
from tqdm import tqdm
from copy import deepcopy


import spacy
nlp = spacy.load("en_core_web_sm")
nlp.remove_pipe("lemmatizer")
nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()
def process_det(s1, s2):
    doc1 = nlp(s1)
    doc2 = nlp(s2)
   
    if doc1[0].pos_ == "DET" and doc2[0].pos_== "DET":
        
        s1 = s1.replace(doc1[0].text, doc2[0].text, 1)
        return s1
    else:
        return s1


mention2filter={"i", "we", "you", "he", "she", "it", "they", "me", "us", "you", "him", "her", "it", "them", "my", "our", "your", "his", "her", "its", "their", "mine", "ours", "yours", "hers", "theirs", "one", "each other", "myself", "ourselves", "yourself", "yourselves", "this", "that", "himself", "herself", "itself", "themselves", "scene_break"}


cannot_contain={"SCENE"}
spkrs_to_filter=["narrator"]

def get_spkr(utter):
    if isinstance(utter, str):
        words=word_tokenize(utter) 
    else:
        words=utter
    found_speaker=False 
    
    for i in range(min(len(words), 4)):
        if words[i]==":":
            spkr=' '.join(words[:i])
            found_speaker=True
            break  
    if not found_speaker:
        spkr="narrator"
    return spkr

def detokenize_transcript(detokenizer, transcript):
    new_transcript=[]
    for utter in transcript:
        utter=detokenizer.detokenize(utter)
        new_transcript.append(utter)
    return new_transcript


def non_dialog_token_ids(line):
    if isinstance(line, str):
        tokenized_line=word_tokenize(line) 
    else:
        tokenized_line=line

    if set(spkrs_to_filter).intersection(set(tokenized_line)):
        return list(range(len(tokenized_line)))
    non_dialog_ids=[]
    non_dialog=False
    for i, tok in enumerate(tokenized_line):
        if tok=="(" or tok=="[":
            non_dialog=True
            non_dialog_ids.append(i)
        elif tok==")" or tok=="]":
            non_dialog=False
            non_dialog_ids.append(i)
        else: 
            if non_dialog:
                non_dialog_ids.append(i)

    return non_dialog_ids


punct_no_white_space={".", ",", "!", "?", ":", ";", ")", "]", "}", ")", "]", "}", "'", '"',}
                      
def remove_white_space_before_punc(line: str) -> str:
    new_line=[]
    line_len=len(line)
    for i, tok in enumerate(line):
        if i+1<line_len and line[i+1] in punct_no_white_space and tok==" ":
            continue
        else:
            new_line.append(tok)
    return "".join(new_line)
    



def postprocess(tv_pred_data_fp, output_dir, word_tokenized_source_transcript, dataset="other", remention_limit=5, dialogue_only=False, extract_multiple_per_cluster=False, extract_multiple_per_transcript=True):
    batch_id=str(tv_pred_data_fp.split("/")[-1].split("-")[0])
    
    
    with open(tv_pred_data_fp, "r") as f:
        tv_pred_data = [json.loads(line) for line in f]


    
    tv_tokenized_data_fp = f"{word_tokenized_source_transcript}/{batch_id}.jsonl"
    with open(tv_tokenized_data_fp, "r") as f:
        tv_tokenized_data=[json.loads(line) for line in f]

    tv_tokenized_data_dict={}
    for item in tv_tokenized_data:
        item['longest_mentions']=[]
        item['re_mentions']=defaultdict(list)
        tv_tokenized_data_dict[item['id']]=item
    coreferences=[]

    min_mention_dist=5
    max_mention_dist=400
    mention_len_limit=10
    longer_mention_min_len=3
    total_cluster_count=0
    min_required_shortening=2
    
    for convo_idx in range(0, len(tv_pred_data)):
        

        if dataset=="tv":
            instance_id=tv_pred_data[convo_idx]['doc_key'].split('.json')[0]+'.json'
        else:
            instance_id=tv_pred_data[convo_idx]['doc_key'].split('_')[0]
        

        convo=tv_tokenized_data_dict[instance_id]

        speakers=[s.lower() for s in convo['speakers']]
        mentions_to_filter=mention2filter.union(set(speakers))

        raw_transcript=convo['sentences']
        transcript_tokens=list(itertools.chain.from_iterable(raw_transcript))
        if dialogue_only:
            non_dialog_token_ids_list=[non_dialog_token_ids(sent) for sent in raw_transcript]
        word_sentence_map=[]
        in_sent_idx_offset=[]
        offset=0
        for i, sent in enumerate(raw_transcript):
            word_sentence_map.extend([i]*len(sent))
            in_sent_idx_offset.extend([offset]*len(sent))
            offset+=len(sent)    

        for word_cluster in tv_pred_data[convo_idx]['predict_clusters']:
            if ('instance_extracted' in tv_tokenized_data_dict[instance_id]) and (tv_tokenized_data_dict[instance_id]['instance_extracted']==True):
                break
            
            if len(word_cluster)<2:
                continue
            total_cluster_count+=1
            mention_lens=[]
            for mention in word_cluster:
                mention_lens.append(mention[1]-mention[0]+1)
            if max(mention_lens)<longer_mention_min_len:
                continue
            
            filtered_mentions=[]
            filtered_mentions_detokenized=[]
            filtered_mentions_indices=[]
            longest_mention_idx=None
            longest_mention_len=None
            
            word_cluster=sorted(word_cluster, key=lambda x: x[0])

            for mention in word_cluster:
                this_mention_len=mention[1]-mention[0]
                if this_mention_len>mention_len_limit:
                    continue
                mention_sent_i=word_sentence_map[mention[0]]
                #check if mention ids are among the non-dialog tokens
                # if so, skip this mention
                mention_start_loc_idx=mention[0]-in_sent_idx_offset[mention[0]]
                if dialogue_only and (mention_start_loc_idx in non_dialog_token_ids_list[mention_sent_i]):
                    continue 
                
                mention_phrase=transcript_tokens[mention[0]:mention[1]+1]
                mention_phrase_detokenized=detokenizer.detokenize(mention_phrase)
                if mention_phrase_detokenized.lower() in mentions_to_filter:
                    continue
                
                valid=True
                for fw in cannot_contain:
                    if fw in mention_phrase_detokenized:
                        valid=False
                        break
                if not valid:
                    continue

                filtered_mentions.append(mention_phrase)  
                filtered_mentions_indices.append(mention)
                filtered_mentions_detokenized.append(mention_phrase_detokenized)
                if word_sentence_map[mention[0]]!=word_sentence_map[mention[1]]:
                    # print('skipped due to mentions across different lines')
                    continue
                if (longest_mention_len is None) or (this_mention_len>longest_mention_len):
                    longest_mention_len=mention[1]-mention[0]
                    longest_mention_idx=len(filtered_mentions)-1
            if longest_mention_len is None:
                continue
            remention_count=0
            if len(filtered_mentions)>1:
                next_mention_idx=longest_mention_idx+1
                mention1=filtered_mentions_detokenized[longest_mention_idx] 
                mention1_ids=filtered_mentions_indices[longest_mention_idx]
                if mention1.endswith("'s"):
                    mention1=mention1[:-2]
                    mention1_ids[1]=mention1_ids[1]-1
                mention1_len=mention1_ids[1]-mention1_ids[0]+1
                mention1_loc_ids=[mention1_ids[0]-in_sent_idx_offset[mention1_ids[0]], mention1_ids[1]-in_sent_idx_offset[mention1_ids[1]]]
                mention1_sent_i=word_sentence_map[mention1_ids[0]]
                if mention1_sent_i!=word_sentence_map[mention1_ids[1]]:
                    continue
                tv_tokenized_data_dict[instance_id]['longest_mentions'].append([mention1_sent_i, mention1_loc_ids[0], mention1_loc_ids[1]])
                
                while next_mention_idx<len(filtered_mentions_indices):
                    mention_dist=filtered_mentions_indices[next_mention_idx][0]-filtered_mentions_indices[longest_mention_idx][1]
                    if mention_dist<min_mention_dist or mention_dist>max_mention_dist:
                        next_mention_idx+=1
                        continue

                    mention2=filtered_mentions_detokenized[next_mention_idx]
                    mention2_ids=filtered_mentions_indices[next_mention_idx]

                    if mention2.endswith("'s"):
                        mention2=mention2[:-2]
                        mention2_ids[1]=mention2_ids[1]-1

                    mention2_len=mention2_ids[1]-mention2_ids[0]+1
                    
                    if len(mention2)<1 or mention2_len>(mention1_len-min_required_shortening):
                        next_mention_idx+=1
                        continue
                    if word_sentence_map[mention2_ids[0]]!=word_sentence_map[mention2_ids[1]]:
                        next_mention_idx+=1
                        continue

                    mention2_loc_ids=[mention2_ids[0]-in_sent_idx_offset[mention2_ids[0]], mention2_ids[1]-in_sent_idx_offset[mention2_ids[1]]]
                    mention2_sent_i=word_sentence_map[mention2_ids[0]]
          
                    tv_tokenized_data_dict[instance_id]['re_mentions'][mention1_sent_i].append({
                        "mention2_sent_i": mention2_sent_i,
                        "mention2_loc_ids": mention2_loc_ids,
                        "mention1_loc_ids": mention1_loc_ids
                    })
      

                    if len(raw_transcript)<40 or dataset!="tv":
                        raw_transcript_seg=raw_transcript
                        mention1_loc_sent_i=mention1_sent_i
                        mention2_loc_sent_i=mention2_sent_i
                    else:
                        if mention2_sent_i-mention1_sent_i<1:
                            start_idx=max(0, mention1_sent_i-1)
                            end_idx=mention2_sent_i+1
                        else:
                            start_idx=mention1_sent_i
                            end_idx=mention2_sent_i+1
                        raw_transcript_seg=raw_transcript[start_idx:end_idx]
                        mention1_loc_sent_i=mention1_sent_i-start_idx
                        mention2_loc_sent_i=-1
           
                    
                    doc_key=tv_pred_data[convo_idx]['doc_key']


                    instance={
                        "doc_key": doc_key,
                        "mention1": mention1,
                        "mention2": mention2,
                        "segment": raw_transcript_seg,
                        "mention1_sent_i": mention1_loc_sent_i,
                        "mention2_sent_i": mention2_loc_sent_i,
                        "mention1_tok_i": mention1_loc_ids,
                        "mention2_tok_i": mention2_loc_ids
                    }


                    coreferences.append(instance)
                    remention_count+=1
                    next_mention_idx+=1
                    if not extract_multiple_per_transcript:
                        tv_tokenized_data_dict[instance_id]['instance_extracted']=True
                    if remention_count>=remention_limit or not extract_multiple_per_cluster:
                        break

    with open(f"{output_dir}/condensation_instances_{batch_id}_in_transcript.json", "w") as f:
        json.dump(tv_tokenized_data_dict, f)
    with open(f"{output_dir}/condensation_instances_{batch_id}.json", "w") as f:
        json.dump(coreferences, f)

     





def failsProperNounFilter(phrase):
    doc = nlp(phrase)  
    propN_count=0 
    pos_list=[]
    for token in doc:
        if token.pos_ == "PROPN":
            propN_count+=1  
        pos_list.append(token.pos_)
    
    if "NOUN" not in pos_list:
        return True
    elif propN_count/len(doc)>=0.5:
        return True
    else:
        return False

def detokenize_transcript(transcript, special_token):
    detokenized_transcript=[]
    for sent in transcript:
        sent=detokenizer.detokenize(sent).replace(' .', '.')
        sent=sent.replace(special_token, ' '+special_token+' ')
        sent=sent.replace('  ', ' ')
        detokenized_transcript.append(sent)

    return detokenized_transcript


special_punctuations={"--", "-", "\n", "\"", "\'", ")", "]", "}",  "(", "[", "{", "``", "..."}
determiner_map={"a": "the", "an": "the", "A": "The", "An": "The"}
def is_special_punctuation(c):
    if (
            c in special_punctuations
            
    ):
        return True
    return False

def is_punctuation(c):
    if (
            c in {".", ",", "?", "!", ";",
                  ":", "--", ")", "]", "}", "-"}
    ):
        return True
    return False

def has_overlap(list_of_spans, new_span):
    for span in list_of_spans:
        if new_span[0]<=span[1] and new_span[1]>=span[0]:
            return True
    return False
     



def update_transcript_after_verbose_insertion(transcript, modified_sent_i, modified_loc_i0, idx_offset):
  
    for sent_i, re_mentions_this_cluster in transcript['re_mentions'].items():
        for other_re_mention in re_mentions_this_cluster:
            if other_re_mention["mention2_sent_i"]==modified_sent_i and other_re_mention["mention2_loc_ids"][0]>modified_loc_i0:
                other_re_mention["mention2_loc_ids"][0]+=idx_offset
                other_re_mention["mention2_loc_ids"][1]+=idx_offset
            if sent_i==modified_sent_i and other_re_mention["mention1_loc_ids"][0]>modified_loc_i0:
                other_re_mention["mention1_loc_ids"][0]+=idx_offset
                other_re_mention["mention1_loc_ids"][1]+=idx_offset

    
def group_mentions_by_first_mention(mentions, max_first_mention_dist=10):
    if len(mentions)==0:
        print("no eligible mentions")
        return []
    first_mention_locs=list(mentions.keys())
    first_mention_locs.sort()
    curr_group_start=first_mention_locs[0]
    groups=[[first_mention_locs[0]]]
    for loc in first_mention_locs[1:]:

        if loc-curr_group_start<max_first_mention_dist:
            groups[-1].append(loc)
        else:
            curr_group_start=loc
            groups.append([loc])
         
    return groups

def __prepare_data_w_mentions(mention1_group, transcript, special_token, is_verbose, max_output_utter=40, batch_fp="batch_X_", is_full_transcript=False, start_idx_offset=0):
    clusters=[]
    re_mention_spans=defaultdict(list)
    input_start_i=max(0, mention1_group[0]-start_idx_offset)
    last_mention_sent_i=input_start_i
    if is_full_transcript:
        input_start_i=0
        max_output_utter=len(transcript['sentences'])

    for mention1_sent_i in mention1_group:
        re_mentions=transcript['re_mentions'][mention1_sent_i]

        first_mention_tokens=deepcopy(transcript['sentences'][mention1_sent_i][re_mentions[-1]['mention1_loc_ids'][0]:re_mentions[-1]['mention1_loc_ids'][1]+1])
        first_mention_tokens=[token.replace(special_token, '') for token in first_mention_tokens]
        if is_punctuation(first_mention_tokens[-1]):
            first_mention_tokens=first_mention_tokens[:-1]
        if is_punctuation(first_mention_tokens[0]):
            if len(first_mention_tokens)>1:
                first_mention_tokens=first_mention_tokens[1:]
            else:
                continue

            

        first_mention_string=detokenizer.detokenize(first_mention_tokens)
        if first_mention_string.isupper():
            continue
        clusters.append({'first_mention': {"text": first_mention_string, "loc":[mention1_sent_i, re_mentions[-1]['mention1_loc_ids'][0], re_mentions[-1]['mention1_loc_ids'][1]]},'re_mentions': []})

        if first_mention_tokens[0] in determiner_map:
            verbose_mention_tokens=deepcopy(first_mention_tokens)
            verbose_mention_tokens[0]=determiner_map[verbose_mention_tokens[0]]
            verbose_mention_string=detokenizer.detokenize(verbose_mention_tokens)
        else:
            verbose_mention_tokens=first_mention_tokens
            verbose_mention_string=first_mention_string


        for re_mention in re_mentions: 
            if (re_mention["mention2_sent_i"]-input_start_i)>max_output_utter-1:
                break
            ori_mention_tokens=transcript['sentences'][re_mention["mention2_sent_i"]][re_mention["mention2_loc_ids"][0]:re_mention["mention2_loc_ids"][1]+1]
            ori_mention_string=detokenizer.detokenize(ori_mention_tokens)

            try:
                if ori_mention_string.isupper() or failsProperNounFilter(ori_mention_string):
                    continue
                if is_punctuation(ori_mention_tokens[-1]):
                    if len(ori_mention_tokens)>1:
                        ori_mention_tokens=ori_mention_tokens[:-1]
                        re_mention["mention2_loc_ids"][1]-=1
                    else:
                        continue
            except Exception as e:
                print("Error", e)
                
            if is_punctuation(ori_mention_tokens[0]):
                if len(ori_mention_tokens)>1:
                    ori_mention_tokens=ori_mention_tokens[1:]
                    re_mention["mention2_loc_ids"][0]+=1
                else:
                    continue

            if special_token not in transcript['sentences'][re_mention["mention2_sent_i"]][re_mention["mention2_loc_ids"][0]]:
                
                if has_overlap(re_mention_spans[re_mention["mention2_sent_i"]], re_mention["mention2_loc_ids"]):
                    continue
                if is_verbose:
                    idx_offset=len(verbose_mention_tokens)-len(ori_mention_tokens)
                    
                    transcript['sentences'][re_mention["mention2_sent_i"]][re_mention["mention2_loc_ids"][0]:re_mention["mention2_loc_ids"][1]+1]=verbose_mention_tokens
                    re_mention['mention2_loc_ids'][1]+=idx_offset

         

                    update_transcript_after_verbose_insertion(transcript, re_mention["mention2_sent_i"], re_mention["mention2_loc_ids"][0], idx_offset)

                transcript['sentences'][re_mention["mention2_sent_i"]][re_mention["mention2_loc_ids"][0]]=special_token+transcript['sentences'][re_mention["mention2_sent_i"]][re_mention["mention2_loc_ids"][0]]

                
                if re_mention["mention2_sent_i"]>last_mention_sent_i:
                    last_mention_sent_i=re_mention["mention2_sent_i"]

                if is_verbose:
                    clusters[-1]['re_mentions'].append({"text": verbose_mention_string, "loc": [re_mention["mention2_sent_i"]]+re_mention["mention2_loc_ids"]})
                else:
                    clusters[-1]['re_mentions'].append({"text": ori_mention_string, "loc": [re_mention["mention2_sent_i"]]+re_mention["mention2_loc_ids"]}) 
                        
                re_mention_spans[re_mention["mention2_sent_i"]].append(re_mention["mention2_loc_ids"])



    if len(re_mention_spans)==0:
        return None, None
    

    sent_w_rementions=list(re_mention_spans.keys())
    sent_w_rementions.sort()

    last_mention_global_sent_i=0
    for s in sent_w_rementions:
        if s==input_start_i or (s-input_start_i)<max_output_utter:
            last_mention_global_sent_i=s

    id_to_save=transcript["id"]+'_'+str(input_start_i)+'_'+str(last_mention_global_sent_i)
    tokenized_instance={
            "id": id_to_save,
            "span_tokenized": transcript['sentences'][input_start_i:last_mention_global_sent_i+1],
            "special_token": special_token.strip(),
            "segment_offset": input_start_i, 
            "batch_fp": batch_fp,
            "clusters": clusters
            }
    
    detokenized_instance={
            "id": id_to_save,
            "span_tokenized": detokenize_transcript(transcript['sentences'][input_start_i:last_mention_global_sent_i+1], special_token),
            "special_token": special_token.strip(),
            "segment_offset": input_start_i, 
            "batch_fp": batch_fp,
            "clusters": clusters,
            }
    return tokenized_instance, detokenized_instance
    


def prepare_data_w_mentions(transcript, mention_special_tokens, batch_fp, is_full_transcript=False):
    tokenized_instances=[]
    detokenized_instances=[]
    for k, cluster in transcript['re_mentions'].items():
        for mention in cluster:
            mention['mention2_loc_ids_ori']=deepcopy(mention['mention2_loc_ids'])
    group_lens=[]
    if is_full_transcript:
        if len(transcript['re_mentions'].keys())==0:
            print("no eligible mentions")
            groups=[]
        else:
            groups=[list(transcript['re_mentions'].keys())]
    else:
       
        groups=group_mentions_by_first_mention(transcript['re_mentions'])
        group_lens.extend([len(group) for group in groups])

        

    for group in groups:
        transcript_tmp=deepcopy(transcript)
        pos_tokenized, pos_detokenized=__prepare_data_w_mentions(group, transcript_tmp, mention_special_tokens[0], is_verbose=False, batch_fp=batch_fp, is_full_transcript=is_full_transcript, start_idx_offset=1)

        if pos_tokenized is None:
            continue
        transcript_tmp=deepcopy(transcript)
        neg_tokenized, neg_detokenized=__prepare_data_w_mentions(group, transcript_tmp, mention_special_tokens[1], is_verbose=True, batch_fp=batch_fp, is_full_transcript=is_full_transcript, start_idx_offset=1)
        tokenized_instances.append(pos_tokenized)
        tokenized_instances.append(neg_tokenized)
        detokenized_instances.append(pos_detokenized)
        detokenized_instances.append(neg_detokenized)
            
    return tokenized_instances, detokenized_instances, group_lens



coref_output_dir="data/coreferences/*-predicts.jsonl"   
final_output_dir="data/coreferences_shortening"
word_tokenized_transcript_dir="data/transcripts"

all_preds_fp = glob.glob(coref_output_dir)

os.makedirs(final_output_dir, exist_ok=True)
for pred_data_fp in tqdm(all_preds_fp):
   postprocess(pred_data_fp, final_output_dir, word_tokenized_transcript_dir, dataset="tv")




save_dir="data/coreferences_segments"
transcripts_fps=glob.glob("data/coreferences_shortening/condensation_instances_*_in_transcript.json")

os.makedirs(save_dir, exist_ok=True)
transcripts_fps.sort()


for fp in tqdm(transcripts_fps):
    all_tokenized=[]
    all_detokenized=[]
    group_lens_all=[]
    print(fp)
    with open(fp) as f:
        transcripts=json.load(f)
    for transcript in transcripts.values():
        transcript['re_mentions']={int(k):v for k, v in transcript['re_mentions'].items()}
        tokenized_instances, detokenized_instances, group_lens=prepare_data_w_mentions(transcript, ["[re-mention natural]", "[re-mention verbose]"], fp, is_full_transcript=False)
        all_tokenized.extend(tokenized_instances)
        all_detokenized.extend(detokenized_instances)
        group_lens_all.extend(group_lens)


    src_basename=os.path.basename(fp).replace("_in_transcript.json", "").replace("condensation_instances_", "")
    with open(os.path.join(save_dir,src_basename+"_tokenized.json"), "w") as f:
        json.dump(all_tokenized, f, indent=4)

    with open(os.path.join(save_dir,src_basename+"_detokenized.json"), "w") as f:
        json.dump(all_detokenized, f, indent=4)
   
    


