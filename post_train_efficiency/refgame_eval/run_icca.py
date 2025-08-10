import pandas as pd
import glob, os, io, random, json, time, argparse, copy, base64, pickle, sys
from datetime import datetime
from utils import *
from tqdm import tqdm
from sklearn.metrics import classification_report
from MLLMs import *
import re
from itertools import chain
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def remove_special_tokens(msg, special_tokens):
    regex_pattern = r"(?:{})".format("|".join(map(re.escape, special_tokens)))
    return re.sub(regex_pattern, "", msg).strip()

def eval_loop(
    trial_entries,
    context_referents,
    iters,
    random_seed,
    spkr_exp_args,
    lsnr_exp_args,
    spkr_model,
    lsnr_model,
    sleep_time=0,
    additional_forbidden_tokens=[]
):
    random.seed(random_seed)
    random_seeds = random.sample(range(0, 1000), iters)
    trials_Records = trial_entries[:iters]

    # prepare the instruction that appears in the beginning of the interaction
    spkr_intro = (
        spkr_model.get_spkr_intro(context_referents)
        if spkr_exp_args.model_type != "Human"
        else []
    )

    spkr_context_referents = context_referents
    lsnr_context_referents = context_referents.copy()
    random.shuffle(lsnr_context_referents)
    lsnr_intro = (
        lsnr_model.get_lsnr_intro(lsnr_context_referents) if lsnr_exp_args.model_type != "oracle" else []
    )

    forbidden_words = trial_entries[0]["forbidden_words"]
    try:
        forbidden_words = json.loads(forbidden_words.replace("'", '"'))
    except:
        forbidden_words = json.loads(forbidden_words.replace("'", '"').replace('"s', "'s"))

    spkr_model.bad_words=forbidden_words
    if isinstance(spkr_model, GemmaModel) or isinstance(spkr_model, LlamaModel):
        bad_words_ids=[]
        for context_item in forbidden_words:
            for word in context_item.split():
                bad_words_ids.append(spkr_model.tokenizer.encode(" " + word, add_special_tokens=False))
                bad_words_ids.append(spkr_model.tokenizer.encode(" " + word.upper(), add_special_tokens=False))
                bad_words_ids.append(spkr_model.tokenizer.encode(word, add_special_tokens=False))
                bad_words_ids.append(spkr_model.tokenizer.encode(word.upper(), add_special_tokens=False))
        spkr_model.bad_words_ids = bad_words_ids

        if trial_entries[0]["context_id"] in additional_forbidden_tokens:
            for word in additional_forbidden_tokens[trial_entries[0]["context_id"]]:
                spkr_model.additional_forbidden_tokens.append(spkr_model.tokenizer.convert_tokens_to_ids(word))
                if isinstance(spkr_model, LlamaModel):
                    spkr_model.additional_forbidden_tokens.append(spkr_model.tokenizer.convert_tokens_to_ids("Ġ"+word))
                else:
                    spkr_model.additional_forbidden_tokens.append(spkr_model.tokenizer.convert_tokens_to_ids("▁"+word))
   

    for t, R_t in enumerate(trials_Records):
        if t != 0:
            time.sleep(sleep_time)
        tgt_name = R_t["target"]  # the target word itself for text-only games
        human_msg = R_t["human_msg"]

        if spkr_exp_args.model_type == "Human":
            gen_msg = human_msg
            spkr_trial_prompt = []
            spkr_prompt = []
        else:
            (
                spkr_prompt,
                spkr_trial_prompt,
                spkr_tgt_referent,
                tgt_label_for_spkr,
                spkr_trial_referents,
            ) = spkr_model.get_spkr_prompt(
                spkr_intro,
                t,
                spkr_context_referents,
                tgt_name,
                spkr_exp_args,
                records=trials_Records,
            )
            R_t["spkr_trial_referent_names"] = spkr_trial_referents

            # query the speaker
            gen_msg = spkr_model.query(spkr_prompt).strip()
            spkr_trial_prompt = spkr_model.update_with_spkr_pred(spkr_trial_prompt, gen_msg)
            R_t["spkr_msg"] = gen_msg
            R_t["tgt_label_for_spkr"] = tgt_label_for_spkr

        # listener prompt prep
        if lsnr_exp_args.model_type == "oracle":
            pred_name, lsnr_trial_prompt, R_t["tgt_label_for_lsnr"], R_t["lsnr_pred"] = (
                spkr_tgt_referent,
                [],
                tgt_label_for_spkr,
                tgt_label_for_spkr,
            )
            lsnr_prompt = []
        else:
            misleading = t in getattr(lsnr_exp_args, "misleading_trials", [])
            (
                lsnr_prompt,
                lsnr_trial_prompt,
                lsnr_tgt_referent,
                tgt_label_for_lsnr,
                lsnr_trial_referents,
                lsnr_trial_referents_lsnr_view,
            ) = lsnr_model.get_lsnr_prompt(
                lsnr_intro,
                t,
                lsnr_context_referents,
                tgt_name,
                msg=gen_msg,
                records=trials_Records,
                random_seed=random_seeds[t],
                no_history=lsnr_exp_args.no_history,
                do_shuffle=lsnr_exp_args.do_shuffle,
                misleading=misleading,
                has_intro=lsnr_exp_args.has_intro,
            )
            R_t["lsnr_trial_fns"] = lsnr_trial_referents

            # query the listener
            lsnr_pred = lsnr_model.query(lsnr_prompt).upper().strip()
            lsnr_trial_prompt = lsnr_model.update_with_lsnr_pred(lsnr_trial_prompt, lsnr_pred)
            R_t["tgt_label_for_lsnr"] = tgt_label_for_lsnr
            R_t["lsnr_pred"] = lsnr_pred

            try:
                pred_name = lsnr_trial_referents[lsnr_model.model_args.label_space.index(lsnr_pred)]
            except ValueError:
                pred_name = "invalid"
            lsnr_feedback = lsnr_model.get_lsnr_feedback(lsnr_pred, lsnr_tgt_referent, lsnr_trial_referents, gen_msg)
            lsnr_trial_prompt.append(lsnr_feedback)

        # get speaker feedback
        if spkr_exp_args.model_type != "Human":
            spkr_feedback = spkr_model.get_spkr_feedback(pred_name, spkr_tgt_referent, spkr_trial_referents)
            spkr_trial_prompt.append(spkr_feedback)

        things_to_print = []
        if spkr_exp_args.model_type != "Human":
            things_to_print.extend([
                {"Gen_msg": gen_msg},
                {"Human_msg": human_msg},
                {"Tgt_name": tgt_name},
            ])
        else:
            things_to_print.extend([{"Human_msg": human_msg}, {"Tgt_name": tgt_name}])

        if lsnr_exp_args.model_type != "oracle":
            things_to_print.extend([
                {"Pred_fn": pred_name},
                {"Pred_label": lsnr_pred},
                {"Tgt_label": tgt_label_for_lsnr},
            ])

        print(" | ".join([f"{k}: {v}".replace("[remention]", "|") for d in things_to_print for k, v in d.items()]))

        R_t["spkr_trial_record"] = spkr_trial_prompt
        R_t["lsnr_trial_record"] = lsnr_trial_prompt

    return trials_Records, spkr_prompt, lsnr_prompt


def run_test(
    context_records_fps,
    num_of_trials,
    random_seed,
    save_suffix,
    dtime,
    spkr_exp_args,
    lsnr_exp_args,
    spkr_model,
    lsnr_model,
    sleep_time=0,
    exp_name=None, 
    input_dir_structure="folder_by_context",
    first_context_i=0,
    last_context_i=None,
    additional_forbidden_token_fp=None,
):
    random.seed(random_seed)
    seeds = random.sample(range(0, 1000), len(context_records_fps))

    if  first_context_i is not None and last_context_i is not None:
        context_records_fps = context_records_fps[first_context_i:last_context_i+1]
    elif last_context_i is not None:
        context_records_fps = context_records_fps[:last_context_i+1]
    elif first_context_i is not None:
        context_records_fps = context_records_fps[first_context_i:]

    for i, context_record_fp in tqdm(enumerate(context_records_fps)):
        datafile_name = os.path.basename(context_record_fp)
        print(f"Working on context {datafile_name}")
        trials_this_context = pd.read_csv(context_record_fp, index_col=0)
        context_referents = trials_this_context.iloc[0]["referents"]
        try:
            context_referents = json.loads(context_referents.replace("'", '"'))
        except:
            context_referents = json.loads(context_referents.replace("'", '"').replace('"s', "'s"))
        trials_this_context = trials_this_context.to_json(orient="records")
        trials_this_context = json.loads(trials_this_context)
        random.seed(i + first_context_i)
        random.shuffle(context_referents)
        with open(additional_forbidden_token_fp) as f:
            additional_forbidden_tokens = json.load(f)
        records, spkr_hist, lsnr_hist = eval_loop(
            trials_this_context,
            context_referents,
            iters=num_of_trials,
            random_seed=seeds[i + first_context_i],
            spkr_exp_args=spkr_exp_args,
            lsnr_exp_args=lsnr_exp_args,
            spkr_model=spkr_model,
            lsnr_model=lsnr_model,
            sleep_time=sleep_time,
            additional_forbidden_tokens=additional_forbidden_tokens,
        )
        records_df = pd.DataFrame(records)
        print(classification_report(records_df["tgt_label_for_lsnr"], records_df["lsnr_pred"], zero_division=0))
        report_to_save = classification_report(records_df["tgt_label_for_lsnr"], records_df["lsnr_pred"], zero_division=0, output_dict=True)
        full_transcripts = {"spkr_hist": spkr_hist, "lsnr_hist": lsnr_hist}
        spkr_model_args = (spkr_model.model_args.__dict__ if spkr_model is not None else None)
        lsnr_model_args = (lsnr_model.model_args.__dict__ if lsnr_model is not None else None)
        report_to_save.update({
            "spkr_model_arg": spkr_model_args,
            "lsnr_model_args": lsnr_model_args,
            "spkr_exp_args": spkr_exp_args.__dict__,
            "lsnr_exp_args": lsnr_exp_args.__dict__,
        })
        if not DEBUG:
            dirname = os.path.basename(os.path.dirname(context_record_fp))
            parent_dirname = os.path.basename(os.path.dirname(os.path.dirname(context_record_fp)))
            dirname_save = os.path.join("output", parent_dirname, dirname)
            os.makedirs(dirname_save, exist_ok=True)
            if os.path.exists(f"{dirname_save}/records_{dtime}_{exp_name}_{save_suffix}_output.csv"):
                for i in range(1, 50):
                    if os.path.exists(f"{dirname_save}/records{i}_{dtime}_{exp_name}_{save_suffix}_output.csv"):
                        continue
                    else:
                        records_df.to_csv(f"{dirname_save}/records{i}_{dtime}_{exp_name}_{save_suffix}_output.csv", mode='x')
                        break
                if i == 50:
                    raise Exception("Too many files with the same name")
            else:
                records_df.to_csv(f"{dirname_save}/records_{dtime}_{exp_name}_{save_suffix}_output.csv", mode='x')
            with open(f"{dirname_save}/records_{dtime}_{exp_name}_{save_suffix}_report.json", "w") as f:
                json.dump(report_to_save, f, indent=4)
            with open(f"{dirname_save}/records_{dtime}_{exp_name}_{save_suffix}_full_transcripts.pickle", "wb") as f:
                pickle.dump(full_transcripts, f)
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spkr_model_type", type=str, default="Llama")
    parser.add_argument("--lsnr_model_type", type=str, default="GPT")
    parser.add_argument("--spkr_intro_version", type=str, default="standard")
    parser.add_argument("--lsnr_intro_version", type=str, default="standard_text_only")
    parser.add_argument("--sleep_time", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="ICCA_data_text_input/ICCA_data_text_game_daily_and_food")
    parser.add_argument("--spkr_api_fp", type=str, default=None)
    parser.add_argument("--lsnr_api_fp", type=str, default="APIs/openai_api.json")
    parser.add_argument("--spkr_model_ckpt", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--isLora", action="store_true")
    parser.add_argument("--lsnr_model_ckpt", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--spkr_intro_texts", type=str, default="args/intro_texts_spkr.json")
    parser.add_argument("--lsnr_intro_texts", type=str, default="args/intro_texts_lsnr.json")
    parser.add_argument("--lsnr_exp_args_fp", type=str, default="args/interaction_args_lsnr.json")
    parser.add_argument("--exp_name", type=str, default="refgame_eval")
    parser.add_argument("--num_of_trials", type=int, default=24)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--last_context_i", type=int, default=None)
    parser.add_argument("--first_context_i", type=int, default=0)
    parser.add_argument("--model_config_dir", type=str, default=None)
    parser.add_argument("--additional_forbidden_token_fp", type=str, default="additional_forbidden_tokens.json")

    args = parser.parse_args()


    with open(args.spkr_intro_texts, "r") as f:
        spkr_intro_texts = json.load(f)

    with open(args.lsnr_intro_texts, "r") as f:
        lsnr_intro_texts = json.load(f)

    spkr_intro_text = (
        None
        if args.spkr_model_type == "Human"
        else spkr_intro_texts[args.spkr_model_type][args.spkr_intro_version]
    )
    lsnr_intro_text = (
        None
        if args.lsnr_model_type == "oracle"
        else lsnr_intro_texts[args.lsnr_model_type][args.lsnr_intro_version]
    )

    with open(args.lsnr_exp_args_fp, "r") as f:
        lsnr_exp_args = json.load(f)

    spkr_exp_args = InteractionArgs(model_type=args.spkr_model_type)
    lsnr_exp_args = InteractionArgs(model_type=args.lsnr_model_type, **lsnr_exp_args)

    if args.spkr_api_fp:
        with open(args.spkr_api_fp) as f:
            spkr_api_info = json.load(f)
    
    if args.lsnr_api_fp:
        with open(args.lsnr_api_fp) as f:
            lsnr_api_info = json.load(f)
    

    
    print("initializing spkr model..")
    match spkr_exp_args.model_type:
        case "GPT":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                max_output_tokens=30,
                intro_text=spkr_intro_text,
            )
            spkr_model = GPTModel(spkr_model_args, **spkr_api_info)
        case "Claude":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                max_output_tokens=30,
                intro_text=spkr_intro_text,
            )
            spkr_model = ClaudeModel(spkr_model_args, **spkr_api_info)
        case "Llama":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                max_output_tokens=32,
                intro_text=spkr_intro_text,
                config_dir=args.model_config_dir
            )
            spkr_model = LlamaModel(spkr_model_args, isLora=args.isLora)
        case "Gemma":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                max_output_tokens=32,
                intro_text=spkr_intro_text,
                config_dir=args.model_config_dir
            )
            spkr_model = GemmaModel(spkr_model_args, isLora=args.isLora)
        case "Human":
            spkr_model = None

    print("initializing lsnr model..")
    match lsnr_exp_args.model_type:
        case "GPT":
            lsnr_model_args = ModelArgs(
                role="lsnr",
                model_ckpt=args.lsnr_model_ckpt,
                max_output_tokens=1,
                intro_text=lsnr_intro_text,
            )
            lsnr_model = GPTModel(lsnr_model_args, **lsnr_api_info)
        case "Claude":
            lsnr_model_args = ModelArgs(
                role="lsnr",
                model_ckpt=args.lsnr_model_ckpt,
                max_output_tokens=1,
                intro_text=lsnr_intro_text,
            )
            lsnr_model = ClaudeModel(lsnr_model_args, **lsnr_api_info)
        case "oracle":
            lsnr_model = None


    print(f"spkr: {spkr_exp_args.model_type} | lsnr: {lsnr_exp_args.model_type} | spkr_intro: {args.spkr_intro_version} | lsnr_intro: {args.lsnr_intro_version}", file=sys.stderr)
    save_suffix = f"{spkr_exp_args.model_type}_{args.spkr_intro_version}_{lsnr_exp_args.model_type}_{args.lsnr_intro_version}"
    test_dir = args.data_dir
    context_records_fps = glob.glob(f"{test_dir}/*/trials_for*.csv")
    context_records_fps.sort()
    dirname = os.path.dirname(context_records_fps[0])
    if not all([os.path.dirname(fp)==dirname for fp in context_records_fps]):
        input_dir_structure="folder_by_context"
    else:
        input_dir_structure="same_folder"
    dtime = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_test(
        context_records_fps,
        num_of_trials=args.num_of_trials,
        random_seed=args.seed,
        save_suffix=save_suffix,
        spkr_exp_args=spkr_exp_args,
        dtime=dtime,
        lsnr_exp_args=lsnr_exp_args,
        spkr_model=spkr_model,
        lsnr_model=lsnr_model,
        sleep_time=args.sleep_time,
        exp_name=args.exp_name,
        input_dir_structure=input_dir_structure,
        first_context_i=args.first_context_i,
        last_context_i=args.last_context_i,
        additional_forbidden_token_fp=args.additional_forbidden_token_fp
    )


if __name__ == "__main__":
    main()
