import pandas as pd
import ast
import argparse


def parse_with_literal_eval(malformed_json_str):
    parsed_data = ast.literal_eval(malformed_json_str)
    return parsed_data

def gpt_out_postprocess(gpt_output_df):
    rectified_winners=[]
    
    for idx, row in gpt_output_df.iterrows():
        if row['winner'] in {'tie', 'cannot decide', 'invalid completion output'}:
            rectified_winners.append(row['winner'])
            continue
        
        json_string=row['evaluator_pred']
        evaluator_pred_json=parse_with_literal_eval(json_string)

        extracted_mention_A=evaluator_pred_json['re-mention in Completion A']
        extracted_mention_B=evaluator_pred_json['re-mention in Completion B']
       
        len_diff=abs(len(extracted_mention_A)-len(extracted_mention_B))

        if extracted_mention_A=='NA' or extracted_mention_B=='NA':
            rectified_winner='cannot decide'

        # #if the length difference between them is less than 3, then it's a tie
        elif len_diff<3:
            rectified_winner='tie'
        else:
            rectified_winner=row['winner']

        rectified_winners.append(rectified_winner)

    return rectified_winners

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess GPT judge results.")
    parser.add_argument("gpt_judge_output_fp", type=str, help="Path to the GPT judge output CSV file")
    args = parser.parse_args()

    gpt_output_df=pd.read_csv(args.gpt_judge_output_fp)
    gpt_output_df['rectified_winner']=gpt_out_postprocess(gpt_output_df)
    print(gpt_output_df["winner"].value_counts())
    print('-'*30)
    print(gpt_output_df["rectified_winner"].value_counts())

    model_win_count=gpt_output_df["rectified_winner"].value_counts().get("model", 0)
    baseline_win_count=gpt_output_df["rectified_winner"].value_counts().get("baseline", 0)
    tie_count=gpt_output_df["rectified_winner"].value_counts().get("tie", 0)
    total = model_win_count + baseline_win_count + tie_count
    print("competence rate: ", (model_win_count+tie_count)/total)