# Post-training for Efficient Communication via Convention Formation

This is the official repo for our paper: [arxiv link]

Yilun Hua, Evan Wang, and Yoav Artzi

## Prerequisites

This repo requires python>=3.11. 

To run model training, please install trl and peft via `pip install -r post_train_efficiency/requirements.txt`

To run evaluation, in the same environment as training, run `pip install -r post_train_efficiency/requirements.eval.txt`

For open-sourced models, you will need a huggingface account to accept their terms and conditions. Please check the respective pages of [Gemma](google/gemma-2-9b-it) and [Llama](https://huggingface.co/meta-llama/Llama-3.1-8B) on huggingface. For proprietary models, you will need API access for them. Please refer to their websites and documentions for API access. 

## Data Processing and Training

The `post_train_efficiency/post-train` folder provides the python scripts and example command for data processing and running the two training stages. 



## Evaluation

See `post_train_efficiency/refgame_eval` for running the text-only reference game evaluation. 

See `post_train_efficiency/doc_grounded_eval` for running the document-grounded evaluation. 



