dataset_name="sft_data"
learning_rate=1e-4
jsd_weight=10.0
save_name="sft_llama"

python sft.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --dataset_name preference_data/${dataset_name} \
    --learning_rate ${learning_rate} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --output_dir out/SFT_output_${save_name} \
    --gradient_checkpointing \
    --save_strategy "steps" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --num_train_epochs 1 \
    --save_steps 100 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --peft_on_token_embed True \
    --train_on_new_embed_only True \
    --use_custom_trainer True \
    --kl_loss True \
    --jsd_weight ${jsd_weight} \
    --loss_on_special_token_only True \