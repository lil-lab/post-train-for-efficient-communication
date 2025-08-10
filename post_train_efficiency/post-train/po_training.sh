dataset_name="dpo_preference_data"
model_name="path_to_sft_model"
learning_rate=1e-5
beta=0.1
loss_type="apo_zero"

python dpo.py \
    --dataset_name preference_data/${dataset_name} \
    --model_name_or_path ${model_name} \
    --learning_rate ${learning_rate} \
    --warmup_steps 100 \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --output_dir out/DPO_out \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --beta ${beta} \
    --eval_strategy steps \
    --eval_steps 500 \
    --loss_type ${loss_type} \
    --max_steps 4000 \