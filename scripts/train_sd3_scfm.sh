export OUTPUT_DIR="output/sd3-scfm"
export MODEL_NAME="stabilityai/stable-diffusion-3.5-large"
export DATA_PATH="./sd3_dataset_1024"
export LORA_LAYERS="time_text_embed.timestep_embedder.linear_1,time_text_embed.timestep_embedder.linear_2,time_text_embed.text_embedder.linear_1,time_text_embed.text_embedder.linear_2,pos_embed.proj,context_embedder,norm1.linear,norm2.linear,attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.net.0.proj,ff.net.2,norm_out.linear,proj_out"
# export NEGA_PROMPT="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

# export EMA_RESTART=99999  more frequent restarts may lead to artifacts, disable this feature if you observe undesired results.
export EMA_RESTART=2000

acc_args=(
--num_machines 1 \
--dynamo_backend no \
--mixed_precision bf16 \
--num_cpu_threads_per_process 8 \
--num_processes 1 \
# --multi_gpu \
)

train_args=(
--pretrained_model_name_or_path $MODEL_NAME \
--train_data_path $DATA_PATH \
--train_batch_size 8 \
--gradient_accumulation_steps 1 \
--dataloader_num_workers 4 \
--resolution 512 \
--learning_rate 2e-5 \
--lr_warmup_steps 0 \
--lr_scheduler constant \
--mixed_precision bf16 \
--optimizer AdamW \
# --use_8bit_adam \
--max_grad_norm 2.0 \
--gradient_checkpointing \
--checkpointing_steps 200 \
--num_train_epochs 10000 \
--output_dir $OUTPUT_DIR \
--max_sequence_length 256 \
--lora_layers $LORA_LAYERS \
--rank 64 \
--rank_alpha 64 \
--t_skip 2 \
--teacher_min_timesteps 32 \
--teacher_max_timesteps 32 \
--guidance_scale 3.5 \
--ema_restart_steps $EMA_RESTART \
--teacher_ratio 0.4 \
--non_shortcut_ratio 0.0 \
)

export TOKENIZERS_PARALLELISM=False
nohup accelerate launch "${acc_args[@]}" trainer/sd3_scfm.py "${train_args[@]}" 2>&1 | tee output/train_sd3.log &