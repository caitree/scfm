export OUTPUT_DIR="output/flux-scfm"
export MODEL_NAME="black-forest-labs/FLUX.1-dev"
export DATA_PATH="./flux_dataset_1024"
export LORA_LAYERS="time_text_embed.timestep_embedder.linear_1,time_text_embed.timestep_embedder.linear_2,time_text_embed.text_embedder.linear_1,time_text_embed.text_embedder.linear_2,time_text_embed.guidance_embedder.linear_1,time_text_embed.guidance_embedder.linear_2,context_embedder,x_embedder,norm.linear,norm1.linear,norm1_context.linear,norm_out.linear,attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.net.0.proj,ff.net.2,ff_context.net.0.proj,ff_context.net.2,proj_mlp,proj_out"

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
--max_grad_norm 10.0 \
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
--teacher_ratio 0.4 \
--non_shortcut_ratio 0.0 \
)

export TOKENIZERS_PARALLELISM=False
nohup accelerate launch "${acc_args[@]}" trainer/flux_scfm.py "${train_args[@]}" 2>&1 | tee output/train.log &