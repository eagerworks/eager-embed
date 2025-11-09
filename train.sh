#!/bin/bash

# To limit the number of examples (useful if you have limited disk space):
# Add --max_train_samples 10000 to limit training examples
# Add --max_corpus_samples 50000 to limit corpus size (helps with 52GB issue)

export PYTHONPATH=/mnt/data/QWEN_EMBEDDINGS/tevatron/src:$PYTHONPATH

deepspeed --include localhost:0 --master_port 60000 --module tevatron.retriever.driver.train_mm \
  --deepspeed ds_zero0_config.json \
  --output_dir retriever-qwen3vl-colpali \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --lora \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --save_steps 25 \
  --save_total_limit 2 \
  --train_yaml dataset_config.yaml \
  --query_prefix "Query: " \
  --passage_prefix "" \
  --bf16 \
  --tf32 True \
  --pooling eos \
  --append_eos_token \
  --normalize \
  --temperature 0.02 \
  --per_device_train_batch_size 4 \
  --gradient_checkpointing \
  --train_group_size 4 \
  --learning_rate 1e-4 \
  --query_max_len 512 \
  --passage_max_len 512 \
  --num_train_epochs 1 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --warmup_ratio 0.005 \
  --dataloader_num_workers 4 \
  --attn_implementation flash_attention_2 \
  #--max_train_samples 10000 \
  #--max_corpus_samples 1000 \