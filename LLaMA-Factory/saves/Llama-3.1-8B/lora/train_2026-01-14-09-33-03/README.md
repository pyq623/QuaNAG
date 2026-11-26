---
library_name: peft
license: other
base_model: /root/llama_fine/llama3.1-8b
tags:
- base_model:adapter:/root/llama_fine/llama3.1-8b
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: train_2026-01-14-09-33-03
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2026-01-14-09-33-03

This model is a fine-tuned version of [/root/llama_fine/llama3.1-8b](https://huggingface.co//root/llama_fine/llama3.1-8b) on the dpo_UTD-MHAD dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-07
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- total_eval_batch_size: 8
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 30
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Framework versions

- PEFT 0.17.1
- Transformers 4.56.2
- Pytorch 2.4.0
- Datasets 4.0.0
- Tokenizers 0.22.1
