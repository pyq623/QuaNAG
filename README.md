# QuaNAG: Architecture and Implementation

## Overview

QuaNAG primarily consists of two core components: **Quantization-Branched MCTS** and **Constraint-to-Quality Curriculum**.

## Repository Structure

### Algorithm Implementation

The complete algorithmic implementation of our approach is contained within the `Quantization-Branched MCTS` directory, which houses all core algorithm code.

### Progressive Dataset Construction

The `QuaNAG` directory contains the implementation for Progressive Dataset construction, with the primary codebase concentrated in the `data_constrution` folder.

## Model Training Configurations and Tokenizers

Training configurations and tokenizer files for QuaNAG are provided in the following directory:

LLaMA-Factory/saves/Llama-3.1-8B/lora

Each benchmark-specific directory includes LoRA adapter configurations, training hyperparameters, and tokenizer files.

For Mhealth, the trained LoRA adapter is also available on [ModelScope](https://modelscope.cn/models/PuppeyYard/Llama-3.1-QuaNAG-Mhealth-LoRA), together with training configurations, tokenizer files, and loading instructions. The adapter is used with the `meta-llama/Llama-3.1-8B` base model.

### Benchmark-Specific Directories

| Directory | Benchmark |
|-----------|-----------|
| `train_2026-01-03-15-22-33` | Mhealth |
| `train_2026-01-07-12-41-41` | MMAct |
| `train_2026-01-13-02-17-52` | Wharf |
| `train_2026-01-13-09-20-52` | USCHAD |
| `train_2026-01-14-09-33-03` | UTD-MHAD |
