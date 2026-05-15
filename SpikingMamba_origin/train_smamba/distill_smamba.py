#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["WANDB_MODE"] = "offline"
import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, set_seed
from transformers import AutoTokenizer
sys.path.append("./alignment")
from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
#from trl import SFTTrainer, setup_chat_format
 
from trainer.mamba_kd_trainer import KDTrainer
from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import MambaConfig
 
from fla_mixer_seq.models.mixer_seq_simple_smamba import MambaLMHeadModel
from fla_mixer_seq.models.mixer_seq_simple import MambaLMHeadModel as Teacher_MambaLMHeadModel

from train_configs import SFTDistillConfig
from util import construct_layer_dict

import torch.distributed as dist
from datetime import timedelta
from datasets import config

logger = logging.getLogger(__name__)

def main():
    
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=360000))

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTDistillConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    )

    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained('/public/data/gpt-neox-20b')
    tokenizer.eos_token ="<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False,
        # use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # model = model_args.model_name_or_path
    # # For ChatML we need to add special tokens and resize the embedding layer
    # if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
    #     model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    #     model, tokenizer = setup_chat_format(model, tokenizer)
    #     model_kwargs = None

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )

    ##########################
    # Decontaminate benchmarks
    ##########################
    if training_args.decontaminate:
        num_raw_train_samples = len(raw_datasets["train"])
        raw_datasets = raw_datasets.filter(decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=data_args.preprocessing_num_workers)
        num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
        logger.info(
            f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
        )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    toks = tokenizer.tokenize(train_dataset[0]["text"])
    print('loaded token: {}'.format(toks))

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    attn_implementation="flash_attention_2"
    if not model_args.use_flash_attention_2:
        attn_implementation="eager"

    teacher_model = Teacher_MambaLMHeadModel.from_pretrained(model_args.model_name_or_path, device=training_args.device, dtype=torch.bfloat16)
    model = MambaLMHeadModel.from_pretrained(model_args.model_name_or_path, device=training_args.device, dtype=torch.bfloat16)


    for name, params in model.named_parameters():
        #print(name)
        if 'embedding' in name:
            params.requires_grad = False
            print('disable: {}'.format(name))
    # for name, m in model.named_modules():
    #     if 'spike_neuron' in name:
    #         m.eval()
    #         for n,p in m.named_parameters():
    #             p.requires_grad = False
    #             print('froze param: {}'.format(n))
    #         print('------------------')
    #         print('froze module: {}'.format(name))
    #         print('==================')

    # model.save_config(training_args.output_dir)
    # model = model.model
    # print("model:", model)

    ########################
    # Initialize the Trainer
    ########################
    trainer = KDTrainer(
        model=model,
        teacher_model=teacher_model,
        #model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_args),
        dataset_kwargs=training_args.dataset_kwargs,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train() #trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
