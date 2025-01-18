#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:35:28 2023

@author: umbertocappellazzo
"""

import torch
from torch.optim import AdamW
from src.AST import AST
from src.AST_LoRA import AST_LoRA, AST_LoRA_ablation
from src.AST_adapters import AST_adapter, AST_adapter_ablation
from src.Wav2Vec_adapter import Wav2Vec, Wav2Vec_adapter
from src.AST_prompt_tuning import AST_Prefix_tuning, PromptAST, Prompt_config
from src.MoA import AST_MoA, AST_SoftMoA
from dataset.fluentspeech import FluentSpeech
from dataset.esc_50 import ESC_50
from dataset.urban_sound_8k import Urban_Sound_8k
from dataset.google_speech_commands_v2 import Google_Speech_Commands_v2
from dataset.iemocap import IEMOCAP
from utils.engine import eval_one_epoch, train_one_epoch
from torch.utils.data import DataLoader
import wandb
import argparse
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
import time
import datetime
import yaml
import os
import copy
from orion.client import report_objective
from types import SimpleNamespace


global args
args = {
    'data_path': '$SLURM_TMPDIR/',
    'seed': 10,
    'device': 'cuda',
    'num_workers': 4,
    'model_ckpt_AST': 'MIT/ast-finetuned-audioset-10-10-0.4593',
    'model_ckpt_wav': 'facebook/wav2vec2-base-960h',
    'max_len_audio': 128000,
    'save_best_ckpt': False,
    'output_path': '/checkpoints',
    'is_AST': True,
    'dataset_name': 'ESC-50',
    'method': 'adapter',
    'seq_or_par': 'parallel',
    'reduction_rate_adapter': 96,
    'adapter_type': 'Pfeiffer',
    'apply_residual': False,
    'adapter_block': 'conformer',
    'kernel_size': 31,
    'is_adapter_ablation': False,
    'befafter': 'after',
    'location': 'FFN',
    'reduction_rate_moa': 128,
    'adapter_type_moa': 'Pfeiffer',
    'location_moa': 'MHSA',
    'adapter_module_moa': 'bottleneck',
    'num_adapters': 7,
    'num_slots': 1,
    'normalize': False,
    'reduction_rate_lora': 64,
    'alpha_lora': 8,
    'is_lora_ablation': False,
    'lora_config': 'Wq,Wv',
    'prompt_len_pt': 24,
    'prompt_len_prompt': 25,
    'is_deep_prompt': True,
    'drop_prompt': 0.,
    'is_few_shot_exp': False,
    'few_shot_samples': 64,
    'use_wandb': True,
    'project_name': 'ASTAdapter',
    'exp_name': '',
    'entity': 'salmanhussainali03-concordia-university',
    'cache_dir' : '/nfs/speed-scratch/s_ssaina'
}


def get_model_details(model):
    n_parameters = sum(p.numel() for p in model.parameters())
    print('Number of params of the model:', n_parameters)
    n_train_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_head_parameters = model.classification_head.weight.numel()
    if model.classification_head.bias is not None:
        n_head_parameters += model.classification_head.bias.numel()
    print('Number of params in classification_head:', n_head_parameters)

    n_train_parameters -= n_head_parameters

    print('Number of trainable params of the model:', n_train_parameters)

    return n_parameters, n_train_parameters


def get_wandb_config():
    return {
        key: getattr(args, key)
        for key in vars(args)
        if not key.startswith('_') and key not in ['use_wandb', 'project_name', 'exp_name', 'entity']
    }


os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_ENTITY"] = "salmanhussainali03-concordia-university"

def train(sweep_config=None):
    start_time = time.time()
    global args

    print(sweep_config)
    print(f'Args: {args}')

    config = sweep_config
    for key, value in config.items():
            if key in args:
                args[key] = value
    
    sweep_config["seq_or_par"] = args["seq_or_par"]

    args = SimpleNamespace(**args)
    

    with wandb.init(project=args.project_name, name=args.adapter_block, entity='salmanhussainali03-concordia-university',tags=[args.dataset_name, args.seq_or_par, args.adapter_block, args.adapter_type], config=sweep_config):
        print(config)


        mamba_config = config['mamba_parameters']

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        device = torch.device(args.device)

        # Fix the seed for reproducibility (if desired).
        if args.seed:
            seed = args.seed
            torch.manual_seed(seed)
            np.random.seed(seed)

        with open('hparams/train.yaml', 'r') as file:
            train_params = yaml.safe_load(file)

        if args.dataset_name == 'FSC':
            max_len_AST = train_params['max_len_AST_FSC']
            num_classes = train_params['num_classes_FSC']
            batch_size = train_params['batch_size_FSC']
            epochs = train_params['epochs_FSC_AST'] if args.is_AST else train_params['epochs_FSC_WAV']
        elif args.dataset_name == 'ESC-50':
            max_len_AST = train_params['max_len_AST_ESC']
            num_classes = train_params['num_classes_ESC']
            batch_size = train_params['batch_size_ESC']
            epochs = train_params['epochs_ESC_AST']
        elif args.dataset_name == 'urbansound8k':
            max_len_AST = train_params['max_len_AST_US8K']
            num_classes = train_params['num_classes_US8K']
            batch_size = train_params['batch_size_US8K']
            epochs = train_params['epochs_US8K']
        elif args.dataset_name == 'GSC':
            max_len_AST = train_params['max_len_AST_GSC']
            num_classes = train_params['num_classes_GSC']
            batch_size = train_params['batch_size_GSC']
            epochs = train_params['epochs_GSC_AST'] if args.is_AST else train_params['epochs_GSC_WAV']
        elif args.dataset_name == 'IEMOCAP':
            max_len_AST = train_params['max_len_AST_IEMO']
            num_classes = train_params['num_classes_IEMO']
            batch_size = train_params['batch_size_IEMO']
            epochs = train_params['epochs_IEMO']
        else:
            raise ValueError('The dataset you chose is not supported as of now.')
        
        final_output = train_params['final_output']
        accuracy_folds = []
        
        if args.dataset_name in ['FSC', 'GSC']:
            fold_number = 1
        elif args.dataset_name == 'ESC-50':
            fold_number = 5
            folds_train = [[1,2,3], [2,3,4], [3,4,5], [4,5,1], [5,1,2]]
            folds_valid = [[4], [5], [1], [2], [3]]
            folds_test = [[5], [1], [2], [3], [4]]
        elif args.dataset_name == 'urbansound8k':
            fold_number = 10
            folds_train = [[1,2,3,4,5,6,7,8,9], [2,3,4,5,6,7,8,9,10], [3,4,5,6,7,8,9,10,1],
                           [4,5,6,7,8,9,10,1,2], [5,6,7,8,9,10,1,2,3], [6,7,8,9,10,1,2,3,4],
                           [7,8,9,10,1,2,3,4,5], [8,9,10,1,2,3,4,5,6], [9,10,1,2,3,4,5,6,7], [10,1,2,3,4,5,6,7,8]]
            folds_test = [[10], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        else:
            assert args.dataset_name == 'IEMOCAP'
            fold_number = 10
            sessions_train = [[1,2,3,4], [1,2,3,4], [2,3,4,5], [2,3,4,5], [3,4,5,1], [3,4,5,1], [4,5,1,2], [4,5,1,2], [5,1,2,3], [5,1,2,3]]
            session_val = [[5], [5], [1], [1], [2], [2], [3], [3], [4], [4]]
            speaker_id_val = ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M']
            speaker_id_test = ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']


        print("Loading dataset")
        for fold in range(0,fold_number):

            # DATASETS
            if args.dataset_name == 'FSC':
                train_data = FluentSpeech(args.data_path, max_len_AST, train= True, apply_SpecAug= False, few_shot= args.is_few_shot_exp, samples_per_class= args.few_shot_samples)
                val_data = FluentSpeech(args.data_path, max_len_AST, train= "valid")
                test_data = FluentSpeech(args.data_path, max_len_AST, train= False)
            elif args.dataset_name == 'ESC-50':
                train_data = ESC_50(args.data_path, max_len_AST, 'train', train_fold_nums= folds_train[fold], valid_fold_nums= folds_valid[fold], test_fold_nums= folds_test[fold], apply_SpecAug= True, few_shot= args.is_few_shot_exp, samples_per_class= args.few_shot_samples)
                val_data = ESC_50(args.data_path, max_len_AST, 'valid', train_fold_nums= folds_train[fold], valid_fold_nums= folds_valid[fold], test_fold_nums= folds_test[fold])
                test_data = ESC_50(args.data_path, max_len_AST, 'test', train_fold_nums= folds_train[fold], valid_fold_nums= folds_valid[fold], test_fold_nums= folds_test[fold])
            elif args.dataset_name == 'urbansound8k':
                train_data = Urban_Sound_8k(args.data_path, max_len_AST, 'train', train_fold_nums= folds_train[fold], test_fold_nums= folds_test[fold], apply_SpecAug=True, few_shot=args.is_few_shot_exp, samples_per_class= args.few_shot_samples,)
                test_data = Urban_Sound_8k(args.data_path, max_len_AST, 'test', train_fold_nums= folds_train[fold], test_fold_nums= folds_test[fold])
            elif args.dataset_name == 'GSC':
                train_data = Google_Speech_Commands_v2(args.data_path, max_len_AST, 'train', apply_SpecAug= False, few_shot= args.is_few_shot_exp, samples_per_class= args.few_shot_samples)
                val_data = Google_Speech_Commands_v2(args.data_path, max_len_AST, 'valid')
                test_data = Google_Speech_Commands_v2(args.data_path, max_len_AST, 'test')
            else:
                train_data = IEMOCAP(args.data_path, args.max_len_audio, max_len_AST, sessions= sessions_train[fold], speaker_id= 'both', is_AST = args.is_AST, apply_SpecAug=False, few_shot=args.is_few_shot_exp, samples_per_class=args.few_shot_samples)
                val_data = IEMOCAP(args.data_path, args.max_len_audio, max_len_AST, sessions= session_val[fold], speaker_id= speaker_id_val[fold], is_AST = args.is_AST)
                test_data = IEMOCAP(args.data_path, args.max_len_audio, max_len_AST, sessions= session_val[fold], speaker_id= speaker_id_test[fold], is_AST = args.is_AST)

            train_loader = DataLoader(train_data, batch_size= batch_size, shuffle= True, num_workers= args.num_workers, pin_memory= True, drop_last= False,)
            test_loader = DataLoader(test_data, batch_size= batch_size, shuffle= False, num_workers= args.num_workers, pin_memory= True, drop_last= False,)

            if args.dataset_name != 'urbansound8k': # US8K does not have the validation set.
                val_loader = DataLoader(val_data, batch_size= batch_size, shuffle= False, num_workers= args.num_workers, pin_memory= True, drop_last= False,)

            # MODEL definition.
            print("Loading Model")
            print(mamba_config)
            
            if args.method == "adapter":
                model = AST_adapter(max_length=max_len_AST, num_classes=num_classes, final_output=final_output,
                                reduction_rate=args.reduction_rate_adapter, adapter_type=args.adapter_type,
                                seq_or_par=args.seq_or_par, apply_residual=args.apply_residual,
                                adapter_block=args.adapter_block, kernel_size=args.kernel_size,
                                model_ckpt=args.model_ckpt_AST, mamba_config=mamba_config, cache_dir=args.cache_dir).to(device)
                lr = config["learning_rate"]
            if args.method == "LoRA":
                model = AST_LoRA(max_length= max_len_AST, num_classes= num_classes, final_output= final_output, rank= args.reduction_rate_lora, alpha= args.alpha_lora, model_ckpt= args.model_ckpt_AST).to(device)
                lr = train_params['lr_LoRA']
            #lr = train_params['lr_adapter']

            # PRINT MODEL PARAMETERS
            if fold == 0:
                n_params, n_trainable_params = get_model_details(model)

            optimizer = AdamW(model.parameters(), lr= lr ,betas= (0.9,0.98),eps= 1e-6, weight_decay= train_params['weight_decay'])

            criterion = torch.nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*(epochs))

            print(f"Learning Rate: {lr}")
            print(f"Start training for {epochs} epochs")

            best_acc = 0.

            for epoch in range(epochs):
                train_loss, train_acc= train_one_epoch(model, train_loader, optimizer, scheduler, device, criterion)
                print(f"Train loss at epoch {epoch}: {train_loss}")

                if args.dataset_name == 'urbansound8k':
                    val_loss, val_acc = eval_one_epoch(model, test_loader, device, criterion)
                else:
                    val_loss, val_acc = eval_one_epoch(model, val_loader, device, criterion)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_params = model.state_dict()

                    if args.save_best_ckpt:
                        torch.save(best_params, os.getcwd() + args.output_path + f'/bestmodel_fold{fold}')

                print("Train intent accuracy: ", train_acc*100)
                print("Valid intent accuracy: ", val_acc*100)

                current_lr = optimizer.param_groups[0]['lr']
                print('Learning rate after initialization: ', current_lr)

                wandb.log({"train_loss": train_loss, "valid_loss": val_loss,
                           "train_accuracy": train_acc, "val_accuracy": val_acc,
                           "lr": current_lr, }
                          )

            best_model = copy.copy(model)
            best_model.load_state_dict(best_params)

            test_loss, test_acc = eval_one_epoch(model, test_loader, device, criterion)

            accuracy_folds.append(test_acc)

        wandb.log({f"Avg Accuracy over {fold_number} folds": np.mean(accuracy_folds),
                   f"Std accuracy over the {fold_number} fold(s): ": np.std(accuracy_folds)})

        print("Folds accuracy: ", accuracy_folds)
        print(f"Avg accuracy over the {fold_number} fold(s): ", np.mean(accuracy_folds))
        print(f"Std accuracy over the {fold_number} fold(s): ", np.std(accuracy_folds))

        print(model)
        
        percent_trained = n_trainable_params/n_params * 100

        wandb.log({"# of params": n_params,
                   "# of trainable params": n_trainable_params,
                   "% params trained": percent_trained,
        })

        report_objective(1-np.mean(accuracy_folds))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--reduction_rate_adapter", type=int, default=64)
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--adapter_block", type=str, default='')
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--adapter_type", type=str, default='Pfeiffer')
    parser.add_argument("--kernel_size", type=int, default=31)
    parser.add_argument("--dataset_name", type=str, default='ESC-50')
    parser.add_argument("--method", type=str, default='adapter')
    parser_args = parser.parse_args()
    
    params = {
        "learning_rate": parser_args.learning_rate,
        "weight_decay": parser_args.weight_decay,
        "batch_size": parser_args.batch_size,
        "mamba_parameters": {
            "d_state": parser_args.d_state,
            "d_conv": parser_args.d_conv,
            "expand": parser_args.expand
        },
        "reduction_rate_adapter": parser_args.reduction_rate_adapter,
        "data_path": parser_args.data_path,
        "adapter_block": parser_args.adapter_block,
        "seed": parser_args.seed,
        "adapter_type": parser_args.adapter_type,
        "kernel_size": parser_args.kernel_size,
        "dataset_name": parser_args.dataset_name,
        "method": parser_args.method,
    }

    train(params)

