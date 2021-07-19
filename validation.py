import yaml
import pandas as pd
import sys
import copy
import numpy as np
import sys
import argparse
import os

from gandlf_csv_adapter import *

def getArgs():
    """
    Establish argument parser and give script description
    """
    parser = argparse.ArgumentParser(
    description = """This program uses the training data split to evaluate the performance with travelling method 0 - ascendent order according to csv file; 1 - descendent order according to csv file; 2 - ascendent order according to # of samples; 3 - descendent order according to # of samples; 4 - random selection """)
    parser.add_argument("-c", type = str, help = "c is the path of the cpt, e.g: C:/Documents/MICCAI_FeTS2021_TrainingData/cpt/cl.path")
    parser.add_argument("-n", type = int, help = "n is the round number for validation")
    # Get your arguments
    return parser.parse_args()

def aggregated_validation(aggregate_model, round_num):
    val_dict = aggregate_model.validate(col_name = 1, round_num = round_num)
    for key, value in val_dict.items():
        print(key, ' : ', value)
    return val_dict

def load_data_to_validate():
    central_data_loader = FeTSChallengeDataLoader(
    data_path = str(int(1)),
    federated_simulation_train_val_csv_path = 'split_single_new.csv',
    training_batch_size = 1,
    q_samples_per_volume = 1,
    q_max_length = 1,
    patch_sampler = 'uniform',
    psize = [128, 128, 128],
    divisibility_factor= 16,
    data_usage = 'train-val',
    q_verbose = False,
    split_instance_dirname = 'fets_phase2_split_1',
    np_split_seed = 8950,
    allow_auto_split = True,
    class_list = ['4', '1||4', '1||2||4'],
    data_augmentation = {
      'noise' : 
        {
        'mean' : 0.0,
        'std' : 0.1,
        'probability' : 0.2
        }
      ,
      'rotate_90':
        {
        'axis' : [1, 2, 3],
        'probability' : 0.5
        }
      ,
      'rotate_180':
        {
        'axis': [1, 2, 3],
        'probability': 0.5
        },
      'flip':
        {
        'axis': [0, 1, 2],
        'probability': 1.0
        }
    },
    data_preprocessing = {
      'crop_external_zero_planes': None,
      'normalize_nonZero_masked': None
    },
    federated_simulation_institution_name = '__USE_DATA_PATH_AS_INSTITUTION_NAME__'                                                                                            
    )
    return central_data_loader


def load_model(data_loader):
    model = Model_Simple( data_loader = data_loader,
            base_filters = 30,
            lr = 0.005,
            loss_function = 'mirrored_brats_dice_loss',
            opt = 'adam',
            use_penalties = False,
            device = 'cuda',
            validate_with_fine_grained_dice = True,
            sigmoid_input_multiplier = 20.0,
            validation_function = 'fets_phase2_validation',
            validation_function_kwargs = { 
            'challenge_reduced_output': True
            }
        )
    return model

val_model = load_model(load_data_to_validate())
#load last cpt after traveling among the collaborators
val_model.load_cpt(str(args.c))
round_num = int(args.n)
metrics[round_num] = aggregated_validation(val_model,round_num)

print("METRIC")
print(metrics)
