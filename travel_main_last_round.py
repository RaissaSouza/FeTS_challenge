import yaml
import pandas as pd
import sys
import copy
import numpy as np
import sys
import argparse
import os

from gandlf_csv_adapter import *

def aggregated_validation(aggregate_model, round_num):
    val_dict = aggregate_model.validate(col_name = 1, round_num = round_num)
    for key, value in val_dict.items():
        print(key, ' : ', value)
    return val_dict
    
def getArgs():
    """
    Establish argument parser and give script description
    """
    parser = argparse.ArgumentParser(
    description = """This program uses the training data split to evaluate the performance with travelling method 0 - ascendent order according to csv file; 1 - descendent order according to csv file; 2 - ascendent order according to # of samples; 3 - descendent order according to # of samples; 4 - random selection """)
    parser.add_argument("-o", type = str, help = "o is the path to save the cpt, e.g: C:/Documents/MICCAI_FeTS2021_TrainingData/cpt")
    parser.add_argument("-c", type = str, help = "c is the last cpt to resume, e.g: C:/Documents/MICCAI_FeTS2021_TrainingData/cpt/cpt_0_1")
    parser.add_argument("-t", type = int, default = 0, help = "t is the type of travelling (default: %(default)s)")
    parser.add_argument("-e", type = float, default = 0.50, help = "e is the number of epochs per round (default: %(default)s)")
    # Get your arguments
    return parser.parse_args()
    
def getCollaboratorsName(t):
    df = pd.read_csv('split_new.csv')
    col_id =df.iloc[:,-1].values
    unique = np.unique(col_id)
    col_names = {}
    if(t==0):
        print("Asc order according to csv file")
        col_names = list(sorted(unique))
    elif(t==1):
        print("Desc order according to csv file")
        col_names = list(sorted(unique, reverse=True))
    elif(t==2):
        print("Asc order according to sample size")
        unique_elements, frequency = np.unique(col_id, return_counts=True)
        sorted_indexes = np.argsort(frequency)[::-1]
        sorted_by_freq = unique_elements[sorted_indexes]
        col_names = sorted_by_freq.tolist()
    elif(t==3):
        print("Desc order according to sample size")
        unique_elements, frequency = np.unique(col_id, return_counts=True)
        sorted_indexes = np.argsort(frequency)[::-1]
        sorted_by_freq = unique_elements[sorted_indexes]
        col_names = sorted_by_freq.tolist()
        col_names.reverse()
    else:
        print("Random selection")
        np.random.seed(0)  
        np.random.shuffle(unique)
        col_names = unique.tolist()
    return col_names
    
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

args = getArgs()
cpt_dir = str(args.o)+"/"
e = float(args.e)
t = int(args.t)

params_dict = {'epochs_per_round': e, 'num_batches': None , 'learning_rate':0.05}
TOTAL_ROUNDS = 5
collaborator_data_loaders = {}
collaborator_names = getCollaboratorsName(t)

print("Collaborators")
print(collaborator_names)

for col in collaborator_names:

    collaborator_data_loaders[col]  = FeTSChallengeDataLoader(
    data_path = str(int(col)),
    federated_simulation_train_val_csv_path = 'split_new.csv',
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

metrics = {}
training_loss = []


print('------------Travel Model Initialized -----------------')

for round_num in range(4,TOTAL_ROUNDS): #cycles
    print("ROUND "+str(round_num))
    save_paths = []
    save_paths.append(str(args.c))
    for col in collaborator_names:
        data_loader = collaborator_data_loaders[col]
        #iniciate model with colaborator FIX ME separete data loader from model
        model = load_model(data_loader)
        if len(save_paths)>0:
            #load previous one
            print("-----LOADING CPT-------"+save_paths[-1])
            model.load_cpt(save_paths[-1])
        new_save_path, loss = model.train_batches(col_name = col, round_num = round_num, params_dict = params_dict, save_path = cpt_dir+'cpt_'+str(round_num)+'_'+str(int(col))+'.pth')
        save_paths.append(new_save_path)
        training_loss.append(loss)
    
    #load model with data to validate
    val_model = load_model(load_data_to_validate())
    #load last cpt after traveling among the collaborators
    val_model.load_cpt(save_paths[-1])
    metrics[round_num] = aggregated_validation(val_model,round_num)
 
     

print('Model Trained')

print("LOSS")
print(training_loss)
print("METRIC")
print(metrics)