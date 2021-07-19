import yaml
import pandas as pd
import sys
import copy
import numpy as np
import textwrap
import os
from gandlf_csv_adapter import *


round_num = sys.argv[1]
DATA_PATH = sys.argv[2] + 'MICCAI_FeTS2021_TrainingData/'
save_path = sys.argv[3]
col_lists = sys.argv[4]


pardir = DATA_PATH

aggregate_path = save_path+"|aggregate"+'|'+str(round_num)+'.pth'

def aggregation_function(vals):
    st = torch.stack(vals, dim = 0).float()
    return torch.mean(st, dim = 0)

def aggregated_validation(aggregate_model, round_num):
    val_dict = aggregate_model.validate(col_name = 1, round_num = round_num)
    for key, value in val_dict.items():
        print(key, ' : ', value)
    return val_dict

def aggregate(model_paths, round_num, aggregate_model, aggregate_path):
    state_dicts = []
    print('Loading Models from following path for aggregation')
    for model_path in model_paths:
        print(model_path)
        state_dicts.append(torch.load(model_path))
    mfd = state_dicts[0]
    for key in mfd:
        vals = []
        for state_dict in state_dicts:
            vals.append(state_dict[key])
        mfd[key] = aggregation_function(vals)  
    torch.save(mfd, aggregate_path)
    aggregate_model.load_cpt(aggregate_path)
    val_dict = aggregated_validation(aggregate_model, round_num)
    return val_dict

col_lists = textwrap.wrap(col_lists, 2)

print('Collaborators to be aggregated')
print(col_lists)

model_paths = []

for col in col_lists:
    model_paths.append(save_path+'|'+str(round_num)+'|'+col+'.pth')

split_subdirs_path = "/home/lakshayt/scratch/FETS/Notebook/Challenge/Task_1/openfl-workspace/fets_challenge_workspace/partitioning_single.csv"
percent_train = 0.8
federated_simulation_train_val_csv_path = "/home/lakshayt/scratch/FETS/Plain/split.csv"
data_config_path = "/home/lakshayt/scratch/FETS/Plain/split.csv"
collaborator_names = construct_fedsim_csv(pardir, split_subdirs_path, percent_train, federated_simulation_train_val_csv_path)
collaborator_data_loaders = {}


df_paths = pd.read_csv('split.csv')
df_paths.columns = ['0', '1', '2', '3', '4', '5', 'TrainOrVal', 'InstitutionName']
df_paths.to_csv('split_new.csv')



data_loader = FeTSChallengeDataLoader(
data_path = str(int('01')),
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



aggregate_model = Model_Simple( data_loader = data_loader,
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
print('Model Initialized')

print('******************************VALIDATION RESULTS FOR ROUND '+str(round_num)+'**********************************')
val_dict = aggregate(model_paths, round_num, aggregate_model, aggregate_path)





