import yaml
import pandas as pd
import sys
import copy
import numpy as np
import sys
import argparse
import os




def getArgs():
	"""
	Establish argument parser and give script description
	"""
	parser = argparse.ArgumentParser(
    description = """This program splits train and testing data and creates the central learning file to compare and evaluate the results """)
	parser.add_argument("-p", type = str, help = "p is the path for the training data, e.g: C:/Documents/MICCAI_FeTS2021_TrainingData")
	parser.add_argument("-f", type = str, default ="partitioning_1.csv", help = "f is the csv file with the collaborators split. (default: %(default)s)")
	parser.add_argument("-s", type = float, default = 0.80, help = "s is the split percentage for training. (default: %(default)s)")
	# Get your arguments
	return parser.parse_args()


 
   
args = getArgs()
data_path = str(args.p)
file_name = str(args.f)
split_subdirs_path = data_path+"/"+file_name
percent_train = float(args.s)

from gandlf_csv_adapter import *
    
federated_simulation_train_val_csv_path = data_path+"/split.csv"
collaborator_names = construct_fedsim_csv(data_path, split_subdirs_path, percent_train, federated_simulation_train_val_csv_path)

#Split of train-val for all institutions
df_paths = pd.read_csv(data_path+"/split.csv")
df_paths.columns = ['0', '1', '2', '3', '4', '5', 'TrainOrVal', 'InstitutionName']
df_paths.to_csv('split_new.csv')

#Generate central learning data
df = pd.read_csv(data_path+"/split.csv")
headers = ['0', '1', '2', '3', '4', '5', 'TrainOrVal', 'InstitutionName']
data =df.iloc[:,:].values
for i in range(len(data)):
    data[:,-1]="1"
df_new = pd.DataFrame(data, index=None, columns = headers)
df_new.to_csv('split_single_new.csv')
