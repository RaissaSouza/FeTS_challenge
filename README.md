# FeTS_challenge

In order to run single weight transfer and cyclical weight transfer follow the steps:

# 1) setup the environment according to the challenge's page: https://github.com/FETS-AI/Challenge/tree/main/Task_1
# 2) run python split_data.py -p "<path for dataset>" -f "<name of partitioning csv file>" -s "<percentage for training>"
#You can check the parameters and their default value running python split_data.py -h for help
-> Data split csvs will be generated for central learning split_single_new.csv and for the travelling models split_new.csv

# 3) run python central_learning.py -o <output path for cpt> -e <epoch per round>
#You can check the parameters and their default value running python central_learning.py -h for help
-> metrics will be displayed once training process and validation is done

# 4) run python travel_main.py -o <output path for cpt> -t <type of travelling>-e <epoch per round>
#You can check the parameters and their default value running python travel_main.py -h for help
#Type of travelling =  order the model will be sent to the next collaborator
0 - ascendent order according to csv file; 
1 - descendent order according to csv file; 
2 - ascendent order according to # of samples; 
3 - descendent order according to # of samples; 
4 - random selection (use seed 0)
-> metrics for each cycle form 1 to 5 will be displayed once training process and validation is done

#CPT file will be saved in output directory with the name cpt_<roundNumber>_<collaboratorID>.pth e.g cpt_0_17.pth for first cycle and collaborator id 17

# 5) To generate the segmented lables in the unseen dataset run python inference.py -d <directory path of validation data> -m <path of model to load e.g cpt file path> -o <directory to save the segmentations>
#You can check the parameters and their default value running python inference.py -h for help

# EXTRA

-> travel_main_last_round.py and travel_main_two_last_round.py can be used if you run out of time in a job and you do not need to retrain everything you can restart from cycle 4 or 5 depending on the last completed round that your trained
#check -h for help and to verify the parameters

-> validation.py can be used to load validation data from the train-val split and a model using cpt to generate the evaluation metrics
#check -h for help and to verify the parameters
