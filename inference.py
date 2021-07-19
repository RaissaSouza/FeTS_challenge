import pandas as pd
import sys
import os
import numpy as np
import torch
import torchio
import SimpleITK as sitk
import sys
import argparse
import os

#print('torchio data')
#print(torchio.DATA)

from gandlf_csv_adapter import *

def getArgs():
    """
    Establish argument parser and give script description
    """
    parser = argparse.ArgumentParser(
    description = """This program uses the validation data and a given model to do brain segmentation that will be sent to FeTs challenge to get evaluated """)
    parser.add_argument("-d", type = str, help = "d is the path to validation dataset, e.g: C:/Documents/MICCAI_FeTS2021_TrainingData/")
    parser.add_argument("-m", type = str, help = "m is the path for the model to load, e.g: C:/Documents/MICCAI_FeTS2021_TrainingData/cpt/cpt_0_1")
    parser.add_argument("-o", type = str, help = "o is the output path, e.g: C:/Documents/inferences")
    # Get your arguments
    return parser.parse_args()


args = getArgs()

DATA_PATH = str(args.d)
model_path = str(args.m)
output_path = str(args.o)

def nan_check(tensor, tensor_description):
    tensor = tensor.cpu()
    if torch.any(torch.isnan(tensor)):
        raise ValueError("A " + tensor_description + " was found to have nan values.")


def binarize(array, threshold=0.5):
    """
    Get binarized output using threshold. 
    """
    
    if (np.amax(array) > 1.0) or (np.amin(array) < 0.0):
        raise ValueError('Voxel value fed to lambda in converting to original labels was out of range.')
    
    # obtain binarized output
    binarized = array.copy()
    zero_mask = (binarized <= threshold)
    binarized[zero_mask] = 0.0
    binarized[~zero_mask] = 1.0
    
    return binarized


def get_binarized_and_belief(array, threshold=0.5):
    
    """
    Get binarized output using threshold and report the belief values for each of the
    channels along the last axis. Belief value is a list of indices along the last axis, and is 
    determined by the order of how close (closest first) each binarization is from its original in this last axis
    so belief should have the same shape as arrray.
    Assumptions:
    - array is float valued in the range [0.0, 1.0]
    """
    
    # check assumption above
    if (np.amax(array) > 1.0) or (np.amin(array) < 0.0):
        raise ValueError('Voxel value fed to lambda in converting to original labels was out of range.')
        
    # obtain binarized output
    binarized = binarize(array=array, threshold=threshold)
    
    # we will sort from least to greatest, so least suspicion is what we will believe
    raw_suspicion = np.absolute(array - binarized)
    
    belief = np.argsort(raw_suspicion, axis=-1)
    
    return binarized, belief


def replace_initializations(done_replacing, array, mask, replacement_value, initialization_value):
    """
    Replace in array[mask] intitialization values with replacement value, 
    ensuring that the locations to replace all held initialization values
    """
    
    # sanity check
    if np.any(mask) and done_replacing:
        raise ValueError('Being given locations to replace and yet told that we are done replacing.')
        
    # check that the mask and array have the same shape
    if array.shape != mask.shape:
        raise ValueError('Attempting to replace using a mask shape: {} not equal to the array shape: {}'.format(mask.shape, array.shape))
    
    # check that the mask only points to locations with initialized values
    if np.any(array[mask] != initialization_value):
        raise ValueError('Attempting to overwrite a non-initialization value.')
        
    array[mask] = replacement_value
    
    done_replacing = np.all(array!=initialization_value)
    
    return array, done_replacing


def check_subarray(array1, array2):
    """
    Checks to see where array2 is a subarray of array1.
    Assumptions:
    - array2 has one axis and is equal in length to the last axis of array1
    """
    
    # check assumption
    if (len(array2.shape) != 1) or (array2.shape[0] != array1.shape[-1]):
        raise ValueError('Attempting to check for subarray equality when shape assumption does not hold.')
        
    return np.all(array1==array2, axis=-1)


def convert_to_original_labels(array, threshold=0.5, initialization_value=999):
    """
    array has float output in the range [0.0, 1.0]. 
    Last three channels are expected to correspond to ET, TC, and WT respecively.
    
    """
    
    binarized, belief = get_binarized_and_belief(array=array, threshold=threshold)
    
    #sanity check
    if binarized.shape != belief.shape:
        raise ValueError('Sanity check did not pass.')
        
    # initialize with a crazy label we will be sure is gone in the end
    slice_all_but_last_channel = tuple([slice(None) for _ in array.shape[:-1]] + [0])
    original_labels = initialization_value * np.ones_like(array[slice_all_but_last_channel])
    
    # the outer keys correspond to the binarized values
    # the inner keys correspond to the order of indices comingn from argsort(ascending) on suspicion, i.e. 
    # how far the binarized sigmoid outputs were from the original sigmoid outputs 
    #     for example, (2, 1, 0) means the suspicion from least to greatest was: 'WT', 'TC', 'ET'
    #     (recall that the order of the last three channels is expected to be: 'ET', 'TC', and 'WT')
    mapper = {(0, 0, 0): 0, 
              (1, 1, 1): 4,
              (0, 1, 1): 1,
              (0, 0, 1): 2,
              (0, 1, 0): {(2, 0, 1): 0,
                          (2, 1, 0): 0, 
                          (1, 0, 2): 1,
                          (1, 2, 0): 1,
                          (0, 2, 1): 0,
                          (0, 1, 2): 1}, 
              (1, 1, 0): {(2, 0, 1): 0,
                          (2, 1, 0): 0, 
                          (1, 0, 2): 4,
                          (1, 2, 0): 4,
                          (0, 2, 1): 4,
                          (0, 1, 2): 4},
              (1, 0, 1): {(2, 0, 1): 4,
                          (2, 1, 0): 2, 
                          (1, 0, 2): 2,
                          (1, 2, 0): 2,
                          (0, 2, 1): 4,
                          (0, 1, 2): 4}, 
              (1, 0, 0): {(2, 0, 1): 0,
                          (2, 1, 0): 0, 
                          (1, 0, 2): 0,
                          (1, 2, 0): 0,
                          (0, 2, 1): 4,
                          (0, 1, 2): 4}}
    
    
    
    done_replacing = False
    
    for binary_key, inner in mapper.items():
        mask1 = check_subarray(array1=binarized, array2=np.array(binary_key))
        if isinstance(inner, int):
            original_labels, done_replacing = replace_initializations(done_replacing=done_replacing, 
                                                                      array=original_labels, 
                                                                      mask=mask1, 
                                                                      replacement_value=inner, 
                                                                      initialization_value=initialization_value)
        else:
            for inner_key, inner_value in inner.items():
                mask2 = np.logical_and(mask1, check_subarray(array1=belief, array2=np.array(inner_key)))
                original_labels, done_replacing = replace_initializations(done_replacing=done_replacing,
                                                                          array=original_labels, 
                                                                          mask=mask2, 
                                                                          replacement_value=inner_value, 
                                                                          initialization_value=initialization_value)
        
    if not done_replacing:
        raise ValueError('About to return so should have been done replacing but told otherwise.')
        
    return original_labels.astype(np.uint8)

def inference_write(inference_loader, model, output_path, outputtag = ''):
    channel_keys = ['1','2','3','4']
    test_loader = inference_loader
    if test_loader == []:
        raise RuntimeError("Attempting to run inference with an empty val loader.")

    for subject in test_loader:
        subfolder = subject['subject_id'][0]
        
        #prep the path for the output file
        if not os.path.exists(output_path):
            os.mkdir(output_path)
         
        inference_outpath = os.path.join(output_path, subfolder + outputtag + '.nii.gz') # the evaluation app requires the format "${subject_id}.nii.gz"
        
        features = torch.cat([subject[key][torchio.DATA] for key in channel_keys], dim=1).float()
        nan_check(tensor=features, tensor_description='features tensor')
          
        model.sanity_check_val_input_shape(features)
        output = model.data.infer_with_patches(model_inference_function=[model.infer_batch_with_no_numpy_conversion], 
                                                              features=features)
        nan_check(tensor=output, tensor_description='model output tensor')
        model.sanity_check_val_output_shape(output)
          
        output = np.squeeze(output.cpu().numpy())

        # GANDLFData loader produces transposed output from what sitk gets from file, so transposing here.
        output = np.transpose(output)

        # process float sigmoid outputs (three channel corresponding to ET, TC, and WT)
        # into original label output (no channels, but values in 0, 1, 2, 4)
        output = convert_to_original_labels(output)
  
        # convert array to SimpleITK image 
        image = sitk.GetImageFromArray(output)

        # get the image information such as affine orientation from the first feature layer
        image.CopyInformation(sitk.ReadImage(subject['1']['path'][0]))

        sitk.WriteImage(image, inference_outpath)
                                        
if __name__ == '__main__':
    data_loader = FeTSChallengeDataLoader(
    data_path = DATA_PATH,
    training_batch_size = 1,
    q_samples_per_volume = 1,
    q_max_length = 1,
    patch_sampler = 'uniform',
    psize = [128, 128, 128],
    divisibility_factor= 16,
    data_usage = 'inference',
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
    )



    print('Model Initialized')
    params_dict = {'epochs_per_round': 0.5, 'num_batches': None , 'learning_rate':0.005}


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
    model.load_cpt(model_path)

    inference_loader = data_loader.get_inference_loader()
    inference_write(inference_loader, model, output_path)