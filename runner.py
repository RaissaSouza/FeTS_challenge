# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Mixin class for FL models. No default implementation.
Each framework will likely have its own baseclass implementation (e.g.
TensorflowTaskRunner) that uses this mixin.
You may copy use this file or the appropriate framework-specific base-class to
port your own models.
"""

from logging import getLogger


class TaskRunner(object):
    """Federated Learning Task Runner Class."""

    def __init__(self, data_loader, tensor_dict_split_fn_kwargs={}, **kwargs):
        """
        Intialize.
        Args:
            data_loader: The data_loader object
            tensor_dict_split_fn_kwargs: (Default=None)
            **kwargs: Additional parameters to pass to the function
        """
        self.data_loader = data_loader
        self.feature_shape = self.data_loader.get_feature_shape()

        # key word arguments for determining which parameters to hold out from
        # aggregation.
        # If set to none, an empty dict will be passed, currently resulting in
        # the defaults:
        # holdout_types=['non_float'] # all param np.arrays of this type will
        # be held out
        # holdout_tensor_names=[]     # params with these names will be held out
        # TODO: params are restored from protobufs as float32 numpy arrays, so
        # non-floats arrays and non-arrays are not currently supported for
        # passing to and from protobuf (and as a result for aggregation) - for
        # such params in current examples, aggregation does not make sense
        # anyway, but if this changes support should be added.
        if type(tensor_dict_split_fn_kwargs) is not dict:
            tensor_dict_split_fn_kwargs = dict()
        self.tensor_dict_split_fn_kwargs = tensor_dict_split_fn_kwargs
        self.set_logger()

    def set_logger(self):
        """Set up the log object."""
        self.logger = getLogger(__name__)

    def set_optimizer_treatment(self, opt_treatment):
        """Change the treatment of current instance optimizer."""
        self.opt_treatment = opt_treatment

    def get_data_loader(self):
        """
        Get the data_loader object.
        Serves up batches and provides info regarding data_loader.
        Returns:
            data_loader object
        """
        return self.data_loader

    def set_data_loader(self, data_loader):
        """Set data_loader object.
        Args:
            data_loader: data_loader object to set
        Returns:
            None
        """
        if data_loader.get_feature_shape() != \
                self.data_loader.get_feature_shape():
            raise ValueError(
                'The data_loader feature shape is not compatible with model.')

        self.data_loader = data_loader

    def get_train_data_size(self):
        """
        Get the number of training examples.
        It will be used for weighted averaging in aggregation.
        Returns:
            int: The number of training examples.
        """
        return self.data_loader.get_train_data_size()

    def get_valid_data_size(self):
        """
        Get the number of examples.
        It will be used for weighted averaging in aggregation.
        Returns:
            int: The number of validation examples.
        """
        return self.data_loader.get_valid_data_size()
