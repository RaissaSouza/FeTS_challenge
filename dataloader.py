from gandlf_data import * 

class FeTSChallengeDataLoader(GANDLFData):

    def get_train_loader(self, batch_size=None, num_batches=None):
        return super().get_train_loader()

    def get_valid_loader(self, batch_size=None):
        return self.get_val_loader()

    def get_train_data_size(self):
        return self.get_training_data_size()

    def get_valid_data_size(self):
        return self.get_validation_data_size()
