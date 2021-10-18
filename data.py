
# This file will handle the datasets for training if needed, not sure right now if we should use this

class CustomDataset:
    def __init__(self, data_dir, image_preprocessing='no_preprocessing'):
        self.image_paths = []
        self.label_paths = []
        self.training_dataset = None
        self.validation_dataset = None
        self.image_preprocessing = image_preprocessing

        # Search through data_dir and collect image and label paths
        pass

    def __getitem__(self, index):
        # Read the image
        # Apply preprocessing step
        pass

    def __len__(self):
        pass

    def get_training_dataset(self):
        return self.training_dataset

    def get_validation_dataset(self):
        return self.validation_dataset


if __name__ == '__main__':
    pass
