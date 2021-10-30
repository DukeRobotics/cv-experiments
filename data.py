import glob
import os
import cv2
from image_preprocessing import *
from configurations import *
from shutil import copyfile


def create_image_processing_folders(image_preprocessing=no_preprocessing):
    assert image_preprocessing in IMAGE_PREPROCESSING_ALGORITHMS, 'image processing technique not added to ' \
                                                                  'image_preprocessing_algorithms '

    # Check that the original image directories are there
    original_files_dir = os.path.join(BASE_DATA_DIR, ORIGINAL_IMAGE_FOLDER)
    original_train = os.path.join(original_files_dir, TRAINING_FOLDER_NAME)
    original_validation = os.path.join(original_files_dir, VALIDATION_FOLDER_NAME)
    assert os.path.exists(original_train), f'{original_train} should exist. Add original training images ' \
                                           'and labels to this directory '
    assert os.path.exists(original_validation), f'{original_validation} should exist. Add original validation ' \
                                                'images and labels to this directory'

    # Create the directory for the new image preprocessing
    new_image_preprocessing_dir = os.path.join(BASE_DATA_DIR, image_preprocessing.__name__)
    if not os.path.exists(new_image_preprocessing_dir):
        os.makedirs(new_image_preprocessing_dir)
    new_train_dir = os.path.join(new_image_preprocessing_dir, TRAINING_FOLDER_NAME)
    new_validation_dir = os.path.join(new_image_preprocessing_dir, VALIDATION_FOLDER_NAME)
    if not os.path.exists(new_train_dir):
        os.makedirs(new_train_dir)
    if not os.path.exists(new_validation_dir):
        os.makedirs(new_validation_dir)

    copy_images_and_labels(original_train, new_train_dir, image_preprocessing)
    copy_images_and_labels(original_validation, new_validation_dir, image_preprocessing)


def copy_images_and_labels(src_dir, dst_dir, image_preprocessing):
    # For each image in the original image training directory, apply the image preprocessing and save into
    images = glob.glob(os.path.join(src_dir, '*'+IMAGE_EXT))
    labels = glob.glob(os.path.join(src_dir, '*'+LABEL_EXT))
    assert len(images) > 0 and len(labels) > 0, f'Could not find images and/or labels to copy over for {src_dir}'

    # Could change the name of the images by appending the preprocessing name to their basename.
    # In this case we would also have to change the filename field in the corresponding xml

    for image in images:
        basename = os.path.basename(image)
        save_path = os.path.join(dst_dir, basename)
        img = image_preprocessing(image)
        cv2.imwrite(save_path, img)
    for label in labels:
        basename = os.path.basename(label)
        dst = os.path.join(dst_dir, basename)
        copyfile(label, dst)


if __name__ == '__main__':
    for image_preprocessing_function in DEFAULT_IMAGE_PREPROCESSING_ALGORITHMS:
        create_image_processing_folders(image_preprocessing_function)
