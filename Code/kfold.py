
import os
import shutil

# Define the source and destination directories
source_dir = "YOUR_SOURCE_DIR/" #newTrainAllImages-2clas/ (classificationDatasetPrep-polygonsCroppedToImage.py OUTPUT_DIR)
dest_dir = "k-fold-all-2clas/"

# Define the split percentage
train_percent = 80
test_percent = 100 - train_percent

# Define the number of folds
num_folds = 5

# Iterate over each directory in the source directory
for class_folder in os.listdir(source_dir):
    class_source_dir = os.path.join(source_dir, class_folder)
    if os.path.isdir(class_source_dir):
        # Create the directory for the current class in the destination folder
        dest_class_dir = os.path.join(dest_dir, class_folder)
        os.makedirs(dest_class_dir, exist_ok=True)

        # Perform k-fold splitting
        for fold in range(1, num_folds + 1):
            # Create fold directories
            fold_train_dir = os.path.join(dest_class_dir, f"fold{fold}", "train")
            fold_test_dir = os.path.join(dest_class_dir, f"fold{fold}", "test")
            os.makedirs(fold_train_dir, exist_ok=True)
            os.makedirs(fold_test_dir, exist_ok=True)

            # List images in the source directory for each class
            for sub_folder in os.listdir(class_source_dir):
                sub_folder_path = os.path.join(class_source_dir, sub_folder)
                if os.path.isdir(sub_folder_path):
                    images = sorted(os.listdir(sub_folder_path))

                    # Calculate split indices
                    split_size = len(images) // num_folds
                    start_index = (fold - 1) * split_size
                    end_index = fold * split_size

                    # Assign images to train and test sets
                    train_images = images[:start_index] + images[end_index:]
                    test_images = images[start_index:end_index]

                    # Create class directories inside train and test directories
                    train_class_dir = os.path.join(fold_train_dir, sub_folder)
                    test_class_dir = os.path.join(fold_test_dir, sub_folder)
                    os.makedirs(train_class_dir, exist_ok=True)
                    os.makedirs(test_class_dir, exist_ok=True)

                    # Copy train images to the destination directory
                    for img in train_images:
                        src = os.path.join(sub_folder_path, img)
                        dst = os.path.join(train_class_dir, img)
                        if os.path.isfile(src):
                            shutil.copy(src, dst)

                    # Copy test images to the destination directory
                    for img in test_images:
                        src = os.path.join(sub_folder_path, img)
                        dst = os.path.join(test_class_dir, img)
                        if os.path.isfile(src):
                            shutil.copy(src, dst)
