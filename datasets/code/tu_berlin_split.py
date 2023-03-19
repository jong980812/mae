import os 
import shutil
import numpy as np


 #Used this fuction to split the dataset into train and test set
def TUberlin_split(original_dataset, out_dir):
    '''
    original_dataset: the path of the original Tu-berlin dataset without any extra split 
    out_dir: the path of the folder where you want to save the train and test set
    This fuction splits the 80% of the dataset into train set and the remaining 20% into test set
    '''
    dir = original_dataset
    out_dir = out_dir 
    train_dir = os.path.join(out_dir, 'train')
    test_dir = os.path.join(out_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)


    class_list = os.listdir(dir)
    for class_name in class_list:
        class_path = os.path.join(dir, class_name)
        img_list = os.listdir(class_path)
        train_dataset = np.random.choice(img_list, int(0.8* len(img_list)), replace=False)
        test_dataset = list(set(img_list).difference(train_dataset))

        # folder_name_train = class_name +'_' + 'train'
        # folder_name_test = class_name +'_' + 'test'
        folder_train = os.path.join(os.path.join(train_dir, class_name))
        folder_test = os.path.join(os.path.join(test_dir, class_name))
        os.makedirs(folder_train, exist_ok=True)
        os.makedirs(folder_test, exist_ok=True)


        for sketch in train_dataset:
            src = os.path.join(class_path, sketch)
            dst = os.path.join(folder_train, sketch)
            shutil.copyfile(src, dst)
        
        for sketch in test_dataset:
            src = os.path.join(class_path, sketch)
            dst = os.path.join(folder_test, sketch)
            shutil.copyfile(src, dst)