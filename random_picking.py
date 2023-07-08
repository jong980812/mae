import os
import subprocess
import random
train_folder='/data/datasets/tu_berlin/01/train'
val_folder='/data/datasets/tu_berlin/01/val'
folders=os.listdir(train_folder)
for folder in folders:
    subprocess.call(['mkdir','-p',f'/data/datasets/tu_berlin/01/val/{folder}'])


for class_name in folders:
    train_folder_path=os.path.join(train_folder,class_name)
    val_folder_path=os.path.join(val_folder,class_name)
    number=len(os.listdir(train_folder_path))
    val_number=int(number/5)
    files=os.listdir(train_folder_path)
    random.shuffle(files)
    for i in range(val_number):
        files_path=os.path.join(train_folder_path,files[i])
        subprocess.call(['mv',files_path,val_folder_path])
    