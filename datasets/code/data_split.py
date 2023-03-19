
import numpy as np

import time
import os
import copy
import shutil
import cv2

def data_split_by_ID_free(dir, out_dir):
    '''
    dir: the path of the dataset to split. ex) dir= ./asd_dataset_eng/free
    out_dir: the path of the dataset to saved, which consists of 5 different train and test sets.
    makes 5 different train and test sets by randomly selecting 80% of the participants(this is the ID) for training and the remaining 20% for testing.
    In the out_dir free_test0,....,free_test4, free_train0,....,free_train4 are created.
    '''

    choice = ['free']
    dignosed_si = ['ASD', 'TD']
    for new in range(5):
        for x in choice:
            type_of_sketch = os.path.join(dir, x)

            for participant in dignosed_si:
                img_path = os.path.join(type_of_sketch,participant )
                img_list = os.listdir(img_path)

                free_participant = list(range(1, 30))
                # person_participant = list(range(1, 26))

                free_train_participant = np.random.choice(free_participant, int(0.8* len(free_participant)), replace=False)
                free_test_participant = list(set(free_participant).difference(free_train_participant))


                folder_name_train = x +'_' + 'train' +str(new)
                folder_name_test = x +'_' + 'test' +str(new)
                folder_train = os.path.join(os.path.join(out_dir, folder_name_train),participant )
                folder_test = os.path.join(os.path.join(out_dir, folder_name_test),participant )

                os.makedirs(folder_train, exist_ok=True)
                os.makedirs(folder_test, exist_ok=True)
                #print(img_path)
                for sketch_name in img_list:
                    number = sketch_name.split('-')
                    number=number[0]
                    number=number.replace(number[0], '')
               
                    for par_num in free_train_participant:
                        if number==str(par_num):
                            src = os.path.join(img_path, sketch_name)
                            dst = os.path.join(folder_train, sketch_name)
                       
                            if os.path.isfile(dst):
                                pass
                            else:
                                shutil.copyfile(src, dst)

                    for par_num in free_test_participant:
                        if number==str(par_num):
                            src = os.path.join(img_path, sketch_name)
                            dst = os.path.join(folder_test, sketch_name)
                            if os.path.isfile(dst):
                                pass
                            else:
                                shutil.copyfile(src, dst)


def data_split_by_person_person(dir, out_dir):
    '''
    dir: the path of the dataset to split. ex) dir= ./asd_dataset_eng/person
    out_dir: the path of the dataset to saved, which consists of 5 different train and test sets.
    makes 5 different train and test sets by randomly selecting 80% of the participants(this is the ID) for training and the remaining 20% for testing.
    In the out_dir person_test0,....,person_test4, person_train0,....,person_train4 are made.
    This out_dir is the folder used for training the models.
    데이터셋을 5개의 다른 트레인과 테스트셋으로 만듭니다.
    80%의 참가자를 랜덤으로 트레인셋으로 선택하고 나머지 20%를 테스트셋으로 선택합니다.
    각 그림이 랜덤으로 선택되는게 아니라, 1번 참가자는 1번 참가자가 그린 두개의 동성과 이성 그림이 둘다 트레인셋에 들어갑니다.
    out_dir에 person_test0,....,person_test4, person_train0,....,person_train4가 만들어집니다.
    '''

    choice = ['']
    dignosed_si = ['ASD', 'TD']
    for new in range(5):
        for x in choice:
            type_of_sketch = os.path.join(dir, x)

            for participant in dignosed_si:
                img_path = os.path.join(type_of_sketch,participant )
                img_list = os.listdir(img_path)

                person_participant = list(range(1, 26))

                out_folder=os.path.join(out_dir,str(new))
                person_train_participant = np.random.choice(person_participant, int(0.8* len(person_participant)), replace=False)
                person_test_participant = list(set(person_participant).difference(person_train_participant))
                os.makedirs(out_folder, exist_ok=True)
                folder_train = os.path.join(os.path.join(out_folder, 'train'),participant )
                folder_test = os.path.join(os.path.join(out_folder, 'val'),participant )

                os.makedirs(folder_train, exist_ok=True)
                os.makedirs(folder_test, exist_ok=True)
           
                for sketch_name in img_list:
                    number = sketch_name.split('-')
                    number=number[0]
                    number=number.replace(number[0], '')
                   
                    for par_num in person_train_participant:
                        if number==str(par_num):
                            src = os.path.join(img_path, sketch_name)
                            dst = os.path.join(folder_train, sketch_name)
                            if os.path.isfile(dst):
                                pass
                            else:
                                shutil.copyfile(src, dst)

                    for par_num in person_test_participant:
                        if number==str(par_num):
                            src = os.path.join(img_path, sketch_name)
                            dst = os.path.join(folder_test, sketch_name)
                            if os.path.isfile(dst):
                                pass
                            else:
                                shutil.copyfile(src, dst)

data_split_by_person_person('/data/datasets/asd/raw_data/hand_crop', '/data/datasets/asd/images/hand_crop/person')
