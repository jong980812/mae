
import cv2
import shutil
import os 
import numpy as np
import pandas as pd
import json


#-------------------
asd =  {'A1':1, 'A2':2, 'A3':5, 'A4':6, 'A5':6, 'A6':5, 'A7':5, 'A8':4,'A9':5, 'A10':3, 
'A11':2, 'A12':6, 'A13':5, 'A4':4, 'A15':4, 'A16':4, 'A17':4, 'A18':5, 'A19':4, 
'A20':5, 'A21':5, 'A22':5, 'A23':5,'A24':3, 'A25':5}

b_asd = {'B1':2, 'B2':1, 'B3':2, 'B4':2, 'B5':0, 'B6':0, 'B7':0, 'B8':2,'B9':1, 'B10':3, 
'B11':0, 'B12':1, 'B13':2, 'B14':0, 'B15':1, 'B16':0, 'B17':1, 'B18':0, 'B19':0, 
'B20':0, 'B21':0, 'B22':0, 'B23':1,'B24':2, 'B25':0}

asd_level = {'ASD': {'A1':1, 'A2':2, 'A3':5, 'A4':6, 'A5':6, 'A6':5, 'A7':5, 'A8':4,'A9':5, 'A10':3, 
'A11':2, 'A12':6, 'A13':5, 'A14':4, 'A15':4, 'A16':4, 'A17':4, 'A18':5, 'A19':4, 
'A20':5, 'A21':5, 'A22':5, 'A23':5,'A24':3, 'A25':5}, 'TD' :  {'B1':2, 'B2':1, 'B3':2, 'B4':2, 'B5':0, 'B6':0, 'B7':0, 'B8':2,'B9':1, 'B10':3, 
'B11':0, 'B12':1, 'B13':2, 'B14':0, 'B15':1, 'B16':0, 'B17':1, 'B18':0, 'B19':0, 
'B20':0, 'B21':0, 'B22':0, 'B23':1,'B24':2, 'B25':0} }
def get_asd_level(dir):
    '''
    데이타셋 경로를 입력하면, ASD 점수를 기준으로 0~6점까지의 threshold를 적용하여 정확도를 출력한다.
    dir: 데이타셋 경로, person_test0, person_test1, ... , person_test4 5개의 폴더가 있어야 합니다. 
    각 폴더에는 ASD와 TD 폴더가 있어야 합니다. 

    dir: path to the dataset, which contains 5 subfolders of person_test0, person_test1, ... , person_test4
    In the subfolders, there are 2 folders of ASD and TD, which contains images of each subjects drawings. 
    The ASD level is stored in asd_level dictionary, which is defined above. 
    It is ASD score from the [ASD 선별을 위한 아동의 미술심리 평가 및 데이터 분석 연구]
    This function prints the average accuracy from 5 different datasets with each threshold level from 0 to 6.
    High ASD score means the subjects are more likely to have ASD.
    
    '''
    for i in range(0,7):
        result = {'person_test_{}'.format(i): []}
        for x in range(5):
            path = dir+'/person_test{}'.format(x)
            class_ = os.listdir(path)
            total_num = 0
            temp_asd_level = {'ASD':{}, 'TD':{}}
            for class_name in class_:
                class_level = asd_level[class_name]
                class_path = os.path.join(path, class_name)
                img_list = os.listdir(class_path)
                for img_name in img_list:
                    subject_name = img_name.split('-')[0]
                    temp_asd_level[class_name][subject_name] = class_level[subject_name]
            #-------------------
            threshold_level = i 
            correct_prediction = []
            asd_keys = list(temp_asd_level.keys())
            for class_name in asd_keys:
                class_level = temp_asd_level[class_name]
                subject_num = class_level.keys()
                total_num += len(subject_num)
                for level in subject_num:
                    if class_level[level] >= threshold_level:
                        if level.startswith('A'):
                            correct_prediction.append(level)
                    if class_level[level] < threshold_level:
                        if level.startswith('B'):
                            correct_prediction.append(level)

            result['person_test_{}'.format(i)].append(len(correct_prediction)/total_num)
            print(total_num)
        result['avg'] = np.mean(result['person_test_{}'.format(i)])
        result['std'] = np.std(result['person_test_{}'.format(i)])
        print( 'avg----',result['avg'])
        print('std---',result['std'])
        print(result)


#get_asd_level('./asd_project/dataset/id_split')


                            
                                    




