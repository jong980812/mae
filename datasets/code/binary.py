import os 
import shutil
import cv2


def binary_img_cv2_asd_level(data_dir, threshold_number):
    ''' 데이터셋을 이진 이미지로 변환하는 함수 
        data_dir: 변환할 데이터셋의 경로
        threshold_number: 이미지를 이진 이미지로 변환할 때 사용하는 숫자
        만약 픽셀 값이 threshold number 보다 크면 픽셀 값을 255으로 변환
        만약 픽셀 값이 threshold number 보다 작으면 픽셀 값을 0으로 변환

        data_dir: the path of the dataset to convert to binary image
        threshold_number: the number that we use to convert the image to binary image
        If the pixel value is higher than the threshold number, the pixel value is 255
        If the pixel value is lower than the threshold number, the pixel value is 0'''
       
    train_test_set = os.listdir(data_dir)
    for x in train_test_set:

        type_path = os.path.join(data_dir, x)
        level_asd = os.listdir(type_path)
        for level in level_asd:

            if level == '.DS_Store':
                pass    
            else:

                img_path = os.path.join(type_path, level)

                for img_name in os.listdir(img_path):
                    path_img = os.path.join(img_path, img_name)
                    image = img = cv2.imread(path_img, 2)
                    ret, bw_img = cv2.threshold(img, threshold_number, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(path_img, bw_img)
def binary_imging(data_dir, output_dir, threshold):
    #data_dir은 split 바로 위 폴더.
    kfold_number=os.listdir(data_dir)
    for number in kfold_number:
        kfold_dir=os.path.join(data_dir,number)
        out_kfold_dir=os.path.join(output_dir,number)
        os.mkdir(out_kfold_dir)
        for split in os.listdir(kfold_dir):
            split_dir=os.path.join(kfold_dir,split)
            out_split_dir=os.path.join(out_kfold_dir,split)
            os.mkdir(out_split_dir)
            for class_name in os.listdir(split_dir):
                class_dir=os.path.join(split_dir,class_name)
                out_class_dir=os.path.join(out_split_dir,class_name)
                os.mkdir(out_class_dir)
                for img_name in os.listdir(class_dir):#real image dir
                    img_path=os.path.join(class_dir,img_name)
                    new_img_path=os.path.join(out_class_dir,img_name)
                    img=cv2.imread(img_path, 2)
                    _,new_img=cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
                    cv2.imwrite(new_img_path,new_img)

if __name__=="__main__":
    binary_imging('/data/dataset/asd/images/original/person','/data/dataset/asd/binary_images/original/person',240)
    binary_imging('/data/dataset/asd/images/half_crop/person','/data/dataset/asd/binary_images/half_crop/person',240)
    binary_imging('/data/dataset/asd/images/hand_crop/person','/data/dataset/asd/binary_images/hand_crop/person',240)
            