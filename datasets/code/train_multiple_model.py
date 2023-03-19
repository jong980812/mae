import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, transforms
import time
import os
import copy
from torchvision.utils import save_image
import argparse
import json
from model_zoo import initialize_model
from model_zoo import model_list
# from top_1 import top1_accuracy
from test_metric import top1_accuracy
from test_metric import top1_accuracy_level
import gc
import shutil


Model_type = ['resnet','densenet','vit', 'mobilenet','convnext' ,'efficientnet','regnet']
def get_args():
    '''
    data_dir: path to the data directory the data diractory should subflder of free_train0...free_train4 and person_train0...person_train4
    and free_test0...free_test4 and person_test0...person_test4
    In the free_train or test folder there should be two fodler ASD and TD

    model: model name in big term EX) resnet, densenet, vit, mobilenet, convnext , efficientnet, regnet
    if yo put the model name resnet it will train all the resnet model listed in the model_zoo.py

    data: free or person the name of the dataset you want to train and test 자유화 or 인물화 

    model_path: where to save the model

    num_classes: number of classes in the dataset in our case ASD or TD so 2

    batch_size: batch size for training

    num_epochs: number of epochs for training, the training epoch for person dataset 인물화 is 20 and free dataset 자유화 is 30

    feature_extract: if you want to finetune the model or not, if you want to finetune the model put False, if you want to train the model from scratch put True

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('-m','--model', type=str, required=True, choices=Model_type)# model name in big term
    parser.add_argument('-d','--data', type=str, required=True)#free or person 
    parser.add_argument('-mp','--model_path', type=str, required=True)#where to save the model
    parser.add_argument('-nc','--num_classes', type=int, default=2)
    parser.add_argument('-bs','--batch_size', type=int, default=25)
    parser.add_argument('-e','--num_epochs', type=int, default=20) # for free dataset use 30 epoch 
    parser.add_argument('-fe','--feature_extract', type=bool, default=True)
    return parser.parse_args()


def train_model(model_path,number , type_name, model_name , model, dataloaders, criterion, optimizer, num_epochs,split_name):
    '''
    model_path: where to save the model
    number: the number of the model in the model list
    type_name: the name of the model in the model list
    model_name: the name of the model in the model list
    model: the model you want to train
    dataloaders: the dataloader for train and test
    criterion: the loss function
    optimizer: the optimizer for the model in our case SGD
    num_epochs: number of epochs for training, the training epoch for person dataset 인물화 is 20 and free dataset 자유화 is 30
    split_name: the name of the split in the pweaon oe free dataset, 자유화 or 인물화
    '''
    gc.collect()
    torch.cuda.empty_cache()
    since = time.time()
    print(number,type_name, model_name)
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs): 
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
               
                inputs = inputs.cuda()
                labels = labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
 
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                trained_model_path = model_path+'/model'+'/{}/{}_{}_{}_{}.pt'.format(type_name,type_name,number,model_name, split_name)
                os.makedirs(model_path+'/model'+'/{}'.format(type_name), exist_ok=True)
                torch.save(model, trained_model_path)
            if phase == 'val':
                val_acc_history.append(epoch_acc)

    

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc*100))

    # load best model weights
    model.load_state_dict(best_model_wts)
    gc.collect()
    torch.cuda.empty_cache()
    return model, best_acc



def run_model(data_dir, model_name,data ,model_path,num_classes,batch_size, num_epochs,feature_extract,train_x, test_x, number,split_name ):
    '''
    data_dir: the path of the dataset
    model_name: the name of the model in the model list
    data: the name of the dataset person or free, 자유화 or 인물화
    model_path: where to save the model
    num_classes: the number of classes in the dataset
    batch_size: the batch size for training
    num_epochs: number of epochs for training, the training epoch for person dataset 인물화 is 20 and free dataset 자유화 is 30
    feature_extract: if you want to extract the feature or not
    train_x: the name of the train folder in the dataset
    test_x: the name of the test folder in the dataset
    number: the number of the model in the model list

    This function is for training the model, it trains the model and saves the model in the model_path
    returns model_org, failed_dict, model_name
    model_org: The dictionary of the model's performance in accuracy, precision, recall, f1 score
    failed_dict: The dictionary of image/drawing path or the name of the image/drawing that the model failed to predict
    model_name: the name of the model in the model list

    '''
    gc.collect()
    torch.cuda.empty_cache()

    train_dir = os.path.join(data_dir, train_x)
    test_dir = os.path.join(data_dir, test_x)

    model_org = {}
    failed_dict = {}
    data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()]),
                'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()])}
    best = []
   
    image_datasets = {'train': datasets.ImageFolder(train_dir,
                                            data_transforms['train']), 'val': datasets.ImageFolder(test_dir,
                                            data_transforms['val'])}
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                shuffle=True, num_workers=4), 'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=len(image_datasets['val']),
                                                shuffle=True, num_workers=4)}
    gc.collect()
    torch.cuda.empty_cache()
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft = nn.DataParallel(model_ft)
    model_ft = model_ft.cuda()

    params_to_update = model_ft.parameters()
    #print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model, best_acc = train_model(model_path,number , data, model_name , model_ft, 
                                    dataloaders, criterion, optimizer_ft, num_epochs,split_name)
    best.append(float(best_acc))

    #-------
    transform_test = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, test_x), transform_test)
    dataset_sizes = len(test_dataset)
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=dataset_sizes, shuffle=False, num_workers=4)
    #-------

    # result, result_dict = top1_accuracy(model, data_loader)
    if num_classes ==2:
        result, result_dict, precision_score, recall_score, f1_score =  top1_accuracy(model, data_loader)
    else:
        result, result_dict, precision_score, recall_score, f1_score =  top1_accuracy_level(model, data_loader)
    print('Test - top1_accuracy: ', result)
    print('Test - precision_score: ', precision_score)
    print('Test - recall_score: ', recall_score)
    print('Test - f1_score: ', f1_score)



    model_org[model_name+'_train'] = []
    model_org[model_name+'_train'].append(best)
    model_org[model_name+'_Top1acc']= []
    model_org[model_name+'_Top1acc'].append(result)
    model_org[model_name+'_precision']= []
    model_org[model_name+'_precision'].append(precision_score)
    model_org[model_name+'_recall']= []
    model_org[model_name+'_recall'].append(recall_score)
    model_org[model_name+'_F1_score']= []
    model_org[model_name+'_F1_score'].append(f1_score)





    gc.collect()
    torch.cuda.empty_cache()
    #return test result and failed case
    return model_org, failed_dict, model_name

def data_list_(data_lists, task_type):
    x_list = []
    for x in data_lists:
        if x.startswith(task_type):
            x_list.append(x)
    return x_list




def run_5_dataset(data_dir, model_name,data ,model_path,num_classes,batch_size, num_epochs,feature_extract):
    '''
    This function is used to run 5 dataset. From train0 to train4

    return model_dict, failed_dict , model_name
    model_dict: model result of 5 datasets and averaged result of accuracy, precision, recall, f1_score
    within 5 different datasets
    failed_dict: failed case of 5 datasets
    model_name: model name

    '''
    data_lists = os.listdir(data_dir)
    dataset_list = data_list_(data_lists, data)#???

    model_dict = {}
    failed_dict = {}
    split_name = data_dir.split('/')[-1]
    if data =='person' or data=='free':
        for number in range(5):
            train_x = data+'_train'+str(number)
            test_x = data+'_test'+str(number)
 

            model_result, failed_case, model_name = run_model(data_dir, model_name,data ,model_path,num_classes,
                                                    batch_size, num_epochs,feature_extract,train_x, test_x, number,split_name )
            model_name_list = list(model_result.keys())
            name = model_name_list[0]
            model_acc = model_name_list[1]
            model_precision = model_name_list[2]
            model_recall = model_name_list[3]
            model_f1_score = model_name_list[4]


            if name in model_dict:
                model_dict[name].extend(model_result[name])
                model_dict[model_acc].extend(model_result[model_acc])
                model_dict[model_precision].extend(model_result[model_precision])
                model_dict[model_recall].extend(model_result[model_recall])
                model_dict[model_f1_score].extend(model_result[model_f1_score])

            else:
                model_dict.update(model_result)

            failed_dict.update(failed_case)
    else:
        character_split_list = os.listdir(data_dir)
        for split_type in character_split_list:
            train_x = split_type+'/train'
            test_x = split_type+'/test'
            model_result, failed_case , model_name= run_model(data_dir, model_name,data ,model_path,num_classes,
                                                    batch_size, num_epochs,feature_extract,train_x, test_x, split_type,split_name )
            model_name_list = list(model_result.keys())
            name = model_name_list[0]
            model_acc = model_name_list[1]
            model_precision = model_name_list[2]
            model_recall = model_name_list[3]
            model_f1_score = model_name_list[4]
            if name in model_dict:
                model_dict[name].extend(model_result[name])
                model_dict[model_acc].extend(model_result[model_acc])
                model_dict[model_precision].extend(model_result[model_precision])
                model_dict[model_recall].extend(model_result[model_recall])
                model_dict[model_f1_score].extend(model_result[model_f1_score])

            else:
                model_dict.update(model_result)

            failed_dict.update(failed_case)

    model_dict[model_name+'_mean'] = np.mean(model_dict[model_acc])
    model_dict[model_name+'_std'] = np.std(model_dict[model_acc])

    model_dict[model_name+'precision_mean'] = np.mean(model_dict[model_name+'_precision'])
    model_dict[model_name+'precision_std'] = np.std(model_dict[model_name+'_precision'])

    model_dict[model_name+'recall_mean'] = np.mean(model_dict[model_name+'_recall'])
    model_dict[model_name+'recall_std'] = np.std(model_dict[model_name+'_recall'])

    model_dict[model_name+'F1_mean'] = np.mean(model_dict[model_name+'_F1_score'])
    model_dict[model_name+'F1_std'] = np.std(model_dict[model_name+'_F1_score'])


    #os.makedirs(model_path+'/{}/{}'.format(data,model_name), exist_ok=True)
    return model_dict, failed_dict , model_name

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = True #this is true for finetuning



def main(data_dir, model,data,model_path,num_classes,batch_size, num_epochs,feature_extract):
    '''
    data_dir: data directory
    model: model name
    data: dataset name
    model_path: model path to save
    num_classes: number of classes
    batch_size: batch size
    num_epochs: number of epochs

    Trains the model on 5 datasets. Saves the results in json file and the model in model_path.
    
    '''
    model_zoo = model_list(model)
    model_totla_dict = {}
    failed_total_dict = {}
    split_name = data_dir.split('/')[-1]
    for model_name in model_zoo:
        model_dict, failed_dict, model_name = run_5_dataset(data_dir, model_name,data ,model_path,num_classes,batch_size, num_epochs,feature_extract)
        model_totla_dict.update(model_dict)
        failed_total_dict.update(failed_dict)
    folder_path = model_path+'/{}/{}'.format(data,model)
    data_path = folder_path + '/data'
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    if split_name =='person':
        split_name = data_dir.split('/')[-2]
    with open(model_path+'/{}/{}'.format(data,model)+'/failed_case{}_{}_batch_{}_{}{}.json'.format(data, model, batch_size, split_name,num_epochs), 'w') as f:
        json.dump(failed_total_dict, f)

    with open(model_path+'/{}/{}'.format(data,model)+'/train_{}_{}_model_batch_{}_{}{}.json'.format(data,model , batch_size, split_name,num_epochs), 'w') as fp:
        json.dump(model_totla_dict, fp)
    print('------------------------------------------------------')
    print('model list - ', model_zoo)
    print('model - ' ,  model)
    print('data - ', data)
    print('model_path/result output folder - ', model_path)
    print('batch size - ', batch_size)
    print('num_epochs - ', num_epochs)
    print('------------------------------------------------------')








if __name__ == '__main__':
    main(**vars(get_args()))

