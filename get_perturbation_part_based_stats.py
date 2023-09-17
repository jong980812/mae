import json
from os.path import join as osp
import torch
from PIL import Image
from torchvision import transforms

def count_iou(mask, points, shape) :
    _shape = shape[1:]
    h, w = _shape
    
    p1, p2 = points
    xmin, ymin = int(p1[0]/w * 168), int(p1[1]/h * 224)
    xmax, ymax = int(p2[0]/w * 168), int(p2[1]/h * 224)
    
    if xmax > xmin and ymax > ymin:
        intersect = torch.sum(mask[ymin:ymax, xmin:xmax])
        # total = torch.sum(mask) + (xmax - xmin) * (ymax - ymin) - intersect
        total = (xmax - xmin) * (ymax - ymin)
        print(intersect, total)
        iou = intersect / total * 100
    else :
        iou = 0
    return iou 

def count_mean_energy(mask, points, shape) :
    _shape = shape[1:]
    h, w = _shape
    
    p1, p2 = points
    xmin, ymin = int(p1[0]/w * 168), int(p1[1]/h * 224)
    xmax, ymax = int(p2[0]/w * 168), int(p2[1]/h * 224)
    
    if xmax > xmin and ymax > ymin:
        energy = torch.mean(mask[ymin:ymax, xmin:xmax])
    else :
        energy = 0
    return energy   

with open("/data/ahngeo11/asd/mae/asd_all_part_annotations_list.txt", 'r') as f :
    data = f.readlines()

# _temp = {"head" : 0, "upper_body" : 0, "lower_body" : 0}
# results = {'ASD' : {"ASD-pred" : _temp.copy(), "TD-pred" : _temp.copy()}, "TD" : {"ASD-pred" : _temp.copy(), "TD-pred" : _temp.copy()}}

_temp = {"eye" : 0, "nose" : 0, "mouth" : 0, "ear" : 0, "head" : 0, "hand" : 0, "foot" : 0, "upper_body" : 0, "lower_body" : 0}
results_template = {'ASD' : {"ASD-pred" : _temp.copy(), "TD-pred" : _temp.copy()}, "TD" : {"ASD-pred" : _temp.copy(), "TD-pred" : _temp.copy()}}
results = []


#! for count iou with hard mask

# for img_path in data :
#     root_dir = "/data/datasets/asd/asd_all_5folds_annotations"
#     with open(osp(root_dir, img_path.replace('\n', '')), 'r') as f :
#         ann = json.load(f)

#     _root = "/data/ahngeo11/mae/mask/perturbation-effi-5split-1e-2-wd1e-1-0.05/mask"
#     for mode in ["ASD", "TD"] :
#         mask = torch.load(osp(_root, img_path.split('/')[-1].split('.')[0] + "-about-{}.pth".format(mode)))
        
#         img = Image.open(osp(root_dir, img_path.split('.')[0] + ".jpg"))
#         totensor = transforms.ToTensor()
#         img = totensor(img)
          
#         _mask = mask.squeeze(0).detach().cpu()
#         _mask = (_mask > 0.4).float().squeeze(0)   #* 224, 224

#         iou_list = []
#         iou_label_list = []
#         for obj_ann in ann['shapes'] :
#             if obj_ann['label'] in ['left_eye', 'right_eye', 'nose', 'mouth', 'head', 'left_hand', 'right_hand', 'left_foot', 'right_foot', 'upper_body', 'lower_body'] :
#             # if obj_ann['label'] in ['head', 'upper_body', 'lower_body'] :
#                 iou_list.append(count_iou(_mask, obj_ann['points'], img.shape))
#                 iou_label_list.append(obj_ann['label'])
              
#         iou_list = torch.tensor(iou_list)
#         idx = torch.argmax(iou_list)
#         iou_max_part = iou_label_list[idx]
        
        
#         img_class = 'ASD' if img_path.split('/')[-1][0] == 'A' else 'TD'   
#         assert img_class in ["ASD", "TD"]
        
#         if 'eye' in iou_max_part :
#             iou_max_part = 'eye'
#         elif 'hand' in iou_max_part :
#             iou_max_part = 'hand'
#         elif 'foot' in iou_max_part :
#             iou_max_part = 'foot'
        
#         results[img_class][f"{mode}-pred"][iou_max_part] += 1
        
#         print(img_path.replace('\n', ''))
#         print("the most important part : " + iou_max_part + "({})".format(int(iou_list[idx]))+'\n')

# print("********** results ***********************************")
# print(results)


#! for count mean energy
for mode in ["ASD", "TD"] :

# mode = "ASD"
    for img_path in data :
        _temp = {"eye" : 0, "hand" : 0, "nose" : 0, "mouth" : 0, "ear" : 0, "foot" : 0, "head" : 0, "upper_body" : 0, "lower_body" : 0}
        results_template = {'ASD' : {"ASD-pred" : _temp.copy(), "TD-pred" : _temp.copy()}, "TD" : {"ASD-pred" : _temp.copy(), "TD-pred" : _temp.copy()}}
        img_results = {'img_name' : {img_path.split('/')[-1]}, "results" : results_template.copy()}
        
        root_dir = "/data/datasets/asd/asd_all_5folds_annotations"
        with open(osp(root_dir, img_path.replace('\n', '')), 'r') as f :
            ann = json.load(f)

        _root = "/data/ahngeo11/asd/mae/mask/perturbation-effi-99/mask"
        mask = torch.load(osp(_root, img_path.split('/')[-1].split('.')[0] + "-about-{}.pth".format(mode)), map_location='cpu')
        
        img = Image.open(osp(root_dir, img_path.split('.')[0] + ".jpg"))
        totensor = transforms.ToTensor()
        img = totensor(img)
        
        _mask = mask.squeeze(0).squeeze(0)

        energy_list = []
        energy_label_list = []
        for obj_ann in ann['shapes'] :
            if obj_ann['label'] in ['left_eye', 'right_eye', "left_ear", "right_ear", 'nose', 'mouth', 'head', 'left_hand', 'right_hand', 'left_foot', 'right_foot', 'upper_body', 'lower_body'] :
                energy_list.append(count_mean_energy(_mask, obj_ann['points'], img.shape))
                energy_label_list.append(obj_ann['label'])
            
        energy_list = torch.tensor(energy_list)
        indices = torch.argsort(energy_list, descending=True)
        
        
        img_class = 'ASD' if img_path.split('/')[-1][0] == 'A' else 'TD'   
        assert img_class in ["ASD", "TD"]

        # logging
        print(img_path.replace('\n', ''))

        for i, idx in enumerate(indices) :
            iou_max_part = energy_label_list[idx]
            if 'eye' in iou_max_part :
                iou_max_part = 'eye'
            elif 'hand' in iou_max_part :
                iou_max_part = 'hand'
            elif 'foot' in iou_max_part :
                iou_max_part = 'foot'
            elif 'ear' in iou_max_part :
                iou_max_part = 'ear'
            
            img_results['results'][img_class][f"{mode}-pred"][iou_max_part] = max(energy_list[idx], img_results['results'][img_class][f"{mode}-pred"][iou_max_part]) 
            
            # logging
            if i == 0 :
                print("the most important part : " + iou_max_part + "({})".format(float(energy_list[idx])))
            elif i == 1 :
                    print("the second important part : " + iou_max_part + "({})".format(float(energy_list[idx])))
            elif i == 2 :
                    print("the third important part : " + iou_max_part + "({})".format(float(energy_list[idx])))

        # results.append(str(img_results))
        results.append(img_results)
        
        
print("\n\n************************* results ***********************************")
print("************************* results ***********************************")
print("************************* results ***********************************\n\n")

# print('\n'.join(results))
import pickle
with open("/data/ahngeo11/asd/mae/part-based-stats-5-splits/effi-99-mean-energy.pkl", 'wb') as f :
    pickle.dump(results, f)