import json
from os.path import join as osp
import torch
from PIL import Image
from torchvision import transforms

def count_iou(mask, points, shape) :
    _shape = shape[1:]
    h, w = _shape
    
    p1, p2 = points
    xmin, ymin = int(p1[0]/w * 224), int(p1[1]/h * 224)
    xmax, ymax = int(p2[0]/w * 224), int(p2[1]/h * 224)
    
    if xmax > xmin and ymax > ymin:
        intersect = torch.sum(mask[xmin:xmax, ymin:ymax])
        total = torch.sum(mask) + (xmax - xmin) * (ymax - ymin) - intersect
        print(intersect, total)
        iou = intersect / total * 100
    else :
        iou = 0
    return iou   

with open("/data/ahngeo11/mae/asd_ver2_part_annotations.txt", 'r') as f :
    data = f.readlines()

results = {'ASD' : {'ASD-pred' : {'head' : 0, 'upper_body' : 0, "lower_body" : 0}, 'TD-pred' : {'head' : 0, 'upper_body' : 0, "lower_body" : 0}}, 'TD' : {'ASD-pred' : {'head' : 0, 'upper_body' : 0, "lower_body" : 0}, "TD-pred" : {'head' : 0, 'upper_body' : 0, "lower_body" : 0}}}

for img_path in data :
    root_dir = "/data/datasets/asd/asd_ver2_part_annotations"
    with open(osp(root_dir, img_path.replace('\n', '')), 'r') as f :
        ann = json.load(f)

    _root = "/local_datasets/asd/perturbation/mask"
    for mode in ["ASD", "TD"] :
        mask = torch.load(osp(_root, img_path.split('/')[-1].split('.')[0] + "-about-{}.pth".format(mode)))
        
        img = Image.open(osp(root_dir, img_path.split('.')[0] + ".jpg"))
        totensor = transforms.ToTensor()
        img = totensor(img)
        
        _mask = mask.squeeze(0).detach().cpu()
        _mask = (_mask > 0.4).float().squeeze(0)

        iou = dict()
        iou_list = []
        for obj_ann in ann['shapes'] :
            if obj_ann['label'] in ["head", "upper_body", "lower_body"] :
                iou[obj_ann['label']] = count_iou(_mask, obj_ann['points'], img.shape)
                iou_list.append(count_iou(_mask, obj_ann['points'], img.shape))
                
        iou_list = torch.tensor(iou_list)
        idx = torch.argmax(iou_list)
        if idx == 0 :   
            iou_max_part = "head"
        elif idx == 1 :
            iou_max_part = "upper_body"
        elif idx == 2 :
            iou_max_part = "lower_body"
        
        img_class = img_path.split('/')[0]
        assert img_class in ["ASD", "TD"]
        results[img_class][f"{mode}-pred"][iou_max_part] += 1
        
        print('\n'+img_path.replace('\n', ''))
        print(iou)
        print("the most important part : " + iou_max_part + "({})".format(int(iou[iou_max_part])))

print("********** results ***********************************")
print(results)