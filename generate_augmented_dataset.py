import os
import sys
import json
import numpy as np
import albumentations as A
import cv2
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from efficientdet.dataset import CocoAlbumentationsDataset

# A dictionary mapping from category id to class
cat_id_to_class = {
    1: 'leukocyte',
    2: 'trophozoite',
    3: 'schizont',
    4: 'ring',
    5: 'gametocyte',
}

# A dictionary mapping from class to category id
class_to_cat_id = {
    key: val for val, key in cat_id_to_class.items()
}

def count(path_json, disp=True):
    cat_id_to_class = {
        1: 'leukocyte',
        2: 'trophozoite',
        3: 'schizont',
        4: 'ring',
        5: 'gametocyte',
    }
    num_per_class = {
        'leukocyte': 0,
        'trophozoite': 0,
        'schizont': 0,
        'ring': 0,
        'gametocyte': 0,
    }
    
    f = open(path_json,)
    dataset = json.load(f)
    f.close()

    for annot in dataset['annotations']:
        category = cat_id_to_class[annot['category_id']]
        num_per_class[category] = num_per_class[category] + 1
    
    if disp:
        print(path_json, '|| num_images:', len(dataset['images']), '||', num_per_class)
    
    return num_per_class
    
class CocoWithoutResizer(CocoAlbumentationsDataset):
    def __init__(self, root_dir, set='train2017', transform=None):
        super().__init__(root_dir, set, transform)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        
        if self.transform:
            # The transformed data should have at least 1 bounding box
            while True:
                category_ids = [ann[-1] for ann in annot]
                temp = self.transform(image=img, bboxes=annot[:, :4],
                                      category_ids=category_ids)
                if len(temp['bboxes']) > 0:
                    break
            
            # Get the transformed image
            img = temp['image']
            
            # Get the transformed annotations
            annot = np.empty((len(temp['bboxes']), 5))
            annot[:, :4] = temp['bboxes']
            annot[:, 4] = temp['category_ids']
        
        # GA PERLU DI-TRANSFORM KARENA MAU DISAVE!
        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        # annot[:, 2] = annot[:, 0] + annot[:, 2]
        # annot[:, 3] = annot[:, 1] + annot[:, 3]
        
        sample = {'img': img, 'annot': annot}
        
        return sample
    
    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        return annotations
    
def generate_augmented_images(dataframe, dataset, target_class,
                              avoided_classes, duplication, num_classes,
                              saved_path, start_from, annotations,
                              dataset_ori, pct=1.0):
    
    # Generate the candidate row
    rows = []
    for _, row in dataframe.sort_values(by=target_class, ascending=False).iterrows():
        if row[target_class] < 1:
            break
            
        contain_avoided_class = False
        for avoided_class in avoided_classes:
            if row[avoided_class] != 0:
                contain_avoided_class = True
        if contain_avoided_class: continue
            
        rows.append(list(row))
    rows = rows[:int(pct * len(rows))]
    
    aug_annots = deepcopy(annotations)
    aug_img_idx = start_from
    for row in tqdm(rows):
        for _ in range(duplication):
            num_cat_target = row[class_to_cat_id[target_class] - 1]
            img_id = row[-1]

            iteration = 0
            max_iteration = 15
            while True:
                #### TYPE I ####
                ## Supaya loop nya ngga infinity kalau ngga ada yang sesuai
                ## Maka kalau iterasi melebihi max_iteration (10), maka
                ## kita ambil dari original data
                # if iteration >= max_iteration:
                #     sample = dataset_ori[img_id]
                # else:
                #     sample = dataset[img_id]
                # iteration = iteration + 1
                
                #### TYPE II ###               
                sample = dataset[img_id]
                img = sample['img']
                annot = sample['annot']

                aug_annot = []
                count_cat_target = 0
                
                if iteration >= max_iteration:
                    break
                iteration = iteration + 1
                
                for ann in annot:
                    if int(ann[-1]) == class_to_cat_id[target_class] - 1:
                        count_cat_target = count_cat_target + 1
                    aug_annot.append([int(ann[0]),
                                      int(ann[1]),
                                      int(ann[2]),
                                      int(ann[3]),
                                      cat_id_to_class[int(ann[-1] + 1)]])

                if num_cat_target == count_cat_target:
                    break
            
            # Kalau iterasi terlalu lama, yaudah continue/skip aja
            if iteration >= max_iteration:
                continue
            
            # Update the num_classes
            for ann in annot:
                ann_class = cat_id_to_class[int(ann[-1]) + 1]
                num_classes[ann_class] = num_classes[ann_class] + 1
            
            # Generate new image path
            while True:
                img_name = f'aug_{target_class}_{aug_img_idx}.png'
                img_path = os.path.join(saved_path, img_name)
                if not os.path.exists(img_path):
                    break
                aug_img_idx = aug_img_idx + 1

            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * 255.)
            aug_annots.append({img_name: {'description': {'height': img.shape[0],
                                                          'width': img.shape[1]},
                                          'annotations': aug_annot}})
            aug_img_idx = aug_img_idx + 1
    
    return rows, num_classes, aug_annots
    
# Cari jumlah class pada tiap image
training_set = CocoWithoutResizer(root_dir=os.path.join('datasets', 'malaria'), set='train',
                                  transform=None
                                 )

img_idx = []
annot = {cat: [] for cat in list(cat_id_to_class.values())}
for idx in tqdm(range(len(training_set))):
    num_per_class = {
        'leukocyte': 0,
        'trophozoite': 0,
        'schizont': 0,
        'ring': 0,
        'gametocyte': 0,
    }
    
    sample = training_set[idx]
    img = sample['img']
    bbs = sample['annot']
    
    for bb in bbs:
        category = cat_id_to_class[int(bb[-1])+1]
        num_per_class[category] = num_per_class[category] + 1
    
    for key in annot.keys():
        annot[key].append(num_per_class[key])
        
    img_idx.append(idx)

# A dataframe containing
# img_id and number of objects for each category
data = deepcopy(annot)
data['img_id'] = img_idx
df = pd.DataFrame(data=data)

# Check the distribution of the image in the dataset
num_classes = count('datasets/malaria/annotations/instances_train.json', False)

# Generate augmented dataset
aug_annots = []
training_set_ori = deepcopy(training_set)
training_set = CocoWithoutResizer(root_dir=os.path.join('datasets', 'malaria'), set='train',
                                  transform=A.Compose([A.OneOf([A.RandomCrop(1200, 1200, p=1.0),
                                                              A.RandomCrop(1100, 1100, p=0.5),
                                                              A.RandomCrop(1000, 1000, p=1.0),
                                                              A.RandomCrop(900, 900, p=0.5),
                                                              A.RandomCrop(800, 800, p=1.0),
                                                              A.RandomCrop(700, 700, p=0.5)],
                                                             p=1.0)],
                                                    bbox_params=A.BboxParams(format='coco',
                                                                             label_fields=['category_ids'],
                                                                             min_visibility=0.2),),
                                 )

##################################################################################################################
##### IMPORTANT !!!!! #####
# Ini FINE TUNING, perlu latihan dulu di google colab
# Supaya jumlah class yang dihasilkan sesuai keinginan
# Belum kebayang dibikin otomatisnya kaya gimana
# Mungkin next-time kalau bener-bener serius mau bikin kodingan yang handle unbalanced dataset 
##### IMPORTANT !!!!! #####
##################################################################################################################
_, num_classes, aug_annots = generate_augmented_images(df, training_set, 'leukocyte', ['trophozoite'], 13,
                                                       num_classes, 'datasets/malaria/augmentation', 0, aug_annots,
                                                       training_set_ori)
_, num_classes, aug_annots = generate_augmented_images(df, training_set, 'schizont', ['trophozoite'], 3,
                                                       num_classes, 'datasets/malaria/augmentation', 0, aug_annots,
                                                       training_set_ori, 0.8)
_, num_classes, aug_annots = generate_augmented_images(df, training_set, 'gametocyte', ['trophozoite'], 5,
                                                       num_classes, 'datasets/malaria/augmentation', 0, aug_annots,
                                                       training_set_ori)
_, num_classes, aug_annots = generate_augmented_images(df, training_set, 'ring', ['trophozoite'], 1,
                                                       num_classes, 'datasets/malaria/augmentation', 0, aug_annots,
                                                       training_set_ori, 0.2)
print("The distribution of the class", num_classes)

# SAVE TO JSON FORMAT
augmented_annotations = {}
for annotation in aug_annots:
    for key, val in annotation.items():
        augmented_annotations[key] = val

json_object = json.dumps(augmented_annotations, indent = 4) 
with open('datasets/malaria/augmentation/augmented_annotations.json', "w") as outfile: 
    outfile.write(json_object) 