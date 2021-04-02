import os
import sys
import json
import numpy as np
import albumentations as A
import cv2
import pandas as pd
import math
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
    
def generate_augmented_images(dataframe, dataset, target_class,
                              avoided_classes, duplication, num_classes,
                              saved_path, start_from, annotations):
    
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
    
    aug_annots = deepcopy(annotations)
    aug_img_idx = start_from
    for row in tqdm(rows):
        for _ in range(duplication):
            num_cat_target = row[class_to_cat_id[target_class] - 1]
            img_id = row[-1]

            while True:
                sample = training_set[img_id]
                img = sample['img']
                annot = sample['annot']

                aug_annot = []
                count_cat_target = 0
                for ann in annot:
                    if int(ann[-1]) == class_to_cat_id[target_class] - 1:
                        count_cat_target = count_cat_target + 1
                    aug_annot.append([math.ceil(ann[0]),
                                      math.ceil(ann[1]),
                                      math.ceil(ann[2]),
                                      math.ceil(ann[3]),
                                      cat_id_to_class[int(ann[-1] + 1)]])

                if num_cat_target == count_cat_target:
                    break
            
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

            # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * 255.)
            aug_annots.append({img_name: aug_annot})
            aug_img_idx = aug_img_idx + 1
    
    return rows, num_classes, aug_annots
    
# Cari jumlah class pada tiap image
training_set = CocoWithoutResizer(root_dir=os.path.join('datasets', 'malaria'), set='train',
                                  transform=A.OneOf([A.RandomCrop(1200, 1200, p=0.5),
                                                     A.RandomCrop(1100, 1100, p=0.5),
                                                     A.RandomCrop(1000, 1000, p=1.0),
                                                     A.RandomCrop(950, 950, p=0.5),
                                                     A.RandomCrop(900, 900, p=1.0),
                                                     A.RandomCrop(850, 850, p=0.5),
                                                     A.RandomCrop(800, 800, p=1.0),
                                                     A.RandomCrop(750, 750, p=0.5),
                                                     A.RandomCrop(700, 700, p=1.0)],
                                                    p=1.0)
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
# FINE TUNING
_, num_classes, aug_annots = generate_augmented_images(df, training_set, 'leukocyte', ['trophozoite'], 12,
                                                       num_classes, 'datasets/malaria/augmentation', 0, aug_annots)
_, num_classes, aug_annots = generate_augmented_images(df, training_set, 'leukocyte', ['trophozoite', 'ring'], 1,
                                                       num_classes, 'datasets/malaria/augmentation', 0, aug_annots)
_, num_classes, aug_annots = generate_augmented_images(df, training_set, 'schizont', ['trophozoite'], 3,
                                                       num_classes, 'datasets/malaria/augmentation', 0, aug_annots)
_, num_classes, aug_annots = generate_augmented_images(df, training_set, 'gametocyte', ['trophozoite'], 5,
                                                       num_classes, 'datasets/malaria/augmentation', 0, aug_annots)
print("The distribution of the class", num_classes)

# SAVE TO JSON FORMAT
augmented_annotations = {}
for annotation in aug_annots:
    for key, val in annotation.items():
        augmented_annotations[key] = val

json_object = json.dumps(augmented_annotations, indent = 4) 
with open('datasets/malaria/augmentation/augmented_annotations.json', "w") as outfile: 
    outfile.write(json_object) 