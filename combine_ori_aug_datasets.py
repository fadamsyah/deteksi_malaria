import os
import json
import copy

# Program ini dibuat untuk menggabungkan annotasi dataset coco
# dengan annotasi dataset yang diaugmentasi

PATH_ANNOT_ORI = 'datasets/malaria/annotations/instances_train.json'
PATH_ANNOT_AUG = 'datasets/malaria/annotations/augmented_annotations.json'
PATH_TARGET = 'datasets/malaria/annotations/instances_train_combined.json'

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

# Fungsi untuk membaca json
def read_json(path_json):
    f = open(path_json,)
    data = json.load(f)
    f.close()
    
    return data

annot_ori = read_json(PATH_ANNOT_ORI)
annot_aug = read_json(PATH_ANNOT_AUG)

img_id = max([img['id'] for img in annot_ori['images']]) + 1
ann_id = max([ann['id'] for ann in annot_ori['annotations']]) + 1

# Membuat anotasi gambar dan objek dataset hasil augmentasi
# sehingga sesuai dengan format COCO
images = []
annotations = []
for key, val in annot_aug.items():
    
    for ann in val['annotations']:
        annotations.append({
            'id': ann_id,
            'bbox': ann[:4],
            'image_id': img_id,
            'category_id': class_to_cat_id[ann[-1]],
            'segmentation': [],
            'area': ann[-2] * ann[-3],
            'iscrowd': 0
        })
        ann_id = ann_id + 1
    
    images.append({
        'file_name': key,
        'height': val['description']['height'],
        'width': val['description']['width'],
        'id': img_id
    })
    img_id = img_id + 1

annot_comb = copy.deepcopy(annot_ori)
annot_comb['images'] = annot_comb['images'] + images
annot_comb['annotations'] = annot_comb['annotations'] + annotations

# SAVE TO JSON FORMAT
json_object = json.dumps(annot_comb, indent = 4) 
with open(PATH_TARGET, "w") as outfile: 
    outfile.write(json_object) 

# print(ann)
# print(annot_ori['annotations'][-1])
# print(annotations[0])
# print(key, val)
# print(annot_ori['images'][-1])
# print(images[0])
# print(annotations[-1])
# print(images[-1])

# print(len(annot_ori['images']), len(images), len(images) + len(annot_ori['images']))
# print(images[-1])


# print(len(annot_ori['images']), len(images), len(images) + len(annot_ori['images']), len(annot_comb['images']))
# print(len(annot_ori['annotations']), len(annotations), len(annotations) + len(annot_ori['annotations']), len(annot_comb['annotations']))

