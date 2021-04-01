import argparse
import json
import os

'''
    FITUR:
        1) Neglect RBCs
        2) Neglect difficult class
        3) Neglect gambar yang hanya mengandung trophozoite saja di Training set
'''

# TRAINING SET
# - leukocyte         103
# - trophozoite      1473
# - schizont          179
# - ring              353
# - gametocyte        144
# - difficult         441

# VALIDATION/TEST SET
# 'leukocyte': 0,
# 'trophozoite': 111,
# 'schizont': 11,
# 'ring': 169,
# 'gametocyte': 12,
# 'difficult': 5

# Karena kelas difficult itu ga jelas dan banyak di training set tapi sedikit di validation set (cuman 5)
# maka gambar yang hanya mengandung kelas difficult bisa dibuang.

# UNTUK DI-TRAINING SET
# Karena kelasnya ini *heavily imbalance* di **Trophozoite**, maka gambar yang hanya mengandung tropozoit saja bisa dibuang.

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str, default='original/training.json')
parser.add_argument('--saved_path', type=str, default='coco_format/instances_train.json')
parser.add_argument('--ignore_trophozoite', type=str, default='true')
args = parser.parse_args()

wh_to_rc = {'width': 'c',
            'height': 'r'}

class_to_id = {
    'leukocyte': 1,
    'trophozoite': 2,
    'schizont': 3,
    'ring': 4,
    'gametocyte': 5,
    'difficult': 6
}
classes_list = list(class_to_id.keys())

categories = [{
    "supercategory": "none",
    "name": key,
    "id": val
} for key, val in class_to_id.items()]

def preprocessing(name, saved_path_json, wh_to_rc=wh_to_rc,
                  class_to_id=class_to_id, categories=categories):    
    f = open(name,)
    dataset = json.load(f) 
    f.close() 

    images = []
    annotations = []

    image_id = 1
    annot_id = 1

    # classes = {}

    for data in dataset:
        contain_class = False
        contain_difficult = False
        only_trophozoite = True
        
        # Get the annotations
        annots = []
        for annot in data['objects']:
            # Check whether the class is inside the classes_list or not
            if annot['category'] not in classes_list:
                continue
            if annot['category'] == 'difficult':
                contain_difficult = True
                break
            if annot['category'] != 'trophozoite':
                only_trophozoite = False
            
            # If the image contain at least one annotation
            contain_class = True
            
            # Get COCO-format bounding box
            bb = annot['bounding_box']
            x = bb['minimum'][wh_to_rc['width']]
            y = bb['minimum'][wh_to_rc['height']]
            w = bb['maximum'][wh_to_rc['width']] - bb['minimum'][wh_to_rc['width']]
            h = bb['maximum'][wh_to_rc['height']] - bb['minimum'][wh_to_rc['height']]

            annots.append({
                "id": annot_id,
                "bbox": [x, y, w, h],
                "image_id": image_id,
                "category_id": class_to_id[annot['category']],
                "segmentation": [],
                "area": w*h,
                "iscrowd": 0
            })

            # Update the annot_id
            annot_id = annot_id + 1

        # If the image does not contain any annotation,
        # just neglect the image
        if (not contain_class) or contain_difficult or (only_trophozoite and (args.ignore_trophozoite.lower()=='true')):
            continue
        
        # Get the annotations
        annotations.extend(annots)
        
        # Get the image description
        image = data['image']
        images.append({
            "file_name": os.path.basename(image['pathname']),
            "height": image['shape'][wh_to_rc['height']],
            "width": image['shape'][wh_to_rc['width']],
            "id": image_id
        })

        # Update the image_id
        image_id = image_id + 1 

    instances = {
        "type": "instances",
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    json_object = json.dumps(instances, indent = 4) 

    # Writing to saved_path_json
    with open(saved_path_json, "w") as outfile: 
        outfile.write(json_object) 
        
if __name__ == "__main__":
    preprocessing(args.path, args.saved_path)