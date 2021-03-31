import argparse
import json
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--path', type=str, default='original/training.json')
parser.add_argument('--saved_path', type=str, default='coco_format/instances_train.json')
args = parser.parse_args()

wh_to_rc = {'width': 'c',
            'height': 'r'}

class_to_id = {
    # 'red blood cell': 1,
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
        
        # Get the annotations
        for annot in data['objects']:
            # Check whether the class is inside the classes_list or not
            if annot['category'] not in classes_list:
                continue
            
            # If the image contain at least one annotation
            contain_class = True
            
            # Get COCO-format bounding box
            bb = annot['bounding_box']
            x = bb['minimum'][wh_to_rc['width']]
            y = bb['minimum'][wh_to_rc['height']]
            w = bb['maximum'][wh_to_rc['width']] - bb['minimum'][wh_to_rc['width']]
            h = bb['maximum'][wh_to_rc['height']] - bb['minimum'][wh_to_rc['height']]

            annotations.append({
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
        if not contain_class: continue
        
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