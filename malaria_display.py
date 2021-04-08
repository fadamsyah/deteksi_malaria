import argparse
import json
import cv2
import numpy as np
from utils.utils import boolean_string, url_to_image, STANDARD_COLORS, standard_to_bgr, get_index_label

# Parameters, yang susah kalau ditaro di CLI(?), harus pake yaml
anchors_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
anchors_ratios = [(0.9, 1.1), (1.0, 1.0), (1.1, 0.9)]
obj_list = ['leukocyte', 'trophozoite',
           'schizont', 'ring', 'gametocyte']
cat_id_to_cat = {i+1: key for i, key in enumerate(obj_list)}

color_list = standard_to_bgr(STANDARD_COLORS)

def get_args():
    parser = argparse.ArgumentParser('Deteksi Sel Telur Fasciola')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='coefficients of efficientdet')
    parser.add_argument('-w', '--weights_path', type=str, default='weights/efficientdet-d2_15_1712.pth',
                        help='path weights-nya')
    parser.add_argument('--use_cuda', type=boolean_string, default=False,
                        help='True kalau mau pake GPU yang ada CUDA nya. False kalau pakai CPU aja')
    parser.add_argument('--img_path', type=str, default='datasets/malaria/train/d9345456-b76d-46cc-b81e-46d4a0a7b652.png',
                        help='path image nya')
    parser.add_argument('--use_url', type=boolean_string, default=False,
                        help='True apabila menggunakan url untuk load gambarnya. False kalau load gambar dari local')
    parser.add_argument('--saved_path_json', type=str, default=None, help='path hasil deteksi (json)')
    #############################
    parser.add_argument('--display', type=boolean_string, default=False)
    parser.add_argument('--saved_path_img', type=str, default=None)
    parser.add_argument('--json_ground_truth', type=str, default=None)
    #############################
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='persentase output min. untuk dianggap sebagai objek (0 - 1)')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='persentase iou max. untuk menganggap 2 objek itu berbeda (0 - 1)')

    args = parser.parse_args()
    return args

def plot_bbox(img, bbox, label, score=None, color=None, line_thickness=3, font_scale=1, font_thickness=1):
    c1, c2 = (int(bbox['xmin']), int(bbox['ymin'])), (int(bbox['xmax']), int(bbox['ymax']))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness)
    if score:
        tf = font_thickness  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)), 0, fontScale=font_scale, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0], c1[1] - 2), 0, font_scale, [0, 0, 0],
                    thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)

    
def display_predictions(pred, ori_img, imshow=True, imwrite=False,
                        target_name='sample.jpg', use_url=False):
    annotations = pred['analysis_results']
    
    img = ori_img.copy()
    if len(annotations) > 0:
        for annot in annotations:
            bbox = annot['bbox']
            category = annot['category']
            score = annot['score']
            
            plot_bbox(img, bbox, label=category, score=score,
                      color=color_list[get_index_label(category, obj_list)])
        
    return img
    
# Fungsi untuk membaca json
def read_json(path_json):
    f = open(path_json,)
    data = json.load(f)
    f.close()
    
    return data

def display_ground_truth(image, annotations, category_id_to_name=cat_id_to_cat):
    
    BOX_COLOR = (255, 0, 0) # Red
    TEXT_COLOR = (255, 255, 255) # White
    
    def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=2):
        """Visualizes a single bounding box on the image"""
        x_min, y_min, x_max, y_max = map(int, bbox[:4])
        x_max = x_max + x_min
        y_max = y_max + y_min
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=BOX_COLOR, thickness=thickness)

        font_scale = 0.75
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)    
        cv2.rectangle(img, (x_min, y_max - int(1.3 * text_height)), (x_min + text_width, y_max), color, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_max - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, 
            color=TEXT_COLOR, 
            lineType=cv2.LINE_AA,
        )
        return img
    
    img = image.copy()
    # for bbox, category_id in zip(bboxes, category_ids):
    for annot in annotations:
        class_name = category_id_to_name[annot['category_id']]
        img = visualize_bbox(img, annot['bbox'], class_name)
    
    return img

if __name__ == "__main__":
    opt = get_args()
    
    # [IMPORTANT] Add additional sys path before importing DetectionModule
    import os, sys
    sys.path.insert(0, os.path.join(os.getcwd(), 'deteksi_malaria'))
    from detector.detector import DetectionModule
    
     # Panggil model
    detector = DetectionModule(opt.compound_coef, obj_list, opt.weights_path,
                               opt.use_cuda, anchors_ratios, anchors_scales)
    
    # Outputnya adalah dictionary 
    output = detector(opt.img_path, opt.use_url,
                      opt.threshold, opt.iou_threshold)
    
    # Kalau mau save json nya
    if opt.saved_path_json:
        json_object = json.dumps(output, indent = 4)
        with open(opt.saved_path_json, "w") as outfile: 
            outfile.write(json_object)
    
    if opt.display or opt.saved_path_img:
        if not opt.use_url: img = cv2.imread(opt.img_path)
        else: img = url_to_image(opt.img_path)
        
        imwrite = True if opt.saved_path_img else False
        img = display_predictions(output, img, opt.display,
                                  imwrite, opt.saved_path_img)
        
        if opt.json_ground_truth:
            datasets = read_json(opt.json_ground_truth)
            img_id = None
            for image in datasets['images']:
                if image['file_name'] == os.path.basename(opt.img_path):
                    img_id = image["id"]
                    break
                
            # Kalau ada annotasi di json groundtruth
            if img_id:
                annotations = []
                for annot in datasets["annotations"]:
                    if annot["image_id"] == img_id:
                        annotations.append(annot)
                img = display_ground_truth(img, annotations)
                
        if imwrite:
            cv2.imwrite(opt.saved_path_img, img)
        
        if opt.display:    
            cv2.imshow('img', img)
            cv2.waitKey(0)