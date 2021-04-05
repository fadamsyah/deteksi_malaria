import sys
import os
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from utils.utils import boolean_string

class DetectionModule():
    def __init__(self, compound_coef, obj_list, weights_path, use_cuda,
                 anchor_ratios = [(0.9, 1.2), (1.0, 1.0), (1.2, 0.8)],
                 anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]):
        # Use Cuda?
        self.use_cuda = use_cuda
        
        # Load input size
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = input_sizes[compound_coef]
        
        # Load Model
        self.model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                    ratios=anchor_ratios, scales=anchor_scales)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.requires_grad_(False)
        self.model.eval()
        if self.use_cuda: self.model = self.model.cuda()
        
        cudnn.fastest = True
        cudnn.benchmark = True
        
        # Object list
        self.obj_list = obj_list
        
        # Color List
        self.color_list = standard_to_bgr(STANDARD_COLORS)
    
    def __call__(self, img_path, use_url,
                 threshold=0.5, iou_threshold=0.5):
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=self.input_size, use_url=use_url)

        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

        out = invert_affine(framed_metas, out)

        output = {"analysis_results": []}
        for i in range(len(out[0]['rois'])):
            (x1, y1, x2, y2) = out[0]['rois'][i]
            entity = {
                "bbox": {
                    "xmax": int(x2 + 0.5),
                    "xmin": int(x1 + 0.5),
                    "ymax": int(y2 + 0.5),
                    "ymin": int(y1 + 0.5)
                },
                "category": self.obj_list[out[0]['class_ids'][i]],
                "score": float(out[0]['scores'][i])
            }
            output["analysis_results"].append(entity)
            
        return output