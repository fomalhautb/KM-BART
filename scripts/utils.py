import sys

import numpy as np
import torch
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures.boxes import Boxes


def print_segment_line(info=''):
    sys.stderr.flush()
    print((' ' + info.strip() + ' ').center(50, '='), flush=True)


def build_cfg(gpu_id=None, threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml")
    if gpu_id is not None:
        cfg.MODEL.DEVICE = 'cuda:' + str(gpu_id)
    else:
        cfg.MODEL.DEVICE = 'cpu'
    return cfg


def extract_features(raw_image, raw_boxes, predictor):
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())

    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]

        # Transform image
        image = predictor.aug.get_transform(raw_image).apply_image(raw_image)

        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)

        # Preprocess image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

        return feature_pooled


def extract_labels(raw_image, raw_boxes, predictor):
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())

    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]

        # Transform image
        image = predictor.aug.get_transform(raw_image).apply_image(raw_image)

        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)

        # Preprocess image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        label_scores, bounding_box_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)

        return label_scores


def extract_bboxes(raw_image, predictor, gpu_id):
    # print(gpu_id)
    raw_boxes = [Boxes(torch.from_numpy(np.array([[0, 0, im.shape[1], im.shape[0]]])).cuda(gpu_id)) for im in raw_image]
    # print(len(raw_boxes))
    with torch.no_grad():
        raw_height = [im.shape[0] for im in raw_image]
        raw_width = [im.shape[1] for im in raw_image]

        # Transform image
        # Option1: first augumentation, then find bbox, same as VCG
        # image = [predictor.aug.get_transform(im).apply_image(im) for im in raw_image]
        # Ootion2: No augumentation, find bbox from original image
        image = []
        for index, im in enumerate(raw_image):
            ret = Image.fromarray(im).transform(
                size=(raw_width[index], raw_height[index]),
                method=Image.EXTENT,
                data=[0, 0, raw_width[index], raw_height[index]]
            )
            image.append(np.asarray(ret))
        # ==================end==============================================

        # Preprocess image
        image = [torch.as_tensor(im.astype("float32").transpose(2, 0, 1)) for im in image]
        inputs = [{"image": image[i], "height": raw_height[i], "width": raw_width[i]} for i in range(len(image))]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        instances, _ = predictor.model.roi_heads(images, features, proposals, None)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [Boxes.cat([x.pred_boxes, raw_boxes[i]]) for i, x in enumerate(instances)]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(features, proposal_boxes)
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        label_scores, _ = predictor.model.roi_heads.box_predictor(feature_pooled)

        return proposal_boxes, feature_pooled, label_scores
