import sys

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup

sys.path.append("bottom-up-attention.pytorch")

from models.bua.layers.nms import nms
from utils.extract_utils import get_image_blob
from models import add_config
from models.bua.box_regression import BUABoxes


class FeatureExtractor:
    def __init__(self, config_path, rank=0):
        self._rank = rank
        self._device = "cuda:{}".format(rank)
        self._cfg = self._get_cfg(config_path)
        self._model = DefaultTrainer.build_model(self._cfg)
        DetectionCheckpointer(self._model, save_dir=self._cfg.OUTPUT_DIR).resume_or_load(self._cfg.MODEL.WEIGHTS)
        self._model.eval()

    def _get_cfg(self, config_path):
        """
        Create configs and perform basic setups.
        """

        class args:
            mode = 'caffe'
            config_file = config_path

        cfg = get_cfg()
        add_config(args, cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.DEVICE = self._device
        cfg.freeze()
        default_setup(cfg, args)
        return cfg

    def extract_feature(self, image, boxes=None):
        if boxes is None:
            return self._extract_feature_without_bbox(image)
        else:
            return self._extract_feature_with_bbox(image, boxes)

    def _extract_feature_with_bbox(self, image, boxes):
        dataset_dict = get_image_blob(image, self._cfg.MODEL.PIXEL_MEAN)

        bbox = torch.from_numpy(np.array(boxes)) * dataset_dict['im_scale']
        raw_boxes = BUABoxes(bbox.cuda(self._rank))

        with torch.no_grad():
            processed_image = self._model.preprocess_image([dataset_dict])
            features = self._model.backbone(processed_image.tensor)
            proposal_boxes = [raw_boxes]
            features = [features[f] for f in self._model.roi_heads.in_features]
            box_features = self._model.roi_heads._shared_roi_transform(features, proposal_boxes)
            features_pooled = box_features.mean(dim=[2, 3])
            scores = self._model.roi_heads.box_predictor(features_pooled)

        dets = proposal_boxes[0].tensor.cpu() / dataset_dict['im_scale']
        scores = F.softmax(scores[0], dim=1).cpu()
        feats = features_pooled.cpu()

        return {
            'boxes': dets.numpy(),
            'scores': scores.numpy(),
            'features': feats.numpy()
        }

    def _extract_feature_without_bbox(self, image):
        MIN_BOXES = self._cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
        MAX_BOXES = self._cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
        CONF_THRESH = self._cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

        dataset_dict = get_image_blob(image, self._cfg.MODEL.PIXEL_MEAN)

        with torch.no_grad():
            # TODO add full image feature
            proposal_boxes, scores, features_pooled, _ = self._model([dataset_dict])

        dets = proposal_boxes[0].tensor.cpu() / dataset_dict['im_scale']
        scores = scores[0].cpu()
        feats = features_pooled[0].cpu()

        max_conf = torch.zeros((scores.shape[0])).to(scores.device)
        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.3)
            max_conf[keep] = torch.where(
                cls_scores[keep] > max_conf[keep],
                cls_scores[keep],
                max_conf[keep]
            )

        keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()

        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]

        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]

        image_feat = feats[keep_boxes].numpy()
        image_bboxes = dets[keep_boxes].numpy()
        image_scores = scores[keep_boxes].numpy()

        return {
            'boxes': image_bboxes,
            'scores': image_scores,
            'features': image_feat
        }
