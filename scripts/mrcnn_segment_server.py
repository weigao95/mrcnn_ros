#! /usr/bin/env python
import os
import torch
import numpy as np
import cv2
from torchvision import transforms as T
import argparse

# The argument
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str,
                    default='/home/wei/catkin_ws/src/mrcnn_ros/config/e2e_mask_rcnn_R_50_FPN_1x_caffe2_singleobj.yaml',
                    help='Absolute path to maskrcnn_benchmark config.')
parser.add_argument('--net_path', type=str,
                    default='/home/wei/data/pdc/coco/output_mug/model_0120000.pth',
                    help='Absolute path to trained network. If empty, use the network in the config file.')

# The maskrcnn staff
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker

# The ros staff
from mrcnn_ros.srv import *
import rospy
from sensor_msgs.msg import RegionOfInterest
from geometry_msgs.msg import Point
from cv_bridge import CvBridge, CvBridgeError

class COCODPredictor(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        # Just return the top predictions
        return top_predictions


class MaskRCNNInstanceSegmentationServer(object):

    def __init__(self, config_path, net_path, use_gpu=False):
        # The converter of opencv image
        self._bridge = CvBridge()

        # Basic check
        assert os.path.exists(config_path)
        assert os.path.exists(net_path)

        # update the config options with the config file
        cfg.merge_from_file(config_path)
        if len(net_path) > 0:
            cfg.merge_from_list(["MODEL.WEIGHT", net_path])
        if not use_gpu:
            cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

        # Construct the predictor
        self._predictor = COCODPredictor(cfg, min_image_size=800, confidence_threshold=0.7)

    def handle_segmentation_request(self, request):  # type: (InstanceSegmentationRequest) -> InstanceSegmentationResponse
        # Decode the image
        try:
            cv_color = self._bridge.imgmsg_to_cv2(request.rgb_image, 'bgr8')
        except CvBridgeError as err:
            print('Image conversion error. Please check the image encoding.')
            print(err.message)
            return self.get_invalid_response()
        
        # Do prediction
        predictions = self._predictor.run_on_opencv_image(cv_color)
        masks = predictions.get_field('mask').numpy()
        labels = predictions.get_field('labels').numpy()
        boxes = predictions.bbox.numpy()
        num_obj = masks.shape[0]

        # Initialize and clear
        response = InstanceSegmentationResponse()
        response.binary_masks = []
        response.object_names = []
        response.bounding_boxes = []

        # Write the result
        for i in range(num_obj):
            # Insert the object name
            label_i = int(labels[i])
            if label_i < len(self._predictor.CATEGORIES):
                name_i = self._predictor.CATEGORIES[label_i]
                response.object_names.append(name_i)
            else:
                response.object_names.append('category-%d' % i)

            # Insert the bounding box
            roi = RegionOfInterest()
            roi.x_offset = int(boxes[i, 0])
            roi.y_offset = int(boxes[i, 1])
            roi.width = int(boxes[i, 2] - boxes[i, 0])
            roi.height = int(boxes[i, 3] - boxes[i, 1])
            response.bounding_boxes.append(roi)

            # Insert the mask
            mask_i = masks[i, 0,:,:].astype(np.uint16)
            mask_i_msg = self._bridge.cv2_to_imgmsg(mask_i)
            response.binary_masks.append(mask_i_msg)

        # Insert the result
        return response
    
    @staticmethod
    def get_invalid_response():  # type: () -> InstanceSegmentationResponse
        response = InstanceSegmentationResponse()
        response.object_names.clear()
        response.bounding_boxes.clear()
        response.binary_masks.clear()
        return response
    
    def run(self):
        rospy.init_node('mrcnn_instance_segmentation_server')
        srv = rospy.Service('instance_segment', InstanceSegmentation, self.handle_segmentation_request)
        print('The server for maskrcnn_benchmark instance segmentation initialization OK!')
        rospy.spin()


def main():
    args = parser.parse_args()
    net_path = args.net_path
    config_path = args.config_path
    server = MaskRCNNInstanceSegmentationServer(config_path, net_path)
    server.run()


if __name__ == '__main__':
    main()    
