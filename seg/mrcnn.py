import numpy as np

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures.instances import Instances


############################## Parameter Setting ##############################


# model config file path
CONFIG_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# model test threshold
SCORE_THRESH_TEST = 0.5


############################## Global Process ##############################


# get model config file
config = get_cfg()
config.merge_from_file(model_zoo.get_config_file(CONFIG_PATH))

# set model threshold
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST

# set model weights (download from web)
config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_PATH)

# predict outputs
predictor = DefaultPredictor(config)

# custom visualizer
class CustomVisualizer(Visualizer):
    def _jitter(self, color):
        return color # never use jitter


############################## Function Definition ##############################


def run(image: np.ndarray) -> Instances:

    # run predictor
    result = predictor(image)

    # move result data into CPU and return it
    return result["instances"].to("cpu")


def visualize(image: np.ndarray, result: Instances, name='Mask R-CNN') -> np.ndarray:

    # convert color format from BGR to RGB
    image = image[:, :, ::-1]

    # visualize result with image
    visualizer = CustomVisualizer(image, MetadataCatalog.get(config.DATASETS.TRAIN[0]),
                                  instance_mode=ColorMode.SEGMENTATION)
    image = visualizer.draw_instance_predictions(result).get_image()

    # convert color format from RGB to BGR
    image = image[:, :, ::-1]

    # return image
    return image
