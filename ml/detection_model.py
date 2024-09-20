import os
from enum import Enum, auto

from imageai.Detection import ObjectDetection
from pprint import pprint


class Model_Paths(Enum):

    yolov3 = "./ml/models/yolov3.pt"
    tiny_yolov3 = "./ml/models/tiny-yolov3.pt"
    retina_net = "./ml/models/retinanet_resnet50_fpn_coco-eeacb38b.pth"


class Model_Handler(Enum):

    yolov3 = auto()
    tiny_yolov3 = auto()
    retina_net = auto()

    @staticmethod
    def set_model_type(model, model_type):
        if model_type == Model_Handler.yolov3:
            model.setModelTypeAsYOLOv3()
        elif model_type == Model_Handler.tiny_yolov3:
            model.setModelTypeAsTinyYOLOv3()
        elif model_type == Model_Handler.retina_net:
            model.setModelTypeAsRetinaNet()
        else:
            raise Exception("No such model type!")

    @staticmethod
    def get_type(type_in_str: str):
        return (
            Model_Handler.yolov3
            if type_in_str == "yolov3"
            else Model_Handler.tiny_yolov3
            if type_in_str == "tiny-yolov3"
            else Model_Handler.retina_net
            if type_in_str == "retina-net"
            else Exception("No such model type!")
        )

    @staticmethod
    def get_path(model_type):
        return (
            Model_Paths.retina_net
            if model_type == Model_Handler.retina_net
            else Model_Paths.tiny_yolov3
            if model_type == Model_Handler.tiny_yolov3
            else Model_Paths.yolov3
            if model_type == Model_Handler.yolov3
            else Exception("No such model!")
        )


class Detection_Model:
    def __init__(self, model_type: Model_Handler, model_path):
        self.detector_model = ObjectDetection()
        self.exec_path = os.getcwd()
        self.model_type = model_type
        self.model_path = model_path

    def initialize(self):
        Model_Handler.set_model_type(self.detector_model, self.model_type)
        self.detector_model.setModelPath(
            os.path.join(self.exec_path, self.model_path.value)
        )
        self.detector_model.loadModel()

    def predict(self, input_image_path, output_image_path, min_perc_prob):
        return self.detector_model.detectObjectsFromImage(
            input_image=os.path.join(self.exec_path, input_image_path),
            output_image_path=os.path.join(
                self.exec_path, output_image_path
            ),
            minimum_percentage_probability=min_perc_prob,
        )


if __name__ == "__main__":
    det = Detection_Model(Model_Handler.retina_net, Model_Paths.retina_net)
    det.initialize()
    detections = det.predict("./input_images/cars.png", "./output_images/cars_retina.png", 30)

    for eachObject in detections:
        print(
            eachObject["name"],
            " : ",
            eachObject["percentage_probability"],
            " : ",
            eachObject["box_points"],
        )
        print("--------------------------------")

    pprint(detections)
