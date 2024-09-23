import os
from pprint import pprint

from imageai.Detection import ObjectDetection

from ml.model_utils import *


class Detection_Model:
    def __init__(self, model_type: Model_Type, model_path):
        self.detector_model = ObjectDetection()
        self.exec_path = os.getcwd()
        self.model_type = model_type
        self.model_path = model_path
        self.initialize()

    def initialize(self):
        Model_Handler.set_model_type(self.detector_model, self.model_type)
        self.detector_model.setModelPath(os.path.join(self.exec_path, self.model_path.value))
        self.detector_model.loadModel()

    def predict(self, input_image_path, output_image_path, min_perc_prob):
        return self.detector_model.detectObjectsFromImage(
            input_image=os.path.join(self.exec_path, input_image_path),
            output_image_path=os.path.join(self.exec_path, output_image_path),
            minimum_percentage_probability=min_perc_prob,
        )


if __name__ == "__main__":
    det = Detection_Model(Model_Type.retina_net, Model_Path.retina_net)
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
