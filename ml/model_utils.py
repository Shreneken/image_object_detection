from enum import Enum, auto

from flask_ml.flask_ml_server.models import CustomInput


class Model_Path(Enum):
    yolov3 = "./ml/models/yolov3.pt"
    tiny_yolov3 = "./ml/models/tiny-yolov3.pt"
    retina_net = "./ml/models/retinanet_resnet50_fpn_coco-eeacb38b.pth"


class Model_Type(Enum):
    yolov3 = auto()
    tiny_yolov3 = auto()
    retina_net = auto()


class Handler:
    pass


class Model_Handler(Handler):

    @staticmethod
    def set_model_type(model, model_type):
        if model_type == Model_Type.yolov3:
            model.setModelTypeAsYOLOv3()
        elif model_type == Model_Type.tiny_yolov3:
            model.setModelTypeAsTinyYOLOv3()
        elif model_type == Model_Type.retina_net:
            model.setModelTypeAsRetinaNet()
        else:
            raise Exception("No such model type!")

    @staticmethod
    def get_type(type_in_str: str):
        return (
            Model_Type.yolov3
            if type_in_str == "yolov3"
            else (
                Model_Type.tiny_yolov3
                if type_in_str == "tiny-yolov3"
                else (
                    Model_Type.retina_net if type_in_str == "retina-net" else Exception("No such model type!")
                )
            )
        )

    @staticmethod
    def get_path(model_type):
        return (
            Model_Path.retina_net
            if model_type == Model_Type.retina_net
            else (
                Model_Path.tiny_yolov3
                if model_type == Model_Type.tiny_yolov3
                else Model_Path.yolov3 if model_type == Model_Type.yolov3 else Exception("No such model!")
            )
        )


class Input_Handler(Handler):

    @staticmethod
    def parse_input(input: list[CustomInput]) -> tuple[str, str, Model_Type, Model_Path]:

        [input_items] = input
        input_values = input_items.input
        input_path = input_values["input"]
        output_path = input_values["output"]
        model_type = Model_Handler.get_type(input_values["model"])
        model_path = Model_Handler.get_path(model_type)

        return (input_path, output_path, model_type, model_path)


class Parameter_Handler(Handler):

    @staticmethod
    def parse_parameter(parameters, str):
        return parameters[str]
