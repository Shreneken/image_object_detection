from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import ResponseModel, CustomInput, ImageResult

from ml.detection_model import Detection_Model, Model_Handler



server = MLServer(__name__)

@server.route('/detect', input_type=DataTypes.CUSTOM)
def detect(input: list[CustomInput], parameters: dict):
    
    print('Inputs:', input)
    print('Parameters:', parameters)
    
    input_path, output_path, model_type = None, None, None
    input_values = input[0].input
    
    input_path = input_values["input"]
    output_path = input_values["output"]

    model_type = Model_Handler.get_type(input_values["model"])
    model_path = Model_Handler.get_path(model_type)
    min_perc_prob = parameters["min_perc_prob"]

    model = Detection_Model(model_type=model_type, model_path=model_path)
    model.initialize()

    results = model.predict(input_path, output_path, min_perc_prob)
    results = [ImageResult(file_path=input_path, result=results)]
    response = ResponseModel(results=results)
    
    return response.get_response()


server.run()

# Expected request json format:
#
# inputs = [
#     {
#         "input": [
#             {
#                 "type": "input",
#                 "value": "./input_images/cars.png",
#             },
#             {
#                 "type": "output",
#                 "value": "./output_images/client_sample_output.png",
#             },
#             {"type": "model", "value": "retina-net"},
#         ]
#     }
# ]