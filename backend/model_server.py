from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import ResponseModel, CustomInput, ImageResult

from ml.detection_model import Detection_Model
from ml.model_utils import * 



class Model_Server:

    server = MLServer(__name__)    
    
    """ 
    Expected request json format in /detect endpoint:

        inputs = [
            {
                "input": [
                    {
                        "type": "input",
                        "value": "./input_images/cars.png",
                    },
                    {
                        "type": "output",
                        "value": "./output_images/client_sample_output.png",
                    },
                    {"type": "model", "value": "retina-net"},
                ]
            }
        ] 

    """
    @server.route('/detect', input_type=DataTypes.CUSTOM)
    def detect(input: list[CustomInput], parameters: dict):
        
        print('Inputs:', input)
        print('Parameters:', parameters)
        
        input_path, output_path, model_type, model_path = Input_Handler.parse_input(input)
        min_perc_prob = Parameter_Handler.parse_parameter(parameters, str="min_perc_prob")

        model = Detection_Model(model_type=model_type, model_path=model_path)

        results = model.predict(input_path, output_path, min_perc_prob)
        results = [ImageResult(file_path=input_path, result=results)]
        response = ResponseModel(results=results)
        
        return response.get_response()
    

    @classmethod
    def start_server(model_server):
        model_server.server.run()

if __name__ == "__main__":
    model_server = Model_Server()
    model_server.start_server()
