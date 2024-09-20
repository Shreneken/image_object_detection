from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

url = "http://127.0.0.1:5000/detect"  # The URL of the server
client = MLClient(url)  

inputs = [
    {
        "input": {
            "input": "./input_images/cars.png",
            "output": "./output_images/client_sample_output.png",
            "model": "retina-net",
        },
    }
] 
input_data_type = DataTypes.CUSTOM  
parameters = {
    "min_perc_prob": 30
}

response = client.request(
    inputs=inputs, data_type=input_data_type, parameters={"min_perc_prob": 30}
)  
print(response)  
