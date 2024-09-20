from flask_ml.flask_ml_client import MLClient
from flask_ml.flask_ml_server.constants import DataTypes

url = "http://127.0.0.1:5000/detect"  # The URL of the server
client = MLClient(url)  # Create an instance of the MLClient object

inputs = [
    {
        "input": {
            "input": "./input_images/cars.png",
            "output": "./output_images/client_sample_output.png",
            "model": "retina-net",
        },
    }
]  # The inputs to be sent to the server
data_type = DataTypes.CUSTOM  # The type of the input data

response = client.request(
    inputs, data_type, parameters={"min_perc_prob": 30}
)  # Send a request to the server
print(response)  # Print the response
