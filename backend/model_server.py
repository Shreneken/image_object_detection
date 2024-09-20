from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import ResponseModel, TextResult

from ml.detection_model import Detection_Model

# create an instance of the model
model = Detection_Model()

# Create a server
server = MLServer(__name__)


# Create an endpoint
@server.route("/detection_model", DataTypes.TEXT)
def process_text(inputs: list, parameters: dict) -> dict:
    results = model.predict(inputs)
    results = [TextResult(text=e["text"], result=r) for e, r in zip(inputs, results)]
    response = ResponseModel(results=results)
    return response.get_response()


# Run the server (optional. You can also run the server using the command line)
server.run()

# Expected request json format:
# {
#     "inputs": [
#         {"text": "Text to be classified"},
#         {"text": "Another text to be classified"}
#     ],
#     "data_type": "TEXT",
#     "parameters": {}
# }
