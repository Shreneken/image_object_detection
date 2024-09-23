This project uses `pipenv` as its virtualenv management tool, which you will need to have installed:

1. Install pipenv
```bash
pip install pipenv
```

Now, just run the following command to install all the dependencies

2. Install dependencies
```bash
pipenv install
```

3. Downloading the models

We want to download the models and store them inside `ml/models/`. To achieve this, you can do the following:
```bash
cd ml
mkdir models
cd models
# Download tiny-yolov3 (size = 34 mb)
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/tiny-yolov3.pt 
# Download yolov3 (size = 237 mb)
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt
# Download retina-net (size = 130 mb)
wget https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/retinanet_resnet50_fpn_coco-eeacb38b.pth

```

### Starting the server

Ensure you are in the project directory and then simply: 
```bash
python -m backend.model_server
```

If you get an import error when running the server, you might need to do the following:
```bash
export PYTHONPATH=pwd # path to this project directory
```
or if you are using Windows:
```pwsh
set PYTHONPATH=%PYTHONPATH%;C:\project_path
```

### Client example

```bash
python client_example.py
```

### Command line tool

For the command line tool, you will need to specify the path to your input image, the path & name that you want for your output image and select one from the following models: retina-net, yolov3, tiny-yolov3.

input image uses the `--in_img` flag, output image uses the `--out_img` flag and specifying the model uses the `--model` flag.

Example usage:
```bash
python cmd_interface.py --in_img ./input_images/cars.png --out_img ./output_images/new_car_2.png --model yolov3
```

There are sample images present in the `input_images` directory and a `output_images` directory where you can store your results as a way to test out this project.