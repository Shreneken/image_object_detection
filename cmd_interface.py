# Command line interface for image object detection
import argparse

from ml.detection_model import Detection_Model
from ml.model_utils import Model_Handler


def main():
    parser = argparse.ArgumentParser(description="Detect Objects in Images")
    parser.add_argument(
        "--in_img",
        type=str,
        help="Your input image file path",
        default=None,
    )
    parser.add_argument(
        "--out_img",
        type=str,
        help="The output image file path where the model will save its result",
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for object detection, [`yolov3`| `tiny-yolov3`| `retina-net`]",
        default=None,
    )
    parser.add_argument(
        "--min_prob",
        type=int,
        help="The min percentage probability for classifying an an object",
        default=30,
    )
    args = parser.parse_args()

    model_type = Model_Handler.get_type(args.model)
    model_path = Model_Handler.get_path(model_type)

    input_image = args.in_img
    output_image = args.out_img
    min_prob = args.min_prob

    if input_image is None or output_image is None or args.model is None:
        raise ValueError("All --in_img, --out_img and --model must be provided")

    model = Detection_Model(model_type, model_path)
    model.initialize()

    detections = model.predict(input_image, output_image, min_prob)

    print(f"Output image for {input_image} saved to output_images/{output_image}")
    for eachObject in detections:
        print(
            eachObject["name"],
            " : ",
            eachObject["percentage_probability"],
            " : ",
            eachObject["box_points"],
        )
        print("--------------------------------")


if __name__ == "__main__":
    main()
