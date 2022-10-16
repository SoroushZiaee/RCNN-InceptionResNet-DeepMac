import argparse
from helper_function import run_app, prepare_RCNN_model, prepare_model
import time
import gc

import imageio


def parse_args():
    parser = argparse.ArgumentParser()
    # For specifying on WANDB
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--output_data", type=str)

    return parser.parse_args()


def run(input_data: str, output_data: str):
    path = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

    detector = prepare_RCNN_model(path)

    path = "models/research/object_detection/test_data/deepmac_1024x1024_coco17/saved_model"

    prediction_function = prepare_model(path)

    video_reader = imageio.get_reader(input_data)

    video_writer = imageio.get_writer(output_data)

    start = time.time()

    for image in video_reader:

        new_frame = run_app(
            image,
            detector,
            prediction_function,
            max_boxes=10,
            min_score=0.3,
            darken_image=0.5,
        )

        del image
        gc.collect()

        video_writer.append_data(new_frame)
    end = time.time()
    print(f"Runtime of the program is {end - start}")

    video_writer.close()


def main(conf):
    print(conf)
    input_data = conf["input_data"]
    output_data = conf["output_data"]
    run(input_data, output_data)


if __name__ == "__main__":
    conf = vars(parse_args())
    main(conf)
