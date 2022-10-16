import tensorflow as tf
import tensorflow_hub as hub

import warnings
import numpy as np
import random
import logging


from skimage import util
from skimage import transform
from skimage.color import rgb_colors
from skimage import color


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib import patches


from object_detection.utils import ops

COLORS = [
    rgb_colors.cyan,
    rgb_colors.orange,
    rgb_colors.pink,
    rgb_colors.purple,
    rgb_colors.limegreen,
    rgb_colors.crimson,
] + [(color) for (name, color) in color.color_dict.items()]
random.shuffle(COLORS)

logging.disable(logging.WARNING)


def load_imgs(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def run_detector(detector, img):

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key: value.numpy() for key, value in result.items()}

    return (
        result["detection_boxes"],
        result["detection_class_entities"],
        result["detection_scores"],
    )


def prepare_RCNN_model(path):
    return hub.load(path).signatures["default"]


def resize_for_display(image, max_height=600):
    height, width, _ = image.shape
    width = int(width * max_height / height)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return util.img_as_ubyte(transform.resize(image, (height, width)))


def get_mask_prediction_function(model):
    """Get single image mask prediction function using a model."""

    @tf.function
    def predict_masks(image, boxes):
        height, width, _ = image.shape.as_list()
        batch = image[tf.newaxis]
        boxes = boxes[tf.newaxis]

        detections = model(batch, boxes)
        masks = detections["detection_masks"]

        return ops.reframe_box_masks_to_image_masks(masks[0], boxes[0], height, width)

    return predict_masks


def plot_image_annotations(image, boxes, masks, darken_image=0.5):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.subplots()
    ax.set_axis_off()

    image = image.numpy()

    image = (image * darken_image).astype(np.uint8)
    ax.imshow(image)

    height, width, _ = image.shape

    num_colors = len(COLORS)
    color_index = 0

    for box, mask in zip(boxes, masks):
        ymin, xmin, ymax, xmax = box
        ymin *= height
        ymax *= height
        xmin *= width
        xmax *= width

        color = COLORS[color_index]
        color = np.array(color)
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        mask = (mask > 0.5).numpy().astype(np.uint8)
        color_image = np.ones_like(image) * color[np.newaxis, np.newaxis, :]
        color_and_mask = np.concatenate([color_image, mask[:, :, np.newaxis]], axis=2)

        ax.imshow(color_and_mask, alpha=0.5)

        color_index = (color_index + 1) % num_colors

    canvas.draw()
    ax = np.array(canvas.renderer.buffer_rgba())

    return ax.astype(np.uint8)


def best_boxes(boxes, scores, max_boxes=10, min_score=0.1):
    boxes = boxes[: len(scores[scores > min_score])]
    num_boxes = min(boxes.shape[0], max_boxes)
    return boxes[:num_boxes]


def prepare_model(model_path):

    print("Loading SavedModel")
    model = tf.keras.models.load_model(model_path)
    return get_mask_prediction_function(model)


def run_app(
    image, detector, prediction_function, max_boxes=10, min_score=0.1, darken_image=0.5
):

    image = tf.convert_to_tensor(np.array(image), dtype="uint8")

    boxes, detection_class_entities, detection_scores = run_detector(detector, image)
    boxes = best_boxes(boxes, detection_scores, max_boxes, min_score)

    masks = prediction_function(
        tf.convert_to_tensor(image), tf.convert_to_tensor(boxes, dtype=tf.float32)
    )
    return plot_image_annotations(image, boxes, masks, darken_image=darken_image)
