# %% Imports Packages
import math
import os
import shutil
import sys
import warnings
from datetime import datetime

sys.path.insert(0, "./evaluate")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import albumentations as A
import cv2
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from tensorflow import keras
from tqdm import tqdm

# %% Misc.
# Directory
DATA_DIR = "./data/2023-datalab-cup2-object-detection"
DATA_PATH = os.path.join(DATA_DIR, "pascal_voc_training_data.txt")
IMAGE_DIR = os.path.join(DATA_DIR, "VOCdevkit_train/VOC2007/JPEGImages/")
AUG_DATA_PATH = "./data/aug/aug_data.txt"
AUG_IMAGE_DIR = "./data/aug/images/"
OUTPUT_DIR = "./output"

# Common params
IMAGE_SIZE = 448
NUM_CLASSES = 20
MAX_OBJECTS_PER_IMAGE = 20
BATCH_SIZE = 8
NUM_IMAGES = len(open(DATA_PATH).readlines())
NUM_AUG_IMAGES = len(open(AUG_DATA_PATH).readlines())

# Model params
CELL_SIZE = 7
BOXES_PER_CELL = 2
OBJECT_SCALE = 1
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 1
COORD_SCALE = 5

# Training params
LEARNING_RATE = 1e-5
EPOCHS = 40

# Other params
AUG_TARGET_NUM = 3000
AUGMENT = False
CONF_THRESHOLD = 0.5
RANDOM_STATE = 417
CLASSES_NAME = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


# Create directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(AUG_IMAGE_DIR):
    os.makedirs(AUG_IMAGE_DIR)

# %% Check GPU
tf.config.list_physical_devices()
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Select GPU number 1
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# %% Read data information
# Annotation format: name x_center y_center width height class ...
def read_data(path):
    image_names = []
    records = []
    num_objs_in_imgs = []

    with open(path, "r") as input_file:
        for line in input_file:
            line_split = line.strip().split(" ")
            image_names.append(line_split[0])
            record = [float(num) for num in line_split[1:]]
            if len(record) > MAX_OBJECTS_PER_IMAGE * 5:
                record = record[: MAX_OBJECTS_PER_IMAGE * 5]
            records.append(record)
            num_objs_in_imgs.append(len(record) // 5)

    bboxes_in_imgs = [np.array(record).reshape(-1, 5) for record in records]

    return image_names, bboxes_in_imgs, num_objs_in_imgs


def analyze_data(path):
    _, bboxes_in_imgs, _ = read_data(path)
    class_count = np.zeros(NUM_CLASSES)

    for bboxes in bboxes_in_imgs:
        classes_in_img = bboxes[:, 4]
        for cls in classes_in_img:
            class_count[int(cls)] += 1

    class_count = pd.DataFrame(
        class_count, index=CLASSES_NAME, columns=["count"], dtype=int
    )
    return class_count


analyze_data(DATA_PATH)


# %% Image augmentation
def augment_data(augmenter, image_name, bboxes):
    img_file = tf.io.read_file(IMAGE_DIR + image_name)
    img = tf.io.decode_jpeg(img_file, channels=3)

    data = {"image": img.numpy(), "bboxes": bboxes}
    augmented = augmenter(**data)

    img_augmented = augmented["image"]
    bboxes_augmented = augmented["bboxes"]

    img_augmented = tf.cast(img_augmented, tf.uint8)
    img_augmented = tf.io.encode_jpeg(img_augmented)

    return img_augmented, bboxes_augmented


def generate_aug_data(augmenter):
    image_names, bboxes_in_imgs, num_objs_in_imgs = read_data(DATA_PATH)
    classes_in_imgs = [bboxes[:, 4].astype(int) for bboxes in bboxes_in_imgs]

    if os.path.exists(AUG_IMAGE_DIR):
        shutil.rmtree(AUG_IMAGE_DIR)

    # Sort image names by number of objects in image
    sort_images = list(
        zip(image_names, bboxes_in_imgs, classes_in_imgs, num_objs_in_imgs)
    )
    sort_images.sort(key=lambda x: x[3])

    # Augment data to balance number of classes
    images_count = np.zeros(NUM_IMAGES, dtype=int)
    class_count = np.zeros(NUM_CLASSES, dtype=int)
    file_write = open(AUG_DATA_PATH, "w")

    # Add original data to file
    for i, (image_name, bboxes, classes, _) in tqdm(
        enumerate(sort_images), desc="Original", total=len(sort_images)
    ):
        if any(class_count > AUG_TARGET_NUM):
            continue
        img_idx = image_names.index(image_name)

        # Write to file
        label = [f"{images_count[img_idx]}_{image_name}"] + [
            str(int(b)) for bboxes in bboxes for b in bboxes
        ]
        if label[0].split("_")[1] != image_name:
            continue

        img = tf.io.read_file(IMAGE_DIR + image_name)

        file_write.write(" ".join(label) + "\n")
        tf.io.write_file(
            AUG_IMAGE_DIR + f"{images_count[img_idx]}_{image_name}", img.numpy()
        )

        images_count[img_idx] += 1
        for _class in classes:
            class_count[_class] += 1

    aug_list = sort_images
    while any(class_count < AUG_TARGET_NUM):
        # Put image with not enough class to the end of list
        tmp_list = []
        for i, (image_name, _, classes, _) in enumerate(aug_list):
            if all(class_count[classes] < AUG_TARGET_NUM):
                tmp_list.append(aug_list[i])
                for _class in classes:
                    class_count[_class] += 1

        # Augment image
        aug_list = tmp_list
        for i, (image_name, bboxes, classes, _) in tqdm(
            enumerate(aug_list), desc="Augment", total=len(aug_list)
        ):
            img_idx = image_names.index(image_name)

            # Generate augment image
            img_augmented, bboxes_augmented = augment_data(
                augmenter, image_name, bboxes
            )

            # Write to file
            aug_label = [f"{images_count[img_idx]}_{image_name}"] + [
                str(int(b)) for bboxes in bboxes_augmented for b in bboxes
            ]
            if aug_label[0].split("_")[1] != image_name:
                continue

            file_write.write(" ".join(aug_label) + "\n")
            tf.io.write_file(
                AUG_IMAGE_DIR + f"{images_count[img_idx]}_{image_name}",
                img_augmented,
            )

            images_count[img_idx] += 1

    file_write.close()


augmenter = A.Compose(
    [
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.GaussNoise(),
        A.OneOf(
            [
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ]
        ),
        A.ShiftScaleRotate(),
        A.OneOf(
            [
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ]
        ),
        A.OneOf(
            [
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ]
        ),
        A.JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
        A.HueSaturationValue(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc"),
)

if AUGMENT:
    generate_aug_data(augmenter)
    NUM_AUG_IMAGES = len(open(AUG_DATA_PATH).readlines())

# %% Analyze augmented data
analyze_data(AUG_DATA_PATH)


# %% Dataset loader
class DatasetGenerator:
    """
    Load pascalVOC 2007 dataset and creates an input pipeline.
    - Reshapes images into 448 x 448
    - converts [0 1] to [-1 1]
    - shuffles the input
    - builds batches
    """

    def __init__(self):
        self.image_names = []
        self.record_list = []
        self.object_num_list = []
        # filling the record_list
        input_file = open(AUG_DATA_PATH, "r")

        for line in input_file:
            line = line.strip()
            ss = line.split(" ")
            self.image_names.append(ss[0])

            self.record_list.append([float(num) for num in ss[1:]])

            self.object_num_list.append(
                min(len(self.record_list[-1]) // 5, MAX_OBJECTS_PER_IMAGE)
            )
            if len(self.record_list[-1]) < MAX_OBJECTS_PER_IMAGE * 5:
                # if there are objects less than MAX_OBJECTS_PER_IMAGE, pad the list
                self.record_list[-1] = self.record_list[-1] + [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ] * (MAX_OBJECTS_PER_IMAGE - len(self.record_list[-1]) // 5)

            elif len(self.record_list[-1]) > MAX_OBJECTS_PER_IMAGE * 5:
                # if there are objects more than MAX_OBJECTS_PER_IMAGE, crop the list
                self.record_list[-1] = self.record_list[-1][: MAX_OBJECTS_PER_IMAGE * 5]

    def _data_preprocess(self, image_name, raw_labels, object_num):
        image_file = tf.io.read_file(AUG_IMAGE_DIR + image_name)
        image = tf.io.decode_jpeg(image_file, channels=3)

        h = tf.shape(image)[0]
        w = tf.shape(image)[1]

        width_ratio = IMAGE_SIZE * 1.0 / tf.cast(w, tf.float32)
        height_ratio = IMAGE_SIZE * 1.0 / tf.cast(h, tf.float32)

        image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 255

        raw_labels = tf.cast(tf.reshape(raw_labels, [-1, 5]), tf.float32)

        xmin = raw_labels[:, 0]
        ymin = raw_labels[:, 1]
        xmax = raw_labels[:, 2]
        ymax = raw_labels[:, 3]
        class_num = raw_labels[:, 4]

        xcenter = (xmin + xmax) * 1.0 / 2.0 * width_ratio
        ycenter = (ymin + ymax) * 1.0 / 2.0 * height_ratio

        box_w = (xmax - xmin) * width_ratio
        box_h = (ymax - ymin) * height_ratio

        labels = tf.stack([xcenter, ycenter, box_w, box_h, class_num], axis=1)

        return image, labels, tf.cast(object_num, tf.int32)

    def generate(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.image_names,
                np.array(self.record_list),
                np.array(self.object_num_list),
            )
        )
        dataset = dataset.shuffle(NUM_AUG_IMAGES + 2000)
        dataset = dataset.map(
            self._data_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=200)

        return dataset


# %% Define Model
def Conv2D_BN_Leaky(x, filters, size, stride=1):
    x = layers.Conv2D(
        filters,
        size,
        stride,
        padding="same",
        kernel_initializer=keras.initializers.TruncatedNormal(),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)

    return x


def yolo_body(input):
    base_model = keras.applications.EfficientNetV2S(
        input_tensor=input, include_top=False, weights="imagenet"
    )
    x = Conv2D_BN_Leaky(base_model.output, 1024, 3, 1)
    x = Conv2D_BN_Leaky(x, 1024, 3, 2)
    x = Conv2D_BN_Leaky(x, 1024, 3, 1)
    x = Conv2D_BN_Leaky(x, 1024, 3, 1)

    x = layers.Flatten()(x)
    x = layers.Dense(
        4096, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01)
    )(x)
    x = layers.LeakyReLU(0.1)(x)
    outputs = layers.Dense(
        1470, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.01)
    )(x)

    return keras.Model(input, outputs)


# %% Create model
YOLO = yolo_body(keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
YOLO.summary()

# %% Define loss function
# base boxes (for loss calculation)
base_boxes = np.zeros([CELL_SIZE, CELL_SIZE, 4])

# initializtion for each cell
for y in range(CELL_SIZE):
    for x in range(CELL_SIZE):
        base_boxes[y, x, :] = [
            IMAGE_SIZE / CELL_SIZE * x,
            IMAGE_SIZE / CELL_SIZE * y,
            0,
            0,
        ]

base_boxes = np.resize(base_boxes, [CELL_SIZE, CELL_SIZE, 1, 4])
base_boxes = np.tile(base_boxes, [1, 1, BOXES_PER_CELL, 1])


def iou(boxes1, boxes2):
    """calculate ious
    Args:
      boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
      boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)

    Return:
      iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
      ====> iou score for each cell
    """

    # boxes1 : [4(xmin, ymin, xmax, ymax), cell_size, cell_size, boxes_per_cell]
    boxes1 = tf.stack(
        [
            boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2,
            boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
            boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2,
            boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2,
        ]
    )

    # boxes1 : [cell_size, cell_size, boxes_per_cell, 4(xmin, ymin, xmax, ymax)]
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])

    boxes2 = tf.stack(
        [
            boxes2[0] - boxes2[2] / 2,
            boxes2[1] - boxes2[3] / 2,
            boxes2[0] + boxes2[2] / 2,
            boxes2[1] + boxes2[3] / 2,
        ]
    )

    # calculate the left up point of boxes' overlap area
    lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
    # calculate the right down point of boxes overlap area
    rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

    # intersection
    intersection = rd - lu

    # the size of the intersection area
    inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

    mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(
        intersection[:, :, :, 1] > 0, tf.float32
    )

    # if intersection is negative, then the boxes don't overlap
    inter_square = mask * inter_square

    # calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (
        boxes1[:, :, :, 3] - boxes1[:, :, :, 1]
    )
    square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

    return inter_square / (square1 + square2 - inter_square + 1e-6)


def losses_calculation(predict, label):
    """
    calculate loss
    Args:
      predict: 3-D tensor [cell_size, cell_size, num_classes + 5 * boxes_per_cell]
      label :  [1, 5]  (x_center, y_center, w, h, class)
    """
    label = tf.reshape(label, [-1])

    # Step A. calculate objects tensor [CELL_SIZE, CELL_SIZE]
    # turn pixel position into cell position (corner)
    min_x = (label[0] - label[2] / 2) / (IMAGE_SIZE / CELL_SIZE)
    max_x = (label[0] + label[2] / 2) / (IMAGE_SIZE / CELL_SIZE)

    min_y = (label[1] - label[3] / 2) / (IMAGE_SIZE / CELL_SIZE)
    max_y = (label[1] + label[3] / 2) / (IMAGE_SIZE / CELL_SIZE)

    min_x = tf.floor(min_x)
    min_y = tf.floor(min_y)

    max_x = tf.minimum(tf.math.ceil(max_x), CELL_SIZE)
    max_y = tf.minimum(tf.math.ceil(max_y), CELL_SIZE)

    # calculate mask of object with cells
    onset = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
    object_mask = tf.ones(onset, tf.float32)

    offset = tf.cast(
        tf.stack([min_y, CELL_SIZE - max_y, min_x, CELL_SIZE - max_x]), tf.int32
    )
    offset = tf.reshape(offset, (2, 2))
    object_mask = tf.pad(object_mask, offset, "CONSTANT")

    # Step B. calculate the coordination of object center and the corresponding mask
    # turn pixel position into cell position (center)
    center_x = label[0] / (IMAGE_SIZE / CELL_SIZE)
    center_x = tf.floor(center_x)

    center_y = label[1] / (IMAGE_SIZE / CELL_SIZE)
    center_y = tf.floor(center_y)

    response = tf.ones([1, 1], tf.float32)

    # calculate the coordination of object center with cells
    objects_center_coord = tf.cast(
        tf.stack(
            [center_y, CELL_SIZE - center_y - 1, center_x, CELL_SIZE - center_x - 1]
        ),
        tf.int32,
    )
    objects_center_coord = tf.reshape(objects_center_coord, (2, 2))

    # make mask
    response = tf.pad(response, objects_center_coord, "CONSTANT")

    # Step C. calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    predict_boxes = predict[:, :, NUM_CLASSES + BOXES_PER_CELL :]

    predict_boxes = tf.reshape(predict_boxes, [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4])
    # cell position to pixel position
    predict_boxes = predict_boxes * [
        IMAGE_SIZE / CELL_SIZE,
        IMAGE_SIZE / CELL_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ]

    # if there's no predict_box in that cell, then the base_boxes will be calcuated with label and got iou equals 0
    predict_boxes = base_boxes + predict_boxes

    iou_predict_truth = iou(predict_boxes, label[0:4])

    # calculate C tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    C = iou_predict_truth * tf.reshape(response, [CELL_SIZE, CELL_SIZE, 1])

    # calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    I = iou_predict_truth * tf.reshape(response, [CELL_SIZE, CELL_SIZE, 1])

    max_I = tf.reduce_max(I, 2, keepdims=True)

    # replace large iou scores with response (object center) value
    I = tf.cast((I >= max_I), tf.float32) * tf.reshape(
        response, (CELL_SIZE, CELL_SIZE, 1)
    )

    # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    no_I = tf.ones_like(I, dtype=tf.float32) - I

    p_C = predict[:, :, NUM_CLASSES : NUM_CLASSES + BOXES_PER_CELL]

    # calculate truth x, y, sqrt_w, sqrt_h 0-D
    x = label[0]
    y = label[1]

    sqrt_w = tf.sqrt(tf.abs(label[2]))
    sqrt_h = tf.sqrt(tf.abs(label[3]))

    # calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    p_x = predict_boxes[:, :, :, 0]
    p_y = predict_boxes[:, :, :, 1]

    p_sqrt_w = tf.sqrt(
        tf.minimum(IMAGE_SIZE * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 2]))
    )
    p_sqrt_h = tf.sqrt(
        tf.minimum(IMAGE_SIZE * 1.0, tf.maximum(0.0, predict_boxes[:, :, :, 3]))
    )

    # calculate ground truth p 1-D tensor [NUM_CLASSES]
    P = tf.one_hot(tf.cast(label[4], tf.int32), NUM_CLASSES, dtype=tf.float32)

    # calculate predicted p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
    p_P = predict[:, :, 0:NUM_CLASSES]

    # class_loss
    class_loss = (
        tf.nn.l2_loss(tf.reshape(object_mask, (CELL_SIZE, CELL_SIZE, 1)) * (p_P - P))
        * CLASS_SCALE
    )

    # object_loss
    object_loss = tf.nn.l2_loss(I * (p_C - C)) * OBJECT_SCALE

    # noobject_loss
    noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * NOOBJECT_SCALE

    # coord_loss
    coord_loss = (
        tf.nn.l2_loss(I * (p_x - x) / (IMAGE_SIZE / CELL_SIZE))
        + tf.nn.l2_loss(I * (p_y - y) / (IMAGE_SIZE / CELL_SIZE))
        + tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w)) / IMAGE_SIZE
        + tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h)) / IMAGE_SIZE
    ) * COORD_SCALE

    return class_loss + object_loss + noobject_loss + coord_loss


def yolo_loss(predicts, labels, objects_num):
    """
    Add Loss to all the trainable variables
    Args:
        predicts:    4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
                     ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
        labels:      3-D tensor of [batch_size, max_objects, 5]
        objects_num: 1-D tensor [batch_size]
    """

    loss = 0.0
    batch_size = len(predicts)
    # you can parallel the code with tf.map_fn or tf.vectorized_map (big performance gain!)
    for i in tf.range(batch_size):
        predict = predicts[i, :, :, :]
        label = labels[i, :, :]
        object_num = objects_num[i]

        for j in tf.range(object_num):
            loss = loss + losses_calculation(predict, label[j : j + 1, :])

    return loss / batch_size


# %% Training setup
dataset = DatasetGenerator().generate()
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
train_loss_metric = tf.keras.metrics.Mean(name="loss")

ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), net=YOLO)
manager = tf.train.CheckpointManager(
    ckpt, "./ckpts/YOLO", max_to_keep=3, checkpoint_name="yolo"
)
ckpt_path = manager.restore_or_initialize()
if ckpt_path:
    print(f"Restored from {ckpt_path}")
else:
    print("Initializing from scratch.")


# %% Training step
@tf.function
def train_step(image, labels, objects_num):
    with tf.GradientTape() as tape:
        outputs = YOLO(image)
        class_end = CELL_SIZE * CELL_SIZE * NUM_CLASSES
        conf_end = class_end + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL
        class_probs = tf.reshape(outputs[:, 0:class_end], (-1, 7, 7, 20))
        confs = tf.reshape(outputs[:, class_end:conf_end], (-1, 7, 7, 2))
        boxes = tf.reshape(outputs[:, conf_end:], (-1, 7, 7, 2 * 4))
        predicts = tf.concat([class_probs, confs, boxes], 3)

        loss = yolo_loss(predicts, labels, objects_num)
        train_loss_metric(loss)

    grads = tape.gradient(loss, YOLO.trainable_weights)
    optimizer.apply_gradients(zip(grads, YOLO.trainable_weights))


# %% Training
print("{}, start training.".format(datetime.now()))
for i in range(EPOCHS):
    train_loss_metric.reset_states()
    ckpt.epoch.assign_add(1)

    for image, labels, objects_num in tqdm(
        dataset, desc=f"Epoch {i+1}", total=int(math.ceil(NUM_AUG_IMAGES // BATCH_SIZE))
    ):
        train_step(image, labels, objects_num)

    print(f"Epoch {i+1}: Average loss {train_loss_metric.result():.2f}")

    save_path = manager.save()
    print(f"Saved checkpoint for epoch {int(ckpt.epoch)}: {save_path}")


# %% Function for process model prediction
def process_outputs(outputs):
    """
    Process YOLO outputs into bounding boxes
    """

    class_end = CELL_SIZE * CELL_SIZE * NUM_CLASSES
    conf_end = class_end + CELL_SIZE * CELL_SIZE * BOXES_PER_CELL
    class_probs = np.reshape(outputs[:, 0:class_end], (-1, 7, 7, 20))
    confs = np.reshape(outputs[:, class_end:conf_end], (-1, 7, 7, 2))
    boxes = np.reshape(outputs[:, conf_end:], (-1, 7, 7, 2 * 4))
    predicts = np.concatenate([class_probs, confs, boxes], 3)

    p_classes = predicts[0, :, :, 0:20]
    C = predicts[0, :, :, 20:22]
    coordinate = predicts[0, :, :, 22:]

    p_classes = np.reshape(p_classes, (CELL_SIZE, CELL_SIZE, 1, 20))
    C = np.reshape(C, (CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 1))

    P = C * p_classes
    # P's shape [7, 7, 2, 20]

    # Choose the one which confidence > threshold
    conf_mask = P >= np.max(P) * CONF_THRESHOLD
    confs = P[conf_mask]
    indexs = np.nonzero(conf_mask)

    class_nums = indexs[3]

    coordinate = np.reshape(coordinate, (CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4))

    coordinates = coordinate[indexs[0], indexs[1], indexs[2], :]

    xs, ys, ws, hs = (
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        coordinates[:, 3],
    )

    xs = (xs + indexs[1]) * (IMAGE_SIZE / float(CELL_SIZE))
    ys = (ys + indexs[0]) * (IMAGE_SIZE / float(CELL_SIZE))
    ws = ws * IMAGE_SIZE
    hs = hs * IMAGE_SIZE

    xmins = xs - ws / 2.0
    ymins = ys - hs / 2.0

    xmaxs = xmins + ws
    ymaxs = ymins + hs

    return xmins, ymins, xmaxs, ymaxs, class_nums, confs


# %% Build test dataset iterator
test_img_files = open(os.path.join(DATA_DIR, "pascal_voc_testing_data.txt"))
test_img_dir = os.path.join(DATA_DIR, "VOCdevkit_test/VOC2007/JPEGImages/")
test_images = []

for line in test_img_files:
    line = line.strip()
    ss = line.split(" ")
    test_images.append(ss[0])

test_dataset = tf.data.Dataset.from_tensor_slices(test_images)


def load_img_data(image_name):
    image_file = tf.io.read_file(test_img_dir + image_name)
    image = tf.image.decode_jpeg(image_file, channels=3)

    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    image = tf.image.resize(image, size=[IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255

    return image_name, image, h, w


test_dataset = test_dataset.map(
    load_img_data, num_parallel_calls=tf.data.experimental.AUTOTUNE
)
test_dataset = test_dataset.batch(32)


# %% Prediction step
@tf.function
def prediction_step(img):
    return YOLO(img, training=False)


# %% Prediction
output_file = open(os.path.join(OUTPUT_DIR, "test_prediction.txt"), "w")
ckpt = tf.train.Checkpoint(net=YOLO)
ckpt.restore("./ckpts/YOLO/yolo-40")

for img_name, test_img, img_h, img_w in test_dataset:
    batch_num = img_name.shape[0]
    for i in range(batch_num):
        xmins, ymins, xmaxs, ymaxs, class_nums, confs = process_outputs(
            prediction_step(test_img[i : i + 1])
        )
        bbox_coords = (
            xmins * (img_w[i : i + 1] / IMAGE_SIZE),
            ymins * (img_h[i : i + 1] / IMAGE_SIZE),
            xmaxs * (img_w[i : i + 1] / IMAGE_SIZE),
            ymaxs * (img_h[i : i + 1] / IMAGE_SIZE),
        )

        bbox_coords = tf.transpose(bbox_coords)

        # img filename, xmin, ymin, xmax, ymax, class, confidence
        for coord, class_num, conf in zip(bbox_coords, class_nums, confs):
            output_file.write(
                img_name[i : i + 1].numpy()[0].decode("ascii")
                + " %d %d %d %d %d %f\n"
                % (coord[0], coord[1], coord[2], coord[3], class_num, conf)
            )

output_file.close()

# %% Evaluate
evaluate.evaluate(
    os.path.join(OUTPUT_DIR, "test_prediction.txt"),
    os.path.join(OUTPUT_DIR, "output.csv"),
)

cap = pd.read_csv("./output/output.csv")["packedCAP"]
print("score: {:f}".format(sum((1.0 - cap) ** 2) / len(cap)))

# %% Visualize
np_img = cv2.imread(
    os.path.join(DATA_DIR, "VOCdevkit_test/VOC2007/JPEGImages/000001.jpg")
)
resized_img = cv2.resize(np_img, (IMAGE_SIZE, IMAGE_SIZE))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
resized_img = np_img
np_img = np_img.astype(np.float32)
np_img = np_img / 255.0
np_img = np.reshape(np_img, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

y_pred = YOLO(np_img, training=False)
xmins, ymins, xmaxs, ymaxs, class_nums, confs = process_outputs(y_pred)
out = zip(xmins, ymins, xmaxs, ymaxs, class_nums, confs)

for xmin, ymin, xmax, ymax, class_num, conf in out:
    class_name = CLASSES_NAME[class_num]
    cv2.rectangle(
        resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), 3
    )
    cv2.putText(resized_img, class_name, (0, 200), 2, 1.5, (0, 255, 255), 2)

plt.imshow(resized_img)
plt.show()
