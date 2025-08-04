import cv2
import random
import numpy as np

xs = []
ys = []

# Points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

# Read data.txt
with open("data/driving_dataset/data.txt") as f:
    for line in f:
        parts = line.split()
        xs.append("data/driving_dataset/" + parts[0])
        ys.append(float(parts[1]) * 3.14159265 / 180)

num_images = len(xs)

# Shuffle the dataset before splitting for better generalization
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

# Train/validation split 80/20
split_index = int(len(xs) * 0.8)
train_xs = xs[:split_index]
train_ys = ys[:split_index]
val_xs = xs[split_index:]
val_ys = ys[split_index:]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        idx = (train_batch_pointer + i) % num_train_images
        img = cv2.imread(train_xs[idx])
        if img is None:
            raise FileNotFoundError(f"Image not found: {train_xs[idx]}")
        # Crop last 150 pixels in height (assuming vertical dimension last)
        img_cropped = img[-150:, :, :]
        img_resized = cv2.resize(img_cropped, (200, 66))
        x_out.append(img_resized / 255.0)
        y_out.append([train_ys[idx]])
    train_batch_pointer += batch_size
    return x_out, y_out


def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        idx = (val_batch_pointer + i) % num_val_images
        img = cv2.imread(val_xs[idx])
        if img is None:
            raise FileNotFoundError(f"Image not found: {val_xs[idx]}")
        # Crop last 150 pixels in height
        img_cropped = img[-150:, :, :]
        img_resized = cv2.resize(img_cropped, (200, 66))
        x_out.append(img_resized / 255.0)
        y_out.append([val_ys[idx]])
    val_batch_pointer += batch_size
    return x_out, y_out


