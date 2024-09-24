import numpy as np
from PIL import Image
import os
import json
import time

def ssd_value(img1, img2):
    return np.sum((img1 - img2) ** 2)

def ncc_value(img1, img2):
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)
    return np.sum(img1 * img2) / np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))


def get_channels(filepath, crop_size):
    img = Image.open(filepath)
    img_np = np.array(img)

    img_size = img_np.shape[0]//3
    min_height = min(img_np[:img_size, :].shape[0], img_np[img_size:2 * img_size, :].shape[0],
                     img_np[2 * img_size:, :].shape[0])

    blue_img = img_np[:min_height, :]
    green_img = img_np[img_size:img_size + min_height, :]
    red_img = img_np[2 * img_size:2 * img_size + min_height, :]

    blue_img = blue_img[crop_size:-crop_size, crop_size:-crop_size]
    green_img = green_img[crop_size:-crop_size, crop_size:-crop_size]
    red_img = red_img[crop_size:-crop_size, crop_size:-crop_size]

    return blue_img, green_img, red_img

def align_channels(relative_img, absolute_img, metric, window):
    if metric == ncc_value:
        score = -np.inf
    else:
        score = np.inf

    displacement = (0, 0)
    for x in range(-window, window+1):
        for y in range(-window, window+1):
            shifted_img = np.roll(np.roll(relative_img, y, axis=0), x, axis=1)
            aux_score = metric(shifted_img, absolute_img)
            if (metric == ncc_value and aux_score > score) or (metric == ssd_value and aux_score < score):
                score = aux_score
                displacement = (x, y)

    aligned_img = np.roll(np.roll(relative_img, displacement[1], axis=0), displacement[0], axis=1)
    return aligned_img, displacement, score

def form_image(red_img, green_img, blue_img, out_path, filename, metric=ncc_value):
    colour_img = np.stack([red_img, green_img, blue_img], axis=-1)
    img = Image.fromarray(colour_img)
    img.save(os.path.join(out_path, metric+filename))


if __name__ == "__main__":
    with open("mgr9_a1_part1a_config.json", "r") as f:
        config = json.load(f)

    metric_map = {
        "ncc_value": ncc_value,
        "ssd_value": ssd_value
    }

    data_path = config["images"]["path"]
    out_direc = config["images"]["out_path"]
    chosen_crop = config["variables"]["crop_size"]
    chosen_window = config["variables"]["window"]
    chosen_metric = metric_map[config["metric"]]
    for file in os.listdir(data_path):
        image_path = os.path.join(data_path, file)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            start_time = time.time()
            blue_frame, green_frame, red_frame = get_channels(image_path, chosen_crop)
            aligned_framesR, shiftR, scoreR   = align_channels(red_frame, blue_frame, chosen_metric, chosen_window)
            aligned_framesG, shiftG, scoreG   = align_channels(green_frame, blue_frame, chosen_metric, chosen_window)
            form_image(aligned_framesR, aligned_framesG, blue_frame, out_direc, file, chosen_metric.__name__)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Image {file} has been aligned with a score of {scoreR} and displacement {shiftR} for red and {scoreG}, {shiftG} for green channel\n")
            print(f"Time taken: {total_time}\n")
