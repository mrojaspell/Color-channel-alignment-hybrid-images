import numpy as np
from PIL import Image
import os
import json
import time


def ncc_value(img1, img2):
    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)
    aux = np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))
    if aux == 0:
        return 0
    return np.sum(img1 * img2) / aux

def ssd_value(img1, img2):
    return np.sum((img1 - img2) ** 2)


def get_channels(filepath, crop_size):
    img = Image.open(filepath)
    if img.mode == 'I;16':
        img = convert_to_8bit(img)

    img_np = np.array(img)
    img_size = img_np.shape[0] // 3
    min_height = min(img_np[:img_size, :].shape[0], img_np[img_size:2 * img_size, :].shape[0],
                     img_np[2 * img_size:, :].shape[0])

    blue_img = img_np[:min_height, :]
    green_img = img_np[img_size:img_size + min_height, :]
    red_img = img_np[2 * img_size:2 * img_size + min_height, :]

    if blue_img.shape[0] > 2 * crop_size and blue_img.shape[1] > 2 * crop_size:
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
    for x in range(-window, window + 1):
        for y in range(-window, window + 1):
            shifted_img = np.roll(np.roll(relative_img, y, axis=0), x, axis=1)
            aux_score = metric(shifted_img, absolute_img)
            if (metric == ncc_value and aux_score > score) or (metric == ssd_value and aux_score < score):
                score = aux_score
                displacement = (x, y)

    aligned_img = np.roll(np.roll(relative_img, displacement[1], axis=0), displacement[0], axis=1)
    return aligned_img, displacement, score

def form_image(red_img, green_img, blue_img, out_path, filename, metric):
    metric_name = metric.__name__

    colour_img = np.stack([red_img, green_img, blue_img], axis=-1).astype(np.uint8)
    img = Image.fromarray(colour_img)
    img.save(os.path.join(out_path, metric_name + "_" + filename))


def downsample(img, factor):
    pil_img = Image.fromarray(img)

    if pil_img.mode not in ["RGB", "L"]:
        pil_img = pil_img.convert("RGB")

    width, height = pil_img.size
    if width // factor > 0 and height // factor > 0:
        downsampled = pil_img.resize((width // factor, height // factor), Image.BICUBIC)
        return np.array(downsampled)
    return img


def pyramid_alignment(relative_img, absolute_img, metric, window, height, factor):
    if height == 0:
        return align_channels(relative_img, absolute_img, metric, window)

    rel_img_downsampled = downsample(relative_img, factor)
    abs_img_downsampled = downsample(absolute_img, factor)

    _, aux_disp, aux_score = pyramid_alignment(rel_img_downsampled, abs_img_downsampled, metric, window, height - 1, factor)
    aux_disp = (aux_disp[0] * factor, aux_disp[1] * factor)
    new_window = max(1, window // 2)

    aligned_frame, displacement, score = align_channels(
        np.roll(np.roll(relative_img, aux_disp[1], axis=0), aux_disp[0], axis=1),
        absolute_img,
        metric,
        new_window
    )
    total_disp = (displacement[0] + aux_disp[0], displacement[1] + aux_disp[1])
    total_score = score + aux_score
    return aligned_frame, total_disp, total_score


def convert_to_8bit(image):
    img_np = np.array(image, dtype=np.uint16)
    img_np_8bit = (img_np / 256).astype(np.uint8)
    img_8bit = Image.fromarray(img_np_8bit)
    return img_8bit

if __name__ == "__main__":
    with open("mgr9_a1_part1b_config.json", "r") as f:
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
    chosen_height = config["height"]
    chosen_factor = config["factor"]

    for file in os.listdir(data_path):
        image_path = os.path.join(data_path, file)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            start_time = time.time()
            blue_frame, green_frame, red_frame = get_channels(image_path, chosen_crop)
            aligned_red, disp_red, score_red = pyramid_alignment(red_frame, blue_frame, chosen_metric, chosen_window, chosen_height, chosen_factor)
            aligned_green, disp_green, score_green = pyramid_alignment(green_frame, blue_frame, chosen_metric, chosen_window, chosen_height, chosen_factor)
            form_image(aligned_red, aligned_green, blue_frame, out_direc, file, chosen_metric)
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Image {file} has been aligned with a score of {score_red} and displacement {disp_red} for red and {score_green}, {disp_green} for green channel in {total_time} time\n")