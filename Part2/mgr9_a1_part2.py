import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage as ndimage
import json
import os
import time

def save_plot(fig, path, title):
    fig.suptitle(title)
    fig.savefig(path)
    plt.close(fig)

def convert_to_8bit(image):
    img_np = np.array(image, dtype=np.uint16)
    img_np_8bit = (img_np / 256).astype(np.uint8)
    img_8bit = Image.fromarray(img_np_8bit)
    return img_8bit

def preprocess_image(image, ):
    gaussian_filter = ndimage.gaussian_filter(image, sigma=1)
    preprocessed_image = image - gaussian_filter
    return preprocessed_image

def get_channels(filepath, crop_size=20):
    img = Image.open(filepath)
    if img.mode == 'I;16':
        img = convert_to_8bit(img)
    img_np = np.array(img)
    img_height = img_np.shape[0]

    channel_height = img_height // 3
    min_height = min(img_np[:channel_height, :].shape[0],
                     img_np[channel_height:2 * channel_height, :].shape[0],
                     img_np[2 * channel_height:, :].shape[0])

    blue_img = img_np[:min_height, :]
    green_img = img_np[channel_height:channel_height + min_height, :]
    red_img = img_np[2 * channel_height:2 * channel_height + min_height, :]

    if blue_img.shape[0] > 2 * crop_size and blue_img.shape[1] > 2 * crop_size:
        blue_img = blue_img[crop_size:-crop_size, crop_size:-crop_size]
        green_img = green_img[crop_size:-crop_size, crop_size:-crop_size]
        red_img = red_img[crop_size:-crop_size, crop_size:-crop_size]

    return blue_img, green_img, red_img

def channel_alignment(relative_frame, absolute_frame):
    f_trans_absolute = np.fft.fft2(absolute_frame)
    f_trans_relative = np.fft.fft2(relative_frame)

    fourier_product = f_trans_absolute * np.conjugate(f_trans_relative)
    inverse_transform = np.fft.ifft2(fourier_product)

    magnitude = np.abs(inverse_transform)
    max_value = np.unravel_index(np.argmax(magnitude), magnitude.shape)

    displacement = np.array(max_value)
    mid_point = np.array(magnitude.shape) // 2
    displacement = (displacement - mid_point) % magnitude.shape - mid_point

    return displacement, magnitude, inverse_transform

def form_image(image_filepath, crop_size, output_path, filename, apply_filter=False):
    blue, green, red = get_channels(image_filepath, crop_size)
    dead_time_start = time.time()

    _, _, b_to_r_no_processing = channel_alignment(red, blue)
    _, _, b_to_g_no_processing = channel_alignment(green, blue)

    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
    axs1[0].imshow(np.log(1 + np.abs(b_to_r_no_processing)), cmap='gray')
    axs1[0].set_title("Blue to Red")
    axs1[1].imshow(np.log(1 + np.abs(b_to_g_no_processing)), cmap='gray')
    axs1[1].set_title("Blue to Green")
    save_plot(fig1, os.path.join(output_path, f"without_processing_{filename}.png"), "Without Processing")

    dead_time_end = time.time()
    total_dead_time = dead_time_end - dead_time_start

    processed_blue = preprocess_image(blue)
    processed_green = preprocess_image(green)
    processed_red = preprocess_image(red)

    red_displacement, red_magnitude, b_to_r = channel_alignment(processed_red, processed_blue)
    green_displacement, green_magnitude, b_to_g = channel_alignment(processed_green, processed_blue)

    red_channel = np.roll(red, shift=red_displacement, axis=(0, 1))
    green_channel = np.roll(green, shift=green_displacement, axis=(0, 1))
    aligned_image = np.stack([red_channel, green_channel, blue], axis=-1)

    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
    axs2[0].imshow(np.log(1 + np.abs(b_to_r)), cmap='gray')
    axs2[0].set_title("Blue to Red")
    axs2[1].imshow(np.log(1 + np.abs(b_to_g)), cmap='gray')
    axs2[1].set_title("Blue to Green")
    save_plot(fig2, os.path.join(output_path, f"with_processing_{filename}.png"), "With Processing")

    return aligned_image, red_displacement, green_displacement, total_dead_time


if __name__ == "__main__":
    with open("mgr9_a1_part2_config.json", "r") as f:
        config = json.load(f)
    data_path = config["images"]["path"]
    out_path = config["images"]["out_path"]
    chosen_crop_size = config["crop_size"]

    for file in os.listdir(data_path):
        image_path = os.path.join(data_path, file)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            start_time = time.time()
            aligned_img, red_displacement, green_displacement ,dead_time = form_image(image_path, chosen_crop_size, out_path, file.split('.')[0])
            end_time = time.time()
            total_time = end_time - start_time - dead_time
            print(f"Image {file} has been aligned with a displacement of {red_displacement} for red channel and {green_displacement} for green channel")
            print(f"Time taken: {total_time}\n")
            plt.imshow(aligned_img.astype(np.uint8))
            plt.title('Aligned RGB frames')
            plt.show()
            Image.fromarray(aligned_img.astype('uint8')).save(os.path.join(out_path, f"aligned_{file}"))
