import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from PIL import Image
import json
import os

def low_pass(img, sigma):
    return ndimage.gaussian_filter(img, sigma=sigma)


def high_pass(img, sigma):
    high_filter = ndimage.gaussian_filter(img, sigma=sigma)
    return img - high_filter


def generate_hybrid(low_filter, high_filter):
    img = (low_filter + high_filter) / 2
    img = np.clip(img, 0, 255)
    return img

if __name__ == "__main__":

    with open("mgr9_a1_part3_config.json", "r") as f:
        config = json.load(f)

    images = [config["hybrid1"], config["hybrid2"], config["hybrid3"]]
    sigmas = [config["sigma_low"], config["sigma_high"]]

    i = 1
    for pair in images:
        img1_path = pair["img1"]
        img2_path = pair["img2"]
        img_1 = Image.open(img1_path).convert('L')
        img_2 = Image.open(img2_path).convert('L')

        low_pass_img = low_pass(img_1, sigmas[0])
        high_pass_img = high_pass(img_2, sigmas[1])

        plt.imsave(os.path.join("./results/", f"mgr9_a1_part3_low_pass_{i}.png"), low_pass_img, cmap='gray')
        plt.imsave(os.path.join("./results/", f"mgr9_a1_part3_high_pass_{i}.png"), high_pass_img, cmap='gray')

        hybrid = generate_hybrid(low_pass_img, high_pass_img)
        low_res_hybrid = np.copy(hybrid)
        low_res_hybrid = low_res_hybrid[::4, ::4]
        plt.imsave(os.path.join("./results/", f"mgr9_a1_part3_low_res_hybrid_{i}.png"), low_res_hybrid, cmap='gray')
        plt.imsave(os.path.join("./results/", f"mgr9_a1_part3_high_res_hybrid_{i}.png"), hybrid, cmap='gray')
        i += 1
