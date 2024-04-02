#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from swift_comet_pipeline.comet_profile import profile_to_image


def make_centered_radii_mesh(width, height):
    assert width % 2 == 1 and height % 2 == 1

    # this assumes both width and height are odd
    center_x = np.floor(width / 2)
    center_y = np.floor(height / 2)

    xs = np.linspace(0, width, num=width, endpoint=False)
    ys = np.linspace(0, height, num=height, endpoint=False)
    x, y = np.meshgrid(xs, ys)
    radius_mesh = np.round(np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)).astype(
        int
    )
    return radius_mesh


def radial_profile(r):
    pixel_profile_xs = np.linspace(1, 100, num=100, endpoint=True)
    pixel_profile = 3 * np.sin(pixel_profile_xs)
    profile_r = np.round(r).astype(int)
    print(f"{r=}, {profile_r=}")
    return pixel_profile[profile_r] ** 3


def subtract_smaller_array(
    large_img, small_img, smaller_centered_on_x, smaller_centered_on_y
):
    large_height, large_width = large_img.shape
    small_height, small_width = small_img.shape

    adjust_x = np.floor(small_width / 2).astype(int)
    adjust_y = np.floor(small_height / 2).astype(int)

    center_x = smaller_centered_on_x
    center_y = smaller_centered_on_y

    # numpy slices do not include the last element, so bump up the range by 1
    y_start_index = center_y - adjust_y
    y_end_index = center_y + adjust_y + 1
    x_start_index = center_x - adjust_x
    x_end_index = center_x + adjust_x + 1

    # do some bounds checking, and return the larger image unaltered if any fail
    if y_start_index < 0 or x_start_index < 0:
        print(f"{y_start_index=}, {x_start_index=}")
        return large_img
    if y_end_index > large_height or x_end_index > large_width:
        print(f"{y_end_index=}, {x_end_index=}")
        return large_img

    large_img[
        y_start_index:y_end_index,
        x_start_index:x_end_index,
    ] -= small_img

    return large_img


def main_old():
    img_height, img_width = 51, 101

    dists = make_centered_radii_mesh(width=img_width, height=img_height)

    img = radial_profile(dists)

    # smaller_img = 5 * np.ones((5, 5))
    smaller_img = np.outer(np.arange(21), np.arange(21))

    sub_img = subtract_smaller_array(
        img, smaller_img, smaller_centered_on_x=90, smaller_centered_on_y=40
    )

    # plt.imshow(img, origin="lower")
    plt.imshow(sub_img, origin="lower")
    plt.show()


def main():
    img_height, img_width = 111, 111
    dists = make_centered_radii_mesh(width=img_width, height=img_height)

    profile_rs = np.linspace(0, 15, num=16, endpoint=True)
    pixels = profile_rs**2

    img = profile_to_image(pixels, dists)
    print(f"{img=}")

    plt.imshow(img, origin="lower")
    plt.show()


if __name__ == "__main__":
    main()
