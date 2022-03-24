#!/usr/bin/env python3

import glob
import os

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import astropy.units as u
import scipy.ndimage as ndi

from astropy.io import fits
from astropy.visualization import make_lupton_rgb


def stack_fits_from_dir(file_dir, file_glob, sigma_low=3, sigma_high=3):
    """
        Computes the average of given image file names with outlier pixels clipped
        from the average.  Only sigma clips once for speed but it should good enough
        if you don't feel like waiting
    """

    # collect list of fits files from file_dir matching given glob
    file_list = glob.glob(file_dir + file_glob)

    # read in all of the files - can be memory hungry if files can't be mmap'd
    fits_data = []
    total_pixels = 0
    for i, f in enumerate(file_list):
        with fits.open(f) as hdul:
            print(f"Loading file {f} ...")
            fits_data.append(hdul[0].data)
            total_pixels += hdul[0].data.size

    print("Initial averaging ...")
    # Simple average for each pixel
    avgs = np.mean(fits_data, axis=0)
    print("Done.")

    print("Computing std dev ...")
    # calculate standard deviations with ddof=1 to divide by (n - 1), a.k.a. Bessel's correction
    std_devs = np.std(fits_data, axis=0, ddof=1)
    # replace standard deviations of 0 (stuck or saturated pixels) with a small dummy value
    std_devs[std_devs == 0] = 0.1
    print("Done.")

    print("Calculating sigmas ...")
    # Calculate how many std devs each pixel in each image is away from the mean
    sigma = (fits_data - avgs) / std_devs
    print("Done.")

    print(f"Sigma clipping: below -{sigma_low} σ and above {sigma_high} σ ...")
    # masks pixels that meet either condition, excluding them from the final average
    masked_images = ma.masked_where(np.logical_or(sigma < -sigma_low, sigma > sigma_high), fits_data)
    total_masked = ma.count_masked(masked_images)
    print(f"Total pixels clipped: {total_masked} out of {total_pixels} ({total_masked/total_pixels:02.3f} %)")
    print("Done.")

    print("Calculating clipped averages ...")
    final_avg = ma.mean(masked_images, axis=0)
    print("Done.")

    return final_avg.data


def stack_fits_from_dir_iterate(file_dir, file_glob, sigma_low=3, sigma_high=3):

    """
        Computes the average of given image file names with outlier pixels clipped
        from the average.  Iterates until all pixels are within {sigma_low, sigma_high}
        but uses masked arrays and is very slow
    """

    # collect list of fits files from file_dir matching given glob
    file_list = glob.glob(file_dir + file_glob)

    # read in all of the files - can be memory hungry if files can't be mmap'd
    fits_data = []
    total_pixels = 0
    for i, f in enumerate(file_list):
        with fits.open(f) as hdul:
            print(f"Loading file {f} ...")
            fits_data.append(hdul[0].data)
            total_pixels += hdul[0].data.size

    # start with empty mask
    mask = ma.empty_like(fits_data)
    masked_images = ma.masked_array(fits_data, mask)
    cur_clip_iters = 1
    change_in_mask = 1
    total_masked = 0

    while change_in_mask != 0:
        print(f"Clipping iteration {cur_clip_iters}")

        print("\tInitial averaging ...")
        # Simple average for each pixel
        avgs = ma.mean(masked_images, axis=0)
        print("\tDone.")

        print("\tComputing std dev ...")
        # calculate standard deviations with ddof=1 to divide by (n - 1), a.k.a. Bessel's correction
        std_devs = ma.std(masked_images, axis=0, ddof=1)
        # replace standard deviations of 0 (stuck or saturated pixels) with a small dummy value
        std_devs[std_devs == 0] = 0.1
        print("\tDone.")

        print("\tCalculating sigmas ...")
        # Calculate how many std devs each pixel in each image is away from the mean
        sigma = (masked_images - avgs) / std_devs
        print("\tDone.")

        print(f"\tSigma clipping: below -{sigma_low} σ and above {sigma_high} σ ...")
        # masks pixels that meet either condition, excluding them from the final average
        new_masked_images = ma.masked_where(np.logical_or(sigma < -sigma_low, sigma > sigma_high), fits_data)
        new_masked = ma.count_masked(new_masked_images)
        print("\tDone.")
        print(f"\tTotal pixels clipped: {new_masked} out of {total_pixels} ({new_masked/total_pixels:02.3f} %)")

        change_in_mask = new_masked - total_masked
        total_masked = new_masked
        masked_images = new_masked_images
        cur_clip_iters += 1

    print("Calculating clipped averages ...")
    final_avg = ma.mean(masked_images, axis=0)
    print("Done.")

    # strip off the masked part and just return the numpy array
    return final_avg.data


def make_master_flat(flat_dir, flat_glob, master_flat_file, master_bias, scale_inv_med=False, sigma_low=3, sigma_high=3):

    """
        bias subtract a list of flats and combine for a master flat
    """

    # suffix to add to bias subtracted flats
    bsf_suffix = '_bias_subtracted.fits'
    bsf_glob = '*' + bsf_suffix

    # flats_list = glob.glob(flat_dir + flat_glob)
    flats_list = list(set(glob.glob(flat_dir + flat_glob)) - set(glob.glob(flat_dir + bsf_glob)))

    # go through the flats, bias subtract, and write result
    for f in flats_list:

        # where to write the bias subtracted flat
        bsf_file = flat_dir + os.path.splitext(os.path.basename(f))[0] + bsf_suffix
        if os.path.isfile(bsf_file):
            print(f"Found {bsf_file}, skipping bias subtraction for {f} ...")
        else:
            with fits.open(f, mode='readonly') as hdul:
                hdul[0].data = hdul[0].data - master_bias
                if scale_inv_med:
                    inverse_median = 1 / np.median(hdul[0].data)
                    hdul[0].data = hdul[0].data * inverse_median
                print("Writing bias subtracted flats to", bsf_file)
                hdul.writeto(bsf_file)

    # send bias subtracted flats to be stacked into master
    master_flat = stack_fits_from_dir(flat_dir, bsf_glob, sigma_low=sigma_low, sigma_high=sigma_high)
    hdul = fits.HDUList([fits.PrimaryHDU(master_flat)])
    hdul.writeto(master_flat_file)


def calibrate_science_images(science_dir, science_glob, master_bias, master_dark, master_flat, science_exposure, dark_exposure):

    """
        Use all relevant calibration info to produce calibrated versions of all
        files specified in the glob
    """

    # names of calibrated science frames
    csf_suffix = '_calibrated.fits'
    csf_glob = '*' + csf_suffix

    # remove the calibrated frames from the processing list
    science_list = list(set(glob.glob(science_dir + science_glob)) - set(glob.glob(science_dir + csf_glob)))

    # bias subtracted dark, adjusted for difference in exposure times
    bias_sub_dark = (master_dark - master_bias) * (science_exposure / dark_exposure)

    # total amount to subtract off science frame
    total_subtraction = bias_sub_dark + master_bias

    for f in science_list:

        # where to write the calibrated science frame
        csf_file = science_dir + os.path.splitext(os.path.basename(f))[0] + csf_suffix
        if os.path.isfile(csf_file):
            # print(f"Found {csf_file}, skipping calibration for {f} ...")
            pass
        else:
            with fits.open(f, mode='readonly') as hdul:
                hdul[0].data = (hdul[0].data - total_subtraction) / master_flat
                print("Writing calibrated science image to", csf_file)
                hdul.writeto(csf_file)


def stack_calibrated(cal_dir, cal_glob):

    """
        Take a list of science images with calibrations already applied and
        return an image using a cumulative pixel-wise sum
    """

    # list of calibrated images to stack
    cal_list = glob.glob(cal_dir + cal_glob)

    # get first one then loop through rest
    final_image = fits.getdata(cal_list[0])

    for f in cal_list[1:]:
        print(f"Adding {f} ...")
        final_image = final_image + fits.getdata(f)

    return final_image


def main():
    """
    " options
    """
    # which type of science frames we have
    mode_selected = 'mono'
    processing_modes = {
            'mono': ['mono'],
            'rgb': ['red', 'green', 'blue']
            }

    # # after applying calibration, stack science frames?
    stack_science_frames = False

    # flats scaling to apply
    scale_inv_med = True

    """
    " input
    """
    # calibration directories
    cal_dirs = {
        'bias': 'calibration/bias/',
        'dark': 'calibration/darks/',
        'flats': {
            # 'red': 'Orion Nebula/calibration/flats red/',
            # 'green': 'Orion Nebula/calibration/flats green/',
            # 'blue': 'Orion Nebula/calibration/flats blue/'
            'mono': 'calibration/flats/'
        }
    }
    # select all .fits files
    cal_glob = "*.fits"

    # science frame directories
    science_dirs = {
        # 'red': 'Orion Nebula/science red/',
        # 'green': 'Orion Nebula/science green/',
        # 'blue': 'Orion Nebula/science blue/',
        'mono': 'science/'
    }
    science_glob = "*.fits"

    # exposure times that we could probably just read from fits header
    exposure_times = {
        'bias': 0.032 * u.ms,
        'dark': 56.192 * u.ms,
        'flats': {
            # 'red': 1.411 * u.s,
            # 'green': 1.411 * u.s,
            # 'blue': 1.411 * u.s
            'mono': 92.013 * u.ms
        },
        'science': {
            # 'red': 3.0 * u.s,
            # 'green': 3.0 * u.s,
            # 'blue': 3.0 * u.s,
            'mono': 56.192 * u.ms
        }
    }

    """
    " output
    """
    master_files = {
        'bias': 'master_bias.fits',
        'dark': 'master_dark.fits',
        'flats': {
            # 'red': 'master_flat_red.fits',
            # 'green': 'master_flat_green.fits',
            # 'blue': 'master_flat_blue.fits'
            'mono': 'master_flat.fits'
        }
    }

    """
    " creation of master bias and dark frames
    """
    # create master bias and dark
    for cal_type in ['bias', 'dark']:
        # check if master already exists in current directory
        if not os.path.isfile(master_files[cal_type]):
            print(f"Master {cal_type} not found, creating ...")
            master_frame = stack_fits_from_dir(cal_dirs[cal_type], cal_glob, sigma_low=2, sigma_high=2)
            hdul = fits.HDUList([fits.PrimaryHDU(master_frame)])
            hdul.writeto(master_files[cal_type])
            del master_frame
            del hdul
        else:
            print(f"{master_files[cal_type]} exists, skipping generation ...")

    # Read in the master bias and dark
    master_bias = fits.getdata(master_files['bias'])
    master_dark = fits.getdata(master_files['dark'])

    # create master flats: red, green, blue or just mono
    for flat_type in processing_modes[mode_selected]:
        # check if master red flat exists
        if not os.path.isfile(master_files['flats'][flat_type]):
            print(f"Master {flat_type} flat not found, creating with {scale_inv_med=} ...")
            make_master_flat(cal_dirs['flats'][flat_type],
                             cal_glob, master_files['flats'][flat_type],
                             master_bias, scale_inv_med=scale_inv_med, sigma_low=2, sigma_high=2)
        else:
            print(f"{master_files['flats'][flat_type]} exists, skipping generation ...")

    """
    " apply calibration to our image collection across all filters to process
    """
    # calibrate science frames
    for science_type in processing_modes[mode_selected]:
        t_dark = exposure_times['dark'].to(u.s).value
        t_sci = exposure_times['science'][science_type].to(u.s).value
        master_flat = fits.getdata(master_files['flats'][science_type])
        calibrate_science_images(science_dirs[science_type], science_glob, master_bias, master_dark, master_flat, t_sci, t_dark)
        del master_flat

    if mode_selected != 'rgb':
        print(f"Mode selected was {mode_selected}, skipping rgb stacking ...")
        return

    if stack_science_frames:
        # add all calibrated science images of each color
        csf_glob = '*_calibrated.fits'
        stacked_suffix = '_stacked.fits'
        for color in processing_modes[mode_selected]:
            if os.path.isfile(color + stacked_suffix):
                print(f"Found final stacked image for {color}, skipping ...")
                continue
            else:
                total_image = stack_calibrated(science_dirs[color], csf_glob)
                hdul = fits.HDUList([fits.PrimaryHDU(total_image)])
                hdul.writeto(color + stacked_suffix)

    """
    " combine final science frames of each color into an rgb image
    """
    rgb_file = 'rgb_stacked.png'
    # set the color scale manually, but we should be using relative brightnesses instead
    r_scale = 3.0
    g_scale = 2.0
    b_scale = 2.0
    # load the calibrated & stacked science frames
    stacked_imgs = {}
    for color in ['red', 'green', 'blue']:
        print("Loading ", color + stacked_suffix, " ...")
        stacked_imgs[color] = fits.getdata(color + stacked_suffix)

    # custom processing for a particular data set
    # try to align green frame to red, after picking pixels that should be aligned
    dst_r = np.array([3355, 1420])
    src_g = np.array([3412, 1434])
    r_to_g_diff = dst_r - src_g
    shift_g = ndi.shift(stacked_imgs['green'], r_to_g_diff)

    # custom processing for a particular data set
    # try to align blue frame to red
    dst_r = np.array([3105, 1107])
    src_b = np.array([3098, 1097])
    r_to_b_diff = dst_r - src_b
    shift_b = ndi.shift(stacked_imgs['blue'], r_to_b_diff)

    # combine to rgb and save
    rgb_data = make_lupton_rgb(stacked_imgs['red'] * r_scale, shift_g * g_scale, shift_b * b_scale, filename=rgb_file,
                               Q=0.1, stretch=0.05)

    # show the final product too
    plt.imshow(rgb_data, origin='lower')
    plt.show()


if __name__ == "__main__":
    main()
