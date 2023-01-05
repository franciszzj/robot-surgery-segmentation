import os
import sys
import cv2
import numpy as np

height, width = 1024, 1280
h_start, w_start = 28, 320

binary_factor = 255
parts_factor = 85
instruments_factor = 32


def process_one_image(image_path, save_image_path, type='image'):
    image = cv2.imread(image_path)
    if type == 'parts':
        image[image == 30] = 1
        image[image == 100] = 2
        image[image == 255] = 3
        image *= parts_factor
    if type == 'instruments':
        image *= instruments_factor
    save_image = image[h_start: h_start + height, w_start: w_start + width]
    save_image = save_image.astype(np.uint8)
    cv2.imwrite(save_image_path, save_image)


def process(image_root, masks_root, save_root):

    for i in range(1, 11):
        image_dir = os.path.join(image_root, 'instrument_dataset_{}/left_frames'.format(i))
        print('process {} ...'.format(image_dir))
        image_files = os.listdir(image_dir)

        save_image_dir = os.path.join(save_root, 'instrument_dataset_{}'.format(i), 'images')
        save_binary_mask_dir = os.path.join(save_root, 'instrument_dataset_{}'.format(i), 'binary_masks')
        save_parts_mask_dir = os.path.join(save_root, 'instrument_dataset_{}'.format(i), 'parts_masks')
        save_instruments_mask_dir = os.path.join(save_root, 'instrument_dataset_{}'.format(i), 'instruments_masks')
        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)
        if not os.path.exists(save_binary_mask_dir):
            os.makedirs(save_binary_mask_dir)
        if not os.path.exists(save_parts_mask_dir):
            os.makedirs(save_parts_mask_dir)
        if not os.path.exists(save_instruments_mask_dir):
            os.makedirs(save_instruments_mask_dir)
        for image_file in image_files:
            print(image_file)
            image_path = os.path.join(image_dir, image_file)
            binary_mask_path = image_path.replace(image_root, masks_root).replace('instrument_dataset_{}/left_frames'.format(
                i), os.path.join('instrument_dataset_{}'.format(i), 'BinarySegmentation')).replace('.jpg', '.png')
            parts_mask_path = image_path.replace(image_root, masks_root).replace('instrument_dataset_{}/left_frames'.format(
                i), os.path.join('instrument_dataset_{}'.format(i), 'PartsSegmentation')).replace('.jpg', '.png')
            instruments_mask_path = image_path.replace(image_root, masks_root).replace('instrument_dataset_{}/left_frames'.format(
                i), os.path.join('instrument_dataset_{}'.format(i), 'TypeSegmentation')).replace('.jpg', '.png')

            save_image_path = os.path.join(save_image_dir, image_file)
            process_one_image(image_path, save_image_path, 'image')
            save_binary_mask_path = os.path.join(save_binary_mask_dir, image_file)
            process_one_image(binary_mask_path, save_binary_mask_path, 'binary')
            save_parts_mask_path = os.path.join(save_parts_mask_dir, image_file)
            process_one_image(parts_mask_path, save_parts_mask_path, 'parts')
            save_instruments_path = os.path.join(save_instruments_mask_dir, image_file)
            process_one_image(instruments_mask_path, save_instruments_path, 'instruments')


if __name__ == '__main__':
    image_root = '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/test'
    masks_root = '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/test/instrument_2017_test'
    save_root = '/jmain02/home/J2AD019/exk01/zxz35-exk01/data/cambridge-1/EndoVis2017/cropped_test'
    process(image_root, masks_root, save_root)
