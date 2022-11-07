import transforms3d
import pandas as pd
import numpy as np
import shutil
import os

if __name__ == '__main__':
    orig_nums = ['3', '5', '6', '7', '9', '10', '101', '103', '105']

    for orig_num in orig_nums:
        orig_dir = f'../indoor_forward_{orig_num}_snapdragon_with_gt/'
        orig_gt_fn = orig_dir + 'groundtruth.txt'
        orig_imfile_fn = orig_dir + 'left_images.txt'
        orig_images_dir = orig_dir + 'img/'

        post_num = '4' + orig_num
        post_gt_fn = f'KITTI/pose_GT/{post_num}.txt'
        post_images_dir = f'KITTI/images/{post_num}/'

        try:
            os.makedirs(post_images_dir)
            print(f'Importing {orig_dir}...')
        except FileExistsError:
            print(f'{orig_dir} already imported, continuing...')
            continue

        orig_gt_df = pd.read_csv(orig_gt_fn, delimiter=' ', skiprows=1, names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        orig_imgs_df = pd.read_csv(orig_imfile_fn, delimiter=' ', skiprows=1, names=['id', 'timestamp', 'image_name', ''])

        first_img = orig_imgs_df[orig_imgs_df.timestamp > orig_gt_df.iloc[0].timestamp].index[0] - 1
        last_img = orig_imgs_df[orig_imgs_df.timestamp > orig_gt_df.iloc[-1].timestamp].index[0] - 1
        viable_imgs_df = orig_imgs_df.iloc[first_img:last_img + 1]

        new_df = pd.DataFrame(index=range(len(viable_imgs_df)), columns=range(12))

        im_row = viable_imgs_df.iloc[0]
        row = orig_gt_df[orig_gt_df.timestamp >= im_row.timestamp].iloc[0]
        R = np.array(transforms3d.quaternions.quat2mat((row['qw'], row['qx'], row['qy'], row['qz'])))
        t = np.array([[row['tx'], row['ty'], row['tz']]])
        Rt0 = np.concatenate((np.concatenate((R, t.T), axis=1), np.array([[0, 0, 0, 1]])))
        Rt0inv = np.linalg.inv(Rt0)

        i = 0
        for (_, img_row), (_, new_row) in zip(viable_imgs_df.iterrows(), new_df.iterrows()):
            gt_row = orig_gt_df[orig_gt_df.timestamp >= img_row.timestamp].iloc[0]
            R = np.array(transforms3d.quaternions.quat2mat((gt_row['qw'], gt_row['qx'], gt_row['qy'], gt_row['qz'])))
            t = np.array([[gt_row['tx'], gt_row['ty'], gt_row['tz']]])
            Rt = np.concatenate((np.concatenate((R, t.T), axis=1), np.array([[0, 0, 0, 1]])))
            Rrel = np.matmul(Rt0inv, Rt)[:3, :]
            R_flat = np.reshape(Rrel, (12,))
            new_row[:] = R_flat
            shutil.copyfile(orig_dir + img_row.image_name, post_images_dir + f'{i:05}.png')
            i += 1

        new_df.to_csv(post_gt_fn, sep=' ', header=False, index=False)
        print(f'Imported {orig_dir}')

    print('Finished all imports')
