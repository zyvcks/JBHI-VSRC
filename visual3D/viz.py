import os
import glob
import math
import numpy as np
import torch
import h5py
import csv
import nibabel as nib
import models as models
from medpy import metric
import torch.nn.functional as F
import SimpleITK as sitk


def test_calculate_metric(args):
    test_checkpoint_path = args.test_checkpoint_path
    model, _ = models.create_model(args)
    epoch = model.restore_checkpoint(test_checkpoint_path)
    model = model.cuda()

    print("init weight from {}".format(test_checkpoint_path))
    model.eval()
    dataset = args.dataset
    num_classes = args.nclass
    patch_size = args.patch_size
    save_viz = args.save_viz
    if dataset == "LA2018":
        avg_metric = test_all_case(args, model, dataset, num_classes=num_classes, patch_size=patch_size,
                               stride_x=18, stride_y=18, stride_z=4, save_result=save_viz)

    else:
        raise Exception("No Dataset Error")

    return avg_metric


def test_all_case(args, model, dataset, num_classes, patch_size, stride_x, stride_y, stride_z, save_result=True,
                  preproc_fn=None):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataset == "PUTH":
        test_save_path = args.root_path + "PUTH/test_save/"
        data_path = args.root_path + "PUTH/val/puth_h5/"
        img_ids = get_train_ids(os.listdir(data_path))
        img_nums = len(img_ids)
    elif dataset == "PICAI_WG":
        test_save_path = args.root_path + "PICAI_WG/test_save/"
        data_path = args.root_path + "PICAI_WG/val/picai_h5/"
        img_ids = get_train_ids_dot(os.listdir(data_path))
        img_nums = len(img_ids)
    elif dataset == "LA2018":
        test_save_path = args.root_path + "LA2018/test_save/"
        data_path = args.root_path + "LA2018/val/"
        img_ids = os.listdir(data_path)
        img_nums = len(img_ids)
    else:
        data_path = None
        img_ids = None
        img_nums = None
        test_save_path = None
        print("data exist error")

    total_metric = 0.0
    eval_metrics = []

    for i in range(img_nums):
        if dataset == "PUTH":
            img_path = os.path.join(data_path, img_ids[i] + '_mri.h5')
        elif dataset == 'PICAI_WG':
            img_path = os.path.join(data_path, img_ids[i] + '.h5')
        elif dataset == "LA2018":
            img_path = os.path.join(data_path, img_ids[i], 'la_mri.h5')

        h5f = h5py.File(img_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction = test_single_case(model, image, stride_x, stride_y, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction[:], label[:])
        total_metric += np.asarray(single_metric)
        eval_metrics.append({'dice': single_metric[0], 'jc': single_metric[1], 'hd': single_metric[2], 'asd': single_metric[3]})

        print('{} predict successed'.format(img_ids[i]))

        if save_result:

            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(pred_itk, test_save_path +
                            "/{}_pred.nii.gz".format(img_ids[i]))

            img_itk = sitk.GetImageFromArray(image)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, test_save_path +
                            "/{}_img.nii.gz".format(img_ids[i]))

            lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
            lab_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(lab_itk, test_save_path +
                            "/{}_lab.nii.gz".format(img_ids[i]))

            print('{} save successed'.format(img_ids[i]))

            eval_head = ['dice', 'jc', 'hd', 'asd']
            with open(test_save_path + "/vsrc_la2018_metrics.csv", "w", encoding='UTF8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=eval_head)
                writer.writeheader()
                writer.writerows(eval_metrics)

    avg_metric = total_metric / img_nums
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(model, image, stride_x, stride_y, stride_z, patch_size, num_classes=1):
    """
    code refer to:
    https://github.com/yulequan/UA-MT/blob/master/code/test_util.py
    """

    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_x) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_y) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_x*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_y * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                y_sdf, y = model.inference(test_patch)
                # y = model.inference(test_patch)

                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                # y = y[0, 0, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    # label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        # label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    # return label_map, score_map
    score_map = score_map[0, :, :, :]
    score_map[score_map <= 0.5] = 0
    score_map[score_map > 0.5] = 1
    return score_map

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd

def get_train_ids(label_names):
    ids = []
    for label_name in label_names:
        name_split = label_name.split("_")
        img_id = name_split[0][:]
        ids.append(img_id)
    return ids

def get_train_ids_dot(label_names):
    ids = []
    for label_name in label_names:
        name_split = label_name.split(".")
        img_id = name_split[0][:]
        ids.append(img_id)
    return ids