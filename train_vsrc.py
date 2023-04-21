import os
import torch
import argparse
import utils as utils
import data_loaders as data_loaders
import models as models
from trainer.base_trainer_VSRC import Base_Trainer_VSRC
from visual3D.viz import test_calculate_metric

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_args():
    parser = argparse.ArgumentParser()
    # base enviroment
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
    parser.add_argument('--nEpochs', type=int, default=500)
    parser.add_argument('--terminal_show_freq', default=1)
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--root_path', type=str, default="./datasets/", help='datasets root path')
    parser.add_argument('--exp', type=str,  default='UAMT_unlabel', help='model_name')
    parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')

    # weight & costs
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
    parser.add_argument('--tau_p', default=0.70, type=float,
                        help='confidece threshold for positive pseudo-labels, default 0.70')
    parser.add_argument('--tau_n', default=0.30, type=float,
                        help='confidece threshold for negative pseudo-labels, default 0.30')
    parser.add_argument('--sdf_threshold', default=0.30, type=float,
                        help='confidece threshold for negative pseudo-labels, default 0.30')
    parser.add_argument('--label_threshold', default=0.5, type=float, metavar='THRESHOLD',
                        help='threshold for stable sample')
    parser.add_argument('--stable_threshold', default=0.01, type=float, metavar='THRESHOLD',
                        help='threshold for stable sample')

    # dataset & model
    # parser.add_argument('--dataset', type=str, default="LA2018", choices=('LA2018', 'PUTH', 'PICAI_WG'))
    parser.add_argument('-d', '--dataset', type=str, default="LA2018")
    parser.add_argument('-p', '--patch_size', nargs="+", type=int, default=(112, 112, 80))
    parser.add_argument('--model', type=str, default='sdf_VNet', choices=('VNET', 'sdf_VNet'))
    parser.add_argument('--split', type=float, default=0.2, help='Select percentage of labeled data(default: 0.2)')
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam'), help='optimizer selection')
    parser.add_argument('--dice_smooth', type=float, default='1e-10')
    parser.add_argument('--logdir', type=str, default='./runs/')

    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('-save_viz', '--save_test_result', action='store_true', default=False)


    parser.add_argument('--pretrained', action='store_true', default=False, help='whether use pretrained model_lastversion')
    parser.add_argument('-lp', '--l_pretrained_path',
                        default='./works/DualModel/l_pretrained/sdf_VNet_30_09___15_19_LA2018_last_epoch.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model_lastversion')
    parser.add_argument('-rp', '--r_pretrained_path',
                        default='./works/DualModel/r_pretrained/sdf_VNet_30_09___15_19_LA2018_last_epoch.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model_lastversion')

    parser.add_argument('-tp', '--test_checkpoint_path',
                        default='./works/DualModel/test_checkpoint/sdf_VNet_LA2018_20.pth',
                        type=str, metavar='PATH',
                        help='test_checkpoint_path')
    args = parser.parse_args()
    args.l_save = './works/DualModel/l_model/' + args.model + '_checkpoints/' + args.model + '_{}_{}'.format(utils.datestr(), args.dataset)
    args.r_save = './works/DualModel/r_model/' + args.model + '_checkpoints/' + args.model + '_{}_{}'.format(utils.datestr(), args.dataset)
    return args

def main():
    args = get_args()
    utils.reproducibility(args, args.seed)
    utils.make_dirs(args.l_save)
    utils.make_dirs(args.r_save)

    train_loader, val_loader = data_loaders.generate_datasets(args, SSL=True)

    l_model, l_optimizer = models.create_model(args)
    r_model, r_optimizer = models.create_model(args)

    pre_epoch = 1
    if args.pretrained:
        l_pre_epoch = l_model.restore_checkpoint(args.l_pretrained_path)
        r_pre_epoch = r_model.restore_checkpoint(args.r_pretrained_path)
        assert l_pre_epoch == r_pre_epoch
        pre_epoch = l_pre_epoch
        print("=====LOAD PRETRAINED MODEL SUCCESSFULLY=====")
    if args.cuda:
        l_model = l_model.cuda()
        r_model = r_model.cuda()
        print("Model transferred in GPU. . . . . .")

    trainer = Base_Trainer_VSRC(args, l_model, r_model, l_optimizer, r_optimizer, train_loader, val_loader,
                                        test_data_loader=None, lr_scheduler=None, pre_epoch=pre_epoch)

    if args.train:
        print("START TRAINING. . . . . . ")
        trainer.training()
    if args.test:
        print("START TESTING. . . . . . ")
        test_calculate_metric(args)


if __name__ == '__main__':
    main()
