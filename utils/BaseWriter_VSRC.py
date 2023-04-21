import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils as utils

dict_class_names = {
                    "LA2018": ["GT"]
                    }

class TensorboardWriter():
    def __init__(self, args):
        name_model = args.model + "_" + args.dataset + "_" + utils.datestr()
        self.writer = SummaryWriter(log_dir=args.logdir + name_model, comment=name_model)
        self.l_save_path = args.l_save
        self.r_save_path = args.r_save

        utils.make_dirs(args.l_save)
        utils.make_dirs(args.r_save)
        self.dataset_name = args.dataset
        self.classes = args.nclass
        self.label_names = dict_class_names[args.dataset]

        self.data = self.create_data_structure()

    def create_data_structure(self, ):
        data = {"train": dict((label, 0.0) for label in self.label_names),
                "val": dict((label, 0.0) for label in self.label_names)}
        data['train']['l_loss'] = 0.0
        data['train']['r_loss'] = 0.0
        data['train']['l_loss_dice'] = 0.0
        data['val']['l_loss_dice'] = 0.0
        data['train']['r_loss_dice'] = 0.0
        data['val']['r_loss_dice'] = 0.0
        data['train']['count'] = 1.0
        data['val']['count'] = 1.0
        return data

    def create_stats_files(self, path):
        train_f = open(os.path.join(path, 'train.csv'), 'w')
        val_f = open(os.path.join(path, 'val.csv'), 'w')
        return train_f, val_f

    def reset(self, mode):
        self.data[mode]['l_loss'] = 0.0
        self.data[mode]['r_loss'] = 0.0
        self.data[mode]['l_loss_dice'] = 0.0
        self.data[mode]['r_loss_dice'] = 0.0
        self.data[mode]['count'] = 1
        for i in range(len(self.label_names)):
            self.data[mode][self.label_names[i]] = 0.0

    def display_terminal(self, iter, epoch, mode='train', summary=False):

        if mode == 'train':
            if summary:
                info_print = "Summary {} --Epoch {:2d}--l_loss:{:.4f}--r_loss:{:.4f}--l_dice:{:.4f}" \
                             "--r_dice:{:.4f}"\
                    .format(mode, epoch, self.data[mode]['l_loss'] /self.data[mode]['count'],
                            self.data[mode]['r_loss'] /self.data[mode]['count'],
                            self.data[mode]['l_loss_dice'] /self.data[mode]['count'],
                            self.data[mode]['r_loss_dice'] /self.data[mode]['count'])

                for i in range(len(self.label_names)):
                    info_print += "--{} : {:.6f}".format(self.label_names[i],
                                                         self.data[mode][self.label_names[i]] / self.data[mode]['count'])

                print(info_print)
            else:

                info_print = "Epoch: {:.2f}--l_loss:{:.4f}--r_loss:{:.4f}--l_dice:{:.4f}--r_dice:{:.4f}"\
                    .format(iter, self.data[mode]['l_loss'] /self.data[mode]['count'],
                            self.data[mode]['r_loss'] /self.data[mode]['count'],
                            self.data[mode]['l_loss_dice'] /self.data[mode]['count'],
                            self.data[mode]['r_loss_dice'] /self.data[mode]['count'])

                for i in range(len(self.label_names)):
                    info_print += "--{}:{:.6f}".format(self.label_names[i],
                                                       self.data[mode][self.label_names[i]] / self.data[mode]['count'])
                print(info_print)
        elif mode == 'val':
            if summary:
                info_print = "Summary {} ---Epoch {:2d}---l_loss_dice:{:.4f}--r_loss_dice:{:.4f}"\
                    .format(mode, epoch, self.data[mode]['l_loss_dice'] / self.data[mode]['count'],
                            self.data[mode]['r_loss_dice'] / self.data[mode]['count'])

                for i in range(len(self.label_names)):
                    info_print += "---{} : {:.6f}".format(self.label_names[i],
                                                         self.data[mode][self.label_names[i]] / self.data[mode]['count'])

                print(info_print)
            else:

                info_print = "Epoch: {:.2f}---l_loss_dice:{:.4f}--r_loss_dice:{:.4f}" \
                    .format(iter, self.data[mode]['l_loss_dice'] / self.data[mode]['count'],
                            self.data[mode]['r_loss_dice'] / self.data[mode]['count']
                            )

                for i in range(len(self.label_names)):
                    info_print += "---{}:{:.6f}".format(self.label_names[i],
                                                       self.data[mode][self.label_names[i]] / self.data[mode]['count'])
                print(info_print)
        else:
            raise Exception("No Mode Error")

    def update_scores(self, iter, l_loss, r_loss, l_loss_dice, r_loss_dice, mode, writer_step):
        """
        :param writer_step: tensorboard writer step
        """
        # WARNING ASSUMING THAT CHANNELS IN SAME ORDER AS DICTIONARY
        l_dsc = (1 - l_loss_dice) * 100
        r_dsc = (1 - r_loss_dice) * 100
        dsc = (l_dsc + r_dsc) / 2

        if mode == 'train':
            self.data[mode]['l_loss'] += l_loss
            self.data[mode]['r_loss'] += r_loss
            self.data[mode]['l_loss_dice'] += l_loss_dice
            self.data[mode]['r_loss_dice'] += r_loss_dice
            self.data[mode]['count'] = iter + 1
            self.data[mode][self.label_names[0]] += dsc

            if self.writer is not None:
                self.writer.add_scalar(mode + '/' + self.label_names[0], l_loss,
                                       global_step=writer_step)

        elif mode == 'val':
            self.data[mode]['l_loss_dice'] += l_loss_dice
            self.data[mode]['r_loss_dice'] += r_loss_dice
            # self.data[mode]['consistency_loss'] += consistency_loss
            self.data[mode]['count'] = iter + 1
            self.data[mode][self.label_names[0]] += dsc

            if self.writer is not None:
                self.writer.add_scalar(mode + '/' + self.label_names[0], l_loss_dice, global_step=writer_step)

        else:
            raise Exception("No Mode Error")


    def write_end_of_epoch(self, epoch):

        self.writer.add_scalars('l_Loss/', {'train': self.data['train']['l_loss'] / self.data['train']['count'],
                                         'val': self.data['val']['l_loss_dice'] / self.data['val']['count'],
                                         }, epoch)
        self.writer.add_scalars('r_Loss/', {'train': self.data['train']['r_loss'] / self.data['train']['count'],
                                          'val': self.data['val']['r_loss_dice'] / self.data['val']['count'],
                                          }, epoch)
        for i in range(len(self.label_names)):
            self.writer.add_scalars(self.label_names[i],
                                    {'train': self.data['train'][self.label_names[i]] / self.data['train']['count'],
                                     'val': self.data['val'][self.label_names[i]] / self.data['val']['count'],
                                     }, epoch)

        train_csv_line = 'Epoch:{:2d} l_Loss:{:.4f} r_Loss:{:.4f}'.format(epoch,
                                                                     self.data['train']['l_loss'] / self.data['train'][
                                                                         'count'],
                                                                     self.data['train']['r_loss'] / self.data['train'][
                                                                         'count'])
        val_csv_line = 'Epoch:{:2d} l_loss_dice:{:.4f} r_loss_dice:{:.4f} GT{:.4f}'\
            .format(epoch, self.data['val']['l_loss_dice'] / self.data['val']['count'],
                    self.data['val']['r_loss_dice'] / self.data['val']['count'],
                    self.data['val'][self.label_names[0]] / self.data['val']['count']
                    )
        train_f = open(os.path.join(self.l_save_path, 'train.csv'), 'a', encoding='utf-8')
        val_f = open(os.path.join(self.l_save_path, 'val.csv'), 'a', encoding='utf-8')
        train_f.write(train_csv_line + '\n')
        val_f.write(val_csv_line + '\n')
        train_f.close()
        val_f.close()