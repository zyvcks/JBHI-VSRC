import torch
from utils import BaseWriter_VSRC as BaseWriter
from losses3D.VSRC_loss import DTU_Student_Loss


class Base_Trainer_VSRC:
    def __init__(self, args, l_model, r_model, l_optimizer, r_optimizer, train_data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None,  pre_epoch=1):
        self.args = args
        self.l_model = l_model
        self.r_model = r_model
        self.l_optimizer = l_optimizer
        self.r_optimizer = r_optimizer
        self.train_data_loader = train_data_loader
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.labeled_bs = self.args.labeled_bs
        self.writer = BaseWriter.TensorboardWriter(args)
        self.save_frequency = 2
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = pre_epoch

    def training(self):
        for epoch in range(self.start_epoch, self.args.nEpochs):
            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            l_val_loss = self.writer.data['val']['l_loss_dice'] / self.writer.data['val']['count']
            r_val_loss = self.writer.data['val']['r_loss_dice'] / self.writer.data['val']['count']

            if self.args.l_save is not None and ((epoch + 1)) % self.save_frequency:
                self.l_model.save_checkpoint(self.args.l_save,
                                                epoch, l_val_loss,
                                                optimizer=self.l_optimizer)
            if self.args.r_save is not None and ((epoch + 1)) % self.save_frequency:
                self.r_model.save_checkpoint(self.args.r_save,
                                                epoch, r_val_loss,
                                                optimizer=self.r_optimizer)
            self.writer.write_end_of_epoch(epoch)

            self.writer.reset('train')
            self.writer.reset('val')

    def train_epoch(self, epoch):
        self.l_model.train()
        self.r_model.train()
        for batch_idx, sampled_batch in enumerate(self.train_data_loader):
            self.l_optimizer.zero_grad()
            self.r_optimizer.zero_grad()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            labeled_volume_batch = label_batch[:self.labeled_bs]
            # unlabeled_volume_batch = volume_batch[self.labeled_bs:]

            noise_1 = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            noise_2 = torch.clamp(torch.randn_like(volume_batch) * 0.2, -0.5, 0.5)

            l_inputs = volume_batch
            r_inputs = volume_batch + noise_1
            le_inputs = volume_batch + noise_2
            re_inputs = volume_batch
            le_inputs.requires_grad = False
            re_inputs.requires_grad = False

            l_outputs_sdf, l_outputs = self.l_model(l_inputs)
            r_outputs_sdf, r_outputs = self.r_model(r_inputs)
            with torch.no_grad():
                le_outputs_sdf, le_outputs = self.l_model(le_inputs)
                re_outputs_sdf, re_outputs = self.r_model(re_inputs)

            dual_student_loss = DTU_Student_Loss(self.args, labeled_volume_batch, l_outputs, r_outputs,
                                                  l_outputs_sdf, r_outputs_sdf, le_outputs, re_outputs,
                                                  le_outputs_sdf, re_outputs_sdf, epoch)

            l_loss, r_loss, l_loss_dice, r_loss_dice = dual_student_loss.cal_train_loss()

            l_loss.backward(retain_graph=True)
            self.l_optimizer.step()
            r_loss.backward()
            self.r_optimizer.step()

            self.writer.update_scores(batch_idx, l_loss.item(), r_loss.item(), l_loss_dice.item(), r_loss_dice.item(),
                                      'train', epoch * self.len_epoch + batch_idx)

            if(batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')
        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.l_model.eval()
        self.r_model.eval()

        for batch_idx, sampled_batch in enumerate(self.valid_data_loader):
            with torch.no_grad():
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                # unlabeled_volume_batch = volume_batch[self.labeled_bs:]
                l_outputs_sdf, l_outputs = self.l_model(volume_batch)
                r_outputs_sdf, r_outputs = self.r_model(volume_batch)
                label_batch_label = label_batch == 1

                dual_student_loss = DTU_Student_Loss(self.args, label_batch_label, l_outputs, r_outputs,
                                                      l_outputs_sdf, r_outputs_sdf, epoch)
                l_loss_dice, r_loss_dice = dual_student_loss.cal_val_loss()

                self.writer.update_scores(batch_idx, None, None, l_loss_dice.item(), r_loss_dice.item(),
                                          'val', epoch * self.len_epoch + batch_idx)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)

