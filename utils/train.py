# coding=utf8
from __future__ import division
import torch
import os, time
from torch.autograd import Variable
import logging
import torch.nn.functional as F
import numpy as np
from logs import configlog
from torch import nn


def trainlog(logfilepath, head='%(asctime)-15s %(message)s'):
    configlog(logfilepath, head='%(asctime)-15s %(message)s')

def train(model,
          epoch_num,
          batch_size,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          dataset,
          data_loader,
          usecuda,
          save_inter,
          save_dir
          ):

    for epoch in range(start_epoch, epoch_num):
        t_s = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step(epoch)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            epoch_loss = 0
            epoch_corrects = 0
            epoch_size = len(dataset[phase]) // batch_size

            t0 = time.time()
            for batch_cnt, data in enumerate(data_loader[phase]):
                # print data
                t1 = time.time()
                since = t1 - t0
                t0 = t1
                imgs,labels = data

                if usecuda:
                    imgs = Variable(imgs.cuda())
                    labels = Variable(labels.cuda())

                else:
                    imgs = Variable(imgs)
                    labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward

                outputs = model(imgs)
                loss = criterion(outputs,labels)
                _, preds = torch.max(outputs.data, 1)


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                epoch_loss += loss.data[0]

                batch_corrects = torch.sum(preds == labels.data)
                batch_acc = batch_corrects / preds.size(0)

                epoch_corrects += batch_corrects

                # batch loss
                if (batch_cnt % 5 == 0) and phase == 'train':
                    logging.info(
                        'epoch[%d]-batch[%d] ||batch-loss: %.4f ||acc @1: %.3f ||%.3f sec/batch '
                        % (epoch, batch_cnt, loss.data[0], batch_acc, since))

            epoch_acc = epoch_corrects / len(dataset[phase])
            epoch_loss = epoch_loss / epoch_size

            if phase == 'train':
                logging.info('epoch[%d]-train-loss: %.4f||train-acc@1: %.4f '
                             % (epoch, epoch_loss, epoch_acc))

            if phase == 'val':
                logging.info('epoch[%d]-val-loss: %.4f ||val-acc@1 : %.4f'
                             % (epoch, epoch_loss, epoch_acc))
                # save model
                if epoch % save_inter == 0:
                    save_path = os.path.join(save_dir,
                                             'weights-%d-[%.4f].pth' % (epoch, epoch_acc))
                    torch.save(model.state_dict(), save_path)
                    logging.info('saved model to %s' % (save_path))
                t_e = time.time()
                logging.info('----time cost: %d sec' % (t_e - t_s))
                logging.info('===' * 20)
