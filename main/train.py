# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import argparse
from config import cfg
import torch
from base import Trainer, Validator
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    cudnn.benchmark = True

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()

    # early stopping
    #early_stopping = EarlyStopping(patience=5, delta=0)
    early_stop = False

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        itrs_log = []
        error_log = []

        if early_stop: break
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            if early_stop: break
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            
            #print(cfg.model_dir)

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(inputs, targets, meta_info, 'train')
            loss = {k:loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            
            if itr % 10 == 0:
                screen = [
                    'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                    'lr: %g' % (trainer.get_lr()),
                    'speed: %.2f(%.2fs r%.2f)s/itr' % (
                        trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                    '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                    ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()


            # validation step
            '''
            if (itr+1) % 500 == 0:
                trainer.logger.info('Evaluating current snapshot...')

                tot_loss = []
                for i, (inputs, targets, meta_info) in enumerate(trainer.val_generator):
                    # forward
                    with torch.no_grad():
                        loss = trainer.model(inputs, targets, meta_info, 'val')
                        loss = {k:loss[k].mean() for k in loss}
                        loss = sum(loss[k] for k in loss) / len(loss)
                        tot_loss.append(loss.item())

                curr_loss = sum(tot_loss) / len(tot_loss)
                itrs_log.append(itr)
                error_log.append(curr_loss)
                trainer.logger.info('Current snapshot loss: ' + str(curr_loss))
                #early_stopping(curr_loss, trainer.model.state_dict())
            '''

            # Early stop
            if itr >= 45000: early_stop = True
        
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)

        '''

        plt.plot(itrs_log, error_log)
        plt.xlabel('Iterations')
        plt.ylabel('Validation error')
        plt.title('Validation error during training')
        plt.savefig('results-dump/val_error.png')

        '''
        
class EarlyStopping:
    def __init__(self, patience=1, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.snapshot = None

    def __call__(self, loss, snapshot):
        score = loss

        if self.best_score == None:
            self.best_score = score
            self.snapshot = snapshot
        else:
            if score > self.best_score + self.delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
    
if __name__ == "__main__":
    main()
