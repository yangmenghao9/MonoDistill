import os
import tqdm

import torch
import numpy as np
import torch.nn as nn
import shutil
import collections
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
# from lib.helpers.save_helper import visualize_feature_map
from lib.helpers.decode_helper import extract_dets_from_outputs

from lib.losses.centernet_loss import compute_centernet3d_loss
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

from lib.helpers.utils_helper import judge_nan

import PIL
import matplotlib.pyplot as plt
import math

# from progress.bar import Bar
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 model_type,
                 root_path,
                 kd_type=None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.class_name = self.test_loader.dataset.class_name
        self.model_type = model_type
        self.root_path = root_path

        # loading pretrain/resume model
        if self.model_type == 'centernet3d':
            if cfg.get('pretrain_model'):
                assert os.path.exists(cfg['pretrain_model'])
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=cfg['pretrain_model'],
                                map_location=self.device,
                                logger=self.logger)

            if cfg.get('resume_model', None):
                assert os.path.exists(cfg['resume_model'])
                self.epoch = load_checkpoint(model=self.model,
                                             optimizer=self.optimizer,
                                             filename=cfg['resume_model'],
                                             map_location=self.device,
                                             logger=self.logger)
                self.lr_scheduler.last_epoch = self.epoch - 1

        if self.model_type == 'distill':
            if cfg.get('pretrain_model'):
                if os.path.exists(cfg['pretrain_model']['rgb']):
                    load_checkpoint(model=self.model.centernet_rgb,
                                    optimizer=None,
                                    filename=cfg['pretrain_model']['rgb'],
                                    map_location=self.device,
                                    logger=self.logger)
                else:
                    self.logger.info("no rgb pretrained model")
                    assert os.path.exists(cfg['pretrain_model']['rgb'])

                if os.path.exists(cfg['pretrain_model']['depth']):
                    load_checkpoint(model=self.model.centernet_depth,
                                    optimizer=None,
                                    filename=cfg['pretrain_model']['depth'],
                                    map_location=self.device,
                                    logger=self.logger)
                else:
                    self.logger.info("no depth pretrained model")
                    assert os.path.exists(cfg['pretrain_model']['depth'])

            if cfg.get('resume_model', None):
                assert os.path.exists(cfg['resume_model'])
                self.epoch = load_checkpoint(model=self.model,
                                             optimizer=self.optimizer,
                                             filename=cfg['resume_model'],
                                             map_location=self.device,
                                             logger=self.logger)
                self.lr_scheduler.last_epoch = self.epoch - 1

        self.gpu_ids = list(map(int, cfg['gpu_ids'].split(',')))
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.kd_type = kd_type

    def update_lr_scheduler(self, epoch):
        if self.warmup_lr_scheduler is not None and epoch < 5:
            self.warmup_lr_scheduler.step()
        else:
            self.lr_scheduler.step()

    def save_model(self):
        if (self.epoch % self.cfg['save_frequency']) == 0:
            os.makedirs(os.path.join(self.cfg['log_dir'], "checkpoints"), exist_ok=True)
            ckpt_name = os.path.join(os.path.join(self.cfg['log_dir'], "checkpoints"), 'checkpoint_epoch_%d' % self.epoch)
            save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)
            #self.inference()
            #self.evaluate()

    def train(self):
        best_ap = {"epoch":0, "ap":0.0}
        # ap_all = self.eval_one_epoch()
        start_epoch = self.epoch

        # progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True,
        #                          desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.epoch += 1
            self.logger.info('------ EPOCH %03d ------' % (self.epoch))

            if self.model_type == 'centernet3d':
                self.train_one_epoch()

            elif self.model_type == 'distill':
                self.train_one_epoch_distill()
                # update learning rate
            self.update_lr_scheduler(epoch)
            self.save_model()

            # progress_bar.update()
            if (self.epoch % self.cfg['save_frequency']) == 0 :
                self.logger.info('------ EVAL EPOCH %03d ------' % (self.epoch))
                ap_all = self.eval_one_epoch()
                # for key in ap_all:
                #     self.logger.info('{}: {}'.format(key, ap_all[key]))
        


    def train_one_epoch(self):
        self.model.train()
        self.stats = {}  # reset stats dict
        self.stats['train'] = {}  # reset stats dict

        loss_stats = ['seg', 'offset2d', 'size2d', 'offset3d', 'depth', 'size3d', 'heading']
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in loss_stats}
        num_iters = len(self.train_loader)
        # bar = Bar('{}/{}'.format("3D", os.path.join(self.cfg['log_dir'], "checkpoints")), max=num_iters)
        end = time.time()

        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            # inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            # train one batch
            self.optimizer.zero_grad()
            # import pdb; pdb.set_trace()
            _, _, outputs = self.model(inputs)

            rgb_loss, rgb_stats_batch = compute_centernet3d_loss(outputs, targets)
            # depth_loss, depth_stats_batch = compute_depth_centernet3d_loss(depth_outputs, targets)
            total_loss = rgb_loss
            total_loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            # Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
            #     self.epoch, batch_idx, num_iters, phase="train",
            #     total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    rgb_stats_batch[l], inputs['rgb'].shape[0])
                # Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            
            if batch_idx % self.cfg['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (batch_idx, num_iters)
                for l in loss_stats:
                    log_str += ' | %s: %.4f' % (l, avg_loss_stats[l].avg)
                self.logger.info(log_str)

        #     bar.next()
        # bar.finish()

    def train_one_epoch_distill(self):
        self.model.train()
        self.stats = {}  # reset stats dict
        self.stats['train'] = {}  # reset stats dict

        loss_stats = ['rgb_loss']
        if 'head_kd' in self.kd_type:
            loss_stats.append('head_loss')
        if 'l1_kd' in self.kd_type:
            loss_stats.append('backbone_loss_l1')
        if 'affinity_kd' in self.kd_type:
            loss_stats.append('backbone_loss_affinity')
        if 'cross_kd' in self.kd_type:
            loss_stats.append('cross_head_loss')
        if 'bkl_kd' in self.kd_type:
            loss_stats.append('bkl_kd')
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = collections.defaultdict(float)
        avg_loss_stats = {l: AverageMeter() for l in loss_stats}
        num_iters = len(self.train_loader)
        # bar = Bar('{}/{}'.format("3D", os.path.join(self.cfg['log_dir'], "checkpoints")), max=num_iters)
        end = time.time()

        disp_dict = collections.defaultdict(float)
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            # start = time.time()
            # import matplotlib.pyplot as plt; plt.imshow(inputs['rgb'][0].permute(1, 2, 0).cpu().numpy()); plt.savefig("rgb.png"); plt.close()
            for key in inputs.keys():
                inputs[key] = inputs[key].to(self.device)
            # inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            # train one batch
            self.optimizer.zero_grad()
            rgb_loss, distill_loss, rgb_loss_dict = self.model(inputs, targets)
            for key in rgb_loss_dict:
                disp_dict[key] += rgb_loss_dict[key]
            # rgb_loss = rgb_loss.mean()
            total_loss = rgb_loss
            # print(total_loss, rgb_loss_dict)
            if 'head_kd' in self.kd_type:
                head_loss = distill_loss['head_loss'].mean()
                disp_dict['head_loss'] += head_loss.item()
                total_loss += head_loss
            if 'l1_kd' in self.kd_type:
                backbone_loss_l1 = distill_loss['backbone_loss_l1'].mean()
                total_loss += backbone_loss_l1 * 10
                disp_dict['backbone_loss_l1'] += backbone_loss_l1.item()
            if 'affinity_kd' in self.kd_type:
                backbone_loss_affinity = distill_loss['backbone_loss_affinity'].mean()
                total_loss += backbone_loss_affinity
                disp_dict['backbone_loss_affinity'] += backbone_loss_affinity.item()
            if 'cross_kd' in self.kd_type:
                cross_head_loss = distill_loss['cross_head_loss'].mean()
                total_loss += cross_head_loss
                disp_dict['cross_head_loss'] += cross_head_loss.item()
            if 'bkl_kd' in self.kd_type:
                bkl_kd_loss = distill_loss['bkl_kd'].mean()
                total_loss += bkl_kd_loss
                disp_dict['bkl_kd'] += bkl_kd_loss.item()
            # print("".join([key + ": " + str(val.item()) + "\n" for key, val in distill_loss.items()]), "total_loss: ", total_loss.item(), "\n", "".join([key + ": " + str(val) + "\n" for key, val in rgb_loss_dict.items()]))
            # print(self.epoch, "%d total_loss: %.4f" % (batch_idx, total_loss.item()))
            disp_dict['total_loss'] += total_loss.item()
            total_loss.backward()
            self.optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx+1) % self.cfg['disp_frequency'] == 0:
                log_str = 'BATCH[%04d/%04d]' % (batch_idx, num_iters)
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] /= self.cfg['disp_frequency']
                    log_str += ' | %s: %.4f' % (key, disp_dict[key])
                    disp_dict[key] = 0
                self.logger.info(log_str)
        #     bar.next()
        # bar.finish()

    def eval_one_epoch(self):
        # torch.set_grad_enabled(False)
        self.model.eval()
        results = {}
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
        with torch.no_grad():
            for batch_idx, (inputs, _, info) in enumerate(self.test_loader):
                # load evaluation data and move data to current device.
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(self.device)
                
                _,  _, outputs = self.model(inputs)
                dets = extract_dets_from_outputs(outputs=outputs, K=50)
                dets = dets.detach().cpu().numpy()

                # get corresponding calibs & transform tensor to numpy
                # calibs = [self.dataloader.dataset.get_calib(index)  for index in info['img_id']]
                calibs = [self.test_loader.dataset.get_calib(index)  for index in info['img_id']]
                info = {key: val.detach().cpu().numpy() for key, val in info.items()}
                cls_mean_size = self.test_loader.dataset.cls_mean_size
                dets = decode_detections(dets=dets,
                                        info=info,
                                        calibs=calibs,
                                        cls_mean_size=cls_mean_size,
                                        threshold=self.cfg.get('threshold', 0.2))
                results.update(dets)
                progress_bar.update()

            progress_bar.close()
            
        out_dir = os.path.join(self.cfg.get('log_dir', 'outputs'))
        self.save_results(results, out_dir)
        return self.test_loader.dataset.eval(results_dir=os.path.join(out_dir, 'data'), logger=self.logger)
    
    def save_results(self, results, output_dir='./rgb_outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if 1:
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.test_loader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.test_loader.dataset.get_sensor_modality(img_id),
                                           self.test_loader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()




