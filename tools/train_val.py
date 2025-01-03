import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime
import os
import socket

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')

args = parser.parse_args()



def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))

    cfg['trainer']['log_dir'] = os.path.join(os.path.dirname(cfg['trainer']['log_dir']), \
                                            datetime.datetime.now().strftime('%Y%m%d%H%M_') + os.path.basename(cfg['trainer']['log_dir']) + '_' + socket.gethostname()[:10] + '_' + '-'.join(cfg['model']['kd_type']))

    if args.evaluate_only:
        log_path = os.path.join(*cfg['tester']['checkpoint'].split('/')[:-2])
    else:
        log_path = cfg['trainer']['log_dir']
    os.makedirs(log_path, exist_ok=True)

    log_file = '/train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = create_logger(log_path, log_file)


    # build dataloader
    train_loader, test_loader  = build_dataloader(cfg['dataset'])


    if args.evaluate_only:
        model = build_model(cfg['model'], 'testing')

        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger)
        tester.test()
        return

    # build model&&build optimizer
    if cfg['model']['type']=='centernet3d' or cfg['model']['type']=='distill':
        model = build_model(cfg['model'],'training')
        optimizer = build_optimizer(cfg['optimizer'], model)
        lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    else:
        raise NotImplementedError("%s model is not supported" % cfg['model']['type'])


    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d'  % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f'  % (cfg['optimizer']['lr']))
    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      model_type=cfg['model']['type'],
                      root_path=ROOT_DIR,
                      kd_type=cfg['model']['kd_type'])
    trainer.train()


if __name__ == '__main__':
    main()