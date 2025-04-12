# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import copy
from typing import Dict, Union, Optional

from mmengine.config import Config, DictAction, ConfigDict
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine._strategy import BaseStrategy

from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.runner._flexible_runner import FlexibleRunner

ConfigType = Union[Dict, Config, ConfigDict]

import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def from_cfg(cfg: ConfigType, strategy: Optional[Union[BaseStrategy, Dict]] = None) -> 'FlexibleRunner':
    """Build a runner from config.

    Args:
        cfg (ConfigType): A config used for building runner. Keys of
            ``cfg`` can see :meth:`__init__`.

    Returns:
        Runner: A runner build from ``cfg``.
    """
    cfg = copy.deepcopy(cfg)
    runner = FlexibleRunner(
        model=cfg['model'],
        work_dir=cfg.get('work_dir', 'work_dirs'),
        experiment_name=cfg.get('experiment_name'),
        train_dataloader=cfg.get('train_dataloader'),
        optim_wrapper=cfg.get('optim_wrapper'),
        param_scheduler=cfg.get('param_scheduler'),
        train_cfg=cfg.get('train_cfg'),
        val_dataloader=cfg.get('val_dataloader'),
        val_evaluator=cfg.get('val_evaluator'),
        val_cfg=cfg.get('val_cfg'),
        test_dataloader=cfg.get('test_dataloader'),
        test_evaluator=cfg.get('test_evaluator'),
        test_cfg=cfg.get('test_cfg'),
        strategy=strategy,
        auto_scale_lr=cfg.get('auto_scale_lr'),
        default_hooks=cfg.get('default_hooks'),
        custom_hooks=cfg.get('custom_hooks'),
        data_preprocessor=cfg.get('data_preprocessor'),
        load_from=cfg.get('load_from'),
        resume=cfg.get('resume', False),
        launcher=cfg.get('launcher'),
        env_cfg=cfg.get('env_cfg'),  # type: ignore
        log_processor=cfg.get('log_processor'),
        log_level=cfg.get('log_level', 'INFO'),
        visualizer=cfg.get('visualizer'),
        default_scope=cfg.get('default_scope', 'mmengine'),
        randomness=cfg.get('randomness', dict(seed=None)),
        cfg=cfg,
    )

    return runner

def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    from functools import partial

    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    size_based_auto_wrap_policy = partial(size_based_auto_wrap_policy,
                                          min_num_params=1e7)
    strategy = dict(
        type='FSDPStrategy',
        model_wrapper=dict(auto_wrap_policy=size_based_auto_wrap_policy))
    runner = from_cfg(cfg, strategy)
    # # build the runner from config
    # if 'runner_type' not in cfg:
    # # build the default runner
    #     runner = FlexibleRunner.from_cfg(cfg)
    # else:
    # # build customized runner from the registry
    # # if 'runner_type' is set in the cfg
    #     runner = RUNNERS.build(cfg)

    print(runner)
    # start training
    runner.train()


if __name__ == '__main__':
    main()