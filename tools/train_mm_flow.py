import os
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation, get_train_augmentation_flow
from semseg.losses import get_loss, compute_eraft_flow_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from semseg.metrics import Metrics
from val_mm_flow import evaluate
import numpy as np
import math
# import Image
from PIL import Image
from torchviz import make_dot

def main(cfg, scene, classes, gpu, save_dir, duration):
    start = time.time()
    best_epe = 1e8
    best_epoch = 0
    num_workers = 4
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL']['RESUME']
    gpus = int(os.environ['WORLD_SIZE'])

    # traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    # traintransform = get_train_augmentation_flow(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    traintransform = None
    # valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    valtransform = None
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'].replace("${DURATION}", str(duration)), 'train', classes, traintransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    # 计算补齐后的目标长度
    if len(trainset) % train_cfg['BATCH_SIZE'] != 0:
        num_batches = math.ceil(len(trainset) / train_cfg['BATCH_SIZE'])
        target_length = num_batches * train_cfg['BATCH_SIZE']
        trainset = ExtendedDSEC_FLOW(trainset, target_length)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'].replace("${DURATION}", str(duration)), 'val', classes, valtransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    # valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'].replace("${DURATION}", str(duration)), 'train', classes, valtransform, dataset_cfg['MODALS'], duration=duration, flow_net_flag=model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'])
    class_names = trainset.SEGMENTATION_CONFIGS[classes]["CLASSES"]

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'], model_cfg['BACKBONE_FLAG'], model_cfg['FLOW_NET_FLAG'], dataset_type=dataset_cfg['TYPE'], anytime_flag=False).flow_net
    for name, param in model.named_parameters():
        print(name)
    print("model: ", model_cfg['FLOW_NET_FLAG'])
    print(model_cfg['RESUME_FLOWNET'])
    print(os.path.isfile(model_cfg['RESUME_FLOWNET']))
    if model_cfg['FLOW_NET_FLAG'] and os.path.isfile(model_cfg['RESUME_FLOWNET']):
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            print('Loading flownet model...')
        flow_net_type = model_cfg['FLOW_NET']
        resume_flownet_path = model_cfg['RESUME_FLOWNET']
        print("flow_net_type: ", flow_net_type)
        print("dataset_cfg['TYPE']: ", dataset_cfg['TYPE'])

        if flow_net_type == 'eraft' or flow_net_type == 'raft_small':
            ## for eraft
            # if dataset_cfg['TYPE'] == 'dsec':
            if dataset_cfg['TYPE'] == 'dsec_':
                flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))['model']
                # flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'), weights_only=True)['model']
            elif dataset_cfg['TYPE'] == 'dsec':
            # elif dataset_cfg['TYPE'] == 'sdsec':
                # flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))
                flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))['model']   # for dsec.tar
                if flow_net_type == 'raft_small':
                    print("Loading raft_small")
                    # 去掉module
                    flownet_checkpoint = {k.replace('module.', ''): v for k, v in flownet_checkpoint.items()}
                    # print("flownet_checkpoint keys: ", flownet_checkpoint.keys())
                # # 筛选出以 'flownet' 为前缀的键
                # flownet_checkpoint = {
                #     key: value for key, value in flownet_checkpoint.items() if key.startswith('flow_net')
                # }
                # # 给所有key去掉前缀 'flow_net.'
                # flownet_checkpoint = {k.replace('flow_net.', ''): v for k, v in flownet_checkpoint.items()}
            if 'fnet.conv1.weight' in flownet_checkpoint:
                # delete weights of the first layer
                flownet_checkpoint.pop('fnet.conv1.weight')
                flownet_checkpoint.pop('fnet.conv1.bias')
            if 'cnet.conv1.weight' in flownet_checkpoint:
                # delete weights of the second layer
                flownet_checkpoint.pop('cnet.conv1.weight')
                flownet_checkpoint.pop('cnet.conv1.bias')
        elif flow_net_type == 'bflow':
            # for bflow
            flownet_checkpoint = torch.load(resume_flownet_path, map_location=torch.device('cpu'))['state_dict']
            # 过滤掉 'flow_network.' 前缀
            # flownet_checkpoint = {k.replace('flow_network.', ''): v for k, v in flownet_checkpoint.items()}
            # 过滤掉 'net.' 前缀
            flownet_checkpoint = {k.replace('net.', ''): v for k, v in flownet_checkpoint.items()}
            if 'fnet_ev.conv1.weight' in flownet_checkpoint:
                # delete weights of the first layer
                flownet_checkpoint.pop('fnet_ev.conv1.weight')
                flownet_checkpoint.pop('fnet_ev.conv1.bias')
            if 'update_block.encoder.convc1.weight' in flownet_checkpoint:
                # delete weights of the first layer
                flownet_checkpoint.pop('update_block.encoder.convc1.weight')
                flownet_checkpoint.pop('update_block.encoder.convc1.bias') 
            # if 'cnet.conv1.weight' in flownet_checkpoint:
            #     # delete weights of the second layer
            #     flownet_checkpoint.pop('cnet.conv1.weight')
            #     flownet_checkpoint.pop('cnet.conv1.bias')

        flownet_msg = model.load_state_dict(flownet_checkpoint, strict=False)
        print("flownet_checkpoint msg: ", flownet_msg)
        logger.info(flownet_msg)
        # exit(0)

    model = model.to(device)

    start_epoch = 0
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None

    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    # trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=0, drop_last=True, pin_memory=True, sampler=sampler, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=True, sampler=sampler_val, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    # valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=0, pin_memory=True, sampler=sampler_val, worker_init_fn=lambda worker_id: np.random.seed(3407 + worker_id))
    iters_per_epoch = len(trainloader)
    print("iters_per_epoch: ", iters_per_epoch)
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs+1)*iters_per_epoch), sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        # logger.info('================== model complexity =====================')
        # cal_flops(model, dataset_cfg['MODALS'], logger)
        logger.info('================== model structure =====================')
        # logger.info(flownet_msg)
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)

        # exit(0)

    for epoch in range(start_epoch, epochs):
        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        for iter, (seq_names, seq_index, sample) in pbar:
            optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device) for x in sample]
            
            with autocast('cuda', enabled=train_cfg['AMP']):
                bin = 5
                ev_t0_t1 = torch.cat([sample[0][:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                ev_before = torch.cat([sample[1][:, bin*i:bin*(i+1)].mean(1).unsqueeze(1) for i in range(20//bin)], dim=1)
                ev_t0_t1 = torch.nn.functional.interpolate(ev_t0_t1, scale_factor=0.5, mode='bilinear', align_corners=False)
                ev_before = torch.nn.functional.interpolate(ev_before, scale_factor=0.5, mode='bilinear', align_corners=False)
                predict_flows = model(ev_before,ev_t0_t1)
                flow_gt = sample[-1]
                flow_gt = torch.nn.functional.interpolate(flow_gt, scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
                loss = compute_eraft_flow_loss(predict_flows, flow_gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8 # minimum of lr
            train_loss += loss.item()
            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        train_loss /= iter+1
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
        # torch.cuda.empty_cache()

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                epe, n1pe, n2pe, n3pe = evaluate(model, valloader, device)
                writer.add_scalar('val/EPE', epe, epoch)

                if epe < best_epe:
                    prev_best_ckp = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_epe}_checkpoint.pth"
                    prev_best = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_epe}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_epe = epe
                    best_epoch = epoch+1
                    cur_best_ckp = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_epe}_checkpoint.pth"
                    cur_best = save_dir / f"model_{scene}_{classes}_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_epe}.pth"
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)
                    # --- 
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_epe': best_epe,
                                '1pe': n1pe,
                                '2pe': n2pe,
                                '3pe': n3pe,
                                }, cur_best_ckp)
                    logger.info(f"EPE: {epe: 2f}, 1PE: {n1pe: 2f}, 2PE: {n2pe: 2f}, 3PE: {n3pe: 2f}")
                    logger.info(f"Best model saved at epoch {best_epoch}")
                logger.info(f"Current epoch:{epoch} EPE: {epe} Best EPE: {best_epe}")

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best EPE', f"{best_epe:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deliver_rgbdel.yaml', help='Configuration file to use')
    parser.add_argument('--scene', type=str, default='night')
    parser.add_argument('--input_type', type=str, default='rgbe')
    parser.add_argument('--classes', type=int, default=11)
    parser.add_argument('--duration', type=int, default=50)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = get_logger(save_dir / f'{args.input_type}_{args.scene}_{args.classes}_{time_}_train.log')
    main(cfg, args.scene, args.classes, gpu, save_dir, args.duration)
    cleanup_ddp()