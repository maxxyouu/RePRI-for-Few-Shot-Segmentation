"""
This is a standlone script mainly to run without using the shell script -- easier to debug
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
# from visdom_logger import VisdomLogger
from collections import defaultdict
from dataset.dataset import get_val_loader
from util import AverageMeter, batch_intersectionAndUnionGPU, get_model_dir, main_process
from util import find_free_port, setup, cleanup, to_one_hot, intersectionAndUnionGPU
from classifier import Classifier
from model.pspnet import get_model
import torch.distributed as dist
from tqdm import tqdm
from util import load_cfg_from_cfg_file, merge_cfg_from_list
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
from visu import make_episode_visualization
from typing import Tuple
from fda_utils import *

def parse_args(data) -> None:
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config', default='config_files/{}.yaml'.format(data), type=str, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg

def vanilla_parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()
    return args

def main_worker(args: argparse.Namespace) -> None:

    # print(f"==> Running DDP checkpoint example on rank {rank}.")
    # setup(args, rank, world_size)

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    # ========== Model  ==========
    # NOTE: here load the model that is pretrained on imagenet if checkpoint is used
    model = get_model(args).to(args.device)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = DDP(model, device_ids=[rank])

    root = get_model_dir(args)

    # NOTE: differ from above, here load the checkpoint pretrained on pascal/coco splits.
    if args.ckpt_used != 'None':
        filepath = os.path.join(root, f'{args.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        print("=> loading weight '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded weight '{}'".format(filepath))
    else:
        print("=> Not loading anything in additional to the imagenet pretrained (if checkpoint is used)")

    # ========== Data  ==========
    episodic_val_loader, _ = get_val_loader(args)

    # ========== Test  ==========
    val_Iou, val_loss = episodic_validate(args=args,
                                            val_loader=episodic_val_loader,
                                            model=model,
                                            use_callback=(args.visdom_port != -1),
                                            suffix=f'test')
    print('validation iou {} with validation loss {}'.format(val_Iou, val_loss))

    # if args.distributed:
    #     dist.all_reduce(val_Iou), dist.all_reduce(val_loss)
    #     val_Iou /= world_size
    #     val_loss /= world_size

    # cleanup()


def episodic_validate(args: argparse.Namespace,
                        val_loader: torch.utils.data.DataLoader,
                        model,
                        use_callback: bool,
                        suffix: str = 'test') -> Tuple[torch.tensor, torch.tensor]:

    print('==> Start testing')

    model.eval()
    nb_episodes = int(args.test_num / args.batch_size_val)

    # ========== Metrics initialization  ==========

    H, W = args.image_size, args.image_size
    # c = model.module.bottleneck_dim
    # h = model.module.feature_res[0]
    # w = model.module.feature_res[1]
    c = model.fea_dim
    h = model.feature_res[0]
    w = model.feature_res[1]

    runtimes = torch.zeros(args.n_runs)
    deltas_init = torch.zeros((args.n_runs, nb_episodes, args.batch_size_val))
    deltas_final = torch.zeros((args.n_runs, nb_episodes, args.batch_size_val))
    val_IoUs = np.zeros(args.n_runs)
    val_losses = np.zeros(args.n_runs)

    # ========== Perform the runs  ==========
    for run in tqdm(range(args.n_runs)):

        # =============== Initialize the metric dictionaries ===============

        loss_meter = AverageMeter()
        iter_num = 0
        cls_intersection = defaultdict(int)  # Default value is 0
        cls_union = defaultdict(int)
        IoU = defaultdict(int)

        mean_img = torch.zeros(1, 1)
        IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        IMG_MEAN = torch.reshape(torch.from_numpy(IMG_MEAN), (1,3,1,1)  )

        # =============== episode = group of tasks ===============
        runtime = 0
        for e in tqdm(range(nb_episodes)):
            t0 = time.time()
            features_s = torch.zeros(args.batch_size_val, args.shot, c, h, w).to(args.device)
            features_q = torch.zeros(args.batch_size_val, 1, c, h, w).to(args.device)
            gt_s = 255 * torch.ones(args.batch_size_val, args.shot, args.image_size,
                                    args.image_size).long().to(args.device)
            gt_q = 255 * torch.ones(args.batch_size_val, 1, args.image_size,
                                    args.image_size).long().to(args.device)
            n_shots = torch.zeros(args.batch_size_val).to(args.device)
            classes = []  # All classes considered in the tasks

            # =========== Generate tasks and extract features for each task ===============
            with torch.no_grad():
                for i in range(args.batch_size_val):
                    try:
                        qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = next(iter_loader) # iter_loader.next()
                    except:
                        iter_loader = iter(val_loader)
                        qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = next(iter_loader) # iter_loader.next()

                    iter_num += 1

                    if args.fda:
                        #---------------------------FDA START----------------------------------------#
                        spprt_imgs_cpy = spprt_imgs.clone() # ([1 B, args.shot, 3 C, 417 H, 417 W])
                        qry_img_cpy = qry_img.clone() # [1, 3, 417, 417]
                        
                        # 1. support in query style, one by one
                        spprts_in_trg = []
                        for k in range(args.shot):
                            spprt = spprt_imgs_cpy[:, k, :] # ([1 B, 3 C, 417 H, 417 W])
                            # convert to np then convert back to tensor type.
                            temp = FDA_source_to_target_np(spprt.squeeze(0).detach().cpu().numpy(), qry_img_cpy.squeeze(0).detach().cpu().numpy(), L=args.LB)
                            temp = torch.from_numpy(temp).unsqueeze(0).to(spprt.device)
                            spprts_in_trg.append(temp)
                            # direct convertion using pytorch, but not working with the new version.
                            # spprts_in_trg.append(FDA_source_to_target(spprt, qry_img_cpy, args.LB))
                        spprts_in_trg = torch.stack(spprts_in_trg, dim=1).float()

                        if mean_img.shape[-1] < 2:
                            B, C, H, W = qry_img_cpy.shape
                            mean_img = IMG_MEAN.repeat(B,1,H,W)
                            
                        # # 2. subtract mean
                        spprt_imgs = spprts_in_trg.clone() - mean_img                                 # src, src_lbl
                        qry_img = qry_img_cpy.clone() - mean_img
                        # spprt_imgs = spprts_in_trg.clone()                          # src, src_lbl
                        # qry_img = qry_img_cpy.clone()
                        #---------------------------FDA END----------------------------------------#

                    q_label = q_label.to(args.device, non_blocking=True)
                    spprt_imgs = spprt_imgs.to(args.device, non_blocking=True) # shape [1, 2(number of shots), 3, 417, 417]
                    s_label = s_label.to(args.device, non_blocking=True)
                    qry_img = qry_img.to(args.device, non_blocking=True) # shape [1, 3, 417, 417]

                    f_s = model.extract_features(spprt_imgs.squeeze(0))
                    # f_s = model.layer4[2].nonclipped_feature #NOTE: if want unclipped feature at the last layer? => uncomment this.
                    f_q = model.extract_features(qry_img)
                    # f_q = model.layer4[2].nonclipped_feature #NOTE: if want unclipped feature at the last layer? => uncomment this.

                    shot = f_s.size(0)
                    n_shots[i] = shot
                    features_s[i, :shot] = f_s.detach()
                    features_q[i] = f_q.detach()
                    gt_s[i, :shot] = s_label
                    gt_q[i, 0] = q_label
                    classes.append([class_.item() for class_ in subcls])

            # =========== Normalize features along channel dimension ===============
            if args.norm_feat:
                features_s = F.normalize(features_s, dim=2)
                features_q = F.normalize(features_q, dim=2)

            # =========== Create a callback is args.visdom_port != -1 ===============
            callback = None #VisdomLogger(port=args.visdom_port) if use_callback else None

            # ===========  Initialize the classifier + prototypes + F/B parameter Π ===============
            classifier = Classifier(args)
            classifier.init_prototypes(features_s, features_q, gt_s, gt_q, classes, callback)
            batch_deltas = classifier.compute_FB_param(features_q=features_q, gt_q=gt_q)
            deltas_init[run, e, :] = batch_deltas.cpu()

            # =========== Perform RePRI inference ===============
            batch_deltas = classifier.RePRI(features_s, features_q, gt_s, gt_q, classes, n_shots, callback)
            deltas_final[run, e, :] = batch_deltas
            t1 = time.time()
            runtime += t1 - t0
            if classifier.swa:
                logits = classifier.get_swa_logits(features_q)
            else:
                logits = classifier.get_logits(features_q)  # [n_tasks, shot, h, w]
            logits = F.interpolate(logits,
                                    size=(H, W),
                                    mode='bilinear',
                                    align_corners=True)
            if classifier.swa:
                probas = classifier.get_swa_probas(logits).detach()
            else:
                probas = classifier.get_probas(logits).detach()
            intersection, union, _ = batch_intersectionAndUnionGPU(probas, gt_q, 2)  # [n_tasks, shot, num_class]
            intersection, union = intersection.cpu(), union.cpu()

            # ================== Log metrics ==================
            one_hot_gt = to_one_hot(gt_q, 2)
            valid_pixels = gt_q != 255
            loss = classifier.get_ce(probas, valid_pixels, one_hot_gt, reduction='mean')
            loss_meter.update(loss.item())
            for i, task_classes in enumerate(classes):
                for j, class_ in enumerate(task_classes):
                    cls_intersection[class_] += intersection[i, 0, j + 1]  # Do not count background
                    cls_union[class_] += union[i, 0, j + 1]

            for class_ in cls_union:
                IoU[class_] = cls_intersection[class_] / (cls_union[class_] + 1e-10)

            if (iter_num % 1 == 0):
                mIoU = np.mean([IoU[i] for i in IoU])
                print('Test: [{}/{}] '
                        'mIoU {:.4f} '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(iter_num,
                                                                                        args.test_num,
                                                                                        mIoU,
                                                                                        loss_meter=loss_meter,
                                                                                        ))

            # ================== Visualization ==================
            if args.visu:
                root = os.path.join('plots', 'episodes')
                os.makedirs(root, exist_ok=True)
                save_path = os.path.join(root, f'run_{run}_episode_{e}.pdf')
                make_episode_visualization(img_s=spprt_imgs[0].cpu().numpy(),
                                            img_q=qry_img[0].cpu().numpy(),
                                            gt_s=s_label[0].cpu().numpy(),
                                            gt_q=q_label[0].cpu().numpy(),
                                            preds=probas[-1].cpu().numpy(),
                                            save_path=save_path)


        runtimes[run] = runtime
        mIoU = np.mean(list(IoU.values()))
        print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
        for class_ in cls_union:
            print("Class {} : {:.4f}".format(class_, IoU[class_]))

        val_IoUs[run] = mIoU
        val_losses[run] = loss_meter.avg

    # ================== Save metrics ==================
    if args.save_oracle:
        root = os.path.join('plots', 'oracle')
        os.makedirs(root, exist_ok=True)
        np.save(os.path.join(root, 'delta_init.npy'), deltas_init.numpy())
        np.save(os.path.join(root, 'delta_final.npy'), deltas_final.numpy())

    print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs.mean()))
    print('Average runtime / run --- {:.4f}.'.format(runtimes.mean()))

    return val_IoUs.mean(), val_losses.mean()


def standard_validate(args: argparse.Namespace,
                        val_loader: torch.utils.data.DataLoader,
                        model,
                        use_callback: bool,
                        suffix: str = 'test') -> Tuple[torch.tensor, torch.tensor]:

    print('==> Standard validation')
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    iterable_val_loader = iter(val_loader)

    bar = tqdm(range(len(iterable_val_loader)))

    loss = 0.
    intersections = torch.zeros(args.num_classes_tr).to(args.device)
    unions = torch.zeros(args.num_classes_tr).to(args.device)

    with torch.no_grad():
        for i in bar:
            images, gt = iterable_val_loader.next()
            images = images.to(args.device, non_blocking=True)
            gt = gt.to(args.device, non_blocking=True)
            logits = model(images).detach()
            loss += loss_fn(logits, gt)
            intersection, union, _ = intersectionAndUnionGPU(logits.argmax(1),
                                                                gt,
                                                                args.num_classes_tr,
                                                                255)
            intersections += intersection
            unions += union
        loss /= len(val_loader.dataset)

    if args.distributed:
        dist.all_reduce(loss)
        dist.all_reduce(intersections)
        dist.all_reduce(unions)

    mIoU = (intersections / (unions + 1e-10)).mean()
    # loss /= dist.get_world_size()
    return mIoU, loss


if __name__ == "__main__":
    args = parse_args('pascal')
    # args = vanilla_parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    # args.debug = False
    if args.debug:
        args.test_num = 500
        args.n_runs = 2

    # non distributed version.
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'  
    torch.manual_seed(args.manual_seed)
    main_worker(args)

    # mp.spawn(main_worker,
    #             args=(world_size, args),
    #             nprocs=world_size,
    #             join=True)