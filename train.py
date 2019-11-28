import argparse
import os
import random
import time
import datetime
import math
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from net import IFENet
from dataloader import THOLoader
from utils import AverageMeter

parser = argparse.ArgumentParser(description='Uisee AI Semi-final training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=10, type=int, metavar='N',
                    help='batch size of training')
parser.add_argument('--speed_weight', default=1, type=float,
                    help='speed weight')
parser.add_argument('--angle-weight', default=1, type=float,
                    help='angle weight')
parser.add_argument('--id', default="test", type=str)
parser.add_argument('--train-dir', default="./dataset/imgs",
                    type=str, metavar='PATH',
                    help='training dataset')
parser.add_argument('--eval-dir', default="./dataset/imgs",
                    type=str, metavar='PATH',
                    help='evaluation dataset')
parser.add_argument('--label-path', default="./dataset/labels.txt",
                    type=str, metavar='PATH',
                    help='evaluation dataset')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate-log', default="",
                    type=str, metavar='PATH',
                    help='path to log evaluation results (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


def output_log(output_str, logger=None):
    """
    standard output and logging
    """
    print("[{}]: {}".format(datetime.datetime.now(), output_str))
    if logger is not None:
        logger.critical("[{}]: {}".format(datetime.datetime.now(), output_str))


def log_args(logger):
    '''
    log args
    '''
    attrs = [(p, getattr(args, p)) for p in dir(args) if not p.startswith('_')]
    for key, value in attrs:
        output_log("{}: {}".format(key, value), logger=logger)


def main():
    global args
    args = parser.parse_args()
    log_dir = os.path.join("./", "logs", args.id)
    run_dir = os.path.join("./", "runs", args.id)
    save_weight_dir = os.path.join("./save_models", args.id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_weight_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, "training.log"),
                        level=logging.ERROR)
    writer = SummaryWriter(log_dir=run_dir)
    log_args(logging)

    if args.gpu is not None:
        output_log('You have chosen a specific GPU. This will completely '
                   'disable data parallelism.', logger=logging)


    model = IFENet()
    # criterion = EgoLoss()
    criterion = nn.MSELoss()

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # TODO check other papers optimizers
    optimizer = optim.Adam(
        model.parameters(), args.lr, betas=(0.7, 0.85))
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.join(save_weight_dir, args.resume)
        if os.path.isfile(args.resume):
            output_log("=> loading checkpoint '{}'".format(args.resume),
                       logging)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            output_log("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.resume, checkpoint['epoch']), logging)
        else:
            output_log("=> no checkpoint found at '{}'".format(args.resume),
                       logging)

    cudnn.benchmark = True

    loader = THOLoader(args.train_dir,args.eval_dir,args.label_path,batch_size=args.batch_size,num_workers=args.workers)

    train_loader = loader.loaders["train"]
    eval_loader = loader.loaders["eval"]

    if args.evaluate:
        args.id = args.id+"_test"
        if not os.path.isfile(args.resume):
            output_log("=> no checkpoint found at '{}'"
                       .format(args.resume), logging)
            return
        if args.evaluate_log == "":
            output_log("=> please set evaluate log path with --evaluate-log <log-path>")

        # TODO add test func
        evaluate(eval_loader, model, criterion, 0, writer)
        return

    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step()
        angle_losses, speed_losses, losses = \
            train(train_loader, model, criterion, optimizer, epoch, writer)

        # remember best prec@1 and save checkpoint
        if (epoch+1)%5==0:
            torch.save(model.state_dict(),args.id+"epoch_"+str(epoch+1)+".pth.tar")


def train(loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    angle_losses = AverageMeter()
    speed_losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    step = epoch * len(loader)
    for i, (img, speed, target_speed, target_angle) in enumerate(loader):
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        img = img.cuda()
        speed = speed.cuda()
        target_speed = target_speed.cuda()
        target_angle = target_angle.cuda()

        pred_speed, pred_angle = model(img, speed)

        angle_loss = criterion(pred_angle, target_angle)
        speed_loss = criterion(pred_speed, speed)

        loss = args.angle_weight * angle_loss + args.speed_weight * speed_loss

        losses.update(loss.item(), args.batch_size)
        angle_losses.update(angle_loss.item(), args.batch_size)
        speed_losses.update(speed_loss.item(), args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(loader):
            writer.add_scalar('train/branch_loss', angle_losses.val, step+i)
            writer.add_scalar('train/speed_loss', speed_losses.val, step+i)
            writer.add_scalar('train/loss', losses.val, step+i)
            output_log(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Angle loss {angle_loss.val:.3f} ({angle_loss.avg:.3f})\t'
                'Speed loss {speed_loss.val:.3f} ({speed_loss.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, angle_loss=angle_losses,
                    speed_loss=speed_losses, loss=losses), logging)

    return angle_losses.avg, speed_losses.avg, losses.avg


def evaluate(loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    step = epoch * len(loader)
    with torch.no_grad():
        end = time.time()
        for i, (img, speed, target, mask) in enumerate(loader):
            img = img.cuda(args.gpu, non_blocking=True)
            speed = speed.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            mask = mask.cuda(args.gpu, non_blocking=True)

            branches_out, pred_speed = model(img, speed)

            mask_out = branches_out * mask
            branch_loss = criterion(mask_out, target) * 4
            speed_loss = criterion(pred_speed, speed)

            loss = args.branch_weight * branch_loss + \
                args.speed_weight * speed_loss

            # measure accuracy and record loss
            losses.update(loss.item(), args.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(loader):
                writer.add_scalar('eval/loss', losses.val, step+i)
                output_log(
                  'Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      i, len(loader), batch_time=batch_time,
                      loss=losses), logging)
    return losses.avg


if __name__ == '__main__':
    main()