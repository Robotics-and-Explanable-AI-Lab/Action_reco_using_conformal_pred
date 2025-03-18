#!/usr/bin/env python3
import os
import time
import shutil
import csv
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool

from tensorboardX import SummaryWriter

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset, args.modality)

    full_arch_name = args.arch
    if args.shift:
        full_arch_name += f'_shift{args.shift_div}_{args.shift_place}'
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         f'e{args.epochs}'])
    if args.pretrain != 'imagenet':
        args.store_name += f'_{args.pretrain}'
    if args.lr_type != 'step':
        args.store_name += f'_{args.lr_type}'
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += f'_{args.suffix}'
    print('storing name: ' + args.store_name)

    check_rootfolders()
    
    # all the modalities use the RGB backbone of the TSM
    if args.modality == 'mono' or args.modality == 'combined':
        args.modality = 'RGB'

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_feature_dim=args.img_feature_dim,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    if args.tune_from:
        print(f"=> fine-tuning from '{args.tune_from}'")
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        
        if args.tune_from == 'pretrained_models/tsm_rgb_epic.ckpt':
            sd = {'.'.join(k.split('.')[1:]): v for k, v in list(sd.items())}
            sd = {''.join('module.'+k): v for k, v in list(sd.items())}
        
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print(f'#### Notice: keys that failed to load: {set_diff}')
        
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        if args.modality == 'Flow' and 'Flow' not in args.tune_from:
            sd = {k: v for k, v in sd.items() if 'conv1.weight' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ]), dense_sample=args.dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(f"group: {group['name']} has {len(group['params'])} params, lr_mult: {group['lr_mult']}, decay_mult: {group['decay_mult']}")

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    # Create directories for logging and also initialize CSV logging
    csv_filename = os.path.join(args.root_log, args.store_name, 'results.csv')
    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'Train Loss', 'Train Prec@1', 'Val Loss', 'Val Prec@1', 'Val Prec@5'])

    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    # Main training loop
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # Train for one epoch. These AverageMeter objects record training metrics.
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.train()
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            loss.backward()
            if args.clip_gradient is not None:
                clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()
            optimizer.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                output_line = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                               'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                               .format(epoch, i, len(train_loader),
                                       batch_time=batch_time, data_time=data_time,
                                       loss=losses, top1=top1, top5=top5,
                                       lr=optimizer.param_groups[-1]['lr'] * 0.1))
                print(output_line)
                log_training.write(output_line + '\n')
                log_training.flush()

        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

        # Evaluate on validation set at specified frequency
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            # Modify validate to return loss and Prec@5 as well
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, epoch, log_training, tf_writer)

            is_best = val_prec1 > best_prec1
            best_prec1 = max(val_prec1, best_prec1)
            tf_writer.add_scalar('acc/test_top1_best', best_prec1, epoch)
            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)
            log_training.write(output_best + '\n')
            log_training.flush()

            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)

            # Log metrics to CSV
            with open(csv_filename, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([epoch, losses.avg, top1.avg, val_loss, val_prec1, val_prec5])


def validate(val_loader, model, criterion, epoch, log=None, tf_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                output_line = ('Test: [{0}/{1}]\t'
                               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                               'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                               .format(i, len(val_loader), batch_time=batch_time,
                                       loss=losses, top1=top1, top5=top5))
                print(output_line)
                if log is not None:
                    log.write(output_line + '\n')
                    log.flush()

    output_line = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                   .format(top1=top1, top5=top5, loss=losses))
    print(output_line)
    if log is not None:
        log.write(output_line + '\n')
        log.flush()
    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
    # Return validation loss, top1 and top5 for logging
    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best):
    filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
