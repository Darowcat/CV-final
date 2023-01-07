"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
# import wandb
from config import cfg
import dataset
import optimizer
import loss
# import resnet
import time
import torch.nn.functional as F
import numpy as np
import random
import model_insightface
from utils import AverageMeter, prep_experiment, evaluate_eval, fast_hist

### set logging
logging.getLogger().setLevel(logging.INFO)

# Argument Parser
parser = argparse.ArgumentParser(description='CV_final')
parser.add_argument('--lr', type=float, default=5e-4)

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_iter', type=int, default=40000)
parser.add_argument('--start_epoch', type=int, default=0)
# parser.add_argument('--rrotate', type=int,
#                     default=0, help='degree of random roate')
# parser.add_argument('--color_aug', type=float,
#                     default=0.0, help='level of color augmentation')
# parser.add_argument('--gblur', action='store_true', default=False,
#                     help='Use Guassian Blur Augmentation')
# parser.add_argument('--bblur', action='store_true', default=False,
#                     help='Use Bilateral Blur Augmentation')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=0.9,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=2,
                    help='Batch size for training per gpu')
parser.add_argument('--bs_mult_val', type=int, default=2,
                    help='Batch size for Validation per gpu') 
# parser.add_argument('--crop_size', type=int, default=720,
#                     help='training crop size')
# parser.add_argument('--pre_size', type=int, default=None,
#                     help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=True)
parser.add_argument('--eval_epoch', type=int, default=1,
                    help='eval interval')
parser.add_argument('--date', type=str, default='default',
                    help='experiment directory date name')
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt',
                    help='Save Checkpoint Point')
parser.add_argument('--tb_path', type=str, default='logs/tb',
                    help='Save Tensorboard Path')

## Arcface loss
parser.add_argument('--af_weight', type=float, default=1.0,
                    help='weight for Arcface loss')
## contrastoive loss
parser.add_argument('--cl_weight', type=float, default=1.0,
                    help='weight for contrastive loss')


## wandb for logs
# parser.add_argument('--wandb_name', type=str, default='cv_final',
#                     help='use wandb and wandb name')

args = parser.parse_args()

# Enable CUDNN Benchmarking optimization
#torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  #304
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

args.world_size = 1


## WORLD_SIZE =  number of machine (distibuted learning)
if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

## local_rank = current GPU ID
torch.cuda.set_device(args.local_rank) 
# torch.cuda.set_device('cuda:1')
print('My Rank:', args.local_rank)



def main():
    """
    Main Function
    """
    # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
    writer = prep_experiment(args, parser)
    # if args.wandb_name:
    #     if args.local_rank == 0:
    #         wandb.init(project='cv_final', name=args.wandb_name, config=args)

    train_loader, val_loader = dataset.get_dataset(args)
    
    net = model_insightface.MobileFaceNet(512).cuda()
    
    optim_net, scheduler_net = optimizer.get_optimizer(args, net)
    # for name, param in net.named_parameters():
    #     print(param)
    #     break

    epoch = 0
    i = 0
    best_rank1 = 0
    best_auc = 0
    best_epoch = 0

    # if args.snapshot:
    #     epoch, mean_iu = optimizer.load_weights(net, optim_net, scheduler_net,
    #                         args.snapshot, args.restore_optimizer)
    #     if args.restore_optimizer is True:
    #         iter_per_epoch = len(train_loader)
    #         epoch = epoch + 1
    #         i = iter_per_epoch * epoch
    #     else:
    #         epoch = 0

    if args.local_rank == 0: # major GPU
        msg_args = ''
        args_dict = vars(args)
        for k, v in args_dict.items():
            msg_args = msg_args + str(k) + ' : ' + str(v) + ', '
        logging.info(msg_args)

    while i < args.max_iter:
        # Update EPOCH CTR
        cfg.immutable(False)
        cfg.ITER = i
        cfg.immutable(True) # ensure at same memory address

        print("#### iteration", i)
        torch.cuda.empty_cache()
        
        
        i = train(train_loader, net, optim_net, epoch, writer, scheduler_net, args.max_iter)

        if (epoch+1) % args.eval_epoch == 0 or i >= args.max_iter:
            torch.cuda.empty_cache()
            if args.local_rank == 0:
                print("Saving pth file...")
                # TODO
                # save model
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optim_net.state_dict(),
                        'scheduler_state_dict': scheduler_net.state_dict(),
                        'best_rank1': best_rank1,
                        'best_auc': best_auc,
                        'best_epoch': best_epoch,
                    }, args.save_path)
                # TODO

            torch.cuda.empty_cache()
            
            print("Extra validating... This won't save pth file")
            rank1, auc = validate(val_loader, net, optim_net, scheduler_net, epoch, writer, i, save_pth=False)
            torch.cuda.empty_cache()
   
            if args.local_rank == 0:
                if rank1 >= best_rank1 and auc > best_auc:
                    best_auc = auc
                    best_rank1 = rank1
                    best_epoch = epoch
                
                msg = 'Best Epoch:{}, Best rank1:{:.5f}, Best auc:{:.5f}'.format(best_epoch, best_rank1, best_auc)
                msg_current = 'Current Epoch:{}, Current rank1:{:.5f}, Current auc:{:.5f}'.format(epoch, rank1, auc)
                # if args.wandb_name:
                #     wandb.log({
                #         'epoch': best_epoch,
                #         'cur_rank1': rank1,
                #         'cur_auc': auc,
                #         'best_auc': best_auc,
                #         'best_rank1': best_rank1
                #     })
                logging.info(msg)
                logging.info(msg_current)

       
        epoch += 1


def train(train_loader, net, optim_net, curr_epoch, writer, scheduler_net, max_iter):
    """
    Runs the training loop per epoch
    train_loader: Data loader for train
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return:
    """
    net.train()
    
    net_total_loss_meter = AverageMeter()
    Arcface_loss_meter = AverageMeter()
    contrastive_loss_meter = AverageMeter()
    rank1_meter = AverageMeter()
    AUC_meter = AverageMeter()
    time_meter = AverageMeter()
    
    curr_iter = curr_epoch * len(train_loader)
    
    for i, (input, gt) in enumerate(train_loader):
        B, C, H, W = input.shape
        # print(input.shape)
        # print(gt.shape)
        start_ts = time.time()
        input, gt = input.cuda(), gt.cuda()
        
        # train main branch
        optim_net.zero_grad()
        output = net(input)
        # print(output.shape)
        
        total_loss = 0
        # TODO
        # loss auc rank1...
        arcface_loss = model_insightface.Arcface()
        af_loss = arcface_loss(output, gt)
        # TODO
        
        # contrstive loss
        # mask = loss.generate_contrastive_mask(gt)
        contrastive_loss = loss.SupConLoss()
        cl_loss = contrastive_loss(output, gt)
        # print(cl_loss)
        
        Arcface_loss_meter.update(af_loss.item(), B)
        contrastive_loss_meter.update(cl_loss.item(), B)

        total_loss = total_loss + args.af_weight * af_loss + args.cl_weight * cl_loss 
        net_total_loss_meter.update(total_loss.item(), B) #devided by batch size
        
        total_loss.backward()
        optim_net.step()            
        time_meter.update(time.time() - start_ts)
            
        if args.local_rank == 0:
            if i % 30 == 29:

                msg = '[epoch {}], [iter {} / {} : {}], [net loss {:0.6f}], [cl loss {:0.6f}], [rank1 {:0.6f}], [AUC {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                curr_epoch, i + 1, len(train_loader), curr_iter, net_total_loss_meter.avg, contrastive_loss_meter.avg, rank1_meter.avg, AUC_meter.avg,
                optim_net.param_groups[-1]['lr'], time_meter.avg)
                
                logging.info(msg)
                # if args.wandb_name:
                #     wandb.log({
                #         'net loss':net_total_loss_meter.avg,
                #         'adain loss':adain_total_loss_meter.avg,
                #         'rc loss':fd_loss_meter.avg,
                #         # 'gram loss':rc_loss_meter.avg,
                #         'similarity loss':similarity_loss_meter.avg, 
                #         'sc loss':sc_loss_meter.avg,
                #     })
                # Log tensorboard metrics for each iteration of the training phase
                
                writer.add_scalar('loss/train_loss', (net_total_loss_meter.avg),
                                curr_iter)
                net_total_loss_meter.reset()
                time_meter.reset()
                if curr_iter >= max_iter:
                    break

        curr_iter += 1
        scheduler_net.step()

    return curr_iter


def validate(val_loader, net, optim, scheduler, curr_epoch, writer, curr_iter, save_pth=True):
    """
    Runs the validation loop after each training epoch
    val_loader: Data loader for validation
    dataset: dataset name (str)
    net: thet network
    optimizer: optimizer
    curr_epoch: current epoch
    writer: tensorboard writer
    return: val_avg for step function if required
    """

    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0
    error_acc = 0
    dump_images = []

    probs = []
    ground_truth = []

    for val_idx, data in enumerate(val_loader):
        inputs, gt_image = data

        inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

        with torch.no_grad():
            output = net(inputs)
        del inputs
        
        # TODO
        # rank1, AUC...
        feat1 = output[0]
        feat2 = output[1]
        similarity = loss.get_cosine_similarity(feat1, feat2)

        probs.append(similarity)
        if gt_cuda[0] == gt_cuda[1]:
            ground_truth.append(1)
        else:
            ground_truth.append(0)

        arcface_loss = model_insightface.Arcface()
        af_loss = arcface_loss(output, gt_cuda)
        contrastive_loss = loss.SupConLoss()
        cl_loss = contrastive_loss(output, gt_cuda)

        total_loss = args.af_weight * af_loss + args.cl_weight * cl_loss 
        # TODO 
        
        # write total loss in params 1.
        val_loss.update(total_loss.item(), inputs.shape[0])

        # Logging
        if val_idx % 20 == 0:
            if args.local_rank == 0:
                logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
        
    # TODO
    # total rank1, AUC...
    f = list(zip(probs, ground_truth))
    rank = [values2 for values1,values2 in sorted(f, key=lambda x:x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    posNum = 0
    negNum = 0
    for i in range(len(ground_truth)):
        if(ground_truth[i] == 1):
            posNum += 1
        else:
            negNum += 1
    auc = 0
    auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    # TODO 

    return 0, auc


if __name__ == '__main__':
    main()