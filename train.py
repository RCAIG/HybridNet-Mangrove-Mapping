import argparse
import datetime
import gorilla
import os
import os.path as osp
import shutil
import time
import wandb
from tqdm import tqdm
import numpy as np
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
import albumentations as A

from torchsummary import summary
import segmentation_models_pytorch as smp

from models.EDGE_unet_mvin_ndwi import * #@To adapt
from utils.eval_mask import *
from utils.dataset_build import *

def get_args():
    parser = argparse.ArgumentParser('HybridNet')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--resume', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', action='store_true', help='skip validation')
    args = parser.parse_args()
    return args


def train(epochs, model, train_loader,val_loader,criterion, optimizer, scheduler, cfg, device):
    #4/15/2025 wandb func
    wandb.init(project=cfg.wandb.project,  name = cfg.wandb.name, save_code=True) #@TOADAPT:Change project name
    model.run_id = wandb.run.id

    train_losses = []
    test_losses = []
    val_iou = []; val_acc = []
    train_iou = []; train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1 ; not_improve=0

    train_miou = []
    train_f1S = []
    train_accs = []

    model.to(device)
    fit_time = time.time()
    best_miou = 0

    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        
        #training loop
        if 1:
            em_t = evaluate_metric(2) #miou
            model.train()
            for i, data in enumerate(tqdm(train_loader)):
                #training phase

                #print(type(i), type(data) )
                image_tiles, mask_tiles = data
                if 0:
                    bs, n_tiles, c, h, w = image_tiles.size()

                    image_tiles = image_tiles.view(-1,c, h, w)
                    mask_tiles = mask_tiles.view(-1, h, w)
                
                image = image_tiles.to(device); mask = mask_tiles.to(device,torch.float)
                b, c, h,w = mask.shape
                #mask = mask.reshape(b,1,h,w)
                mask_squueze = torch.argmax(mask, 1) #b,h, w
                mask_squueze = mask_squueze.reshape(b,1,h,w)
                mask_edge = getSobel(mask_squueze, device) #b, 1, h, w
                #forward
                edge, output = model(image)
                loss = criterion(output, mask, edge, mask_edge)


                #evaluation metrics,
                # iou_score += mIoU(output, mask)
                # accuracy += pixel_accuracy(output, mask)

                if len(mask.shape) > 3:
                    em_t.addBatch(torch.argmax(mask, dim=1).view(-1).cpu().numpy(), torch.argmax(output, dim=1).contiguous().view(-1).cpu().numpy())                        
                else:
                    em_t.addBatch(mask.view(-1).cpu().numpy(), torch.argmax(output, dim=1).contiguous().view(-1).cpu().numpy())
                
                #backward
                loss.backward()
                optimizer.step() #update weight          
                optimizer.zero_grad() #reset gradient
                
                #step the learning rate
                lrs.append(get_lr(optimizer))
                scheduler.step() 
                
                running_loss += loss.item()
                train_losses.append(loss.item())


                train_iou_score, train_f1, train_accuracy  = em_t.getMetrics()

                train_miou.append(train_iou_score)
                train_f1S.append(train_f1)
                train_accs.append(train_accuracy)


        #test loop  
        if 1:
            print("Start to validate ....")
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0

            em =  evaluate_metric(2) #miou

            #validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    #reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if 0:
                        bs, n_tiles, c, h, w = image_tiles.size()

                        image_tiles = image_tiles.view(-1,c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)
                    
                    image = image_tiles.to(device); mask = mask_tiles.to(device,torch.float);
                    b, c, h,w = mask.shape
                    #mask = mask.reshape(b,1,h,w)
                    mask_squueze = torch.argmax(mask, 1) #b,h, w
                    mask_squueze = mask_squueze.reshape(b,1,h,w)
                    mask_edge = getSobel(mask_squueze,device) #b, 1, h, w
                    
                    #forward
                    edge, output = model(image)
                    loss = criterion(output, mask, edge, mask_edge)
                    #evaluation metrics
#                     val_iou_score +=  mIoU(output, mask)
#                     test_accuracy += pixel_accuracy(output, mask)
                    #loss
                    if len(mask.shape) > 3:
                        em.addBatch(torch.argmax(mask, dim=1).view(-1).cpu().numpy(), torch.argmax(output, dim=1).contiguous().view(-1).cpu().numpy())                        
                    else:
                        em.addBatch(mask.view(-1).cpu().numpy(), torch.argmax(output, dim=1).contiguous().view(-1).cpu().numpy())
                
                    #loss = criterion(output, mask)                                  
                    test_loss += loss.item() #
                    test_losses.append(loss.item())
            
            test_meters = np.array(test_losses,dtype=np.float32)
            test_losses.clear()
            val_loss = test_meters.mean()

            val_iou_score, f1, test_accuracy  = em.getMetrics()
            if val_iou_score > best_miou:
                best_miou = val_iou_score
                print('saving model...')
                if(epochs >=10):                
                    torch.save(model, 'HybridNet.pt')                            
                                
            if min_loss > val_loss:
                print('Loss Decreasing.. {:.3f} >> {:.3f} '.format(min_loss, val_loss))
                min_loss = val_loss
                decrease += 1

        train_meters = np.array(train_losses, dtype=np.float32)
        train_losses.clear()
        train_lossm = train_meters.mean()

        #publish to wandb
        wandb.log({'epoch':e, 'train_loss': train_lossm, 'val_loss':val_loss,"Train mIoU:":train_miou[0],
                "Val mIoU:": val_iou_score,
                "Train Acc:": train_accs[0],
                "Val Acc:":test_accuracy})

        #reset
        train_miou.clear()
        train_accs.clear()
        #print(em.confusionMat)
        #iou
        val_iou.append(val_iou_score)
        train_iou.append(iou_score/len(train_loader))
        train_acc.append(accuracy/len(train_loader))
        val_acc.append(test_accuracy)

    # log and save
    #save_file = osp.join(cfg.work_dir, 'lastest.pth')
    #meta = dict(epoch=epochs)
    #gorilla.save_checkpoint(model, save_file, optimizer, lr_scheduler, meta)



def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./exps', osp.splitext(osp.basename(args.config))[0])

    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')

    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))


    #device for running the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed
    gorilla.set_random_seed(cfg.train.seed)

    # model
    model = EDGE_net(cfg.model.num_layer)

    count_parameters = gorilla.parameter_count(model)['']
    #logger.info(f'Parameters: {count_parameters / 1e6:.2f}M')

    # optimizer and scheduler

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=cfg.weight_decay)

    #lr_scheduler = gorilla.build_lr_scheduler(optimizer, cfg.lr_scheduler)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, cfg.max_lr, epochs=cfg.epoch,
                                                steps_per_epoch=cfg.steps_per_epoch)
    # pretrain or resume
    # train and val dataset
    # A refers to Albumentations is a computer vision tool that boosts the performance of deep convolutional neural networks.
    t_train = A.Compose([ A.HorizontalFlip(), A.VerticalFlip(), 
                        A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                        A.GaussNoise()])

    # H * W = 704 * 1056
    t_val = A.Compose([ A.HorizontalFlip(),
                    A.GridDistortion(p=0.2)])

    # To add dataset root here
    trainingData = cfg.dataloader.train
    testData = cfg.dataloader.test
    #train_dataset = build_dataset(cfg.data.train, logger)
    train_set = SentinelDataset(trainingData, cfg.smin, cfg.smax, one_hot = True, EDGE_B = False,transform= t_train, patch=False)
    val_set = SentinelDataset(testData, cfg.smin, cfg.smax, one_hot = True, EDGE_B = False,transform=  t_val, patch=False)

    #train_loader = build_dataloader(train_dataset, **cfg.dataloader.train)
    train_loader = build_dataset(train_set, cfg.batch_size)
    val_loader = build_dataset(val_set, cfg.batch_size)  


    train(cfg.train.epochs, model, train_loader, val_loader, criterion, optimizer, sched, cfg, device)



if __name__ == '__main__':
    main()