from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms as T


class SentinelDataset(Dataset):
    
    def __init__(self, data, smin, smax, one_hot = False, EDGE_B = True,transform=None, patch=False):
        self.feature, self.label = data
        self.transform = transform
        self.patches = patch
        self.min = smin
        self.max = smax
        self.one_hot = one_hot
        self.EDGE_B = EDGE_B
        
        self.SOBEL = nn.Conv2d(1, 2, 1, padding=1, padding_mode='replicate', bias=False)
        self.SOBEL.weight.requires_grad = False
        self.SOBEL.weight.set_(torch.Tensor([[
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]],
           [[-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]]).reshape(2, 1, 3, 3))

        
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        #main idea, each scene will contain several small patches with size of 256 * 256
#         indice = idx // 100
#         img = self.feature[indice]
#         mask = self.label[indice]
        
#         h = img.shape[0]
#         cur_center = np.random.choice(h-257, 2)
#         #img1 = np.flipud(np.transpose(img, [1, 2, 0]))# H, W, C
#         img1 = img[ cur_center[0]:cur_center[0]+256, cur_center[1]:cur_center[1]+256,: ]
#         mask = mask[cur_center[0]:cur_center[0]+256, cur_center[1]:cur_center[1]+256 ]
        img1, mask = self.feature[idx], self.label[idx]
        #print(img1.shape)

        img_splt1 = img1[:,:,2:]
        img_splt2 = img1[:,:,:2]
        img1 = np.concatenate([img_splt1, img_splt2],axis=2)
        #print(img_splt1.shape)
        #print(img1.shape)
        if self.transform is not None:
            aug = self.transform(image=img1, mask=mask)
            img1 = aug['image']
            mask = aug['mask']
        
        if self.transform is None:
            img1 = img1
        
        #t = T.Compose([T.ToTensor(), T.Normalize(self.min, self.max)])
        
        #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        t = T.ToTensor() 
        img1 = t(img1)
        #one_hot
        if self.one_hot:
            if not self.EDGE_B:
                mask = np.eye(2)[mask,:]
                mask = np.transpose(mask, (2,0 ,1))
                #mask = torch.from_numpy(mask).long()
                mask = torch.tensor(mask)


            else:
                mask = torch.tensor(mask)
                mask1 = torch.any(self.SOBEL(mask) != 0, dim=1, keepdims=True).float() 
                mask = torch.unsqueeze(mask, dim=0)
                mask1 = torch.unsqueeze(mask1, dim=0)


                print(mask1.numpy().shape)
                mask = torch.cat([mask, mask1], dim=0) #2, H ,W
            
        else:
            mask = torch.from_numpy(mask).long()
        
        if self.patches:
            img1, mask = self.tiles(img1, mask)
            
        return img1, mask
    
    def tiles(self, img, mask):
        # contiguous() simply make the memory stored the array be continuous
        # while view() do the same thing like reshape()
        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768) 
        img_patches  = img_patches.contiguous().view(3,-1, 512, 768) 
        img_patches = img_patches.permute(1,0,2,3)
        
        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)
        
        return img_patches, mask_patches
    


def build_dataset(train_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader

