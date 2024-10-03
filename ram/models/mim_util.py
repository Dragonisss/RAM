import numpy as np
import torch
class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6,mask_type='binary'):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = torch.FloatTensor(mask)
        mask_token = None
        
        if self.mask_type == 'rgba':
            mask = mask * torch.rand((self.rand_size,self.rand_size))
            # print(mask)
          
        if self.mask_type != 'binary':
            mask_token = torch.rand((3,self.rand_size,self.rand_size))
        else:
            mask_token = torch.zeros((3,self.rand_size,self.rand_size))
        return mask, mask_token


class TestMaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6,mask_type='binary'):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self,h=None,w=None):
        mask_idx=None
        if h is not None and w is not None:
            self.token_count = h * w
            self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
            mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        else:
            mask_idx = np.random.permutation(self.token_count)[:self.mask_count]

        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((h, w))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = torch.FloatTensor(mask)
        mask_token = None
        
        if self.mask_type == 'rgba':
            mask = mask * torch.rand((h,w))
            # print(mask)
          
        if self.mask_type != 'binary':
            mask_token = torch.rand((3,h,w))
        else:
            mask_token = torch.zeros((3,h,w))
        return mask, mask_token