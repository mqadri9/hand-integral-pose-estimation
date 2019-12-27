import numpy as np
import torch
import copy, os, sys
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def convert_cvimg_to_tensor(cvimg):
    # from h,w,c(OpenCV) to c,h,w
    tensor = cvimg.copy()
    tensor = np.transpose(tensor, (2, 0, 1))
    # from BGR(OpenCV) to RGB
    tensor = tensor[::-1, :, :]
    # from int to float
    tensor = tensor.astype(np.float32)
    return tensor

from config import cfg
from base import Trainer

def main():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..', '..')
    common_dir = os.path.join(root_dir, 'common')
    main_dir = os.path.join(root_dir, 'main')
    util_dir = os.path.join(root_dir, 'common', 'utils')
    
    sys.path.insert(0, os.path.join(common_dir))
    sys.path.insert(0, os.path.join(main_dir))
    sys.path.insert(0, os.path.join(util_dir))
    cfg.batch_size = 256
    trainer = Trainer(cfg)
    trainer._make_batch_generator(main_loop=False)
    
    #===========================================================================
    # class MyDataset(Dataset):
    #     def __init__(self): 
    #         self.data = torch.randn(100, 3, 24, 24)
    #         
    #     def __getitem__(self, index):
    #         x = self.data[index]
    #         return x
    # 
    #     def __len__(self):
    #         return len(self.data)
    #     
    # 
    # dataset = MyDataset()
    # loader = DataLoader(
    #     dataset,
    #     batch_size=10,
    #     num_workers=1,
    #     shuffle=False
    # )
    # 
    #===========================================================================
    
    mean = 0.
    std = 0.
    nb_samples = 0.
    #===========================================================================
    # for data in loader:
    #     batch_samples = data.size(0)
    #     data = data.view(batch_samples, data.size(1), -1)
    #     mean += data.mean(2).sum(0)
    #     std += data.std(2).sum(0)
    #     nb_samples += batch_samples
    #===========================================================================
    

    for itr, data in enumerate(trainer.batch_generator):
        data = np.transpose(data, (0, 3, 1, 2)).type(torch.float64)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples*224
    std /= nb_samples*224
    print(mean)
    print(std)    
    
    
if __name__ == "__main__":
    main()