import os, sys
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from config import cfg
from dataset import DatasetLoader
from timer import Timer
from logger import colorlogger
from nets.balanced_parallel import DataParallelModel, DataParallelCriterion
from nets import loss
from model import get_pose_net

#for p in sys.path:
#    print(p)

# dynamic dataset import
#print('from ' + cfg.trainset[0] + ' import ' + cfg.trainset[0])
#print('from ' + cfg.testset + ' import ' + cfg.testset)

for i in range(len(cfg.trainset)):
    exec('from ' + cfg.trainset[i] + ' import ' + cfg.trainset[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.txt'):
        
        self.cfg = cfg
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

    def save_model(self, state, epoch):
        file_path = osp.join(self.cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer, scheduler):
        model_file_list = glob.glob(osp.join(self.cfg.model_dir,'*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt = torch.load(osp.join(self.cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')) 
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        return start_epoch, model, optimizer, scheduler
    

class Trainer(Base):
    
    def __init__(self, cfg):
        self.JointLocationLoss = DataParallelCriterion(loss.JointLocationLoss())
        super(Trainer, self).__init__(cfg, log_name = 'train_logs.txt')    

    def get_optimizer(self, optimizer_name, model):
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd) 
        else:
            print("Error! Unknown optimizer name: ", optimizer_name)
            assert 0

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.lr_dec_epoch, gamma=self.cfg.lr_dec_factor)
        return optimizer, scheduler
    
    def _make_batch_generator(self):
        self.logger.info("Creating dataset...")
        trainset_list = []
        for i in range(len(self.cfg.trainset)):
            trainset_list.append(eval(self.cfg.trainset[i])("training"))
    
        trainset_loader = DatasetLoader(trainset_list, True, 
                                        transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.cfg.pixel_mean, std=self.cfg.pixel_std)]))
        print(len(trainset_loader))
        batch_generator = DataLoader(dataset=trainset_loader, 
                                     batch_size=self.cfg.num_gpus*self.cfg.batch_size, 
                                     shuffle=True, 
                                     num_workers=self.cfg.num_thread, 
                                     pin_memory=True)

        self.joint_num = trainset_loader.joint_num[0]
        self.itr_per_epoch = math.ceil(trainset_loader.__len__() / self.cfg.num_gpus / self.cfg.batch_size)
        self.batch_generator = batch_generator       
        
    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_pose_net(self.cfg, True, self.joint_num)
        model = DataParallelModel(model).cuda()
        optimizer, scheduler = self.get_optimizer(self.cfg.optimizer, model)
        if self.cfg.continue_train:
            start_epoch, model, optimizer, scheduler = self.load_model(model, optimizer, scheduler)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler         
        
