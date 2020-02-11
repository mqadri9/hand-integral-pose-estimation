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
import config_panet as config_panet
import PANet_reconstruction
import copy
from pycrayon import CrayonClient
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
    
    def compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly')
        

class Trainer(Base):
    
    def __init__(self, cfg):
        if cfg.loss == "L_combined":
            self.CombinedLoss = DataParallelCriterion(loss.CombinedLoss())
        else: 
            self.JointLocationLoss = DataParallelCriterion(loss.JointLocationLoss())
            self.JointLocationLoss2 = DataParallelCriterion(loss.JointLocationLoss2())        
        super(Trainer, self).__init__(cfg, log_name = 'train_logs.txt')    

    def get_optimizer(self, optimizer_name, model):
        if optimizer_name == 'adam':
            lr = self.cfg.lr
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd) 
        else:
            print("Error! Unknown optimizer name: ", optimizer_name)
            assert 0

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.cfg.lr_dec_epoch, gamma=self.cfg.lr_dec_factor)
        return optimizer, scheduler

    def load_nrsfm_tester(self):
        self.nrsfm_tester = PANet_reconstruction.NRSfM_tester(pts_num=config_panet.pts_num)
        self.nrsfm_tester.load_model(config_panet.pretrain_model)
        self.nrsfm_tester.nrsfm_net = DataParallelModel(self.nrsfm_tester.nrsfm_net).cuda()
        self.logger.info("loaded Procrustes Analysis Network")
    
    def load_regressor_teacher(self):
        ckpt = torch.load(cfg.teacher_model_path)
        model = get_pose_net(self.cfg, True, self.joint_num)
        model = DataParallelModel(model).cuda()
        optimizer, scheduler = self.get_optimizer(self.cfg.optimizer, model)
        model.load_state_dict(ckpt['network'])
#         optimizer.load_state_dict(ckpt['optimizer'])
#         scheduler.load_state_dict(ckpt['scheduler'])    
        self.teacher_network = model
        self.teacher_network.eval()
        self.logger.info("Loaded teacher pose regressor")
        return copy.deepcopy(model)
       
    def _make_batch_generator(self, main_loop=True):
        self.logger.info("Creating dataset...")
        trainset_list = []
        for i in range(len(self.cfg.trainset)):
            trainset_list.append(eval(self.cfg.trainset[i])("training"))
    
        trainset_loader = DatasetLoader(trainset_list, True, 
                                        transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.cfg.pixel_mean, std=self.cfg.pixel_std)]),
                                        main_loop=main_loop)
        print("batch size ")
        print(self.cfg.num_gpus*self.cfg.batch_size)
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
            if cfg.loss == "L_combined":
                self.load_nrsfm_tester()
                self.load_regressor_teacher()
                assert not self.teacher_network.training
        elif cfg.loss == "L_combined":
            self.load_nrsfm_tester()
            _model = self.load_regressor_teacher()
            assert not self.teacher_network.training
            start_epoch = 0
        else:
            start_epoch = 0
        model.train()
#         optimizer, scheduler = self.get_optimizer(self.cfg.optimizer, model)
        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compare_models(self.teacher_network, self.model)
        assert self.model.training
        
class Tester(Base):
    
    def __init__(self, cfg, test_epoch):
        self.coord_out = loss.softmax_integral_tensor
        self.test_epoch = int(test_epoch)
        self.JointLocationLoss = DataParallelCriterion(loss.JointLocationLoss())
        self.JointLocationLoss2 = DataParallelCriterion(loss.JointLocationLoss2())
        super(Tester, self).__init__(cfg, log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset = eval(self.cfg.testset)("testing")
        testset_loader = DatasetLoader(testset, False, transforms.Compose([transforms.ToTensor(), 
                                                                           transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]))
        batch_generator = DataLoader(dataset=testset_loader, batch_size=self.cfg.num_gpus*self.cfg.test_batch_size, shuffle=False, num_workers=self.cfg.num_thread, pin_memory=True)
        
        self.testset = testset
        self.joint_num = testset_loader.joint_num
        self.skeleton = testset_loader.skeleton
        self.tot_sample_num = testset_loader.__len__()
        self.batch_generator = batch_generator
        self.num_samples = testset.num_samples
        print("Number of testing samples is {}".format(self.num_samples))

    def load_nrsfm_tester(self):
        #### Loaded to calculate testing loss only #####
        self.logger.info("loading Procrustes Analysis Network")
        self.nrsfm_tester = PANet_reconstruction.NRSfM_tester(pts_num=config_panet.pts_num)
        self.nrsfm_tester.load_model(config_panet.pretrain_model)     
        self.nrsfm_reconstruction_func = PANet_reconstruction.PANet_reconstruction
    
    def load_regressor_teacher(self):
        #### Loaded to calculate testing loss only #####
        self.logger.info("Loading teacher pose regressor")
        ckpt = torch.load(cfg.teacher_model_path)
        teacher_model = get_pose_net(self.cfg, True, self.joint_num)
        teacher_model = DataParallelModel(teacher_model).cuda() 
        ckpt = torch.load(cfg.teacher_model_path)
        teacher_model.load_state_dict(ckpt['network'])
        teacher_model.eval()
        self.teacher_network = teacher_model
       
    def _make_model(self):
        
        model_path = os.path.join(self.cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_pose_net(self.cfg, False, self.joint_num)
        model = DataParallelModel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()
        self.model = model
        if cfg.loss == "L_combined":
            self.load_nrsfm_tester()
            self.load_regressor_teacher()
        
    def _evaluate(self, preds, label_list, augmentation_list, result_save_path):
        self.testset.evaluate(preds, label_list, augmentation_list,  result_save_path)


class Evaluator(Base):

    def __init__(self, cfg, evaluation_epoch):
        self.evaluation_epoch = int(evaluation_epoch)
        super(Evaluator, self).__init__(cfg, log_name = 'evaluate_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        evaluationset = eval(self.cfg.testset)("evaluation", is_eval=True)
        evaluationset_loader = DatasetLoader(evaluationset, 
                                             False, 
                                             transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]),
                                             is_eval=True)
        batch_generator = DataLoader(dataset=evaluationset_loader, batch_size=self.cfg.num_gpus*self.cfg.eval_batch_size, shuffle=False, num_workers=self.cfg.num_thread, pin_memory=True)
        
        self.evaluationset = evaluationset
        self.joint_num = evaluationset_loader.joint_num
        self.skeleton = evaluationset_loader.skeleton
        self.tot_sample_num = evaluationset_loader.__len__()
        self.batch_generator = batch_generator
        self.num_samples = evaluationset.num_samples
        print("Number of evaluation samples is {}".format(self.num_samples))
    
    def _make_model(self):
        model_path = os.path.join(self.cfg.model_dir, 'snapshot_%d.pth.tar' % self.evaluation_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_pose_net(self.cfg, False, self.joint_num)
        model = DataParallelModel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()
        self.model = model

    def _evaluate(self, preds, params, result_save_path):
        self.evaluationset.evaluate_evaluations(preds, params, result_save_path)
        
    