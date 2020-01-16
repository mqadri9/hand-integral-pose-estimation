import argparse
from config import cfg
from base import Trainer
import torch.backends.cudnn as cudnn
import sys
import numpy as np
from base import Tester
from tqdm import tqdm
from torch.nn.parallel.scatter_gather import gather
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train)
    # Refer to https://github.com/soumith/cudnn.torch for a detailed 
    # explanation of these parameters
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()
    tester = Tester(cfg, trainer.start_epoch)
    tester._make_batch_generator()
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.scheduler.step()
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (img_patch, label, label_weight, augmentation) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            trainer.optimizer.zero_grad()
            img_patch = img_patch.cuda()
            label = label.cuda()
            label_weight = label_weight.cuda()
            
            center_x = augmentation["bbox"][0].cuda()
            center_y = augmentation["bbox"][1].cuda()
            width = augmentation["bbox"][2].cuda()
            height = augmentation["bbox"][3].cuda()
            scale = augmentation["scale"].cuda()
            R = augmentation["R"].cuda()
            trans = augmentation["trans"].cuda()
            zoom_factor = augmentation["zoom_factor"].cuda()
            z_mean = augmentation["z_mean"].cuda()
            f = augmentation["f"].cuda()
            K = augmentation["K"].cuda()
            joint_cam = augmentation["joint_cam"].cuda()
            heatmap_out = trainer.model(img_patch)
            JointLocationLoss = trainer.JointLocationLoss(heatmap_out, label, label_weight)
            JointLocationLoss2 = trainer.JointLocationLoss2(heatmap_out, label, label_weight, joint_cam, center_x,
                                                           center_y, width, height, scale, R, trans,
                                                           zoom_factor, z_mean, f, K)

            loss = JointLocationLoss

            loss.backward()
            trainer.optimizer.step()
            
            trainer.gpu_timer.toc()

            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.scheduler.get_lr()[0]),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                '%s: %.4f | %.4f' % ('loss_loc', JointLocationLoss.detach(), JointLocationLoss2.detach()) #.detach()),
                ]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        print("Finished epoch {}. Calculating test error".format(epoch))
        with torch.no_grad():
            loss_sum = 0
            loss_sum2 = 0
            tester.test_epoch = epoch
            i = 0
            for itr, (img_patch, label, label_weight, augmentation) in enumerate(tqdm(tester.batch_generator)):
                i+=1
                img_patch = img_patch.cuda()
                label = label.cuda()
                label_weight = label_weight.cuda()
                
                center_x = augmentation["bbox"][0].cuda()
                center_y = augmentation["bbox"][1].cuda()
                width = augmentation["bbox"][2].cuda()
                height = augmentation["bbox"][3].cuda()
                scale = augmentation["scale"].cuda()
                R = augmentation["R"].cuda()
                trans = augmentation["trans"].cuda()
                zoom_factor = augmentation["zoom_factor"].cuda()
                z_mean = augmentation["z_mean"].cuda()
                f = augmentation["f"].cuda()
                K = augmentation["K"].cuda()
                joint_cam = augmentation["joint_cam"].cuda()
                heatmap_out = trainer.model(img_patch)
                #if cfg.num_gpus > 1:
                #    heatmap_out = gather(heatmap_out,0)
                JointLocationLoss = trainer.JointLocationLoss(heatmap_out, label, label_weight)
                JointLocationLoss2 = trainer.JointLocationLoss2(heatmap_out, label, label_weight, joint_cam, center_x,
                                                               center_y, width, height, scale, R, trans,
                                                               zoom_factor, z_mean, f, K)

                loss_sum2 += JointLocationLoss2.detach()
                loss_sum += JointLocationLoss.detach()
            screen = [
               'Epoch %d/%d' % (epoch, cfg.end_epoch),
               '%s: %.4f | %.4f' % ('Average loss on test set', loss_sum/i, loss_sum2/i),
               ]
            tester.logger.info(' '.join(screen))
            
        if epoch >= 0:
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'scheduler': trainer.scheduler.state_dict(),
            }, epoch)
  
if __name__ == "__main__":
    main()