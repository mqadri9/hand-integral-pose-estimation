import argparse
from config import cfg
from base import Trainer
import torch.backends.cudnn as cudnn
import sys
import numpy as np

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

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.scheduler.step()
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        for itr, (img_patch, label, label_weight) in enumerate(trainer.batch_generator):
        #for itr, kk in enumerate(trainer.batch_generator):
            #print(kk)
            #sys.exit()
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            trainer.optimizer.zero_grad()
            #===================================================================
            # print("==============================================")
            # print(input_img.shape)
            # print(joint_img.shape)
            # print(joint_vis.shape)
            # print(joints_have_depth.shape)
            # #print(joints_have_depth)
            # 
            # print("===============================================")
            #===================================================================
            img_patch = img_patch.cuda()
            label = label.cuda()
            label_weight = label_weight.cuda()
            
            heatmap_out = trainer.model(img_patch)
            #print(len(heatmap_out))
            JointLocationLoss = trainer.JointLocationLoss(heatmap_out, label, label_weight)

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
                '%s: %.4f' % ('loss_loc', JointLocationLoss.detach()),
                ]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'scheduler': trainer.scheduler.state_dict(),
        }, epoch)
  
if __name__ == "__main__":
    main()