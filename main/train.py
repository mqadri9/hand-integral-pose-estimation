import argparse
from config import cfg
from base import Trainer
import torch.backends.cudnn as cudnn
import sys, time
import numpy as np
from base import Tester
from tqdm import tqdm
from torch.nn.parallel.scatter_gather import gather
import torch
import config_panet
from nets.loss import softmax_integral_tensor, JointLocationLoss
from FreiHand_config import FreiHandConfig


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

        for itr, (img_patch, params) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            trainer.optimizer.zero_grad()
            img_patch = img_patch.cuda()
            #trainer.compare_models(trainer.teacher_network, trainer.model)
            heatmap_out = trainer.model(img_patch)
            #===================================================================
            # heatmap_teacher_out = trainer.teacher_network(img_patch)
            # print(heatmap_out)
            # print(heatmap_teacher_out)
            # sys.exit()
            #===================================================================
            label = params["label"].cuda()
            label_weight = params["label_weight"].cuda()
            labelled = params["labelled"].cuda()
            trans = params["trans"].cuda()
            bbox = params["bbox"].cuda()
            K = params["K"].cuda()
            scale = params["scale"].cuda()
            joint_cam_normalized = params["joint_cam_normalized"].cuda()
            tprime = params["tprime_torch"].cuda()
            R = params["R"].cuda()
            if cfg.loss == "L_combined":
                hm_width = cfg.output_shape[0]
                hm_height = cfg.output_shape[0]
                hm_depth = cfg.depth_dim
                with torch.no_grad():
                    heatmap_teacher_out = trainer.teacher_network(img_patch)
                coord_out_teacher = []
                if cfg.num_gpus > 1:
                    for i in range(len(heatmap_teacher_out)):
                        coord_out_teacher.append(softmax_integral_tensor(heatmap_teacher_out[i], FreiHandConfig.num_joints, 
                                                                         hm_width, hm_height, hm_depth))
                else:
                        coord_out_teacher.append(softmax_integral_tensor(heatmap_teacher_out, FreiHandConfig.num_joints,
                                                                        hm_width, hm_height, hm_depth))                 
                del heatmap_teacher_out
                coord_out_teacher = torch.stack(coord_out_teacher, dim=0)
                combinedLoss, student_mpjpe, teacher_mpjpe, loss_supervised, loss_unsupervised = trainer.CombinedLoss(heatmap_out, coord_out_teacher, label, label_weight, labelled, 
                                                                                                                              tprime, trans, bbox, K, R, scale, joint_cam_normalized,
                                                                                                                              trainer.nrsfm_tester)
                loss = combinedLoss
            else:
                JointLocationLoss = trainer.JointLocationLoss(heatmap_out, label, label_weight)
                loss = JointLocationLoss
                student_mpjpe = 1
                teacher_mpjpe = 1
                loss_supervised = 1
                loss_unsupervised = 1
            
            loss.backward()
            trainer.optimizer.step()
            
            trainer.gpu_timer.toc()         
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.scheduler.get_lr()[0]),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                '%s: %.4f | %s %.4f | %s %.4f | %s %.4f  | %s %.4f' % ('loss_loc', loss.detach(), 'student_mpjpe', student_mpjpe*1000, 'teacher_mpjpe', teacher_mpjpe*1000,
                                                                        'loss_supervised', loss_supervised,  'loss_unsupervised', loss_unsupervised)
                ]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        if epoch >= 0:
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'scheduler': trainer.scheduler.state_dict(),
            }, epoch)
        print("Finished epoch {}. Calculating test error".format(epoch))
        with torch.no_grad():
            loss_sum = 0
            loss_sum2 = 0
            tester.test_epoch = epoch
            i = 0
            for itr, (img_patch, params) in enumerate(tqdm(tester.batch_generator)):
                i+=1
                img_patch = img_patch.cuda()
                label = params["label"].cuda()
                label_weight = params["label_weight"].cuda()
                heatmap_out = trainer.model(img_patch)
                #if cfg.num_gpus > 1:
                #    heatmap_out = gather(heatmap_out,0)
                JointLocationLoss = tester.JointLocationLoss(heatmap_out, label, label_weight)
                #JointLocationLoss2 = tester.JointLocationLoss2(heatmap_out, label, label_weight, joint_cam, joint_cam_normalized, center_x,
                #                                               center_y, width, height, scale, R, trans, K, tprime)

                loss_sum2 += 0 #JointLocationLoss2.detach()
                loss_sum += JointLocationLoss.detach()
            screen = [
               'Epoch %d/%d' % (epoch, cfg.end_epoch),
               '%s: %.4f | %.4f' % ('Average loss on test set', loss_sum/i, loss_sum2/i),
               ]
            tester.logger.info(' '.join(screen))
  
if __name__ == "__main__":
    main()