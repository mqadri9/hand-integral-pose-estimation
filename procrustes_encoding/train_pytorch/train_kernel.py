import torch
import cv2
from torch import optim
import os
import numpy as np
import argparse
import copy
from tensorboardX import SummaryWriter

""" template code for training """

seeds = [
	3203477698260160801,
	9055020843370939117,
    5027425335615802694,
    5477599281297670245,
    1418598265549597701,
    1360158215207868720,
    6972948577695775127,
    1219328218444875891,
    1294029598755011749,
    1668434612404921824,
    7707263122193563216,
    9051751849877968013
	]

class TrainKernel(torch.nn.Module):

	def load_model(self, ckpt_file):
		pass

	def save_model(self, ckpt_file):
		pass

	def get_parameters(self):
		pass

	def init_weights(self):
		pass

	def forward(self, input, do_log_im, do_log_scalar):
		# return loss, logs_to_write
		pass


def array2im(input):
	
	# first, normalize the array
	xmin = input.min()
	xmax = input.max()
	x = np.clip((input-xmin)/(xmax-xmin+1e-30)*255, 0, 255).astype(np.uint8)

	# convert to image according to colormap
	colormap = cv2.COLORMAP_JET
	im = cv2.applyColorMap(x, colormap)
	return im

def MatAngleAxisToR(angle_axis):
    """ Convert 3d vector of axis-angle rotation to 3x3 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 3, 3)`

    Example:
        >>> input = torch.rand(N, 3)  # Nx3
        >>> output = angle_axis_to_rotation_matrix(input)  # Nx3x3
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)    

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix    
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.zeros(batch_size, 3, 3, device='cuda:0')    
    rotation_matrix = torch.eye(3).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx3x3


def add_train_args(arg_parser):
	arg_parser.add_argument('--root', type=str, help='parent folder of the dataset')
	arg_parser.add_argument('--trlist', type=str, help='training set list text file')
	arg_parser.add_argument('--logdir', type=str, help='training log and checkpoint output folder')
	arg_parser.add_argument('--epoch', type=int, default=10, help='number of epoch to train')
	arg_parser.add_argument('--which', type=int, default=-2, help='-1 start from most recent check point, \
		>=0 load the corresponding check point; else random initial weights')
	arg_parser.add_argument('--pretrain_model', type=str, default=None, help='path to a pretrained model, \
		set this if want to start with the pretrained model')
	arg_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
	arg_parser.add_argument('--bsize', type=int, default=1, help='batch size')
	arg_parser.add_argument('--print_freq', type=int, default=10, help='print info every # iteration')
	arg_parser.add_argument('--disp_freq', type=int, default=100, help='display image every # iteration')
	arg_parser.add_argument('--save_freq', type=int, default=1000, help='save model every # iteration')
	arg_parser.add_argument('--validation_freq', type=int, default=1000, help='Validation model every # iteration')
	arg_parser.add_argument('--thread_num', type=int, default=4, help='number of threads for data loading')

	arg_parser.add_argument('--seed_id', type=int, default=0, help='id to pick random seed from a predefined list')

	# parameters for training with data on memory
	arg_parser.add_argument('--maxitr', type=int, default=200000, help='maximum number of iterations')
	arg_parser.add_argument('--lr_decay_step', type=int, default=10000, help='learning rate decay step')
	arg_parser.add_argument('--lr_decay_rate', type=float, default=0.95, help='learning rate decay rate')

	# relu vs bst for NRSfM encoder
	arg_parser.add_argument('--encode_with_relu', type=int, default=1, help='choose between ReLU or BST for encoder')

	# option to augment the data
	arg_parser.add_argument('--augmentation', type=int, default=0, help='option to augment the data by rotating roll, pitch, yaw')
	arg_parser.add_argument('--aug_rotate_val', type=float, default=0.15, help='augment the data by rotating with the following radians')


def evaluate_validation_set(validation_data_loader, train_kernel):
	error_list = []
	with torch.no_grad():
		for ii, data in enumerate(validation_data_loader):
			loss, log_dict = train_kernel.forward(
				data, do_log_im=False, do_log_scalar=True)
			error_list.append(log_dict['error'])
	mean_error = np.asarray(error_list).mean()
	print('-- Validation Error: %f --' % mean_error)
	return mean_error

def evaluate_validation_set_batch(validation_data, validation_kernel, batch_size=256):
	num_sample = validation_data[0].shape[0]
	num_batch = num_sample // batch_size
	batch_start_index = np.floor(np.linspace(0, num_sample, num_batch)).astype(np.int64).tolist()
	error_list = []
	log_dict_all = {}
	with torch.no_grad():
		for idx in range(len(batch_start_index) - 1):
			start_id = batch_start_index[idx]
			end_id = batch_start_index[idx+1]
			minbatch_data = tuple([x[start_id:end_id, ...] for x in list(validation_data)])
			loss, log_dict = validation_kernel.forward(
				minbatch_data,
				do_log_im=False,
				do_log_scalar=True)

			if idx == 0:
				# create entries in dict
				for tag in log_dict.keys():
					val = log_dict[tag]
					if val.size == 1:
						log_dict_all[tag] = []

			for tag in log_dict.keys():
				val = log_dict[tag]
				if val.size == 1:
					log_dict_all[tag].append(val)

			error_list.append(log_dict['error'])

		for tag in log_dict_all.keys():
			log_dict_all[tag] = np.asarray(log_dict_all[tag]).mean()

		mean_error = np.asarray(error_list).mean()

	print('-- Validation Error: %f --' % mean_error)

	return mean_error, log_dict_all

def write_log_images(tfb_writer, log_dict, itr_num):
	if tfb_writer is None:
		return
	for tag in log_dict.keys():
		val = log_dict[tag]
		if val.size > 1:
			tfb_writer.add_image(tag, val, itr_num)

def write_log_scalars(tfb_writer, log_dict, epoch,
					  itr_num, itr_num_per_epoch, tag_prefix='train'):
	str_to_print = '%s -- Iteration [%d][%d] %d/%d: ' %(
					tag_prefix,
					itr_num,
					epoch,
					itr_num-epoch*itr_num_per_epoch,
					itr_num_per_epoch)

	for tag in log_dict.keys():
		val = log_dict[tag]
		if val.size == 1:
			str_to_print += '%s %f | '%(tag_prefix + '_' + tag, val)
			tfb_writer.add_scalar(tag_prefix + '_' + tag, val, itr_num)
	print(str_to_print)


def train_main(train_set, train_kernel, validation_set, args):
	seed = seeds[args.seed_id]
	torch.manual_seed(seed)

	# options:
	# dataroot = args.root
	# train_list_file = args.trlist
	log_dir = args.logdir
	epoch_num = args.epoch
	which_epoch = args.which

	# training parameters
	lr = args.lr
	batch_size = args.bsize

	# log options
	print_freq = args.print_freq
	display_freq = args.disp_freq
	save_latest_freq = args.save_freq
	validation_freq = args.validation_freq

	## create tensorboard writer
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)

	training_writer = SummaryWriter(log_dir)

	# create data loader
	train_data_loader = torch.utils.data.DataLoader(train_set,
						batch_size=batch_size, shuffle=True,
						num_workers=args.thread_num, pin_memory=True)
	validation_data_loader = torch.utils.data.DataLoader(validation_set,
						batch_size=batch_size, shuffle=False,
						num_workers=args.thread_num, pin_memory=True)

	# create model
	if which_epoch >= 0:
		train_kernel.load_model(os.path.join(log_dir, 'model_%04d.pth' % (which_epoch)))
	elif which_epoch == -1:
		train_kernel.load_model(os.path.join(log_dir, 'model_best.pth'))
	else:
		train_kernel.init_weights()
	train_kernel.cuda()

	# creat the optimizer
	optimizer = optim.Adam(train_kernel.get_parameters(), lr=lr)


	## start training
	itr_num_per_epoch = int(np.ceil(len(train_set)/batch_size))
	itr_num = max(0, which_epoch) * itr_num_per_epoch
	start_epoch = max(0, which_epoch)
	max_epoch = start_epoch + epoch_num

	lowest_validation_error,_ = evaluate_validation_set(validation_data_loader, train_kernel)
	print('-- Current lowest validation error: %f --' % lowest_validation_error)

	for epoch in range(max(0, which_epoch), max_epoch):
		for ii, data in enumerate(train_data_loader):
			do_log_im = np.mod(itr_num, display_freq) == 0
			do_log_scalar = np.mod(itr_num, print_freq) == 0

			optimizer.zero_grad()

			loss, log_dict = train_kernel.forward(data, do_log_im, do_log_scalar)

			# Do backward only when loss is not NAN
			loss.backward()
			if torch.isnan(loss).data.cpu().numpy():
				print('NAN!')
				continue

			optimizer.step()

			if do_log_im:
				write_log_images(training_writer, log_dict, itr_num)

			if do_log_scalar:
				write_log_scalars(training_writer, log_dict, epoch, itr_num, itr_num_per_epoch)

			if np.mod(itr_num, save_latest_freq) == 0 and itr_num>0:
				train_kernel.save_model(os.path.join(log_dir, 'model_cur.pth'))
				train_kernel.save_model(os.path.join(log_dir, 'model_%04d.pth' % (epoch)))

			if np.mod(itr_num, validation_freq) == 0:
				validation_error = evaluate_validation_set(validation_data_loader, train_kernel)
				print('-- Current best validation error: %f --' % lowest_validation_error)
				training_writer.add_scalar('Validation error', validation_error, itr_num)
				# save model if it is current best model with lowest error
				if validation_error < lowest_validation_error:
					lowest_validation_error = validation_error
					train_kernel.save_model(
						os.path.join(log_dir, 'model_best.pth'))
					print('Saving current best model!')

			itr_num+=1

	# Perform validation and save model before exiting
	train_kernel.save_model(os.path.join(log_dir, 'model_cur.pth'))
	validation_error, validation_log = evaluate_validation_set(validation_data_loader, train_kernel)
	training_writer.add_scalar('Validation_error', validation_error, itr_num)
	# Save model if it is current best model with lowest error
	if validation_error < lowest_validation_error:
		lowest_validation_error = validation_error
		train_kernel.save_model(
			os.path.join(log_dir, 'model_best.pth'))
		print('Saving current best model')

	print('-- Training end, best validation error %f --' % lowest_validation_error)


def train_batch_main(train_data, train_kernel, validation_data, args):
	seed = seeds[args.seed_id]
	torch.manual_seed(seed)

	print_freq = args.print_freq
	display_freq = args.disp_freq
	save_latest_freq = args.save_freq
	lr = args.lr
	log_dir = args.logdir

	min_batch_size = args.bsize
	max_itr = args.maxitr
	lr_decay_step = args.lr_decay_step
	lr_decay_rate = args.lr_decay_rate

	# Create tensorboard writer
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)
	training_writer = SummaryWriter(log_dir)

	train_kernel.init_weights()
	if args.pretrain_model is not None:
		train_kernel.load_model(args.pretrain_model)
	train_kernel.cuda()

	# Set optimizer and scheduler
	optimizer = optim.Adam(train_kernel.get_parameters(), lr=lr)
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
		milestones=np.arange(lr_decay_step, max(max_itr, lr_decay_step), lr_decay_step).tolist(),
		gamma=lr_decay_rate)

	# Load train data directly to GPU
	train_data = tuple([x.cuda() for x in list(train_data)])
	validation_data = tuple([x.cuda() for x in list(validation_data)])


	# print(train_data[0])
	# print(size(train_data))

	lowest_validation_error, _ = evaluate_validation_set_batch(validation_data, train_kernel)

	num_min_batch = train_data[0].shape[0] // min_batch_size
	itr_num_in_epoch = 0

	rnd_index = torch.randperm(train_data[0].shape[0])

	for itr_num in range(max_itr):
		do_log_im = np.mod(itr_num, display_freq) == 0
		do_log_scalar = np.mod(itr_num, print_freq) == 0

		if min_batch_size < train_data[0].shape[0]:
			index = torch.randint(0, train_data[0].shape[0], (min_batch_size,))
			index = rnd_index[index]
			train_minbatch_data = tuple([x[index, ...] for x in list(train_data)])
		else:
			train_minbatch_data = train_data

		if (args.augmentation == 1):			
			#angles = args.aug_rotate_val*np.random.randn(args.bsize, 3)
			angles = np.random.randn(args.bsize, 3) * (args.aug_rotate_val + args.aug_rotate_val) + args.aug_rotate_val
			angles = torch.from_numpy(angles)
			R = MatAngleAxisToR(angles)
			R = R.cuda().float()
			train_minbatch_data = list(train_minbatch_data)
			train_minbatch_data[0] = torch.matmul(train_minbatch_data[0], R)
			train_minbatch_data = tuple(train_minbatch_data)

		optimizer.zero_grad()
		loss, log_dict = train_kernel.forward(
			train_minbatch_data, do_log_im=do_log_im, do_log_scalar=do_log_scalar)
		loss.backward()
		optimizer.step()

		if do_log_im:
			write_log_images(training_writer, log_dict, itr_num)

		if do_log_scalar:
			write_log_scalars(training_writer, log_dict, 0, itr_num, max_itr)

		if (np.mod(itr_num, save_latest_freq)==0 or np.mod(itr_num, args.validation_freq)==0) and itr_num>0:
			train_kernel.save_model(os.path.join(log_dir, 'model_cur.pth'))
			cur_error, validation_log = evaluate_validation_set_batch(validation_data, train_kernel)
			print('-- Current best: %f --' % lowest_validation_error)
			write_log_scalars(training_writer, validation_log, 0, itr_num, max_itr, 'validation')

			if lowest_validation_error > cur_error:
				lowest_validation_error = cur_error
				train_kernel.save_model(os.path.join(log_dir, 'model_best.pth'))

		scheduler.step()

def train_composite_model(train_data, train_kernel, composite_kernel_constructor, args):

	log_dir = args.logdir

	comp_num = args.comp_num

	# create workspace
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)

	# move the pretrained network to the current dir, and treat it as the #0 component
	os.system('cp %s %s' % (
		args.pretrain_model,
		os.path.join(log_dir, 'model_comp_00.pth')))

	train_data = tuple([x.cuda() for x in list(train_data)])
	train_sample_num = train_data[0].shape[0]

	for comp_id in range(1, comp_num):
		args_cur = copy.deepcopy(args)
		args_cur.logdir = os.path.join(log_dir, 'comp%02d'%comp_id)
		args_cur.pretrain_model = os.path.join(log_dir, 'model_comp_%02d.pth' % (comp_id-1))
		args_cur.comp_num = comp_id

		# detect hard samples
		# step 1: Load the composite model
		with torch.no_grad():
			composite_model = composite_kernel_constructor(args=args_cur)
			composite_model.load_model(log_dir)
			composite_model.cuda()

			# step 2: Compute loss per sample
			loss_per_sample, _ = predict_batch_main(train_data, composite_model, args)

			# step 3: Pick the last 10% of the samples
			print(loss_per_sample.shape)
			loss_validation_sorted, index_sorted = loss_per_sample.sort(descending=True)
			hard_sample_num = train_sample_num // 10
			hard_sample_index = index_sorted[:hard_sample_num]
			hard_sample = tuple([x[hard_sample_index, ...] for x in list(train_data)])

		# Train with the detected hard_samples
		train_batch_main(hard_sample, train_kernel, hard_sample, args_cur)

		# Copy the best model to be the next component model
		os.system('cp %s %s' % (
			os.path.join(args_cur.log_dir, 'model_best.pth'),
			os.path.join(log_dir, 'model_comp_%02d.pth'%comp_id)))
		print('Comp #%02d saved --- ' % comp_id)


def predict_batch_main(test_data, test_kernel, args):
	
	test_kernel.load_model(args.pretrain_model)
	test_kernel.cuda()

	data_size = test_data[0].shape[0]
	batch_size = min(args.bsize, data_size)
	batch_start_id_list = np.asarray(np.arange(0, data_size, batch_size))
	batch_end_id_list = batch_start_id_list + batch_size
	batch_end_id_list[-1] = data_size

	test_data = tuple(x.cuda() for x in list(test_data))
	output_list = []
	for idx in range(batch_start_id_list.shape[0]):
		start_id = batch_start_id_list[idx]
		end_id = batch_end_id_list[idx]
		batch_data = tuple([x[start_id:end_id, ...] for x in list(test_data)])

		output_batch = test_kernel.forward(batch_data)

		if len(output_batch) == 1:
			output_batch = [output_batch]
		else:
			output_batch = list(output_batch)

		if idx == 0:
			output_list = [[x.data.cpu()] for x in output_batch]
		else:
			for output_id in range(len(output_list)):
				output_list[output_id].append(output_batch[output_id].data.cpu())

	output = tuple(torch.cat(x, 0) for x in output_list)
	return output




















