import logging
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_censored
from data import DataBowl3Classifier
import model
import argparse

import random
import numpy as np
import os

import datetime
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from model import get_inplanes, BasicBlock, ResNet, Bottleneck

from lifelines.utils import concordance_index

# wandb is optional — disabled when --no_wandb flag is set
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
best_con = 0.
best_acc = 1e10
l1_loss = torch.nn.SmoothL1Loss()

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

parser = argparse.ArgumentParser()
parser.add_argument('--data_root',
                    default='./data/',
                    type=str,
                    help='Root directory path')
parser.add_argument('--ckpt_path',
                    default=None,
                    type=str,
                    help='Annotation file path')
parser.add_argument('--result_path',
                    default='./',
                    type=str,
                    help='Result directory path')
parser.add_argument('--suffix',
                    default=None,
                    type=str,
                    help='Result directory path')
parser.add_argument('--alpha',
                    default=0.5,
                    type=float,
                    help='trade-off parameter')
parser.add_argument('--beta',
                    default=0.5,
                    type=float,
                    help='trade-off parameter')
parser.add_argument('--no_wandb',
                    action='store_true',
                    help='Disable wandb logging')
parser.add_argument('--mode',
                    choices=['train', 'test'],
                    default='train',
                    help='train: full training loop; test: load checkpoint and evaluate on test split')
parser.add_argument('--checkpoint',
                    default=None,
                    type=str,
                    help='Path to .pth weights (required for --mode test; e.g. best.pth)')

args = parser.parse_args()

# Never use wandb in test mode (no training metrics to log)
USE_WANDB = (
    _WANDB_AVAILABLE and not args.no_wandb and args.mode == "train"
)
if USE_WANDB:
    wandb.init(project='Lite-ProTransformer')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def _to_float(x):
    """Convert scalar tensor or Python number to float for logging/plotting."""
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def train_one_epoch(net, data_loader, entropy_loss, optimizer, exp_lr_scheduler, scaler):
	#####################  train model	##########################
	# one epoch
	global best_con, best_acc
	train_loss = 0
	count = 0
	net.train()
	correct = 0.0
	event_c=torch.tensor([],dtype=torch.bool)
	label_c=torch.tensor([]).to(DEVICE)
	outputs_c=torch.tensor([]).to(DEVICE)
	for i, data in enumerate(data_loader):
		inputs, clinical, label, event = data
		inputs = inputs.unsqueeze(1)
		label = label.unsqueeze(1)

		inputs = inputs.repeat(1, 3, 1, 1, 1)
		inputs = inputs.float().to(DEVICE)
		label = label.float().to(DEVICE)
		clinical = clinical.float().to(DEVICE)

		optimizer.zero_grad()
		with torch.amp.autocast('cuda'):
			outputs, z1, z2 = net(inputs, clinical)
			loss = entropy_loss(outputs, label) + l1_loss(outputs, label)
			loss_1 = entropy_loss(z1, label)
			loss_2 = l1_loss(z2, label)
			loss_ = loss + args.alpha*loss_1 + args.beta*loss_2

		# calculate the gradient and update weights
		scaler.scale(loss_).backward()
		scaler.step(optimizer)
		scaler.update()

		train_loss = train_loss + loss_.item()
		count = count + 1
		correct += torch.sum(torch.abs(outputs - label)).data
		event_c=torch.cat((event_c,event))
		label_c=torch.cat((label_c,label))
		outputs_c=torch.cat((outputs_c,outputs))

	ctd = concordance_index_censored(event_c.view(-1).cpu().detach().numpy(), label_c.view(-1).cpu().detach().numpy(),
									 outputs_c.view(-1).cpu().detach().numpy())
	accracy = correct / len(data_loader.dataset)
	train_loss = train_loss / count
	train_acc = _to_float(accracy)
	exp_lr_scheduler.step()
	if (1-ctd[0]) > best_con:
		best_con = 1 - ctd[0]
	if train_acc < best_acc:
		best_acc = train_acc
	if USE_WANDB:
		wandb.log({'Best concordance': best_con,
		           'Current concordance': 1-ctd[0],
		           'Training loss': train_loss,
		           'Lowest MAE': best_acc})
	print('training loss: %.4f, accuracy= %.4f, concordance = %.4f, current best concordance = %.4f, current lowest MAE = %.4f' % (train_loss, train_acc, 1-ctd[0], best_con, best_acc))
	LOGGER.info('training loss: %.4f, accuracy= %.4f, concordance = %.4f' % (train_loss, train_acc, 1-ctd[0]))

	return train_loss, train_acc


def validate_one_epoch(net, test_loader, entropy_loss):
	# test the mode
	# one epoch
	net.eval()
	test_loss = 0
	count = 0
	correct = 0
	event_c=torch.tensor([],dtype=torch.bool)
	label_c=torch.tensor([]).to(DEVICE)
	outputs_c=torch.tensor([]).to(DEVICE)
	for i, data in enumerate(test_loader):
		inputs, clinical, label, event = data
		inputs = inputs.unsqueeze(1)        # (B,Z,Y,X) → (B,1,Z,Y,X)
		label = label.unsqueeze(1)
		inputs = inputs.repeat(1, 3, 1, 1, 1)
		inputs = inputs.float().to(DEVICE)
		label = label.float().to(DEVICE)
		clinical = clinical.float().to(DEVICE)

		outputs, _, _ = net(inputs,clinical)
		# calculate the cross entropy loss
		loss = entropy_loss(outputs, label)

		test_loss = test_loss + loss.item()
		count = count + 1
		correct += torch.sum(torch.abs(outputs - label)).data
		event_c=torch.cat((event_c,event))
		label_c=torch.cat((label_c,label))
		outputs_c=torch.cat((outputs_c,outputs))

	ctd = concordance_index_censored(event_c.view(-1).cpu().detach().numpy(), label_c.view(-1).cpu().detach().numpy(),
									 outputs_c.view(-1).cpu().detach().numpy())
	val_c_index = 1 - ctd[0]
	val_loss = test_loss / count
	val_acc = correct / len(test_loader.dataset)
	val_acc = _to_float(val_acc)

	print('validate loss: %.4f, accuracy= %.4f, concordance= %.4f' % (val_loss, val_acc, val_c_index))
	LOGGER.info('validate loss: %.4f, accuracy= %.4f, concordance= %.4f' % (val_loss, val_acc, val_c_index))

	return val_loss, val_acc, val_c_index


def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
	if pretrain_path:
		print('loading pretrained model {}'.format(pretrain_path))
		pretrain=torch.load(pretrain_path, map_location='cpu')

		model.load_state_dict(pretrain['state_dict'])
		# model.load_state_dict(pretrain)
		tmp_model=model
		if model_name == 'densenet':
			tmp_model.classifier=nn.Linear(tmp_model.classifier.in_features,
											 n_finetune_classes)
		else:
			tmp_model.fc=nn.Linear(tmp_model.fc.in_features,
									 n_finetune_classes)

	return model

def train():
	################  load data ##########################
	batch_size=1
	workers=4
	train_path=os.path.join(args.data_root, "train")
	val_path=os.path.join(args.data_root, "val")
	# path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/evaluat"
	#
	# if phase == "train":
	#	  path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/train"

	logging.basicConfig(
		level=logging.DEBUG,
		format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
		filename="training.log")



	# path = "/data/yujwu/NSCLC/survival_estimate/tumor_segment/seg_output"
	dataset_train=DataBowl3Classifier(
		train_path, phase='train', isAugment=True)
	dataset_val=DataBowl3Classifier(val_path, phase='val', isAugment=False)

	# if phase == "evaluate":
	#	  path = "/data/yujwu/NSCLC/survival_estimate/survival_est_xh/data/evaluat"
	#
	#	  dataset = DataBowl3Classifier(path, phase = 'evaluate')
	#
	#
	train_loader_case=DataLoader(
		dataset_train, batch_size=batch_size*64, shuffle=True)
	val_loader_case=DataLoader(
		dataset_val, batch_size=batch_size, shuffle=True)

	################  define model, loss and optimizer ##########################
	# net = model.DPN92_3D()
	# r3d18_K_200ep.pth: --model resnet --model_depth 18 --n_pretrain_classes 700
	# output_class=1
	# # net=torchvision.models.video.r3d_18(pretrained=True)
	# net=torchvision.models.video.r3d_18(pretrained=True)
	# for param in net.parameters():
	#	  param.requires_grad=True
	#
	# num_featdim=net.fc.in_features
	# net.fc=nn.Linear(num_featdim, output_class)
	# net.fc=nn.Linear(num_featdim, 50)

	net = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes())

	# Load MedNet pretrained 3D ResNet-10 weights
	mednet_path = '/kaggle/input/datasets/jirkaborovec/meidcalnet-pretrained-3d-resnet-weights/resnet_10.pth'
	if os.path.isfile(mednet_path):
		pretrain = torch.load(mednet_path, map_location='cpu')
		state_dict = pretrain.get('state_dict', pretrain)
		# Remove 'module.' prefix if saved with DataParallel
		state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
		# Adapt conv1: MedNet was trained with 1-channel CT input.
		# We use 3-channel input (via .repeat(1,3,1,1,1)), so we replicate the
		# single-channel weights across 3 channels and scale by 1/3 to preserve
		# the overall activation magnitude.
		if 'conv1.weight' in state_dict:
			w = state_dict['conv1.weight']       # shape: [64, 1, 7, 7, 7]
			state_dict['conv1.weight'] = w.repeat(1, 3, 1, 1, 1) / 3.0  # [64, 3, 7, 7, 7]
		missing, unexpected = net.load_state_dict(state_dict, strict=False)
		print(f"Loaded MedNet weights from {mednet_path}")
		print(f"  Missing keys (new heads — expected): {len(missing)}")
		print(f"  Unexpected keys (ignored): {len(unexpected)}")
	else:
		print(f"[WARNING] MedNet weights not found at {mednet_path} — training from scratch")

	if USE_WANDB:
		wandb.watch(net)
	net.to(DEVICE)

	# define loss
	# entropy_loss = torch.nn.CrossEntropyLoss()
	mse_loss=torch.nn.MSELoss()
	# mse_loss = nn.BCELoss()
	# define optimizer
	learnable_params=filter(lambda p: p.requires_grad, net.parameters())
	optimizer=torch.optim.Adam(learnable_params, lr=0.001, weight_decay=0.001)

	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
	scaler = torch.amp.GradScaler('cuda')

	################  train  model ##########################
	num_epoch=80
	best_loss=9999.9
	best_acc=0
	train_loss_list, train_acc_list = [], []
	val_loss_list, val_acc_list = [], []

	writer = SummaryWriter()

	for i in range(num_epoch):
		print("----epoch %d:" % i)
		LOGGER.info("----epoch %d:" % i)
		train_loss, train_acc=train_one_epoch(
			net, train_loader_case, mse_loss, optimizer, exp_lr_scheduler, scaler)

		val_loss, val_acc, val_c_index=validate_one_epoch(net, val_loader_case, mse_loss)
		val_acc1 = val_acc

		print(len(train_loader_case.dataset),len(val_loader_case.dataset))

		# save the best model
		if val_acc1 < best_loss:
			best_loss=val_acc1
			if args.ckpt_path:
				torch.save(net.state_dict(), os.path.join(args.ckpt_path, "best.pth"))

		# save the last trained model
		if args.ckpt_path:
			torch.save(net.state_dict(), os.path.join(args.ckpt_path, "checkpoint.pth"))

		writer.add_scalar('Loss/train', train_loss, i)
		writer.add_scalar('Loss/test', val_loss, i)
		writer.add_scalar('Acc/train', train_acc, i)
		writer.add_scalar('Acc/test', val_acc, i)

		train_loss_list.append(train_loss)
		train_acc_list.append(train_acc)
		val_loss_list.append(val_loss)
		val_acc_list.append(val_acc)

	print("test complete")
	LOGGER.info("test complete")

	x = np.arange(num_epoch)
	#plt.subplot(211)
	l1 = plt.plot(x, train_loss_list, 'r--', label='training_loss')
	l2 = plt.plot(x, val_loss_list, 'b--', label='testing_loss')
	plt.title('Loss')
	plt.xlabel('Number of epochs')
	plt.ylabel('Loss values')
	plt.grid()
	plt.legend()
	#plt.show()
	plt.savefig('train_.png')

	#plt.subplot(212)
	l3 = plt.plot(x, train_acc_list, 'g--', label='training_acc')
	l4 = plt.plot(x, val_acc_list, 'y--', label='testing_acc')
	plt.title('Accuracy')
	plt.xlabel('Number of epochs')
	plt.ylabel('Accuracy')
	plt.grid()
	plt.legend()
	#plt.show()
	plt.savefig('test_.png')

	writer.close()


def load_checkpoint_weights(net, checkpoint_path):
	"""Load state_dict from torch.save(net.state_dict()) or nested dict."""
	state = torch.load(checkpoint_path, map_location=DEVICE)
	if isinstance(state, dict) and 'state_dict' in state:
		state = state['state_dict']
	net.load_state_dict(state, strict=True)
	print(f"Loaded weights from {checkpoint_path}")


def _batch_bool_event(event):
	"""Normalize event tensor/scalar to a single bool (batch_size==1)."""
	if torch.is_tensor(event):
		return bool(event.view(-1)[0].item())
	return bool(event)


def test_run():
	"""Evaluate a saved checkpoint on data_root/test (same preprocessing as val)."""
	if not args.checkpoint or not os.path.isfile(args.checkpoint):
		raise SystemExit(
			"--mode test requires --checkpoint /path/to/best.pth (or checkpoint.pth)"
		)

	test_path = os.path.join(args.data_root, "test")
	if not os.path.isdir(test_path):
		raise SystemExit(f"Test folder not found: {test_path}")

	dataset_test = DataBowl3Classifier(test_path, phase='test', isAugment=False)
	test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)

	net = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes())
	load_checkpoint_weights(net, args.checkpoint)
	net.to(DEVICE)
	net.eval()

	mse_loss = torch.nn.MSELoss()
	test_loss = 0.0
	count = 0
	abs_err = 0.0

	event_list, label_list, pred_list = [], [], []

	with torch.no_grad():
		for data in test_loader:
			inputs, clinical, label, event = data
			inputs = inputs.unsqueeze(1)
			label = label.unsqueeze(1)
			inputs = inputs.repeat(1, 3, 1, 1, 1)
			inputs = inputs.float().to(DEVICE)
			label = label.float().to(DEVICE)
			clinical = clinical.float().to(DEVICE)

			outputs, _, _ = net(inputs, clinical)
			loss = mse_loss(outputs, label)
			test_loss += loss.item()
			count += 1
			abs_err += torch.sum(torch.abs(outputs - label)).item()

			event_list.append(_batch_bool_event(event))
			label_list.append(label.detach().view(-1).cpu().numpy())
			pred_list.append(outputs.detach().view(-1).cpu().numpy())

	if count == 0:
		raise SystemExit("Test loader is empty.")

	test_loss /= count
	mae = abs_err / len(dataset_test)

	ev = np.asarray(event_list, dtype=bool)
	y_time = np.concatenate(label_list).astype(np.float64)
	y_pred = np.concatenate(pred_list).astype(np.float64)

	ctd = concordance_index_censored(ev, y_time, y_pred)
	c_index = float(1.0 - ctd[0])

	# MAE on uncensored only (paper-style)
	unc = ~ev
	if unc.any():
		mae_unc = float(np.mean(np.abs(y_pred[unc] - y_time[unc])))
	else:
		mae_unc = float("nan")

	print("=" * 60)
	print(f"TEST  checkpoint : {args.checkpoint}")
	print(f"TEST  samples   : {len(dataset_test)}")
	print(f"TEST  MSE loss  : {test_loss:.6f}")
	print(f"TEST  MAE (all) : {mae:.6f}")
	print(f"TEST  MAE (unc.) : {mae_unc:.6f}")
	print(f"TEST  C-index   : {c_index:.4f}  (1 - sksurv concordance)")
	print("=" * 60)

	out_txt = os.path.join(args.result_path, "test_metrics.txt")
	try:
		os.makedirs(args.result_path, exist_ok=True)
		with open(out_txt, "w") as f:
			f.write(f"checkpoint={args.checkpoint}\n")
			f.write(f"n={len(dataset_test)}\n")
			f.write(f"mse={test_loss}\n")
			f.write(f"mae_all={mae}\n")
			f.write(f"mae_uncensored={mae_unc}\n")
			f.write(f"c_index={c_index}\n")
		print(f"Wrote metrics to {out_txt}")
	except OSError as e:
		print(f"Could not write {out_txt}: {e}")


if __name__ == '__main__':
	if args.mode == 'train':
		train()
		print('5_2_hiddenx4layers')
	else:
		test_run()
