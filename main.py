import logging
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedShuffleSplit
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
import wandb

wandb.init(project='Lite-ProTransformer', mode="disabled")

random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True

best_con = 0.
best_acc = 1e10
l1_loss = torch.nn.SmoothL1Loss()

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

parser = argparse.ArgumentParser()

# ── Kaggle data paths (replaces the old --data_root folder structure) ──────
parser.add_argument(
    '--metadata_csv',
    default='/kaggle/input/datasets/manuelcldg/radiomics-nsclc/phase1_metadata.csv',
    type=str,
    help='Path to phase1_metadata CSV (patient_id, ct_path, seg_path)',
)
parser.add_argument(
    '--clinical_csv',
    default='/kaggle/input/datasets/saibhossain/clinical-data-of-nsclc-lungi1/'
            'NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv',
    type=str,
    help='Path to NSCLC-Radiomics clinical CSV',
)
parser.add_argument(
    '--val_split',
    default=0.2,
    type=float,
    help='Fraction of data to use for validation',
)
parser.add_argument(
    '--gtv_margin',
    default=10,
    type=int,
    help='Voxel margin for GTV bounding-box crop (0 = no crop)',
)

# ── Original args (unchanged) ──────────────────────────────────────────────
parser.add_argument('--ckpt_path',
                    default='./',
                    type=str,
                    help='Directory to save checkpoints')
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
parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    help='Training batch size')
parser.add_argument('--epochs',
                    default=800,
                    type=int,
                    help='Number of training epochs')
parser.add_argument('--lr',
                    default=0.0005,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--num_workers',
                    default=2,
                    type=int,
                    help='DataLoader worker count')

args = parser.parse_args()


def train_one_epoch(net, data_loader, entropy_loss, optimizer, exp_lr_scheduler):
	#####################  train model	##########################
	global best_con, best_acc
	train_loss = 0
	count = 0
	net.train()
	correct = 0.0
	event_c=torch.tensor([],dtype=torch.bool)
	label_c=torch.tensor([]).to("cuda:0")
	outputs_c=torch.tensor([]).to("cuda:0")
	for i, data in enumerate(data_loader):
		inputs, clinical, label, event = data
		inputs = inputs.unsqueeze(1)
		label = label.unsqueeze(1)

		inputs = inputs.repeat(1, 3, 1, 1, 1)
		inputs = inputs.float().to("cuda:0")
		label = label.float().to("cuda:0")
		clinical = clinical.float().to("cuda:0")
		optimizer.zero_grad(set_to_none=True)

		outputs, z1, z2 = net(inputs,clinical)
		loss = entropy_loss(outputs, label) + l1_loss(outputs, label)
		loss_1 = entropy_loss(z1, label)
		loss_2 = l1_loss(z2, label)
		loss_ = loss + args.alpha*loss_1 + args.beta*loss_2
		if not torch.isfinite(loss_):
			continue

		loss_.backward()
		torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
		optimizer.step()

		exp_lr_scheduler.step()

		train_loss = train_loss + loss_.item()
		count = count + 1
		correct += torch.sum(torch.abs(outputs - label)).data
		event_c=torch.cat((event_c,event))
		label_c=torch.cat((label_c,label))
		outputs_c=torch.cat((outputs_c,outputs))

	# Guard against occasional NaNs/Inf in predictions so C-index computation doesn't crash.
	est = outputs_c.view(-1).cpu().detach().numpy()
	est = np.nan_to_num(est, nan=0.0, posinf=1e6, neginf=-1e6)
	ctd = concordance_index_censored(
		event_c.view(-1).cpu().detach().numpy(),
		label_c.view(-1).cpu().detach().numpy(),
		est,
	)
	accracy = correct/len(data_loader.dataset) if len(data_loader.dataset) > 0 else torch.tensor(0.0)
	train_loss = train_loss / max(count, 1)
	if (1-ctd[0]) > best_con:
		best_con = 1 - ctd[0]
	if accracy < best_acc:
		best_acc = accracy
	wandb.log({'Best concordance': best_con})
	wandb.log({'Current concordance': 1-ctd[0]})
	wandb.log({'Training loss':train_loss})
	wandb.log({'Lowest MAE':best_acc})
	print('training loss: %.4f, accuracy= %.4f, concordance = %.4f, current best concordance = %.4f, current lowest MAE = %.4f' % (train_loss, accracy, 1-ctd[0], best_con, best_acc))
	LOGGER.info('training loss: %.4f, accuracy= %.4f, concordance = %.4f' % (train_loss, accracy, 1-ctd[0]))
	train_acc = accracy

	return train_loss, train_acc


def validate_one_epoch(net, test_loader, entropy_loss):
	net.eval()
	test_loss = 0
	count = 0
	correct = 0
	for i, data in enumerate(test_loader):
		inputs, clinical, label, event = data
		inputs = inputs.unsqueeze(0)
		label = label.unsqueeze(0)
		inputs = inputs.repeat(1, 3, 1, 1, 1)
		inputs = inputs.float().to("cuda:0")
		label = label.float().to("cuda:0")
		clinical = clinical.float().to("cuda:0")

		outputs, _, _ = net(inputs,clinical)
		loss = entropy_loss(outputs, label)

		test_loss = test_loss + loss.item()
		count = count + 1
		correct += torch.sum(torch.abs(outputs - label)).data

	val_loss = test_loss / count
	val_acc=correct / len(test_loader.dataset)

	print('validate loss: %.4f, accuracy= %.4f' % (val_loss, val_acc))
	LOGGER.info('validate loss: %.4f, accuracy= %.4f' % (val_loss, val_acc))

	return val_loss, val_acc


def train():
	logging.basicConfig(
		level=logging.DEBUG,
		format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
		filename="training.log")

	# ── Build full dataset, then stratified-split into train / val ──────────
	# Use survival-time quantile as a proxy stratification label so both
	# splits have a similar distribution of short/long survivors.
	full_dataset = DataBowl3Classifier(
		metadata_csv=args.metadata_csv,
		clinical_csv=args.clinical_csv,
		phase='train',
		isAugment=False,          # augmentation applied per-item in train subset
		gtv_margin=args.gtv_margin,
	)

	n = len(full_dataset)
	# Bin survival into 4 quantile buckets for stratification
	survs = full_dataset.data["Survival.time"].values
	bins  = np.percentile(survs, [25, 50, 75])
	strat_labels = np.digitize(survs, bins)   # 0,1,2,3

	splitter = StratifiedShuffleSplit(
		n_splits=1, test_size=args.val_split, random_state=7,
	)
	train_idx, val_idx = next(splitter.split(np.zeros(n), strat_labels))

	# Train subset — rebuild with augmentation enabled
	train_dataset = DataBowl3Classifier(
		metadata_csv=args.metadata_csv,
		clinical_csv=args.clinical_csv,
		phase='train',
		isAugment=True,
		gtv_margin=args.gtv_margin,
		indices=train_idx.tolist(),
	)
	val_dataset = DataBowl3Classifier(
		metadata_csv=args.metadata_csv,
		clinical_csv=args.clinical_csv,
		phase='val',
		isAugment=False,
		gtv_margin=args.gtv_margin,
		indices=val_idx.tolist(),
	)

	print(f"Dataset: {n} total | {len(train_dataset)} train | {len(val_dataset)} val")

	train_loader_case = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers,
		pin_memory=True,
	)
	val_loader_case = DataLoader(
		val_dataset,
		batch_size=1,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=True,
	)

	################  define model, loss and optimizer ##########################
	net = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes())
	wandb.watch(net)
	net.to("cuda:0")

	mse_loss = torch.nn.MSELoss()

	learnable_params = filter(lambda p: p.requires_grad, net.parameters())
	optimizer = torch.optim.Adam(learnable_params, lr=args.lr, weight_decay=0.001)

	exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

	################  train  model ##########################
	num_epoch = args.epochs
	best_loss = 9999.9
	best_acc  = 0
	train_loss_list, train_acc_list = [], []
	val_loss_list,   val_acc_list   = [], []

	writer = SummaryWriter()

	os.makedirs(args.ckpt_path, exist_ok=True)

	for i in range(num_epoch):
		print("----epoch %d:" % i)
		LOGGER.info("----epoch %d:" % i)
		train_loss, train_acc = train_one_epoch(
			net, train_loader_case, mse_loss, optimizer, exp_lr_scheduler)

		val_loss, val_acc = validate_one_epoch(net, val_loader_case, mse_loss)
		val_acc1 = val_acc.item()
		if not np.isfinite(val_acc1):
			val_acc1 = 1e10

		print(len(train_loader_case.dataset), len(val_loader_case.dataset))

		if val_acc1 < best_loss:
			best_loss = val_acc1
			torch.save(net.state_dict(), os.path.join(args.ckpt_path, "best.pth"))

		torch.save(net.state_dict(), os.path.join(args.ckpt_path, "checkpoint.pth"))

		writer.add_scalar('Loss/train', train_loss, i)
		writer.add_scalar('Loss/test',  val_loss,   i)
		writer.add_scalar('Acc/train',  train_acc,  i)
		writer.add_scalar('Acc/test',   val_acc,    i)

		train_loss_list.append(train_loss)
		train_acc_list.append(train_acc)
		val_loss_list.append(val_loss)
		val_acc_list.append(val_acc)

	print("test complete")
	LOGGER.info("test complete")

	x = np.arange(num_epoch)
	plt.plot(x, train_loss_list, 'r--', label='training_loss')
	plt.plot(x, val_loss_list,   'b--', label='testing_loss')
	plt.title('Loss')
	plt.xlabel('Number of epochs')
	plt.ylabel('Loss values')
	plt.grid()
	plt.legend()
	plt.savefig('train_.png')

	plt.clf()
	plt.plot(x, train_acc_list, 'g--', label='training_acc')
	plt.plot(x, val_acc_list,   'y--', label='testing_acc')
	plt.title('Accuracy')
	plt.xlabel('Number of epochs')
	plt.ylabel('Accuracy')
	plt.grid()
	plt.legend()
	plt.savefig('test_.png')

	writer.close()

if __name__ == '__main__':
	train()
	print('5_2_hiddenx4layers')
