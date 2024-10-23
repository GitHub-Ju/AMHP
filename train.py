import os

import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
import clip
import random
from AMHP import AMHP
from utils import set_dataset, _preprocess1, _preprocess2, convert_models_to_fp32, emd_loss
from itertools import product

from scipy.stats import pearsonr
from scipy.stats import spearmanr


aesthetics = ['bad', 'poor', 'fair', 'good', 'perfect']

seed = 20240412
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
train_patch = 1
emd2 = emd_loss(dist_r=2)

aes_texts = torch.cat(
    [clip.tokenize(f"A photo with {a} aesthetics.")
     for a in product(aesthetics)
    ]
).to(device)

preprocess_train = _preprocess1()
preprocess_test = _preprocess2()


def compute_val_metrics(pred_score, true_score):
    srcc_mean = spearmanr(pred_score, true_score)
    lcc_mean = pearsonr(pred_score, true_score)
    return srcc_mean[0], lcc_mean[0]

def train(model):


    num_steps_per_epoch = 415

    model.eval()

    loaders = []
    for loader in train_loaders:
        loaders.append(iter(loader))

    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])

    for step in range(num_steps_per_epoch):

        img_batch = []
        aes_distri_batch = []
        att_texts_batch = []

        for dataset_idx, loader in enumerate(loaders, 0):
            try:
                sample_batched = next(loader)
            except StopIteration:
                loader = iter(train_loaders[dataset_idx])
                sample_batched = next(loader)
                loaders[dataset_idx] = loader

            x, aes_distri, att_texts = (
                sample_batched['I'],
                sample_batched['aes_distri'],
                sample_batched['att_texts'],
            )

            x = x.to(device)
            img_batch.append(x)
            aes_distri = aes_distri.to(device)
            aes_distri_batch.append(aes_distri)
            att_texts_token = clip.tokenize(att_texts)
            att_texts_token = att_texts_token.to(device)
            att_texts_batch.append(att_texts_token)

        img_batch = torch.cat(img_batch, dim=0)  # tensor(bs, 1, 3, 224, 224)
        aes_distri_batch = torch.cat(aes_distri_batch, dim=0)  # tensor(bs, 5)
        att_texts_batch = torch.cat(att_texts_batch, dim=0)  # tensor(bs, 77)

        optimizer.zero_grad()

        logits_aesthetic = model(img_batch, aes_texts, att_texts_batch)

        # loss
        loss_aes = emd2(logits_aesthetic, aes_distri_batch.detach())
        total_loss = loss_aes

        total_loss.backward()

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()

        if ((step + 1) % 415 == 0):
            format_str = ('(E:%d, S:%d / %d)')
            print(format_str % (epoch + 1, step + 1, num_steps_per_epoch))

    SRCC = 0.0
    PLCC = 0.0

    SRCC, PLCC = eval(PARA_test_loader)
    print('-------------------------------------------------------------------------------------------------------')
    print('Epoch {}:  SRCC:{:.3}  PLCC:{:.3}'.format(epoch + 1, SRCC, PLCC))
    print('-------------------------------------------------------------------------------------------------------')

    return SRCC, PLCC


def eval(loader):
    model.eval()

    aes_gt = []
    aes_pre = []

    for step, sample_batched in enumerate(loader, 0):

        x, aes_mean, att_texts = (
            sample_batched['I'],
            sample_batched['aes_mean'],
            sample_batched['att_texts'],
        )

        x = x.to(device)
        att_texts_token = clip.tokenize(att_texts)
        att_texts_token = att_texts_token.to(device)

        aes_gt = aes_gt + aes_mean.cpu().tolist()

        with torch.no_grad():
            logits_aesthetic = model(x, aes_texts, att_texts_token)

        aesthetic_preds = 1 * logits_aesthetic[:, 0] + 1.75 * logits_aesthetic[:, 1] + 2.75 * logits_aesthetic[:, 2] + 3.75 * logits_aesthetic[:, 3] + 5 * logits_aesthetic[:, 4]

        aes_pre = aes_pre + aesthetic_preds.cpu().tolist()

    SRCC, PLCC = compute_val_metrics(aes_pre, aes_gt)

    return SRCC, PLCC

num_workers = 8
num_epoch = 5

initial_lr = 2e-6
PARA_set = '/media/boot/4T/dataset/PARA/imgs'
PARA_train_csv = "/media/boot/4T/dataset/PARA/annotation/PARA-GiaaTrain.csv"
PARA_test_csv = "/media/boot/4T/dataset/PARA/annotation/PARA-GiaaTest.csv"
PARA_train_loader = set_dataset(PARA_train_csv, 68, PARA_set, num_workers, preprocess_train, train_patch, False)
PARA_test_loader = set_dataset(PARA_test_csv, 16, PARA_set, num_workers, preprocess_test, 1, True)
train_loaders = [PARA_train_loader]

model = AMHP(input_dim=1024)
model = model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=initial_lr,
    weight_decay=0.001)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

start_epoch = 0
i = 0
for epoch in range(0, num_epoch):
    SRCC, PLCC = train(model)
    scheduler.step()





