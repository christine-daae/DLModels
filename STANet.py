# STANet

import os

from sklearn.metrics import f1_score

from Datasets.LoadData3 import Get_data
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import random
import time

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

subject_list = ['VP001-EEG', 'VP002-EEG', 'VP003-EEG', 'VP004-EEG', 'VP005-EEG', 'VP006-EEG', 'VP007-EEG', 'VP008-EEG', 'VP009-EEG', 'VP010-EEG', 'VP011-EEG',
                'VP012-EEG', 'VP013-EEG', 'VP014-EEG', 'VP015-EEG', 'VP016-EEG', 'VP017-EEG', 'VP018-EEG', 'VP019-EEG', 'VP020-EEG', 'VP021-EEG', 'VP022-EEG',
                'VP023-EEG', 'VP024-EEG', 'VP025-EEG', 'VP026-EEG', 'VP027-EEG', 'VP028-EEG', 'VP029-EEG']


# Spatial Attention Module
class SpatialAtt(nn.Module):
    def __init__(self):
        # self.patch_size = patch_size
        super().__init__()

        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 3, (1, 1), (1, 1)),
            nn.ELU(),
        )

        self.pooling1 = nn.AdaptiveAvgPool2d((1, None))

        self.FC = nn.Sequential(
            nn.Linear(16, 8),
            nn.ELU(),
            nn.Linear(8, 16),
        )

        self.ConvBlock = nn.Sequential(
            nn.Conv2d(1, 5, (1, 16), (1, 1)),
            nn.ELU(),
            nn.MaxPool2d((4, 1))
        )

    def forward(self, x: Tensor) -> Tensor:
        x = torch.transpose(x, 3, 2)
        b, _, _, _ = x.shape
        out = self.Conv1(x)
        # print("shape", out.shape)
        out, _ = torch.max(out, dim=1)
        out = torch.unsqueeze(out, dim=1)
        # print("out shape", out.shape)
        x = x * out
        x = self.FC(x)
        # print("x shape", x.shape)
        x = self.ConvBlock(x)
        x = torch.squeeze(x, dim=-1)
        x = torch.transpose(x, 2, 1)
        # print(x.shape)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 修改线性层的输入和输出维度
        self.keys = nn.Linear(emb_size, 50)
        self.queries = nn.Linear(emb_size, 50)
        self.values = nn.Linear(emb_size, emb_size)  # values的维度保持不变
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = 50 ** (1 / 2)  # 修改scaling的计算，使其与新的维度相匹配
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=1,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(640, 2),
            nn.Dropout(0.5),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # print("FC", x.shape)
        x = x.contiguous().view(x.size(0), -1)
        # print("FC2", x.shape)
        out = self.fc(x)
        return x, out


class STANet(nn.Sequential):
    def __init__(self, emb_size=5, depth=1, n_classes=2, **kwargs):
        super().__init__(

            SpatialAtt(),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 16
        self.n_epochs = 30
        self.lr = 0.001
        self.nSub = nsub

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = STANet().cuda()
        # self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        # summary(self.model, (1, 22, 1000))

    def get_source_data(self):

        # sub_list = []
        # sub_list.append(subject_list[self.nSub - 1])
        # print(sub_list)
        X_train, X_test, y_train, y_test = Get_data(self.nSub)
        print(y_train.shape, X_test.shape)
        self.train_data = X_train
        self.train_label = y_train

        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        print(self.train_data.shape)

        shuffle_num = np.random.permutation(len(self.train_data))
        self.train_data = self.train_data[shuffle_num, :, :]
        self.train_label = self.train_label[shuffle_num]

        self.test_data = X_test
        self.test_label = y_test

        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label


        # standardize
        target_mean = np.mean(self.train_data)
        target_std = np.std(self.train_data)
        self.train_data = (self.train_data - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.train_data, self.train_label, self.testData, self.testLabel


    def train(self):

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=0)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=False, num_workers=0)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        averLoss = 0
        bestf1 = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        for e in range(self.n_epochs):
            torch.cuda.empty_cache()
            self.model.train()
            tick = 0
            for img, label in self.dataloader:

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                tok, outputs = self.model(img)
                # print("output")
                # print(outputs)
                # print("loss", tick)
                loss = self.criterion_cls(outputs, label)
                tick += 1

                # print(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # out_epoch = time.time()


            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                num = num + 1
                averAcc = averAcc + acc
                averLoss = averLoss + loss_test.detach().cpu().numpy()
                f1 = f1_score(test_label.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')
                if acc > bestAcc:
                    bestAcc = acc
                    bestf1 = f1
                    Y_true = test_label
                    Y_pred = y_pred

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        torch.save(self.model.state_dict(), 'model.pth')
        averAcc = averAcc / num
        averLoss = averLoss / num
        print('The average accuracy is:', averAcc)
        print('The average loss is:', averLoss)
        print('The best accuracy is:', bestAcc)
        print('The related f1 is:', bestf1)

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def main():
    best = 0
    aver = 0

    for i in range(13, 14):
        seed_n = np.random.randint(2023)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i+1))
        exp = ExP(i + 1)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))

        best = best + bestAcc
        aver = aver + averAcc


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
