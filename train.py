import os
import torch

from myDataset import myDataset as myDataset
from torch import nn
from myModel import myCNN
import numpy as np
from torch.utils.data import DataLoader
# from metric import f1_score, classification_report, confusion_matrix
from tensorboardX import SummaryWriter
import time

def train(net, gpu, position, feature, obj, ith, traindata, trainlabel, testdata, testlabel, EPOCH, LR, WD, DROP, BATCH, adjustlr, optimizername, day):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    max_test_acc = 0
    train_ls, test_ls = [], []
    train_acc, test_acc = [], []
    train_f1, test_f1 = [], []

    # writer = SummaryWriter('runs/' + position + '/' + feature + '/' + 'fold' + str(k+1) + '/s' + str(obj) + '_' + str(time.strftime("%m-%d-%H-%M", time.localtime())))
    writer = SummaryWriter('runs/'+position+'/'+feature+'/fold' + str(ith + 1) + '/D' + str(day) + '_' + net.modelname+'_s' + str(obj) + '_' + str(
        time.strftime("%m-%d-%H-%M", time.localtime())))
    data_read_train = myDataset(traindata, testdata, trainlabel, testlabel, train=True)
    loader_train = DataLoader(dataset=data_read_train, batch_size=BATCH, shuffle=True, drop_last=True)
    data_read_test = myDataset(traindata, testdata, trainlabel, testlabel, train=False)
    loader_test = DataLoader(dataset=data_read_test, batch_size=BATCH, shuffle=False, drop_last=True)#drop_last在Batch比较大的时候容易把最后一个标签的样本都丢了

    if torch.cuda.is_available():
        print('available')

    model = net.cuda()
    # print(model)

    if optimizername == 'SGD':
        print('Use SGD')
        optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=WD, momentum=0.9)  # optimize all parameters
    else:
        print('Use Adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    loss_func = nn.CrossEntropyLoss()
    traintime = 0
    for epoch in range(EPOCH):
# 训练阶段
        total_acc_train = 0
        total_f1_train = 0
        total_loss_train = 0
        class_correct = list(0. for i in range(17))
        class_total = list(0. for i in range(17))
        model.train()
        for step, (x, y) in enumerate(loader_train):  # gives batch data
            # 整理x, y成为模型输入所需的shape和type: x->(b, 409, 6), y->(64, 1)
            if net.modelname == 'cnn':
                x = torch.Tensor(np.array(x, dtype='float32'))
                x = torch.unsqueeze(x, 1)  # x(b,1,200,6)
            if net.modelname == 'tcn':
                x = torch.Tensor(np.transpose(np.array(x, dtype='float32'), (0, 2, 1)))
            if net.modelname == 'lstm':
                h0 = torch.randn(net.num_layers, BATCH, net.hidden_size)
                c0 = torch.randn(net.num_layers, BATCH, net.hidden_size)
                # h0 = torch.Tensor(np.random.uniform(0, 0.05, (net.num_layers, BATCH, net.hidden_size))) # 3层，每层128个，第二个维度是batch_size
                # c0 = torch.Tensor(np.random.uniform(0, 0.05, (net.num_layers, BATCH, net.hidden_size)))
                x = torch.Tensor(np.array(x, dtype='float32'))
            if net.modelname == 'gru':
                x = torch.Tensor(np.array(x, dtype='float32'))
            if net.modelname == 'transformer':
                x = torch.Tensor(np.transpose(np.array(x, dtype='float32'), (0, 2, 1)))
                x = torch.unsqueeze(x, 1)  # x(b,1,200,6)
            y = y.type(torch.LongTensor)
            y = torch.squeeze(y)

            # 拟合数据得到模型输出
            if net.modelname == 'lstm':
                output, (h_state, c_state), feat = model(x.cuda(), h0.cuda(), c0.cuda())
            else:
                s = time.time()
                output, feat = model(x.cuda())
            pred_y = torch.max(output, 1)[1]

            # 计算acc和f1
            # c = torch.eq(pred_y, y.data.cuda())
            # total_acc_train += (c.sum()) / float(y.size(0))
            # for i in range(BATCH):
            #     label = y[i]
            #     label = label.int()
            #     class_correct[label] += c[i].item()
            #     class_total[label] += 1
            # total_f1_train += f1_score(y.cpu(), pred_y.cpu())

            # 计算loss, bp
            loss_train = loss_func(output, y.cuda())
            total_loss_train += loss_train
            optimizer.zero_grad()  # clear gradients for this training step
            loss_train.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients to update weights
            e = time.time()
            traintime += (e-s)
            print(e-s)
    print(traintime)
    #     avr_acc_train = total_acc_train / (step + 1)
    #     avr_f1_train = total_f1_train / (step + 1)
    #     avr_loss_train = total_loss_train / (step + 1)
        # print("Epoch: %d, trainAcc: %.2f %%" % (epoch + 1, avr_acc_train * 100))
        # for i in range(17):
        # #     print('Train accuracy of %d : %.2f %%' % (i+1, (class_correct[i] / (class_total[i]))*100))
        #     writer.add_scalar('train/Movement' + str(i + 1) + 'acc', class_correct[i] / (class_total[i]), epoch)
        # writer.add_scalar('train/Accurancy', avr_acc_train, epoch)
        # train_acc.append(avr_acc_train.item())
        # writer.add_scalar('train/F1', avr_f1_train, epoch)
        # train_f1.append(avr_f1_train)
        # writer.add_scalar('train/Loss', avr_loss_train, epoch)
        # train_ls.append(avr_loss_train.item())

# 调整学习率
#         if adjustlr == 1:
#             if epoch+1 == EPOCH/2:
#                 for p in optimizer.param_groups:
#                     p['lr'] *= 0.1
#                     LR = p['lr']
#
#         writer.add_scalar('Hyperpram/LR', LR, epoch)
#         writer.add_scalar('Hyperpram/BATCH', BATCH, epoch)
#         writer.add_scalar('Hyperpram/WD', WD, epoch)
#         writer.add_scalar('Hyperpram/DROPOUT', DROP, epoch)
#
# # 测试阶段
#         total_acc_test = 0
#         total_f1_test = 0
#         total_loss_test = 0
#         sum_time = 0
#         class_correct_test = list(0. for i in range(17))
#         class_total_test = list(0. for i in range(17))
#         model.eval()
#         with torch.no_grad():
#             for step, (x, y) in enumerate(loader_test):
#                 # 整理x, y成为网络需要的shape和type
#                 if net.modelname == 'cnn':
#                     # x = torch.Tensor(np.transpose(np.array(x, dtype='float32'), (0, 2, 1)))
#                     x = torch.Tensor(np.array(x, dtype='float32'))
#                     x = torch.unsqueeze(x, 1)  # x(b,1,200,6)
#                 if net.modelname == 'tcn':
#                     x = torch.Tensor(np.transpose(np.array(x, dtype='float32'), (0, 2, 1)))
#                 if net.modelname == 'lstm':
#                     torch.Tensor(np.random.uniform(0, 0.05, (net.num_layers, BATCH, net.hidden_size)))  # 3层，每层128个，第二个维度是batch_size
#                     torch.Tensor(np.random.uniform(0, 0.05, (net.num_layers, BATCH, net.hidden_size)))
#                     x = torch.Tensor(np.array(x, dtype='float32'))
#                 if net.modelname == 'gru':
#                     x = torch.Tensor(np.array(x, dtype='float32'))
#                 if net.modelname == 'transformer':
#                     x = torch.Tensor(np.transpose(np.array(x, dtype='float32'), (0, 2, 1)))
#                     x = torch.unsqueeze(x, 1)  # x(b,1,200,6)
#                 y = y.type(torch.LongTensor)
#                 y = torch.squeeze(y)
#
#                 # 拟合数据得到模型输出
#                 if net.modelname == 'lstm':
#                     output_test, (h_state, c_state), feat_test = model(x.cuda(), h0.cuda(), c0.cuda())
#                 else:
#                     start = time.time()
#                     output_test, feat_test = model(x.cuda())
#                     end = time.time()
#                     sum_time += (end-start)
#                     # print("infer time:", end-start)
#                 pred_y_test = torch.max(output_test, 1)[1]
#
#                 # 计算acc，f1, loss
#                 c_test = (torch.eq(pred_y_test, y.data.cuda()))
#                 total_acc_test += (c_test.sum()) / float(y.size(0))
#                 for i in range(BATCH):
#                     label_test = y[i]
#                     label_test = label_test.int()
#                     class_correct_test[label_test] += c_test[i].item()
#                     class_total_test[label_test] += 1
#                 total_f1_test += f1_score(y.cpu(), pred_y_test.cpu())
#                 loss_test = loss_func(output_test, y.cuda())
#                 total_loss_test += loss_test
#             avr_acc_test = total_acc_test / (step + 1)
#             avr_f1_test = total_f1_test / (step + 1)
#             avr_loss_test = total_loss_test / (step + 1)
#             print("Epoch: %d, testAcc: %.2f %%" % (epoch + 1, avr_acc_test * 100))
#             infer = sum_time/(step+1)
#             print("average infer time:", infer)
#             for i in range(17):
#                 # print('Test accuracy of %d : %.2f %%' % (i+1, (class_correct_test[i] / (class_total_test[i]))*100))
#                 writer.add_scalar('test/Movement' + str(i + 1) + 'acc', class_correct_test[i] / (class_total_test[i]),epoch)
#             writer.add_scalar('test/Accurancy', avr_acc_test, epoch)
#             test_acc.append(avr_acc_test.item())
#             writer.add_scalar('test/F1', avr_f1_test, epoch)
#             test_f1.append(avr_f1_test)
#             writer.add_scalar('test/Loss', avr_loss_test, epoch)
#             test_ls.append(avr_loss_test.item())
#
#             # 保存testacc最大时对应的模型参数
#             if epoch + 1 >= EPOCH / 2:
#                 if avr_acc_test > max_test_acc:
#                     max_test_acc = avr_acc_test
#                     print('max_test_acc:', max_test_acc, 'epoch:', epoch)
#                     torch.save(model, 'save/D' + str(day) + '_s' + str(obj) + '_' + net.modelname + position + '_fold' + str(ith+1) + '.pkl')
#             writer.add_scalar('test/Maxtestacc', max_test_acc, epoch)

    return train_ls, train_acc, train_f1, test_ls, test_acc, test_f1
