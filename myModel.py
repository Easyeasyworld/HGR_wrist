import torch
import torch.nn as nn
from tcn import TemporalConvNet as TCN
from TEMGNET import VisionTransformer


class myCNN(nn.Module):
    def __init__(self, batch, drop, in_channels):
        super(myCNN, self).__init__()
        self.batch = batch
        self.drop = drop
        self.in_channels = in_channels

        self.modelname = 'cnn'

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            # nn.LayerNorm([20, 7]),
            nn.ReLU(),
            nn.Dropout(self.drop),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            # nn.LayerNorm([20, 7]),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2),
            nn.BatchNorm2d(64),
            # nn.LayerNorm([18, 5]),
            nn.ReLU(),
            nn.Dropout(self.drop),
        )
        self.fc1 = nn.Linear(64*203*1, 17)
        self.drop1 = nn.Dropout(self.drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        feat = torch.flatten(x, start_dim=1, end_dim=3) #与x = x.view(x.size(0), -1)一个效果
        # print(x.shape)
        x = x.view(self.batch, -1)
        x = self.fc1(x)
        x = self.drop1(x)
        return x, feat

class mytcn(nn.Module):
    def __init__(self, num_inputs, num_channels, winsize, dropout, batch, kernel_size=3):
        super(mytcn, self).__init__()
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.drop = dropout
        self.batch = batch

        self.modelname = 'tcn'

        self.tcn = TCN(self.num_inputs, self.num_channels, self.kernel_size, self.drop)
        self.fc = nn.Linear(17*winsize, 17)
        self.drop = nn.Dropout(self.drop)

    def forward(self, x):
        x = self.tcn(x)
        feat = torch.flatten(x, start_dim=1, end_dim=2)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.drop(x)
        return x, feat

class mylstm(nn.Module):
    def __init__(self, DROPOUT, in_channels, hidden_size, num_layers):
        super(mylstm, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = DROPOUT

        self.modelname = 'lstm'

        self.lstm = nn.LSTM(
            input_size=self.in_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.drop,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidden_size*200, 17)

    def forward(self, inputs, h0, c0):
        '''Inputs have to have dimension （seq_len, batch, input_size)'''
        # lstm_out (seq_len, batch_size, hidden_size)
        lstm_out, (ht, ct) = self.lstm(inputs, (h0, c0))
        out = self.linear(lstm_out.reshape(lstm_out.shape[0], lstm_out.shape[1]*lstm_out.shape[2]))
        return out, (ht, ct)

class mygru(nn.Module):
    def __init__(self, DROPOUT, in_channels, hidden_size, num_layers):
        super(mygru, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop = DROPOUT

        self.modelname = 'gru'

        self.gru = nn.GRU(
            input_size=self.in_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.drop,
            batch_first=True
        )
        self.fc1 = nn.Linear(self.hidden_size, 17)
        self.drop1 = nn.Dropout(self.drop)
        # self.fc2 = nn.Linear(64, 17)
        # self.drop2 = nn.Dropout(self.drop)

    def forward(self, inputs):
        '''Inputs have to have dimension （seq_len, batch, input_size)'''
        # gru_out (seq_len, batch_size, hidden_size)
        gru_out, ht = self.gru(inputs) #gru_out(b, seq_len, hidden_size), ht(num_layers, b, hidden_size)
        # tmp = torch.equal(torch.squeeze(gru_out[:,-1,:]), torch.squeeze(ht[-1])) #运行结果：True
        out = self.fc1(ht[-1]) #这样是只拿最后一个时间步的输出做分类（相当于相信gru最后一个时间步能包含前面所有时间步的信息），像上面lstm那种是用所有时间步的输出展平做分类，也有道理，这样可能信息更多
        out = self.drop1(out)
        # out = self.fc2(out)
        # out = self.drop2(out)
        return out, ht[-1]

def get_net(batch, drop, in_channels, winsize, modelname):
    if modelname == 'cnn':
        net = myCNN(batch, drop, in_channels)
    if modelname == 'tcn':
        net = mytcn(in_channels, [32, 64, 17], winsize, drop, batch)
    if modelname == 'lstm':
        net = mylstm(drop, in_channels, 128, 3)
    if modelname == 'gru':
        net = mygru(drop, in_channels, 32, 2)
    if modelname == 'transformer':
        net = VisionTransformer(drop_ratio=drop, attn_drop_ratio=drop, drop_path_ratio=0.)
    # c初始化参数
    # for param in net.parameters():
    #     # nn.init.normal_(param, mean=0, std=0.01)
    #     torch.nn.init.xavier_uniform_(param, gain=torch.nn.init.calculate_gain('relu'))
    #     # p = param
    # for m in net.modules():
    #     if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
    #         torch.nn.init.xavier_uniform_(m.weight, gain = torch.nn.init.calculate_gain('relu'))
    return net