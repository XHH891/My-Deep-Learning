import torch
import text_dataset
import train
from torch import nn
from torch.nn import functional as F

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)
    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state
    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,batch_size, self.num_hiddens),device=device)
        else:
            return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

def try_gpu():
    """如果GPU可用，返回第一个GPU设备；否则返回CPU。"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')  # 使用第一个GPU
    return torch.device('cpu')

batch_size,num_steps = 32,35
train_iter, vocab = text_dataset.load_data_time_machine(batch_size,num_steps)

vocab_size,num_hiddens,num_layers = len(vocab),256,10
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs,num_hiddens,num_layers)

device = try_gpu()
net = RNNModel(lstm_layer, vocab_size=len(vocab))
net = net.to(device)
num_epochs, lr = 500, 1
train.train_ch8(net, train_iter, vocab, lr, num_epochs,device)
