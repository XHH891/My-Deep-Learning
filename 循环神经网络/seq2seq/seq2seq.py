import collections
import math
import torch
from torch import nn
import encoder_decoder

def grad_clipping(net, theta): #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class Seq2SeqEncoder(encoder_decoder.Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0, **kwargs):
        super(Seq2SeqEncoder,self).__init__(**kwargs)
        self.embedding= nn.Embedding(vocab_size,embed_size)
        self.rnn= nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)
    def forward(self,X, *args):
        X= self.embedding(X)
        X= X.permute(1, 0, 2)
        output,state = self.rnn(X)
        return output,state

class Seq2SeqDecoder(encoder_decoder.Decoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0, **kwargs):
        super(Seq2SeqDecoder,self).__init__(**kwargs)
        self.embedding= nn.Embedding(vocab_size,embed_size)
        self.rnn= nn.GRU(embed_size+ num_hiddens,num_hiddens,num_layers,dropout=dropout)
        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs,*args):
        return enc_outputs[1]

    def forward(self,X,state):
        X= self.embedding(X).permute(1, 0,2)
        context= state[-1].repeat(X.shape[0], 1,1)
        X_and_context= torch.cat((X,context), 2)
        output,state = self.rnn(X_and_context,state)
        output= self.dense(output).permute(1, 0,2)
        return output,state

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self,pred,label,valid_len):
        weights= torch.ones_like(label)
        weights= sequence_mask(weights,valid_len)
        self.reduction='none'
        unweighted_loss=super(MaskedSoftmaxCELoss,self).forward(pred.permute(0, 2,1),label)
        weighted_loss= (unweighted_loss*weights).mean(dim=1)
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    # 损失函数的标量进⾏"反向传播"
    grad_clipping(net, 1)
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(num_epochs):
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            optimizer.step()
    print(f'tokens/sec on {str(device)}')