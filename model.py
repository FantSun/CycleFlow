import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


    
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal
    
    
    
class Encoder_r(nn.Module):
    """Rhythm Encoder
    """
    def __init__(self, hparams):
        super().__init__()

        self.dim_neck_r = hparams.dim_neck_r
        self.freq_r = hparams.freq_r
        self.dim_freq = hparams.dim_freq
        self.dim_enc_r = hparams.dim_enc_r
        self.chs_grp = hparams.chs_grp
        
        convolutions = []
        for i in range(1):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_freq if i==0 else self.dim_enc_r,
                         self.dim_enc_r,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_r//self.chs_grp, self.dim_enc_r))
            convolutions.append(conv_layer)
        self.convolutions_r = nn.ModuleList(convolutions)
        
        self.lstm_r = nn.LSTM(self.dim_enc_r, self.dim_neck_r, 1, batch_first=True, bidirectional=True)
        

    def forward(self, x, mask):
                
        for conv in self.convolutions_r:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm_r.flatten_parameters()
        outputs, _ = self.lstm_r(x)
        if mask is not None:
            outputs = outputs * mask
        out_forward = outputs[:, :, :self.dim_neck_r]
        out_backward = outputs[:, :, self.dim_neck_r:]
            
        codes = torch.cat((out_forward[:,self.freq_r-1::self.freq_r,:], out_backward[:,::self.freq_r,:]), dim=-1)

        return codes        
    
    
    
class Encoder_cf(nn.Module):
    """Sync Encoder module
    """
    def __init__(self, hparams):
        super().__init__()

        self.dim_neck_c = hparams.dim_neck_c
        self.freq_c = hparams.freq_c
        self.freq_f = hparams.freq_f
        self.dim_enc_c = hparams.dim_enc_c
        self.dim_enc_f = hparams.dim_enc_f
        self.dim_freq = hparams.dim_freq
        self.chs_grp = hparams.chs_grp
        self.register_buffer('len_org', torch.tensor(hparams.max_len_pad))
        self.dim_neck_f = hparams.dim_neck_f
        self.dim_f0 = hparams.dim_f0
        
        # convolutions for content
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_freq if i==0 else self.dim_enc_c,
                         self.dim_enc_c,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_c//self.chs_grp, self.dim_enc_c))
            convolutions.append(conv_layer)
        self.convolutions_c = nn.ModuleList(convolutions)
        
        self.lstm_c = nn.LSTM(self.dim_enc_c, self.dim_neck_c, 2, batch_first=True, bidirectional=True)
        
        # convolutions for pitch
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(self.dim_f0 if i==0 else self.dim_enc_f,
                         self.dim_enc_f,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.GroupNorm(self.dim_enc_f//self.chs_grp, self.dim_enc_f))
            convolutions.append(conv_layer)
        self.convolutions_f = nn.ModuleList(convolutions)
        
        self.lstm_f = nn.LSTM(self.dim_enc_f, self.dim_neck_f, 1, batch_first=True, bidirectional=True)
        
        self.interp = InterpLnr(hparams)

        
    def forward(self, x_f0):
        
        x = x_f0[:, :self.dim_freq, :]
        f0 = x_f0[:, self.dim_freq:, :]
        
        for conv_1, conv_2 in zip(self.convolutions_c, self.convolutions_f):
            x = F.relu(conv_1(x))
            f0 = F.relu(conv_2(f0))
            x_f0 = torch.cat((x, f0), dim=1).transpose(1, 2)
            x_f0 = self.interp(x_f0, self.len_org.expand(x.size(0)))
            x_f0 = x_f0.transpose(1, 2)
            x = x_f0[:, :self.dim_enc_c, :]
            f0 = x_f0[:, self.dim_enc_c:, :]
            
            
        x_f0 = x_f0.transpose(1, 2)    
        x = x_f0[:, :, :self.dim_enc_c]
        f0 = x_f0[:, :, self.dim_enc_c:]
        
        # code 1
        x = self.lstm_c(x)[0]
        f0 = self.lstm_f(f0)[0]
        
        x_forward = x[:, :, :self.dim_neck_c]
        x_backward = x[:, :, self.dim_neck_c:]
        
        f0_forward = f0[:, :, :self.dim_neck_f]
        f0_backward = f0[:, :, self.dim_neck_f:]
        
        codes_c = torch.cat((x_forward[:,self.freq_c-1::self.freq_c,:], 
                             x_backward[:,::self.freq_c,:]), dim=-1)
        
        codes_f = torch.cat((f0_forward[:,self.freq_f-1::self.freq_f,:], 
                              f0_backward[:,::self.freq_f,:]), dim=-1)
        
        return codes_c, codes_f      
    
    
    
class Decoder_S(nn.Module):
    """Decoder module
    """
    def __init__(self, hparams):
        super().__init__()
        self.dim_neck_c = hparams.dim_neck_c
        self.dim_neck_r = hparams.dim_neck_r
        self.dim_neck_f = hparams.dim_neck_f
        self.dim_emb = hparams.dim_spk_emb
        
        self.lstm = nn.LSTM(self.dim_neck_c*2+self.dim_neck_r*2+self.dim_neck_f*2+self.dim_emb, 
                            512, 3, batch_first=True, bidirectional=True)
        
        self.linear_projection = LinearNorm(1024, self.dim_freq)

    def forward(self, x):
        
        outputs, _ = self.lstm(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output          
    
    
    
class Decoder_P(nn.Module):
    """For F0 converter
    """
    def __init__(self, hparams):
        super().__init__()
        self.dim_neck_r = hparams.dim_neck_r
        self.dim_neck_f = hparams.dim_neck_f
        self.dim_f0 = hparams.dim_f0
        
        self.lstm = nn.LSTM(self.dim_neck_r*2+self.dim_neck_f*2, 
                            256, 2, batch_first=True, bidirectional=True)
        
        self.linear_projection = LinearNorm(512, self.dim_f0)

    def forward(self, x):
        
        outputs, _ = self.lstm(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output         
    
    

class Encoder_cyc(nn.Module):
    """Encoder"""
    def __init__(self, hparams):
        super().__init__()

        self.encoder_cf = Encoder_cf(hparams)
        self.encoder_r = Encoder_r(hparams)
        self.encoder_t = Encoder_t(hparams)

    def forward(self, x_f0, x_org):

        x_1 = x_f0.transpose(2,1)
        codes_c, codes_f = self.encoder_cf(x_1)

        x_2 = x_org.transpose(2,1)
        codes_r = self.encoder_r(x_2, None)

        return codes_r, codes_c, codes_f


class Decoder_cyc(nn.Module):
    """CycleFlow model"""
    def __init__(self, hparams):
        super().__init__()

        self.decoder_S = Decoder_S(hparams)
        self.decoder_P = Decoder_P(hparams)

        self.freq_c = hparams.freq_c
        self.freq_r = hparams.freq_r
        self.freq_f = hparams.freq_f
        self.freq_t = hparams.freq_t


    def forward(self, codes_t, codes_r, codes_c, codes_f, mode="test"):

        code_exp_c = codes_c.repeat_interleave(self.freq_c, dim=1) # content
        code_exp_f = codes_f.repeat_interleave(self.freq_f, dim=1) # pitch
        code_exp_r = codes_r.repeat_interleave(self.freq_r, dim=1) # rhythm
        code_exp_t = codes_t.repeat_interleave(self.freq_t, dim=1) # timbre

        Z_S = torch.cat((code_exp_c, code_exp_f, code_exp_r, code_exp_t), dim=-1)
        mel_outputs = self.decoder_S(Z_S)

        if mode == 'train':
            Z_P = torch.cat((code_exp_r, code_exp_f), dim=-1)
            f0_outputs = self.decoder_P(Z_P)
            return mel_outputs, f0_outputs
        else:
            return mel_outputs



class InterpLnr(nn.Module):
    
    def __init__(self, hparams):
        super().__init__()
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad
        
        self.min_len_seg = hparams.min_len_seg
        self.max_len_seg = hparams.max_len_seg
        
        self.max_num_seg = self.max_len_seq // self.min_len_seg + 1
        
        
    def pad_sequences(self, sequences):
        channel_dim = sequences[0].size()[-1]
        out_dims = (len(sequences), self.max_len_pad, channel_dim)
        out_tensor = sequences[0].data.new(*out_dims).fill_(0)
        
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, :] = tensor[:self.max_len_pad]
            
        return out_tensor 
    

    def forward(self, x, len_seq):  
        
        if not self.training:
            return x
        
        device = x.device
        batch_size = x.size(0)
        
        # indices of each sub segment
        indices = torch.arange(self.max_len_seg*2, device=device)\
                  .unsqueeze(0).expand(batch_size*self.max_num_seg, -1)
        # scales of each sub segment
        scales = torch.rand(batch_size*self.max_num_seg, 
                            device=device) + 0.5
        
        idx_scaled = indices / scales.unsqueeze(-1)
        idx_scaled_fl = torch.floor(idx_scaled)
        lambda_ = idx_scaled - idx_scaled_fl
        
        len_seg = torch.randint(low=self.min_len_seg, 
                                high=self.max_len_seg, 
                                size=(batch_size*self.max_num_seg,1),
                                device=device)
        
        # end point of each segment
        idx_mask = idx_scaled_fl < (len_seg - 1)
       
        offset = len_seg.view(batch_size, -1).cumsum(dim=-1)
        # offset starts from the 2nd segment
        offset = F.pad(offset[:, :-1], (1,0), value=0).view(-1, 1)
        
        idx_scaled_org = idx_scaled_fl + offset
        
        len_seq_rp = torch.repeat_interleave(len_seq, self.max_num_seg)
        idx_mask_org = idx_scaled_org < (len_seq_rp - 1).unsqueeze(-1)
        
        idx_mask_final = idx_mask & idx_mask_org
        
        counts = idx_mask_final.sum(dim=-1).view(batch_size, -1).sum(dim=-1)
        
        index_1 = torch.repeat_interleave(torch.arange(batch_size, 
                                            device=device), counts)
        
        index_2_fl = idx_scaled_org[idx_mask_final].long()
        index_2_cl = index_2_fl + 1
        
        y_fl = x[index_1, index_2_fl, :]
        y_cl = x[index_1, index_2_cl, :]
        lambda_f = lambda_[idx_mask_final].unsqueeze(-1)
        
        y = (1-lambda_f)*y_fl + lambda_f*y_cl
        
        sequences = torch.split(y, counts.tolist(), dim=0)
       
        seq_padded = self.pad_sequences(sequences)
        
        return seq_padded    
