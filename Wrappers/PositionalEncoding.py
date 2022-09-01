import torch.nn as nn
import numpy as np
import torch


class PositionalEncodingSuperPixel(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncodingSuperPixel, self).__init__()
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, seq_len, features)
        :return: Positional Encoding Matrix of size (batch_size, seq_len, features)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, seq, feat = tensor.shape
        pos_x = tensor[:, :, 0].type(self.inv_freq.type())
        pos_y = tensor[:, :, 1].type(self.inv_freq.type())

        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq) # batch, seq, feat/4
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq) # batch, seq, feat/4
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1) # batch, seq, feat/2
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1) # batch, seq, feat/2
        emb = torch.zeros((batch_size, seq, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb





class PositionalEncodingDict(nn.Module):
    def __init__(self, width, height, dim_head):
        """
        :param width: Width of dictionary
        :param height: Height of dictionary
        """
        super(PositionalEncodingDict, self).__init__()
        self.width = width
        self.height = height
        self.x_encodings = nn.Parameter(torch.randn(1, width, dim_head))
        self.y_encodings = nn.Parameter(torch.randn(1, height, dim_head))

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, seq_len, features)
        :return: Positional Encoding Matrix of size (batch_size, seq_len, features)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, seq, feat = tensor.shape
        pos_x = (tensor[:, :, 0]*self.width).type(torch.long).unsqueeze(2).repeat(1, 1, self.x_encodings.size(2)) # batch, seq_len, 1
        pos_y = (tensor[:, :, 1]*self.height).type(torch.long).unsqueeze(2).repeat(1, 1, self.x_encodings.size(2)) # batch, seq_len, 1

        x_encodings = self.x_encodings.repeat(batch_size, 1, 1) # batch, width, features
        y_encodings = self.y_encodings.repeat(batch_size, 1, 1) # batch, height, features

        emb_x = torch.gather(x_encodings, 1, pos_x) # batch, seq_len, features
        emb_y = torch.gather(y_encodings, 1, pos_y) # batch, seq_len, features

        return emb_x, emb_y