import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_angles(pos, i, emb_size):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(emb_size))
  return pos * angle_rates

def positional_encoding(position, emb_size):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(emb_size)[np.newaxis, :],
                          emb_size)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return torch.tensor(pos_encoding, dtype=torch.float32).to(device)


def create_padding_mask(prods):
    mask = (prods == 0).int()
    return mask.reshape(mask.shape[0], 1, 1, mask.shape[1])

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Size of each hidden vector of a head
        self.depth = self.args.emb_size // self.args.num_heads

        self.W_q = nn.Linear(self.args.emb_size, self.args.emb_size)
        self.W_k = nn.Linear(self.args.emb_size, self.args.emb_size)
        self.W_v = nn.Linear(self.args.emb_size, self.args.emb_size)

        # W for multi-head attention
        self.mha_W = nn.Linear(self.args.emb_size, self.args.emb_size)

    def split_heads(self, x):
        # (batch size, num tokens,  num heads, depth)
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1, self.args.num_heads, self.depth))
        
        # (batch size, num heads, num tokens, depth)
        return x.transpose(2,1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]

        q = self.split_heads(self.W_q(q))
        k = self.split_heads(self.W_k(k))
        v = self.split_heads(self.W_v(v))

        qk = torch.matmul(q, k.transpose(3,2))
        dk = torch.tensor(q.shape[-1], dtype=torch.float32)
        
        scaled_att_logits = qk / torch.sqrt(dk)
        if mask is not None:
            scaled_att_logits += (mask * -1e9)
            
        att_weights = F.softmax(scaled_att_logits, dim=-1)

        scaled_att = torch.matmul(att_weights, v)

        # (batch_size, invs+hidden+enc_state, num_heads, depth)
        scaled_att = scaled_att.transpose(2,1)

        # Squeeze MHA together for each embedding
        scaled_att = torch.reshape(scaled_att, (batch_size, -1, self.args.emb_size))

        # Combine the MHA for each embedding
        out = self.mha_W(scaled_att)
        return out, att_weights

class PointWiseFFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.emb_size, self.args.dff)
        self.fc2 = nn.Linear(self.args.dff, self.args.emb_size)

    def forward(self, x):
        pw_out = F.gelu(self.fc1(x))
        pw_out = self.fc2(pw_out)

        return pw_out


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mha = MultiHeadAttention(self.args)

        self.pw_ffn = PointWiseFFN(self.args)

        self.dropout1 = nn.Dropout(self.args.drop_rate)
        self.dropout2 = nn.Dropout(self.args.drop_rate)

        self.norm1 = nn.LayerNorm(self.args.emb_size, eps=self.args.eps)
        self.norm2 = nn.LayerNorm(self.args.emb_size, eps=self.args.eps)

    def forward(self, x, enc_padding_mask=None):
        # Perform MHA attention
        att_out, _ = self.mha(x, x, x, enc_padding_mask)
        att_out = self.dropout1(att_out)
        mha_out = self.norm1(x + att_out)

        # Plug through Point-wise FFN
        pw_ffn_out = self.pw_ffn(mha_out)
        pw_ffn_out = self.dropout2(pw_ffn_out)
        out = self.norm2(mha_out + pw_ffn_out)

        return out

class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.mha1 = MultiHeadAttention(self.args)
        self.mha2 = MultiHeadAttention(self.args)

        self.pw_ffn = PointWiseFFN(self.args)

        self.dropout1 = nn.Dropout(self.args.drop_rate)
        self.dropout2 = nn.Dropout(self.args.drop_rate)
        self.dropout3 = nn.Dropout(self.args.drop_rate)
        
        self.norm1 = nn.LayerNorm(self.args.emb_size, eps=self.args.eps)
        self.norm2 = nn.LayerNorm(self.args.emb_size, eps=self.args.eps)
        self.norm3 = nn.LayerNorm(self.args.emb_size, eps=self.args.eps)
    
    def forward(self, x, enc_output, enc_padding_mask=None, dec_padding_mask=None):
        # Perform self attention over available invs
        att_out, att_weights1 = self.mha1(x, x, x, dec_padding_mask)
        att_out = self.dropout1(att_out)
        mha_out1 = self.norm1(x + att_out)

        # Perform attention over encoder output and inv embs
        enc_att_out, att_weights2 = self.mha2(
            mha_out1, enc_output, enc_output, enc_padding_mask)
        enc_att_out = self.dropout2(enc_att_out)
        mha_out2 = self.norm2(mha_out1 + enc_att_out)

        pw_ffn_out = self.pw_ffn(mha_out2)
        pw_ffn_out = self.dropout3(pw_ffn_out)
        out = self.norm3(mha_out2 + pw_ffn_out)

        return out, att_weights1, att_weights2

class Encoder(nn.Module):
    def __init__(self, args, num_enc_layers=1):
        super().__init__()
        self.args = args
        
        #self.enc_layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(num_enc_layers)])

        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.args.emb_size,
                dim_feedforward=self.args.dff,
                activation="gelu",
                nhead=self.args.num_heads,
                dropout=self.args.drop_rate) 
            for _ in range(num_enc_layers)])
        
        self.dropout = nn.Dropout(self.args.drop_rate)
    
        self.pos_enc = positional_encoding(self.args.max_pos_enc, self.args.emb_size)

    def forward(self, x, enc_padding_mask=None):
        #print("torch.sqrt(torch.tensor(self.args.emb_size, dtype=torch.float32))", torch.sqrt(torch.tensor(self.args.emb_size, dtype=torch.float32)))
        #x = x * torch.sqrt(torch.tensor(self.args.emb_size, dtype=torch.float32))
        #x = x + self.pos_enc[:, :x.shape[1], :]
        x = self.dropout(x)

        # Run through all encoders
        for enc_layer in self.enc_layers:
            x = enc_layer(x, enc_padding_mask)
        
        # Return output of last encoder layer
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.pos_enc = positional_encoding(self.args.max_pos_enc, self.args.emb_size)

        self.dec_layers = nn.ModuleList([DecoderLayer(self.args) for _ in range(self.args.num_enc_layers)])
        
        self.dropout = nn.Dropout(self.args.drop_rate)
    
    def forward(self, x, enc_output, enc_padding_mask=None, dec_padding_mask=None):
        """Args:
            x: Tensor of inv embeddings.
            enc_output: Tensor of output of last encoder layer.
        """
        num_fulfill = x.shape[1]
        x = x * torch.sqrt(torch.tensor(self.args.emb_size, dtype=torch.float32))
        #x = x + self.pos_enc[:, :num_fulfill, :]
        x = self.dropout(x)
        
        att_weights = {}

        # Run through all decoder layers
        for i, dec_layer in enumerate(self.dec_layers):
            x, att_weights1, att_weights2 = dec_layer(
                x, enc_output, enc_padding_mask, dec_padding_mask)

            att_weights['decoder_inv{}_weights1'.format(i+1)] = att_weights1
            att_weights['decoder_inv{}_weights2'.format(i+1)] = att_weights2

        return x, att_weights


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = Encoder(self.args, self.args.num_enc_layers)
        self.decoder = Decoder(self.args)
        

    def forward(self, enc_input, dec_input, enc_padding_mask=None, dec_padding_mask=None):
        enc_output = self.encoder(enc_input, enc_padding_mask)
        
        dec_output, att_weights = self.decoder(
            dec_input, enc_output, enc_padding_mask, dec_padding_mask)
        return dec_output, att_weights