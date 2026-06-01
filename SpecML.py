import torch.nn as nn
import torch.nn.functional as F


#------------------------------------------HYPER PARAMETERS----------------------------------------------------#
#these will change the size of the model
D_emb = 64 #Embedding Dimension 
n_heads = 4 #The number of parallel attentions you do at the same time 
n_layers = 4  #The number of times a block is stacked, each time different weights will be learnt 
ffn_dim = 4 * D_emb #Dimension of Feed Forward Network, FFN is how the vectors communicate within the vectors 

#------------------------------------------ATTENTION----------------------------------------------------#

class SpectralAttention(nn.Module):
    def __init__(self, d, h): #d and h defined in SpecML class
        super().__init__()
        assert d % h == 0 #Error if not True
        self.h, self.dh = h, d // h #self.h = h (number of heads), self.dh = d//h (dimension of heads)
        self.qkv = nn.Linear(d, 3 * d) # Matrix 1 (learnt qkv matrix) QKV are all defined in one tensor. Linearised by taking something with dimension d and returning 3*d to account for Q K and V.
        self.out = nn.Linear(d, d) #Matrix 2 (learnt out matrix) Transforms output from d to d 
        # Local positional embedding: per-head, per-channel 3-tap filter.
        self.local = nn.Conv1d(self.dh, self.dh, 3, padding=1, groups=self.dh, bias=False)

    def forward(self, x, validity):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)  # [B, H, T, dh]
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)

        # Depthwise conv on V, applied independently per head.
        v_flat = v.reshape(B * self.h, T, self.dh).transpose(1, 2)  # [B*H, dh, T]
        local = self.local(v_flat).transpose(1, 2).view(B, self.h, T, self.dh)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=validity[:, None, None, :]
        )  # [B, H, T, dh]
        y = y + local
        y = y * validity[:, None, :, None].to(x.dtype)  # zero invalid queries

        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out(y)
    


#------------------------------------------SPECTRAL TRANSFORMER BLOCK----------------------------------------------------#

class SpectralBlock(nn.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d) #Layer norm twice for individual trained parameters that learn differently in each LN to be applied to different sections
        self.attn = SpectralAttention(d, h)
        self.ln2 = nn.LayerNorm(d) 
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d)) #Feed Forward Network, GELU more suited for Transformer Architecture
    def forward(self, x, validity): #Call spectral block, will then call the forward and it will add the attn from ln1 only with valid tokens to x and also apply the ffn to x
        x = x + self.attn(self.ln1(x), validity) 
        x = x + self.ffn(self.ln2(x)) 
        return x



#------------------------------------------THE MODEL: SpecML----------------------------------------------------#

class SpecML(nn.Module): 
    def __init__(self, patch_dim, d = D_emb, h = n_heads, n_layers = n_layers, ff = ffn_dim): #avengers assemble
        super().__init__() 
        self.embed = nn.Linear(patch_dim, d) #takes in dimensions from patch_dim and gives something with dimensions d_emb
        nn.init.trunc_normal_(self.embed.weight, mean=0.0, std=1 / d, a=-3 / d, b=3 / d)
        self.blocks = nn.ModuleList([SpectralBlock(d, h, ff) for _ in range(n_layers)]) #.blocks is a list of n_layer spectral blocks 
        self.norm = nn.LayerNorm(d) #normalises all the vectors in that matrix to ensure everything is in the same range
        self.head = nn.Linear(d, patch_dim) #this is the decoder

    def _encode(self, X, V, P): #function for encoding that will take in X and perform the embedding to create x, applying the validity mask 
        x = self.embed(X) + P 
        for blk in self.blocks:
            x = blk(x, V) #pass x through all the self.blocks we made above 
        return self.norm(x)  # [B, T, D] we want to normalise x as it stablises training more

    def forward(self, X, V, P): #march on, call forward which will call encode etc
        return self.head(self._encode(X, V, P))

    def encode(self, X, V, P):
        x = self._encode(X, V, P)  # [B, T, D]
        mask = V.unsqueeze(-1).to(x.dtype)  # [B, T, 1]
        return (x * mask).sum(dim=1) / mask.sum(dim=1)  # [B, D]



