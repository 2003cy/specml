import re as _re
import torch
import torch.nn as nn
import torch.nn.functional as F



#------------------------------------------HYPER PARAMETERS----------------------------------------------------#
#these will change the size of the model
D_emb = 384 #Embedding Dimension
n_heads = 8 #The number of parallel attentions you do at the same time
n_layers = 8  #The number of times a block is stacked, each time different weights will be learnt
ffn_dim = 4 * D_emb #Dimension of Feed Forward Network, FFN is how the vectors communicate within the vectors
DROPOUT = 0.0  # dropout probability applied after attention and FFN residuals

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
    def __init__(self, d, h, ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d) #Layer norm twice for individual trained parameters that learn differently in each LN to be applied to different sections
        self.attn = SpectralAttention(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d)) #Feed Forward Network, GELU more suited for Transformer Architecture
        self.drop = nn.Dropout(dropout)

    def forward(self, x, validity): #Call spectral block, will then call the forward and it will add the attn from ln1 only with valid tokens to x and also apply the ffn to x
        x = x + self.drop(self.attn(self.ln1(x), validity))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x



#------------------------------------------THE MODEL: SpecML----------------------------------------------------#

class SpecML(nn.Module):
    def __init__(self, patch_dim, d=D_emb, h=n_heads, n_layers=n_layers, ff=ffn_dim, dropout=DROPOUT,
                 patch_size=None, overlap=None): #avengers assemble
        super().__init__()
        self.embed = nn.Linear(patch_dim, d) #takes in dimensions from patch_dim and gives something with dimensions d_emb
        nn.init.trunc_normal_(self.embed.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)
        self.blocks = nn.ModuleList([SpectralBlock(d, h, ff, dropout) for _ in range(n_layers)]) #.blocks is a list of n_layer spectral blocks
        self.norm = nn.LayerNorm(d) #normalises all the vectors in that matrix to ensure everything is in the same range
        self.head = nn.Linear(d, patch_dim) #this is the decoder
        # Store training config as a non-gradient buffer so it's saved inside every .pt/.ckpt
        # Layout: [patch_dim, patch_size, overlap, D_emb, n_heads, n_layers, ffn_dim]
        ps = patch_size if patch_size is not None else (patch_dim - 2)
        ol = overlap    if overlap    is not None else 0
        self.register_buffer('_config', torch.tensor(
            [patch_dim, ps, ol, d, h, n_layers, ff], dtype=torch.int64
        ))

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


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────────────────────────────────────

def load_specml(path, device='cpu'):
    """
    Load any SpecML checkpoint and return (model, cfg).

    All architecture and tokenisation parameters are read from the _config
    buffer stored inside the checkpoint.  For older checkpoints that pre-date
    this buffer, the architecture is inferred from the weight shapes and the
    overlap is parsed from the filename (e.g. '10PS 4OL.pt').

    cfg keys: patch_dim, patch_size, overlap, step, D_emb, n_heads, n_layers, ffn_dim
    """
    raw   = torch.load(path, map_location=device, weights_only=False)
    # Support both flat state-dicts and Lightning-style {'state_dict': ...} wrappers
    state = raw.get('state_dict', raw) if isinstance(raw, dict) else raw

    if '_config' in state:
        # New-style checkpoint: params are self-described
        patch_dim, ps, ol, d, h, nl, ff = [int(x) for x in state['_config'].tolist()]
        model = SpecML(patch_dim=patch_dim, d=d, h=h, n_layers=nl, ff=ff,
                       patch_size=ps, overlap=ol)
        model.load_state_dict(state)
    else:
        # Old-style checkpoint: infer architecture from weight shapes
        patch_dim = int(state['embed.weight'].shape[1])
        d         = int(state['embed.weight'].shape[0])
        nl        = sum(1 for k in state if k.startswith('blocks.') and k.endswith('.ln1.weight'))
        dh        = int(state['blocks.0.attn.local.weight'].shape[0])
        h         = d // dh
        ff        = int(state['blocks.0.ffn.0.weight'].shape[0])
        ps        = patch_dim - 2
        m         = _re.search(r'(\d+)\s*[Oo][Ll]', str(path))
        ol        = int(m.group(1)) if m else 0
        model = SpecML(patch_dim=patch_dim, d=d, h=h, n_layers=nl, ff=ff,
                       patch_size=ps, overlap=ol)
        model.load_state_dict(state, strict=False)   # _config key absent in old checkpoint

    model.to(device).eval()
    cfg = dict(
        patch_dim  = patch_dim,
        patch_size = ps,
        overlap    = ol,
        step       = ps - ol,
        D_emb      = d,
        n_heads    = h,
        n_layers   = nl,
        ffn_dim    = ff,
    )
    return model, cfg



