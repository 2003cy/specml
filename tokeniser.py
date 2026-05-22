import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from astropy.table import Table
from astropy.io import fits
from astropy import units as u

import torch
import torch as t 
from torch import nn as nn
from torch import optim as optim
import torch.utils.data as data
import math
import copy


#------------------------------------------CONFIGURATION----------------------------------------------------#

#can test for one or two spectrums to test that its wokring for the big batch 

data = Table.read(f'dja_msaexp_emission_lines_v4.4.prism_spectra.fits')

#for flux values current dims after running this section amount of spectrum,473
f = data['flux']
f = np.array(f)
#f = f[:, :amount of spectrum] #f values were matrix of all spectrums so we only wanted to take 100 on the columns of the matrix
f = f.T #this transposes the matrix to make batch the first dimension for using pytorch ltr
#dimension = rows,columns = B,L = (amount of spectrum,473) where B is amount of spectra/batches and L is observation length of flux array

#for wavelengths
w = data['wave'] #w is not a matrix bc maybe same wavelength values for the grating
#dims = L = (473)

f /= w**2 #units
f_normalised = f / np.nanmean(f,axis=1,keepdims=True) # flux_conversion(data['flux']) #is this the correct normalising, in paper when they normalise they subtract mean and divide by std
#this is now changed because before it was a vector now is a matrix, so we want ot make sure we are doing the division just to the rows which is axis =1 
#dimensions of f_normalised = amount of spectrum,473

# --------------- making the big for loop ------------------ # 

####creating overlapping patches 
patch_size = 20
overlap = 10
L = f_normalised.shape[1] #to get the 473 for length
B = f_normalised.shape[0] #to get the amount of spectrum for length

step_size = patch_size - overlap 

#this does the for loop but faster 
pad_length = patch_size - L % step_size
f_normalised = np.append(f_normalised, np.ones((B,pad_length))*np.nan, axis=1) #B, pad_length makes it pad only to the columns (473), axis = 1 applys the np.append along the L dimension which is along the 473


x_t = sliding_window_view(f_normalised, patch_size, axis = 1)[:,::step_size] #(amount of spectrum,48,20) amount of spectrum (34000) is B, 48 is how many patches w fit into the thing which is t and 20 is patch_size
X = np.concatenate([np.nanmean(x_t,axis=2, keepdims = True), np.nanstd(x_t,axis=2, keepdims = True), x_t], axis=2) #adding mean and std to create matrix, axis = 2 adds to flux values
# print(X.shape) dimensions B x T x p+2, 34949, 48, 22


# print(f_normalised.shape, x_t.shape, X.shape)
# exit()

#------------------------------------------CONTIGUOUS CHUNK MASKING SCHEME----------------------------------------------------#
#define mask ratio gamma is either 1 or 0 where mask ratio is 0.75
mask_ratio = 0.75
w = 2.5 * patch_size #patch size is where it is mathematically failing
T = X.shape[1]#amount of x_t patches in X, 1 returns the rows which will be the amount of vectors x_t since the vectors x_t are the rows, 0 would have returned B which is the 2 dimensional stacking of the matrix for each spectra
N = int(T//(2*w)) #N is the number if chunks we take from X with width w and non-overlapping
gamma = int(mask_ratio * N) #randomly selecting M amount of non overlapping chunks to mask
# M is a matrix of dimensions B x T (this is the shape), not including p+2 in dimensions because we apply patch to everything in xt so it doesnt matter the dimension of x_t
random_masked = np.random.choice(N, gamma, replace = False) 
M = np.zeros((X.shape[0],T)) #ones by default matrix for M, give it the shape as tuples, we get B dimension from the X matrix
M[:, random_masked] = 1 

print(random_masked)


#define X_tilda as X after masking


#-------------- need to build the binary mask -----------------#

#revisit below we might have done this wrong
#this is so that for x_t vectors that include 50% nans or more, the model knows these are invalid

threshold = 0.00 #this allows the mask to mark a patch as invalid if it has any nans

V = (np.isnan(X).sum(axis=2) / patch_size) >= threshold #the mask criteria put together, check the axis here ltr
 

#--------------- builting the wavelength embedding ------------------#

#need to pad wavelength array without adding nans

lam = np.append(w, w[-1] + np.arange(1, pad_length+1)*(w[-1] - w[-2])) #this linearly extrapolates wavelength values at the end of the array to meet the required pad_length

x_lam = sliding_window_view(lam, patch_size)[::step_size] #the patches for the wavelength array slightly overlapping

W_lam = np.mean(x_lam, axis=1, keepdims = True) #a vector of the means from the patches

#------------- sinousiodal wavelength encoder ----------------------#

# create p that is t long and d wide (define abritrary) each element in p must satisfy c onfitions in paper       
t = X.shape[1] #T
D_emb = 64 #properly define now, something smaller than embedding dimensions in paper 

P = np.zeros((B,t,D_emb)) #empty matrix of 0's of dimensions B t and D_emb


omegas = 10000 ** (- 2 * np.arange(D_emb // 2) / D_emb) #what is written in paper using int division


product = W_lam * omegas # this is from paper! added 10000, removed for now but why should we add it in again


# applied to each element
sines = np.sin(product)
coses = np.cos(product) 


even_mask = np.arange(D_emb) % 2 == 0 # np.arange(D_emb) makes (0, 1, 2,. .. , D_emb-1),  % 2 == 0 means make it True if it is divisible by two, otherwise it is False


# : means just do this for all of the B's and all of the T's
P[:, :,  even_mask] = sines[np.newaxis,...] #adds a batch dimension to sines and coses to shapes match up, numpy new axis adds a new dimension 1 to make it consistent because we are assigning it to something with a batch dimension so it needs to have something w that shape
P[:, :, ~even_mask] = coses[np.newaxis,...]


#---------------------------------------CREATING Z----------------------------------------------#

#put X through a linear layer

# linear_layer = nn.Linear(patch_size + 2, D_emb, bias=True) #no need to manually create a W matrix, the nn.linear creates and manages weight matrix internally 
# #takes input_features which we want to be the size of x_t and output_features which is size of projection or D_emb, at end of input to linear ', device=None, dtype=None' are defaults and i have dropped them to keep it clean 
# Z_x = linear_layer(X) # B x T x D_emb, includes all patches (x_t) from X as a new compressed element z_t with compression dimension D_emb into a matrix Z_x

# #initialising weights, no need to assign a variable to it because it uses pytorch
# nn.init.trunc_normal_(linear_layer.weight, mean=0.0, std=(1/D_emb), a =-3.0, b=3.0) #this allows for truncation on the normal distribution to -3 to 3sigma, also isolates weight from previously computed nn.linear layer
# #this would go in the intialisation in the class, what am i calling once
# #finally make Z, but Z_x is a pytorch tensor and P_global is a numpy 
# #need to convert P_global to pytorch tensor, Z will be inside the class

# class Embedder(nn.Module):
#     def __init__(self, patch_size, D_emb): #self applies it to this current instance
#         super(Embedder, self).__init__()
#         self.linear_layer = nn.Linear(patch_size + 2, D_emb, bias=True) #linear layer
#         nn.init.trunc_normal_(self.linear_layer.weight, mean=0.0, std=(1/D_emb), a =-3.0, b=3.0)

#     def forward(self, x):
#         return self.linear_layer(x)



#------------------------------------------TRANSFORMER BLOCK----------------------------------------------------#

#self attention because it was originally cross attention for like writing in english and chanigng it to french but self attention is like writing and reading at the same time
#self attention is like the vectors talking to eachother, like its talking to itself, moving information around between vectors
N_HEADS = 4 #how many parallel attention things you do at the same time
N_LAYERS = 4 #how many times the block is stacked, with different weights each time that are learnt 
FFN_DIM = D_emb #this is the same as ff? check spectral block, usually make ff bigger to move information around within vectors easier

class SpecML(nn.Module): #specML is the child of nn.Module so it inherits a whole bunch of code from it
    def __init__(self, d= D_emb, h=N_HEADS, n_layers=N_LAYERS, ff=FFN_DIM, patch_dim= patch_size + 2): #avengers assemble
        super().__init__() #super() mean to tell it to intiialise the parent, and init() is just to initialise on the parent
        self.embed = nn.Linear(patch_dim, d) #takes in dimensions from patch_dim and gives something with dimensions d_emb
        self.blocks = nn.ModuleList([SpectralBlock(d, h, ff) for _ in range(n_layers)]) #.blocks is a list of n_layer spectral blocks 
        self.norm = nn.LayerNorm(d) #layer norm allows all the inputs to be in the same range, and d is the dimension of the layer norm, dimension of vectors it normalises, normalises all the vectors in that matrix to ensure everything is in the same range
        #layernorm(d) only takes in d because the new dimensions of X after compression are of D_emb size vectors and is saying it is a normalisation inside each batch (vector) instead of across all batches which would be batch norm
        self.head = nn.Linear(d, patch_dim) #maybe this is the decoder or the unembedder?
        nn.init.trunc_normal_(self.embed.weight, mean=0.0, std=1/d, a=-3 / d, b=3 / d) #divide truncations by demb because in paper it is -3sigma and sigma is 1/demb... why is it not sigma squared in the ND

    def _encode(self, X, V, P): #function for encoding that will take in X and perform the embedding to create Z (labelled x in this case), applying the validity mask to ensure this is only being done to valid tokens and also using the sinusoidal function P to add it to Z(x) to make final Z
        x = self.embed(X) + P #this is making the Z from the paper
        for blk in self.blocks:
            x = blk(x, V) #pass x through all the self.blocks we made above and x is the Z
        return self.norm(x)  # [B, T, D] we want to normalise Z (or x) as it stablises training more, does the normalises vectors of size d, every vector inside of x is of size d


    def forward(self, X, V, P): #march on
        #when you want to call SpecML, you dont call encode, you call forward then forward will call encode and that is why we are not using little x hee, but the big X because when it calls encode it will make the little x
        # return self.head(self._encode(X, V, P))
        Z = self._encode(X, V, P) #Z is our latent vector 
        return self.head(Z),Z # i want to also return Z itself, self.head will do the decoder to turn Z back into X but also we wanted to return Z so returned Z as well
#so when you call SpecML and you call forward, it will tell it to go through the encode process which is defined prior, and the forward calls on the encode function and then it goes through the decode which is the .head so this forward is the autoencoder

class SpectralBlock(nn.Module):
    """Transformer Block Section"""
    def __init__(self, d, h, ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d) #layer norm is a trained parameter so we do layer norm twice... read layer norm paper, in the documentation this step has beta and gamma and they are learnt parameters that learn where it wants the mean and the variance to be 
        self.attn = SpectralAttention(d, h)
        self.ln2 = nn.LayerNorm(d) #a second one because it will be applied in different sections with the learn beta and gamma, each one initialisies a new beta and gamma that learn differently
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d)) # a ffn is a neural network, attention moves information between vectors and then the ffn moves information within vectors
        #sequential means apply all the following sequentially, GELU is a less sharp function than RELU function it is better for training because the gradient descent is how the nn updates itself, and gelu has nicer gradients than relu, and also it works better on transformers
        #ffn moves information within vectors d to ff, so it goes d to ff and then ff to d so the information is really only travelling d to d, which is wihtin the vector
    def forward(self, x, validity): #so now when we call spectral block, it calls the forward and it will add the attn from layer norm 1 only with valid tokens to x and also apply the ffn to x, this is exactly as seen in paper
        x = x + self.attn(self.ln1(x), validity) #ln1 has a beta and gamma that relate and learn from the attention in this residual structure
        x = x + self.ffn(self.ln2(x)) #then this ln2 has a new beta and gamma that is applied to the x we just made the line above, and learns a new beta and gamma based off the ffn neural network initiated above
        return x


class SpectralAttention(nn.Module):
    """Figure 1""" #figure 1?
    def __init__(self, d, h): #h is number of heads
        super().__init__()
        assert d % h == 0 #asset just means to throw an error if its not true, d has to be a multiple of h 
        self.h, self.dh = h, d // h #self.h = h which is number of heads and i think self.dh is dimension of heads = the dimensions of d//h
        self.qkv = nn.Linear(d, 3 * d) # matrix 1 that is learnt, is the q k and v matrix. defining Q K and V all in one tensor, defining it all in one tensor, linearising taking it something of dimension d and then giving out 3*d because q k and v is three
        self.out = nn.Linear(d, d) #matrix 2 that is learnt is the out matrix which is like transforming the outut from d to d 
        # Local positional embedding: per-head, per-channel 3-tap filter.
        self.local = nn.Conv1d(
            self.dh, self.dh, 3, padding=1, groups=self.dh, bias=False
        )

    def forward(self, x, validity):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)  # [B, H, T, dh]
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)

        # Depthwise conv on V, applied independently per head.
        v_flat = v.reshape(B * self.h, T, self.dh).transpose(1, 2)  # [B*H, dh, T]
        local = self.local(v_flat).transpose(1, 2).view(B, self.h, T, self.dh)

        # Additive mask: -inf at invalid key positions, 0 elsewhere.
        mask = torch.zeros(B, 1, 1, T, device=x.device, dtype=x.dtype)
        mask.masked_fill_(~validity[:, None, None, :], float('-inf'))

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # [B, H, T, dh]
        y = y + local
        y = y * validity[:, None, :, None].to(x.dtype)  # zero invalid queries

        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out(y)








