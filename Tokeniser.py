import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from astropy.table import Table
from SpecML import D_emb

#------------------------------------------CONFIGURATION/DATA SETUP----------------------------------------------------#

# Parameters
patch_size = 4
overlap = 2
step_size = patch_size - overlap 


#Load 4.5 Data, use Cache = True to store on memory/RAM or locally after data is read for first time
data = Table.read('https://s3.amazonaws.com/msaexp-nirspec/extractions/dja_msaexp_emission_lines_v4.5.prism_spectra.fits', cache=True,)
#notes on data: There will be columns such as flux values etc, there is a total of 473 wavelength values for this spectrum, and there will be 30000+ recorded spectra values for flux etc at each wavelength. 
#some will be invalid, perhaps at the beginning or end of the spectrum range. You can look into the table to see this. We only want to work with spectra/wavelengths that have valid data.


#Set up Flux after normalisation and Wavelength variables, only using valid spectrum and wavelengths from the data, 'Valid' is a column name in data file
valid_w = np.any(data['valid'],1) #Filtering for valid wavelengths by finding valid data points in the data file from column 2
valid_spectrum = np.any(data['valid'],0) #Filtering for spectrums with valid data at each wavelength by finding valid data points in the data file from column 1

w = data['wave'][valid_w] #Assigns the wavelength variable as w, draws from the wave column of data file, filtering for valid data using the valid filter previously defined
f = data['flux'][np.ix_(valid_w, valid_spectrum)].T / (w**2) #Converts Flux values

#Shape of w and f
# w = 469 -> amount of valid wavelengths
# f = B x L = 42195, 469 -> B is the batch amount e.g. amount of spectrum with valid data,, L is the observation length e.g. amount of valid wavelength values

# scale = np.nanmedian(np.abs(f), axis=1, keepdims=True).clip(1e-30) #identifies a scale of the median flux values
# f_norm = np.arcsinh(f / scale) #normalises the flux values using the scale previously defined and an arcsin
f_norm = (f -  np.mean(f,keepdims=True,axis=1)) / np.std(f,keepdims=True,axis=1)

# Data Quality validity — (B, L): True where the detector pixel is good, set up for later validity masking, assesses if a pixel/data point is valid using the valid column of the data file
dq = data['valid'][np.ix_(valid_w, valid_spectrum)].T  # (B, L)




#------------------------------------------TOKENSIER----------------------------------------------------#


# #this does the for loop but faster 
# pad_length = patch_size - L % step_size
# f_normalised = np.append(f_normalised, np.ones((B,pad_length))*np.nan, axis=1) #B, pad_length makes it pad only to the columns (473), axis = 1 applys the np.append along the L dimension which is along the 473

#Create vectors x_t for tokensing, (B x T x P) where T is the amount of patches (total amount of x_t made from the observation lenth for each spectrum), and P is the patch size, currently just consisting of 20 flux values
x_t = sliding_window_view(f_norm, patch_size, axis = 1)[:,::step_size] 

#Create a matrix X consisting of the vectors x_t after concatenating the mean and std of each patch to the patch_size (vector x_t)
X = np.concatenate([np.nanmean(x_t,axis=2, keepdims = True), np.nanstd(x_t,axis=2, keepdims = True), x_t], axis=2) # B x T x (P + 2), now accounting for the +2 of the mean and std for each patch





#------------------------------------------VALIDITY MASK----------------------------------------------------#

# Validity mask — (N, T): creates vectors of booleans over the patches where True is only when every pixel in the patch is DQ-valid, meaning it has valid data points, assessed in the data column of the data file
dq_patches = sliding_window_view(dq, patch_size, axis=1)[:, ::step_size]  # (B, T, P)
V = np.array(dq_patches.all(axis=2))  # (B, T), tests whether array elements along P axis of dq_patches evaluates to true and creates an array of this
X[~V] = 0.0 #if it is not True, or the opposite of the Validity mask we just defined, then it is invalid and will be replaced by 0.0


#------------------------------------------WAVELENGTH ENCODING----------------------------------------------------#


# Wavelength encoding — (T, D_EMB), shared across all spectra.
# Assumes a common wavelength grid; extend to (B, T, D_EMB) when that breaks.

#create patches over the wavelength array
w_patches = sliding_window_view(w, patch_size)[::step_size].mean(axis=1)  # (T,) amount of patches made from wavelength array, the vector w_patches consists of the mean from that patch
omegas = 10000 ** (-2 * np.arange(D_emb // 2) / D_emb) #Defined from OmniSpec paper (Md Khairul Islam1 & Judy Fox, 2026)
product = np.outer(w_patches * 1e4, omegas)  # (T, D_EMB//2) 

#create empty P matrix of dimensions T x D_emb, T is the amount of patches x_t in X, and D_emb is our pre-defined embedding dimension imported from model script
P = np.empty((X.shape[1], D_emb)) #only required these two dimensions for P because it is already going over the batch size within X.shape[1]
P[:, 0::2] = np.sin(product)
P[:, 1::2] = np.cos(product)
#this completes the sinusoidal condition, where elements within the 0 and 1st index of P to meet sin and cosine conditions from (Md Khairul Islam1 & Judy Fox, 2026)


# # lam = np.append(w, w[-1] + np.arange(1, pad_length+1)*(w[-1] - w[-2])) #this linearly extrapolates wavelength values at the end of the array to meet the required pad_length



