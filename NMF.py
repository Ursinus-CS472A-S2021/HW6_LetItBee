"""
Purpose: To implementing the NMF techniques in [1]
[1] Driedger, Jonathan, Thomas Praetzlich, and Meinard Mueller. 
"Let it Bee-Towards NMF-Inspired Audio Mosaicing." ISMIR. 2015.
"""
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import time
import librosa


def create_musaic(S, WComplex, win_length, hop_length, L, r=3, p=10, c=3):
    """
    Implement the technique from "Let It Bee-Towards NMF-Inspired
    Audio Mosaicing"

    Parameters
    ----------
    S: ndarray(M, N, dtype=np.complex)
        A M x N nonnegative target matrix
    WComplex: ndarray(M, K, dtype=np.complex) 
        An M x K matrix of template sounds in some time order along the second axis
    win_length: int
        Window length of STFT (used in Griffin Lim)
    hop_length: int
        Hop length of STFT (used in Griffin Lim)
    L: int
        Number of iterations
    r: int
        Half of the width of the repeated activation filter
    p: int
        Degree of polyphony; i.e. number of values in each column of H which should be 
        un-shrunken
    c: int
        Half length of time-continuous activation filter
    """
    V = np.abs(S) # V is the absolute magnitude spectrogram, keeping it nonnegative
    W = np.abs(WComplex) # W is the absolute magnitude spectrogram of WComplex
    N = V.shape[1]
    K = W.shape[1]
    WDenom = np.sum(W, 0)
    WDenom[WDenom == 0] = 1

    # Random nonnegative initialization of H
    H = np.random.rand(K, N)
    for l in range(L):
        print(l, end='.') # Print out iteration number for progress
             
        # Step 1: Avoid repeated activations
        
        ## TODO: Fill this in
        
        # Step 2: Restrict number of simultaneous activations
               
        ## TODO: Fill this in
        
        # Step 3: Supporting time-continuous activations
        
        ## TODO: Fill this in
        
        # Step 4: Match target with an iteration of KL-based NMF, keeping
        # W fixed
        WH = W.dot(H)
        WH[WH == 0] = 1 # Prevent divide by 0
        VLam = V/WH
        H = H*((W.T).dot(VLam)/WDenom[:, None])
    
    y = librosa.istft(WComplex.dot(H), win_length=win_length, hop_length=hop_length)
    ## TODO: Use 10 iterations of Griffin-Lim instead of a straight-up STFT
    return y
