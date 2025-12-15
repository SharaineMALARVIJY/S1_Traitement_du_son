# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:57:40 2020

@author: argan
"""

#import matplotlib.pyplot as plt
#import matplotlib
#from matplotlib import rc
import numpy as np
#import scipy.io
#import scipy.io.wavfile as wav


def TFCT(Nwin, Nhop, Nfft, x_vect,Fe):
    L = int(np.floor(1+(len(x_vect)-1)/Nhop)) #nb de colonnes
    nl=int(np.floor(Nfft//2+1)) #nb de lignes
    x_mat=np.zeros([nl,L],dtype=complex)
    Lx2=(L-1)*Nhop+Nwin
    x_vect2=np.append(x_vect,np.zeros([1,Lx2-len(x_vect)]))
    freqspectro_vect=np.linspace(0,Fe/2,nl)
    #t_vect0=np.arange(Nwin//2,Nwin//2+L*Nhop,Nhop)/Fe
    w_vect=np.hamming(Nwin)
    #t_vect=np.linspace(0,(len(x_vect)-1)/Fe,len(x_vect))
    tspectro_vect=(np.arange(0,L,1)*Nhop+Nwin/2)/Fe

    for l in range(0,L):
        trame=x_vect2[l*Nhop:l*Nhop+Nwin]*w_vect
        trame_fft=np.fft.fft(trame,Nfft)
        x_mat[:,l]=trame_fft[0:nl]
        
    return x_mat,tspectro_vect,freqspectro_vect