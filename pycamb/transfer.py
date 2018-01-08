import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower

#GENERATING TRANSFERS
#NOTE TO SELF: for normalization purposes, the ell for subhorizon/superhorizon crossing is 150, which corresponds to transfer.l[36] (this may change depending on accuracy setttings)

pars=camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars)
data = camb.get_transfer_functions(pars)
transfer = data.get_cmb_transfer_data()
totCL=powers['total']

dTl5=np.loadtxt("/Users/darshkodwani/Desktop/CAMB-0.1.6.1/pycamb/decay_trnsf_l150.txt",unpack=True)

#MAKE PLOTS

# plt.semilogx(transfer.q,transfer.delta_p_l_k[0,36,:],'b',label="Grow l=150")
# plt.semilogx(transfer.q,dTl5,'r',label="Decay l=150")
# 
# plt.legend()
# plt.show()

#Cls comparison

clgrow=np.loadtxt("/Users/darshkodwani/Desktop/CAMB-0.1.6.1/pycamb/grow_cl.txt",unpack=True)
plt.loglog(4.82*totCL[:,0])
plt.loglog(clgrow)

plt.show()

