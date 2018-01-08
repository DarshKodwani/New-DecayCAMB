import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
#from astropy.io import fits
import numpy as np
import camb
from matplotlib import colors, ticker
from camb import model, initialpower

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0,epsX=0.,kXpivot=0.)
#calculate results for these parameters
results = camb.get_results(pars)
powers =results.get_cmb_power_spectra(pars)

#plot the total lensed CMB power spectra versus unlensed, and fractional difference
backgroundCL=powers['total']
unlensedCL=powers['unlensed_scalar']
#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
ls = np.arange(backgroundCL.shape[0])
#lmax=np.size(backgroundCL[:,0])


###Detector noise###

#Using the defintiion of detector noise used in "1206.2832" by Santos et al (2012). See Eq A9 in this paper.
# Defining the noise paramaters - all quantities taken from the paper given above Table VII - we only take the 143 GHz channel

thetaarcmin = 7.1
thetaarcmin100 = 9.5
thetaarcmin217 = 5.0
thetaarcmin353 = 5.0
thetarad = thetaarcmin/3437.75
thetarad100 = thetaarcmin100/3437.75
thetarad217 = thetaarcmin217/3437.75
thetarad353 = thetaarcmin353/3437.75


sigmaT = 6.0016*(10**(-6))
sigmaT100 = 6.82*(10**(-6))
sigmaT217 = 13.0944*(10**(-6))
sigmaT353 = 40.1016*(10**(-6))

sigmaP = 11.4576*(10**(-6))
sigmaP100 = 10.9120*(10**(-6))
sigmaP217 = 26.7644*(10**(-6))
sigmaP353 = 81.2944*(10**(-6))

def dnoise(l):
    return (((thetaarcmin*sigmaT)**(-2))*np.exp(-l*(l+1)*(thetarad**2)/(8*np.log(2)))
    + ((thetaarcmin100*sigmaT100)**(-2))*np.exp(-l*(l+1)*(thetarad100**2)/(8*np.log(2)))
    + ((thetaarcmin217*sigmaT217)**(-2))*np.exp(-l*(l+1)*(thetarad217**2)/(8*np.log(2)))
    + ((thetaarcmin353*sigmaT353)**(-2))*np.exp(-l*(l+1)*(thetarad353**2)/(8*np.log(2))))**(-1)
    
def dnoiseP(l):
    return (((thetaarcmin*sigmaP)**(-2))*np.exp(-l*(l+1)*(thetarad**2)/(8*np.log(2)))
    + ((thetaarcmin100*sigmaP)**(-2))*np.exp(-l*(l+1)*(thetarad100**2)/(8*np.log(2)))
    + ((thetaarcmin217*sigmaP217)**(-2))*np.exp(-l*(l+1)*(thetarad217**2)/(8*np.log(2)))
    + ((thetaarcmin353*sigmaP353)**(-2))*np.exp(-l*(l+1)*(thetarad353**2)/(8*np.log(2))))**(-1)
    
###Planck noise###

# full_planck = fits.open('./COM_PowerSpect_CMB_R202.fits')
# 
# #Low l TT spectrum
# 
# TTLOLUNB = full_planck[1].data
# 
# xTTLOLUNB = TTLOLUNB.field(0)
# yTTLOLUNB = TTLOLUNB.field(1)
# ypeTTLOLUNB = TTLOLUNB.field(2)
# ymeTTLOLUNB = TTLOLUNB.field(3)
# yTTLOLUNBerr =  [ymeTTLOLUNB, ypeTTLOLUNB]
# 
# #Low l TE spectrum
# 
# TELOLUNB = full_planck[2].data
# 
# xTELOLUNB = TELOLUNB.field(0)
# yTELOLUNB = TELOLUNB.field(1)
# ypeTELOLUNB = TELOLUNB.field(2)
# ymeTELOLUNB = TELOLUNB.field(3)
# yTELOLUNBerr =  [ymeTELOLUNB, ypeTELOLUNB]
# 
# #Low l EE spectrum
# 
# EELOLUNB = full_planck[3].data
# 
# xEELOLUNB = EELOLUNB.field(0)
# yEELOLUNB = EELOLUNB.field(1)
# ypeEELOLUNB = EELOLUNB.field(2)
# ymeEELOLUNB = EELOLUNB.field(3)
# yEELOLUNBerr =  [ymeEELOLUNB, ypeEELOLUNB]
# 
# #High l TT spectrum 
# 
# TTHILUNB = full_planck[8].data
# 
# xTTHILUNB = TTHILUNB.field(0)
# yTTHILUNB = TTHILUNB.field(1)
# ypeTTHILUNB = TTHILUNB.field(2)
# ymeTTHILUNB = -TTHILUNB.field(2)
# yTTHILUNBerr =  [ymeTTHILUNB, ypeTTHILUNB]
# 
# #High l TE spectrum 
# 
# TEHILUNB = full_planck[10].data
# 
# xTEHILUNB = TEHILUNB.field(0)
# yTEHILUNB = TEHILUNB.field(1)
# ypeTEHILUNB = TEHILUNB.field(2)
# ymeTEHILUNB = -TEHILUNB.field(2)
# yTEHILUNBerr =  [ymeTEHILUNB, ypeTEHILUNB]
# 
# #High l EE spectrum 
# 
# EEHILUNB = full_planck[12].data
# 
# xEEHILUNB = EEHILUNB.field(0)
# yEEHILUNB = EEHILUNB.field(1)
# ypeEEHILUNB = EEHILUNB.field(2)
# ymeEEHILUNB = -EEHILUNB.field(2)
# yEEHILUNBerr =  [ymeEEHILUNB, ypeEEHILUNB]
# 
# #Combining the full TT spectrum
# 
# xTTFULLUNB = np.append(xTTLOLUNB, xTTHILUNB)
# yTTFULLUNB = np.append(yTTLOLUNB, yTTHILUNB)
# ypeTTFULLUNB = np.append(ypeTTLOLUNB, ypeTTHILUNB)
# ymeTTFULLUNB = np.append(ymeTTLOLUNB, ymeTTHILUNB)
# yTTFULLUNBerr =  [ymeTTFULLUNB, ypeTTFULLUNB]
# 
# #Combining the full TE spectrum
# 
# xTEFULLUNB = np.append(xTELOLUNB, xTEHILUNB)
# yTEFULLUNB = np.append(yTELOLUNB, yTEHILUNB)
# ypeTEFULLUNB = np.append(ypeTELOLUNB, ypeTEHILUNB)
# ymeTEFULLUNB = np.append(ymeTELOLUNB, ymeTEHILUNB)
# yTEFULLUNBerr =  [ymeTEFULLUNB, ypeTEFULLUNB]
# 
# #Combining the full EE spectrum
# 
# xEEFULLUNB = np.append(xEELOLUNB, xEEHILUNB)
# yEEFULLUNB = np.append(yEELOLUNB, yEEHILUNB)
# ypeEEFULLUNB = np.append(ypeEELOLUNB, ypeEEHILUNB)
# ymeEEFULLUNB = np.append(ymeEELOLUNB, ymeEEHILUNB)
# yEEFULLUNBerr =  [ymeEEFULLUNB, ypeEEFULLUNB]
# 
# #Computing the naive noise
# 
# TTn = np.sqrt(np.square(ypeTTFULLUNB) + np.square(ymeTTFULLUNB))
# TEn = np.sqrt(np.square(ypeTEFULLUNB) + np.square(ymeTEFULLUNB))
# EEn = np.sqrt(np.square(ypeEEFULLUNB) + np.square(ymeEEFULLUNB))

###Fisher###
lmax=np.size(backgroundCL[:,0])
ksmin = 10**(-3)
ksmax = 5*10**(-1)
numks = 100
ks = np.linspace(ksmin,ksmax,numks)
TTp = np.zeros((lmax,numks))
EEp = np.zeros((lmax, numks))
TEp = np.zeros((lmax,numks))
TTm = np.zeros((lmax,numks))
EEm = np.zeros((lmax, numks))
TEm = np.zeros((lmax,numks))
epspower = 0.01


countp = 0 
for k in ks:
    add_initial_powerp = initialpower.InitialPowerParams()
    add_initial_powerp.set_params(kXpivot=k, epsX = epspower)
    results.power_spectra_from_transfer(add_initial_powerp)
    clp = results.get_total_cls(lmax-1)
    TTp[:,countp] = clp[:,0]
    EEp[:,countp] = clp[:,1]
    TEp[:,countp] = clp[:,3]
    countp += 1

countm = 0 
for k in ks:
    add_initial_powerm = initialpower.InitialPowerParams()
    add_initial_powerm.set_params(kXpivot=k, epsX = -epspower)
    results.power_spectra_from_transfer(add_initial_powerm)
    clm = results.get_total_cls(lmax-1)
    TTm[:,countm] = clm[:,0]
    EEm[:,countm] = clm[:,1]
    TEm[:,countm] = clm[:,3]
    #plt.loglog(np.arange(lmax+1),clm[:,0])
    countm += 1

FFReal = np.zeros((numks, numks)) 
countbeta=0
for i in np.arange(numks):
    countalpha=0
    for j in np.arange(numks):
        fishtemp=[]
        for l in range(2,lmax):
            TTalpha=TTp[:,countalpha]-TTm[:,countalpha]
            TTbeta=TTp[:,countbeta]-TTm[:,countbeta]
            EEalpha=EEp[:,countalpha]-EEm[:,countalpha]
            EEbeta=EEp[:,countbeta]-EEm[:,countbeta]
            TEalpha=TEp[:,countalpha]-TEm[:,countalpha]
            TEbeta=TEp[:,countbeta]-TEm[:,countbeta]
            TTback=backgroundCL[:,0]
            EEback=backgroundCL[:,1]
            TEback=backgroundCL[:,3]
            cov=1/((TEback[l]**2 - (EEback[l]+dnoiseP(l))*(TTback[l]+dnoise(l))**2))
            tempfish= ((2*(l)+1)/2)*cov*(TTalpha[l]*TTbeta[l]/((EEback[l]+dnoiseP(l))**2)
            +TTalpha[l]*TEbeta[l]/((TEback[l])**2) 
            +TTalpha[l]*EEbeta[l]/(-2*(TEback[l])*(EEback[l]+dnoiseP(l)))
            +TEalpha[l]*TTbeta[l]/(TEback[l]**2)
            +TEalpha[l]*TEbeta[l]/((TTback[l]+dnoise(l))**2)
            +TEalpha[l]*EEbeta[l]/(-2*TEback[l]*(TTback[l]+dnoise(l)))
            +EEalpha[l]*TTbeta[l]/(-2*TEback[l]*(EEback[l]+dnoiseP(l)))
            +EEalpha[l]*TEbeta[l]/(-2*TEback[l]*(TTback[l]+dnoise(l)))
            +EEalpha[l]*EEbeta[l]/(2*(TEback[l]+(EEback[l]+dnoiseP(l))*(TTback[l]+dnoise(l)))))
            
            fishtemp.append(np.sqrt(tempfish**2))
        FFReal[countalpha, countbeta] = sum(fishtemp)/(4*epspower**2)
        countalpha+=1
    countbeta+=1

#Save the Fisher matrix values
np.savetxt('/Users/darshkodwani/Desktop/CAMB-0.1.6.1/pycamb/Full_fisher_D.txt', FFReal)
#Save diagonalized Fisher
diag_fisher = np.diag(FFReal)
#Normalization
norm_fisher =diag_fisher/(max(diag_fisher))
np.savetxt('/Users/darshkodwani/Desktop/CAMB-0.1.6.1/pycamb/DiagPCA_fisher_D.txt', diag_fisher)
# Making a plot of the diagonal fisher
plt.figure()
plt.plot(ks,norm_fisher)
plt.show()
plt.savefig('/Users/darshkodwani/Desktop/CAMB-0.1.6.1/pycamb/DiagPCA_plot_D.pdf')

#Finding the principle components
pc_ind = norm_fisher.argsort()[-3:][::-1]  #Finds the index of the three highes numbers in the diagonalized Fisher
pca = [[ks[pc_ind[0]], norm_fisher[pc_ind[0]]], [ks[pc_ind[1]], norm_fisher[pc_ind[1]]], [ks[pc_ind[2]], norm_fisher[pc_ind[2]]]] #The array of pc's and there indices
print pca
np.savetxt('/Users/darshkodwani/Desktop/CAMB-0.1.6.1/pycamb/PCA_D.txt', pca)

#Make the plot
#Noramlize fisher
norm_FFReal = FFReal/(np.max(FFReal))
plt.figure()
#CS = plt.contour(ks, ks, FFReal,cmap = plt.cm.bone, locator=ticker.LogLocator())
CS = plt.contour(ks, ks, norm_FFReal, 100)
#plt.clabel(CS)
cbar = plt.colorbar(CS)
plt.title('Fisher information $I(k_1, k_2)$')
plt.ylabel('$k_1$ ($Mpc^{-1}$)')
plt.xlabel('$k_2$ ($Mpc^{-1}$)')
plt.yscale('log')
plt.xscale('log')
plt.show()
plt.savefig('/Users/darshkodwani/Desktop/CAMB-0.1.6.1/pycamb/Full_fisher_plot_D.pdf')
