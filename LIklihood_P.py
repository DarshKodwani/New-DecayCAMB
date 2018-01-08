import matplotlib
matplotlib.use('Agg')
import sys, platform, os
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colors, ticker
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
#uncomment this if you are running remotely and want to keep in synch with repo changes
#if platform.system()!='Windows':
#    !cd $HOME/git/camb; git pull github master; git log -1
import camb
from camb import model, initialpower

print('Using CAMB installed at '+ os.path.realpath(os.path.join(os.getcwd(),'..')))
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))

#For sending email
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders

#Send the email when the script is run

sender = 'darsh1993@gmail.com'
password = 'Cr7redevilg'
receivers = ['ddkcdk@yahoo.co.uk']

message = """
Subject: Likelihood script started

Likelihood script started on Lobster.cita. with pure Ad features
"""

s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login(sender, password)
s.sendmail(sender, receivers, message)

# This script is an attempt to find the Fisher info kernel for adiabatic/non-adiabatic modes 

#0) Planck noise

full_planck = fits.open('./COM_PowerSpect_CMB_R202.fits')

#Low l TT spectrum

TTLOLUNB = full_planck[1].data

xTTLOLUNB = TTLOLUNB.field(0)
yTTLOLUNB = TTLOLUNB.field(1)
ypeTTLOLUNB = TTLOLUNB.field(2)
ymeTTLOLUNB = TTLOLUNB.field(3)
yTTLOLUNBerr =  [ymeTTLOLUNB, ypeTTLOLUNB]

#Low l TE spectrum

TELOLUNB = full_planck[2].data

xTELOLUNB = TELOLUNB.field(0)
yTELOLUNB = TELOLUNB.field(1)
ypeTELOLUNB = TELOLUNB.field(2)
ymeTELOLUNB = TELOLUNB.field(3)
yTELOLUNBerr =  [ymeTELOLUNB, ypeTELOLUNB]

#Low l EE spectrum

EELOLUNB = full_planck[3].data

xEELOLUNB = EELOLUNB.field(0)
yEELOLUNB = EELOLUNB.field(1)
ypeEELOLUNB = EELOLUNB.field(2)
ymeEELOLUNB = EELOLUNB.field(3)
yEELOLUNBerr =  [ymeEELOLUNB, ypeEELOLUNB]

#High l TT spectrum 

TTHILUNB = full_planck[8].data

xTTHILUNB = TTHILUNB.field(0)
yTTHILUNB = TTHILUNB.field(1)
ypeTTHILUNB = TTHILUNB.field(2)
ymeTTHILUNB = -TTHILUNB.field(2)
yTTHILUNBerr =  [ymeTTHILUNB, ypeTTHILUNB]

#High l TE spectrum 

TEHILUNB = full_planck[10].data

xTEHILUNB = TEHILUNB.field(0)
yTEHILUNB = TEHILUNB.field(1)
ypeTEHILUNB = TEHILUNB.field(2)
ymeTEHILUNB = -TEHILUNB.field(2)
yTEHILUNBerr =  [ymeTEHILUNB, ypeTEHILUNB]

#High l EE spectrum 

EEHILUNB = full_planck[12].data

xEEHILUNB = EEHILUNB.field(0)
yEEHILUNB = EEHILUNB.field(1)
ypeEEHILUNB = EEHILUNB.field(2)
ymeEEHILUNB = -EEHILUNB.field(2)
yEEHILUNBerr =  [ymeEEHILUNB, ypeEEHILUNB]

#Combining the full TT spectrum

xTTFULLUNB = np.append(xTTLOLUNB, xTTHILUNB)
yTTFULLUNB = np.append(yTTLOLUNB, yTTHILUNB)
ypeTTFULLUNB = np.append(ypeTTLOLUNB, ypeTTHILUNB)
ymeTTFULLUNB = np.append(ymeTTLOLUNB, ymeTTHILUNB)
yTTFULLUNBerr =  [ymeTTFULLUNB, ypeTTFULLUNB]

#Combining the full TE spectrum

xTEFULLUNB = np.append(xTELOLUNB, xTEHILUNB)
yTEFULLUNB = np.append(yTELOLUNB, yTEHILUNB)
ypeTEFULLUNB = np.append(ypeTELOLUNB, ypeTEHILUNB)
ymeTEFULLUNB = np.append(ymeTELOLUNB, ymeTEHILUNB)
yTEFULLUNBerr =  [ymeTEFULLUNB, ypeTEFULLUNB]

#Combining the full EE spectrum

xEEFULLUNB = np.append(xEELOLUNB, xEEHILUNB)
yEEFULLUNB = np.append(yEELOLUNB, yEEHILUNB)
ypeEEFULLUNB = np.append(ypeEELOLUNB, ypeEEHILUNB)
ymeEEFULLUNB = np.append(ymeEELOLUNB, ymeEEHILUNB)
yEEFULLUNBerr =  [ymeEEFULLUNB, ypeEEFULLUNB]

#Computing the naive noise

NoiseTT = np.sqrt(np.square(ypeTTFULLUNB) + np.square(ymeTTFULLUNB))
NoiseTE = np.sqrt(np.square(ypeTEFULLUNB) + np.square(ymeTEFULLUNB))
NoiseEE = np.sqrt(np.square(ypeEEFULLUNB) + np.square(ymeEEFULLUNB))


# 1) Defining detector noise

#Using the defintiion of detector noise used in "Forecasting isocurvature models with CMB lensing information" by Santos et al (2012). See Eq A9 in this paper.
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


#
#def dnoise(l):
#    return ((thetaarcmin*sigmaT)**2)*np.exp(l*(l+1)*thetarad**2/(8*np.log(2)))
#    
#def dnoiseP(l):
#    return ((thetaarcmin*sigmaP)**2)*np.exp(l*(l+1)*thetarad**2/(8*np.log(2)))

def dnoise(l):
    return ( ((thetaarcmin*sigmaT)**(-2))*np.exp(-l*(l+1)*(thetarad**2)/(8*np.log(2)))
    + ((thetaarcmin100*sigmaT100)**(-2))*np.exp(-l*(l+1)*(thetarad100**2)/(8*np.log(2)))
    + ((thetaarcmin217*sigmaT217)**(-2))*np.exp(-l*(l+1)*(thetarad217**2)/(8*np.log(2)))
    + ((thetaarcmin353*sigmaT353)**(-2))*np.exp(-l*(l+1)*(thetarad353**2)/(8*np.log(2))))**(-1)
    
def dnoiseP(l):
    return (  ((thetaarcmin*sigmaP)**(-2))*np.exp(-l*(l+1)*(thetarad**2)/(8*np.log(2)))
    + ((thetaarcmin100*sigmaP)**(-2))*np.exp(-l*(l+1)*(thetarad100**2)/(8*np.log(2)))
    + ((thetaarcmin217*sigmaP217)**(-2))*np.exp(-l*(l+1)*(thetarad217**2)/(8*np.log(2)))
    + ((thetaarcmin353*sigmaP353)**(-2))*np.exp(-l*(l+1)*(thetarad353**2)/(8*np.log(2))))**(-1)
    
    
# 2) Setting up CAMB to obtain the C_ls

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()

# Here we set the initial condition for the Fluctuations.

#epsX here is the amplitude of the power we are adding to the power spectrum at a given k and kXpivot is the initial k 
pars.InitPower.set_params(ns=0.965, r=0, kXpivot= 0, epsX = 0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);

#calculate transfer functions for these parameters
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars)
for name in powers: print name
totCL=powers['total']
unlensedCL=powers['unlensed_scalar']
print totCL.shape

ls = np.arange(totCL.shape[0])
lmax = 1995

#Setting up intial ks

ksmin = 10**(-4)
ksmax = 5*10**(-1)
numks = 50
xs = np.linspace(ksmin,ksmax,numks)
ks = xs


#The adiabtic, unmodified, C_ls 

Unmod_ad_totCl = np.genfromtxt('./test_scalCls.dat')
#Trans = np.genfromtxt('/Users/darshkodwani/Documents/Darsh/Toronto/Research/CAMB-May2016/test_transfer_out.dat')

#Norm = 1
Norm = 7.42835025**(-12)
        
#Creating a set of Cls from normal/fiducial power spectrum 

add_initial_power1 = initialpower.InitialPowerParams()
add_initial_power1.set_params(kXpivot=0, epsX = 0, As = 2.1*10**(-9))
Unmodcls = results.get_total_cls(lmax)
#Unmodcls = np.zeros((2001,4))

#Creating a set of Cls from modified power spectrum 

#Adiabatic Cls
#
Allcls_ad = np.zeros((lmax+1, numks))
AllclsEE_ad = np.zeros((lmax+1, numks))
AllclsBB_ad = np.zeros((lmax+1,numks))
AllclsTE_ad = np.zeros((lmax+1, numks))
epspower_ad = 10**(-9)
count_ad = 0 


pars.scalar_initial_condition = 1
pars.InitialConditionVector = (-1.,0.,0., 0., 0., 0., 0., 0., 0.)
results_ad = camb.get_results(pars)
for k in ks:
    add_initial_power = initialpower.InitialPowerParams()
    add_initial_power.set_params(kXpivot=k, epsX = epspower_ad, As = 2.1*10**(-9))
    results_ad.power_spectra_from_transfer(add_initial_power)
    cl_ad = results_ad.get_total_cls(lmax)
    Allcls_ad[:,count_ad] = cl_ad[:,0]
    AllclsEE_ad[:,count_ad] = cl_ad[:,1]
    AllclsBB_ad[:,count_ad] = cl_ad[:,2]
    AllclsTE_ad[:,count_ad] = cl_ad[:,3]
    count_ad += 1
    plt.plot(np.arange(lmax+1),cl_ad[:,0])
    
# 3) Computing the fisher info kernel

FFReal = np.zeros((numks, numks)) 

#Full Cls

Allcls = np.zeros((lmax+1, numks))
AllclsEE = np.zeros((lmax+1, numks))
AllclsBB = np.zeros((lmax+1,numks))
AllclsTE = np.zeros((lmax+1, numks))

Allcls = Allcls_ad 
#Allcls = (- Allcls_ad - AllclsCDM_iso + AllclsCDM_isoad)
#Allcls = 0.5*(-AllclsCDM_iso +AllclsCDM_isoad)

#This is the full Fisher info with the full covariance

#Simulated noise
#countreala = 0
#for k in ks: 
#    countrealb = 0
#    for j in ks:
#        xtemp1 = []
#        for i in range(2, lmax):
#            tempd1 = ((2*i+1)/2)*((Unmodcls[i,0] - Allcls[i, countreala])*(Unmodcls[i,0] - Allcls[i,countrealb])*(((Norm*Unmod_ad_totCl[i,2] + dnoiseP(i))/((Norm*Unmod_ad_totCl[i,3])**2 - (Norm*Unmod_ad_totCl[i,2] + dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1]+ dnoise(i))))**2)
#            + ((Unmodcls[i,0] - Allcls[i, countreala])*(Unmodcls[i,1] - AllclsEE[i, countrealb])*(((Norm*Unmod_ad_totCl[i,3])**2 - (Norm*Unmod_ad_totCl[i,2] + dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1]+ dnoise(i)))**2))
#            + ((Unmodcls[i,0] - Allcls[i, countrealb])*(Unmodcls[i,1] - AllclsEE[i, countreala])*(((Norm*Unmod_ad_totCl[i,3])**2 - (Norm*Unmod_ad_totCl[i,2] + dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1]+ dnoise(i)))**2))
#            - (Unmodcls[i,0] - Allcls[i, countreala])*(Unmodcls[i,3] - AllclsTE[i, countrealb])*((2*(Norm*Unmod_ad_totCl[i,2]+ dnoiseP(i))*(Norm*Unmod_ad_totCl[i,3]))/(((Norm*Unmod_ad_totCl[i,3])**2 - (Norm*Unmod_ad_totCl[i,2]+ dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1]+ dnoise(i)))**2))
#            - (Unmodcls[i,0] - Allcls[i, countrealb])*(Unmodcls[i,3] - AllclsTE[i, countreala])*((2*(Norm*Unmod_ad_totCl[i,2]+ dnoiseP(i))*(Norm*Unmod_ad_totCl[i,3]))/(((Norm*Unmod_ad_totCl[i,3])**2 - (Norm*Unmod_ad_totCl[i,2]+ dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1]+ dnoise(i)))**2))
#            + (Unmodcls[i,1] - AllclsEE[i, countreala])*(Unmodcls[i,1] - AllclsEE[i, countrealb])*(((Norm*Unmod_ad_totCl[i,1] + dnoise(i))/((Norm*Unmod_ad_totCl[i,3])**2 - (Norm*Unmod_ad_totCl[i,2] + dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1] + + dnoiseP(i))))**2)
#            - (Unmodcls[i,1] - AllclsEE[i, countreala])*((Unmodcls[i,3] - AllclsTE[i, countrealb]))*((2*(Norm*Unmod_ad_totCl[i,3])*(Norm*Unmod_ad_totCl[i,1] + dnoise(i)))/(((Norm*Unmod_ad_totCl[i,3])**2 - (Norm*Unmod_ad_totCl[i,2] + dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1] + dnoise(i)))**2))
#            - (Unmodcls[i,1] - AllclsEE[i, countrealb])*((Unmodcls[i,3] - AllclsTE[i, countreala]))*((2*(Norm*Unmod_ad_totCl[i,3])*(Norm*Unmod_ad_totCl[i,1]+ dnoise(i)))/(((Norm*Unmod_ad_totCl[i,3])**2 - (Norm*Unmod_ad_totCl[i,2] + dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1] +dnoise(i)))**2))
#            + (Unmodcls[i,3] - AllclsTE[i, countreala])*(Unmodcls[i,3] - AllclsTE[i, countrealb])*((2*((Norm*Unmod_ad_totCl[i,3])**2 - (Norm*Unmod_ad_totCl[i,2]+ dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1] + dnoise(i))))/((Norm*Unmod_ad_totCl[i,3]**2 - (Norm*Unmod_ad_totCl[i,2]+ dnoiseP(i))*(Norm*Unmod_ad_totCl[i,1]+ dnoise(i))))))
#            xtemp1.append(tempd1)
#        FFReal[countreala, countrealb] = sum(xtemp1)/(4*epspower_ad**2)
#        countrealb += 1
#    countreala += 1

#Planck noise

countreala = 0
for k in ks: 
    countrealb = 0
    for j in ks:
        xtemp1 = []
        for i in range(2, lmax):
            tempd1 = ((2*i+1)/2)*((Unmodcls[i,0] - Allcls[i, countreala])*(Unmodcls[i,0] - Allcls[i,countrealb])*(((Norm*yEEFULLUNB[i] + Norm*NoiseEE[i])/((Norm*yTEFULLUNB[i])**2 - (Norm*yEEFULLUNB[i] + Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i]+ Norm*NoiseTT[i])))**2)
            + ((Unmodcls[i,0] - Allcls[i, countreala])*(Unmodcls[i,1] - AllclsEE[i, countrealb])*(((Norm*yTEFULLUNB[i])**2 - (Norm*yEEFULLUNB[i] + Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i]+ Norm*NoiseTT[i]))**2))
            + ((Unmodcls[i,0] - Allcls[i, countrealb])*(Unmodcls[i,1] - AllclsEE[i, countreala])*(((Norm*yTEFULLUNB[i])**2 - (Norm*yEEFULLUNB[i] + Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i]+ Norm*NoiseTT[i]))**2))
            - (Unmodcls[i,0] - Allcls[i, countreala])*(Unmodcls[i,3] - AllclsTE[i, countrealb])*((2*(Norm*yEEFULLUNB[i]+ Norm*NoiseEE[i])*(Norm*yTEFULLUNB[i]))/(((Norm*yTEFULLUNB[i])**2 - (Norm*yEEFULLUNB[i]+ Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i]+ Norm*NoiseTT[i]))**2))
            - (Unmodcls[i,0] - Allcls[i, countrealb])*(Unmodcls[i,3] - AllclsTE[i, countreala])*((2*(Norm*yEEFULLUNB[i]+ Norm*NoiseEE[i])*(Norm*yTEFULLUNB[i]))/(((Norm*yTEFULLUNB[i])**2 - (Norm*yEEFULLUNB[i]+ Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i]+ Norm*NoiseTT[i]))**2))
            + (Unmodcls[i,1] - AllclsEE[i, countreala])*(Unmodcls[i,1] - AllclsEE[i, countrealb])*(((Norm*yTTFULLUNB[i] + Norm*NoiseTT[i])/((Norm*yTEFULLUNB[i])**2 - (Norm*yEEFULLUNB[i] + Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i] + + Norm*NoiseEE[i])))**2)
            - (Unmodcls[i,1] - AllclsEE[i, countreala])*((Unmodcls[i,3] - AllclsTE[i, countrealb]))*((2*(Norm*yTEFULLUNB[i])*(Norm*yTTFULLUNB[i] + Norm*NoiseTT[i]))/(((Norm*yTEFULLUNB[i])**2 - (Norm*yEEFULLUNB[i] + Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i] + Norm*NoiseTT[i]))**2))
            - (Unmodcls[i,1] - AllclsEE[i, countrealb])*((Unmodcls[i,3] - AllclsTE[i, countreala]))*((2*(Norm*yTEFULLUNB[i])*(Norm*yTTFULLUNB[i]+ Norm*NoiseTT[i]))/(((Norm*yTEFULLUNB[i])**2 - (Norm*yEEFULLUNB[i] + Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i] +Norm*NoiseTT[i]))**2))
            + (Unmodcls[i,3] - AllclsTE[i, countreala])*(Unmodcls[i,3] - AllclsTE[i, countrealb])*((2*((Norm*yTEFULLUNB[i])**2 - (Norm*yEEFULLUNB[i]+ Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i] + Norm*NoiseTT[i])))/((Norm*yTEFULLUNB[i]**2 - (Norm*yEEFULLUNB[i]+ Norm*NoiseEE[i])*(Norm*yTTFULLUNB[i]+ Norm*NoiseTT[i])))))
            xtemp1.append(tempd1)
        FFReal[countreala, countrealb] = sum(xtemp1)/(4*epspower_ad**2)
        countrealb += 1
    countreala += 1


#Fihser info for TT only (not full covariance)

#Simulated noise

#countreala = 0
#for k in ks: 
#    countrealb = 0
#    for j in ks:
#        xtemp1 = []
#        for i in range(2, lmax):
#            tempd1 = ((2*i+1)/2)*((Unmodcls[i,0] - Allcls[i, countreala])*(Unmodcls[i,0] - Allcls[i,countrealb])*((Norm*Unmod_ad_totCl[i,1] + dnoise(i))**(-2)))
#            xtemp1.append(tempd1)
#        FFReal[countreala, countrealb] = sum(xtemp1)/(4*epspower_ad*epspowerCDM_iso)
#        countrealb += 1
#    countreala += 1
    
#Planck noise 

#countreala = 0
#for k in ks: 
#    countrealb = 0
#    for j in ks:
#        xtemp1 = []
#        for i in range(2, lmax):
#            tempd1 = ((2*i+1)/2)*((Unmodcls[i,0] - Allcls[i, countreala])*(Unmodcls[i,0] - Allcls[i,countrealb])*((Norm*yTTFULLUNB[i] + Norm*NoiseTT[i])**(-2)))
#            xtemp1.append(tempd1)
#        FFReal[countreala, countrealb] = sum(xtemp1)/(4*epspower_ad*epspowerCDM_iso)
#        countrealb += 1
#    countreala += 1


#Save the Fisher matrix values
np.savetxt('/home/dkodwani/CAMB-Nov2016/Fisher_data/50k_Ad_fisher_null_eps10m9.txt', FFReal)
#Save diagonalized Fisher
diag_fisher = np.diag(FFReal)
np.savetxt('/home/dkodwani/CAMB-Nov2016/Fisher_data/50k_Ad_fisherdiag_null_eps10m9.txt', diag_fisher)
# Making a plot of the diagonal fisher
plt.figure()
plt.plot(ks,diag_fisher)
plt.show()
plt.savefig('/home/dkodwani/CAMB-Nov2016/Fisher_plots/50k_Ad_pc_null_eps10m9.pdf')

#Finding the principle components
pc_ind = diag_fisher.argsort()[-3:][::-1]  #Finds the index of the three highes numbers in the diagonalized Fisher
pca = [[ks[pc_ind[0]], diag_fisher[pc_ind[0]]], [ks[pc_ind[1]], diag_fisher[pc_ind[1]]], [ks[pc_ind[2]], diag_fisher[pc_ind[2]]]] #The array of pc's and there indices
print pca
np.savetxt('/home/dkodwani/CAMB-Nov2016/Fisher_data/50k_Ad_pca_null_eps10m9.txt', pca)

#Make the plot
plt.figure()
#CS = plt.contour(ks, ks, FFReal,cmap = plt.cm.bone, locator=ticker.LogLocator())
CS = plt.contour(ks, ks, FFReal, 100)
#plt.clabel(CS)
cbar = plt.colorbar(CS)
plt.title('Fisher information $I(k_1, k_2)$')
plt.ylabel('$k_1$ ($Mpc^{-1}$)')
plt.xlabel('$k_2$ ($Mpc^{-1}$)')
plt.yscale('log')
plt.xscale('log')
plt.show()
plt.savefig('/home/dkodwani/CAMB-Nov2016/Fisher_plots/50k_Ad_fisher_null_eps10m9.pdf')

#Send the email when its done

sender = 'darsh1993@gmail.com'
password = 'Cr7redevilg'
receivers = ['ddkcdk@yahoo.co.uk']

msg = MIMEMultipart()

msg["Subject"] = "Plots of Ad fluctuations for 50ks"
body = "Finished running script on Lobster.cita"

msg.attach(MIMEText(body, 'plain'))

filename = "50k_Ad_fisher_null_eps10m9.pdf"
attachment = open("/home/dkodwani/CAMB-Nov2016/Fisher_plots/50k_Ad_fisher_null_eps10m9.pdf", "rb")

part = MIMEBase('application', 'octet-stream')
part.set_payload((attachment).read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

msg.attach(part)

s = smtplib.SMTP('smtp.gmail.com', 587)
s.starttls()
s.login(sender, password)
text=msg.as_string()
s.sendmail(sender, receivers, text)
