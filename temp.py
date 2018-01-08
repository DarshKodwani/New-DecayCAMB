import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import *
from pylab import *

#ax1 = subplot(2,1,1)
#ax2 = subplot(2,1,2)

######################plotting current camb Cls

ax1 = subplot(1,1,1)


temp = np.genfromtxt('/Users/darshkodwani/Desktop/CAMB-0.1.6.1/test_scalCls.dat')
tempclass = np.genfromtxt('/Users/darshkodwani/Desktop/class_decay/output/decaycls_addcs.dat')
normal = np.genfromtxt('/Users/darshkodwani/Documents/Darsh/Research/Sound_modes/ModifiedCAMB-DecayingMode/decay_out/normalgrowingmodecls.dat')
normalclass = np.genfromtxt('/Users/darshkodwani/Desktop/class_decay/output/decaycls_ad.dat')
xdata = temp[:,0]
ytemp = temp[:,1]
ynormal = normal[:,1]
ytempclass=tempclass[:,1]
ynormalclass=normalclass[:,1]
#RatioCLASSCAMB=ytemp/ytempclass

ax1.loglog(xdata,ytemp, label="Decaying mode")
#ax1.loglog(xdata,ytempclass, label="Class decaying mode")
ax1.loglog(xdata,ynormal, label="Growing mode ")
#ax1.loglog(xdata,ynormalclass, label="Class growing mode")
ax1.set_xlabel("$l$")
ax1.set_ylabel("$C^{TT}_l l(l+1)$")
plt.legend()

#ax1.loglog(xdata,RatioCLASSCAMB, label="Class/Camb Cls")
#ax1.set_xlabel("$l$")
#ax1.set_ylabel("$C^{TT}_l l(l+1)[class]/C^{TT}_l l(l+1)[camb]$")
#plt.legend()

show()
