# Agustin Laguarda

#Ineicher P. Conversion function between the Linke turbidity and the atmospheric
#water vapor and aerosol content. Sol Energy 2008;82:1095–7
# saca TL con RTM y lo parametriza.

# aod550 = [adim] 550 nm aerosol optical depth (from 0 to 0.6) urban aerosol type.
# w      = [cm  ] water vapor (from 0.2 to 10 cm) 
# alt    = [mts ] asnm (from 0 to 7000 m) 
import numpy as np

def TL_de_var_atm(aod550, w, alt):
    pc = 1/ np.exp(-alt /8434.5)  # es el inverso de Pcoef
    TL = ( 3.91*np.exp(0.689*pc)*aod550 + 0.376*np.log(w) + (2. + 0.54*pc - 0.5*pc**2 + 0.16*pc**3) )*0.8662
    return TL

