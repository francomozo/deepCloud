# modelos FRACCION DIFUSA
#---Agustin Laguarda
import numpy as np

def Fraccion_Difusa_Horaria_Erbs(kt): # modelo Erbs 1982
    #---29/06/2015
    #---25/09/2018-version Python
    N=len(kt)
    fd=0.165 * np.ones(N)  #kt>=0.8
    fd[kt<0.8]=0.9511 - 0.16*kt[kt<0.8]+4.388*kt[kt<0.8]**2-16.638*kt[kt<0.8]**3+12.334*kt[kt<0.8]**4 #kt entre 0.22 y 0.8
    fd[kt<0.22]= 1 - 0.09*kt[kt<0.22]  #kt <0.2
    return fd
