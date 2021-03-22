#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 19:26:11 2019

@author: agustin
"""
import numpy as np
import pylab as plt
# graficar circul0
#radio r

#ejemplo para graficar

def semicirculo(r, paso):
    #paso= 0.01
    x = np.arange(-r,r+paso,paso)
    y = np.sqrt( r**2 -np.round(x**2,3))    
    return x, y


def graficar_circulo_MBD_sigma(CIRC, est):
    MBD = CIRC.loc['MBD']*100
    sigma = CIRC.loc['sigma']*100
    modelos = CIRC.columns
    
#    CIRC = pd.DataFrame([] ,columns=modelos, index = ['MBD', 'sigma'])
#    CIRC.loc['MBD']= MET.loc['rMBD']
#    CIRC.loc['sigma']= np.sqrt( MET.loc['rRMS']**2 - MET.loc['rMBD']**2)
    
    #MBD = np.array([1.2,-1.1, 0.5, 2.3, 2.9])
#RMS = np.array([3.2, 2.8, 4.0, 3.8, 4.0])
#model=['modelo1','modelo2', 'modelo3', 'modelo4','modelo5']
#sigma= np.sqrt(RMS**2 - MBD**2)
    formas= ['o', 'v', 's', 'D', '^', 'H','<', 'P','>', 'X', 'p', 'o', 'v', 's', 'D' ]
    fig = plt.figure(figsize=(12,6))#, constrained_layout=True)
    lim = 6
    ax = fig.add_subplot(1, 1, 1)
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.grid(linestyle='-', linewidth=1, color='lightgray')
    plt.title('estación  : '+ est)
    
    for radio in np.arange(1,lim+1):
        paso = 0.01
        plt.plot(semicirculo(radio, paso)[0],semicirculo(radio, paso)[1], 'k--', lw=0.5)
    
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    #ax.spines['bottom'].set_position(0)
    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    plt.ylim((0,lim))
    plt.xlim((-lim,lim))
    plt.xlabel('rMBD')
    plt.ylabel('$\sigma$')
    plt.text(-lim/1.1, lim/1.1, 'rRMSD', fontsize=18)

    for j in range(len(MBD)):
        plt.plot(MBD[j], sigma[j], formas[j], markersize=18, label=modelos[j])
    plt.legend( fontsize = 12)
#plt.xlabel(r'Ángulo azimutal [$^o$]')
#plt.ylabel(r'Altura solar [$^o$]')
#plt.tight_layout()
