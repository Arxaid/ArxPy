# This file is part of ArxPy statistics library.
#
# Copyright (c) 2022 Vladislav Sosedov.

import numpy as np
import pandas as pd
import scipy.stats as st

from math import sqrt, erf, exp
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')

def DataLoading_csv(name):
    # Data loading from .csv file
    # Returns ndarray w/sorted data
    datasheet = pd.read_csv(name, header=None).to_numpy()
    datasheet.sort()
    return datasheet

def DataLoading_txt(name):
    # Data loading from .txt file
    # Returns list w/sorted data
    datasheet = []
    with open(name,'r') as reader:
        for value in reader:
            datasheet.append(float(value.rstrip('\n\r')))
    datasheet.sort()
    return datasheet

def DatasheetMathExp(datasheet):
    # Mathmatical expectation value of empirical distribution
    mx = sum(datasheet)/len(datasheet)
    return mx

def DatasheetDisp(datasheet):
    # Dispertion value of empirical distribution
    mx = DatasheetMathExp(datasheet=datasheet)
    dx = 0
    for x in datasheet:
        dx = dx + (x - mx)**2
    dx = dx/(len(datasheet) - 1)
    return dx

def DatasheetSigma(dx):
    # Standart deviation value of empirical distribution
    sigma = sqrt(dx)
    return sigma

def Expo(datasheet):
    # Exponential distribution pdf
    expo_pdf = []
    mx = DatasheetMathExp(datasheet=datasheet)
    for x in datasheet:
        expo_pdf.append(1 - exp(-(1/mx) * x))
    return expo_pdf

def Norm(datasheet):
    # Normal distribution pdf
    norm_pdf = []
    mx = DatasheetMathExp(datasheet=datasheet)
    dx = DatasheetDisp(datasheet=datasheet)
    for x in datasheet:
        norm_pdf.append(0.5 * (1 + erf((x - mx)/sqrt(2 * dx))))
    return norm_pdf

def ExpoScipy(datasheet):
    # Exponential distribution pdf via SciPy and NumPy libs
    params = st.expon.fit(datasheet)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    if arg:
        expo_pdf = st.expon.pdf(np.arange(30), *arg, loc=loc, scale=scale) * len(datasheet)
    else:
        expo_pdf = st.expon.pdf(np.arange(30), loc=loc, scale=scale) * len(datasheet)
    return expo_pdf

def NormScipy(datasheet):
    # Normal distribution pdf via SciPy and NumPy libs
    params = st.norm.fit(datasheet)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    if arg:
        norm_pdf = st.norm.pdf(np.arange(30), *arg, loc=loc, scale=scale) * len(datasheet)
    else:
        norm_pdf = st.norm.pdf(np.arange(30), loc=loc, scale=scale) * len(datasheet)
    return norm_pdf

def Histogram(datasheet, expo_pdf, norm_pdf):
    # Datasheet histogram w/distributions above it
    plt.hist(datasheet, bins=20, color='blue', edgecolor='black', label='Data')
    plt.plot(expo_pdf, label='Exponential distribution', color='red')
    plt.plot(norm_pdf, label='Normal distribution', color='orange')
    plt.legend(loc='upper right')
    plt.show()

def KSCritical(alpha):
    # Critical value for Kolmogorov–Smirnov statistics test
    critLambda = 0
    if alpha == 0.1:
        critLambda = 1.224
    if alpha == 0.05:
        critLambda = 1.358
    if alpha == 0.01:
        critLambda = 1.628
    return critLambda
    
def KSTest(datasheet, mx, dx, dist='norm', alpha=0.05, show=True):
    # Kolmogorov–Smirnov statistics test
    critLambda = KSCritical(alpha=alpha)
    dValue = 0

    if dist == 'norm':
        for counter in range(0, len(datasheet)):
            currentValue = abs((counter + 1)/len(datasheet) - 0.5 * (1 + erf((datasheet[counter] - mx)/sqrt(2 * dx))))
            dValue = max(dValue, currentValue)

    if dist == 'expo':
        for counter in range(0, len(datasheet)):
            currentValue = abs((counter + 1)/len(datasheet) - (1 - exp(-(1/mx) * datasheet[counter])))
            dValue = max(dValue, currentValue)
        
    currentLambda = (6 * len(datasheet) * dValue + 1)/(6 * sqrt(len(datasheet)))
    
    if show == True:
        if currentLambda <= critLambda:
            print('Kolmogorov–Smirnov statistics:   ', dValue)
            print('H0 accepted at the', alpha, 'significance level\n')
        if currentLambda > critLambda:
            print('Kolmogorov–Smirnov statistics:   ', dValue)
            print('H0 rejected at the', alpha, 'significance level\n')

    return dValue, currentLambda

def TTest(datasheet1, datasheet2, alpha=0.05, show=True):
    # Student t-test
    critLambda = KSCritical(alpha=alpha)

    mx1 = DatasheetMathExp(datasheet=datasheet1)
    dx1 = DatasheetDisp(datasheet=datasheet1)
    d, Lambda = KSTest(datasheet1, mx1, dx1, 'norm', alpha=alpha, show=False)
    if Lambda > critLambda:
        print('First datasheet doesnt follow normal distribution')
        exit(-1)

    mx2 = DatasheetMathExp(datasheet=datasheet2)
    dx2 = DatasheetDisp(datasheet=datasheet2)
    d, Lambda = KSTest(datasheet2, mx2, dx2, 'norm', alpha=alpha, show=False)
    if Lambda > critLambda:
        print('Second datasheet doesnt follow normal distribution')
        exit(-1)
    
    S = (dx1 * (len(datasheet1) - 1) + dx2 * (len(datasheet2) - 1))/(len(datasheet1) + len(datasheet2) - 2)
    tValue = (mx1 - mx2)/(sqrt(S * (1/len(datasheet1) + 1/len(datasheet2))))
    dof = len(datasheet1) + len(datasheet2) - 2

    tValueSepar = (mx1 - mx2)/(sqrt((dx1/len(datasheet1)) + (dx2/len(datasheet2))))
    dofSepar = ((dx1/len(datasheet1) + dx2/len(datasheet2))**2)/(((dx1/len(datasheet1))**2)/(len(datasheet1) - 1) + ((dx2/len(datasheet2))**2)/(len(datasheet2) - 1))


    if show == True:
        print('Mean Group 1:                    ',mx1)
        print('Mean Group 2:                    ',mx2)
        print('t-value:                         ',tValue)
        print('Degrees of freedom:              ',dof)
        print('t-value separated:               ',tValueSepar)
        print('Degrees of freedom:              ',dofSepar)

    return tValue