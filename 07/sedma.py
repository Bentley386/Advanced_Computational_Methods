# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:35:07 2018

@author: Admin
"""
import time
from PIL import Image
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.integrate import ode
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg as lin
from scipy.optimize import fsolve
from scipy.linalg import solve
from scipy.linalg import solve_banded
from scipy.special import jn_zeros #prvi parameter je order, drugi št. ničel
from scipy.special import jv #prvi order drugi argument
#from scipy.special import beta
import scipy.special as spec
import scipy.spatial as spat
import scipy.sparse
from scipy.optimize import root
from scipy.integrate import quad
from scipy.integrate import romb
from scipy.integrate import complex_ode
from scipy.integrate import trapz
from scipy.optimize import linprog
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.linalg import svd
from matplotlib.patches import Ellipse
from matplotlib import gridspec
from numba import jit
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc("text",usetex=True)
matplotlib.rcParams["text.latex.unicode"] = True
plt.close("all")
PI = np.pi


#HARMONSKI OSCILATOR ========================================00

@jit(nopython=True)
def V(x,lamb):
    return 0.5*x*x + lamb*x*x*x*x

@jit(nopython=True)
def V2(stara,beta,M,lamb):
    return V(stara[0],lamb)
    """
    en = 0.
    for j in range(M):
        en += 1/M * V(stara[j],lamb)
    return en
    """

@jit(nopython=True)
def K2(stara,beta,M,lamb):
    en = 0.5*M/beta
    for j in range(M):
        en -= 0.5*M/(beta*beta)*(stara[(j+1)%M]-stara[j])**2
    return en

@jit(nopython=True)
def K(beta):
    return -1/beta*np.log(np.sqrt(1/(2*PI*beta)))

@jit(nopython=True)
def poteza(stara,M,eps):
    nova = np.copy(stara)
    j = np.random.randint(0,M) #0 inclusive N exclusive
    nova[j]=stara[j]+eps*(2*np.random.rand()-1)
    return (nova,j)

@jit(nopython=True)
def prehodna(a,b,beta,M,lamb):
    return np.math.exp(-0.5*M/beta*(b-a)*(b-a)-beta/M*V(a,lamb))

@jit(nopython=True)
def E(stara,beta,M,lamb):
    en = 0.5*M/beta
    for j in range(M):
        en -= 0.5*M/(beta*beta)*(stara[(j+1)%M]-stara[j])**2
        en += 1/M * V(stara[j],lamb)
    return en

@jit(nopython=True)
def genZac(M):
    return 2*np.random.rand(M)-np.ones(M)

@jit(nopython=True)
def metropolisBrez(nsteps,beta,M,lamb,eps,zacetni):
    #sprejeti = 0
    prejsnji = np.copy(zacetni)
    trenutni = np.copy(prejsnji)
    for i in range(nsteps):
        prejsnji = trenutni
        trenutni, j = poteza(prejsnji,M,eps)
        a=prehodna(prejsnji[(j-1)%M],prejsnji[j],beta,M,lamb)*prehodna(prejsnji[j],prejsnji[(j+1)%M],beta,M,lamb)
        if a==0:
            continue
        verj = prehodna(prejsnji[(j-1)%M],trenutni[j],beta,M,lamb)*prehodna(trenutni[j],prejsnji[(j+1)%M],beta,M,lamb)/a
        if verj > 1:
            #sprejeti+=1
            continue
        else:
            U = np.random.rand()
            if U < verj:
                #sprejeti+=1
                continue
            else:
                trenutni=prejsnji
    #return sprejeti/nsteps
    return trenutni


@jit(nopython=True)
def metropolisKor(nsteps,beta,M,lamb,eps,zacetni):
    #vrnjene = np.zeros((int(nsteps/50),M))
    vrnjene=np.zeros(nsteps)
    prejsnji = np.copy(zacetni)
    trenutni = np.copy(prejsnji)
    for i in range(nsteps):
        """
        if i%50==0:
            indeks = int(i/50)
            rez[indeks][0]=E(trenutni,beta,M,lamb)
            rez[indeks][1]=V2(trenutni,beta,M,lamb)
            rez[indeks][2]=rez[indeks][0]-rez[indeks][1]
            #rez[indeks][2]=K2(trenutni,beta,M,lamb)
        """
        vrnjene[i]=E(trenutni,beta,M,lamb)
        #vrnjene[i]=trenutni[0]
        prejsnji = trenutni
        trenutni, j = poteza(prejsnji,M,eps)
        a=(prehodna(prejsnji[(j-1)%M],prejsnji[j],beta,M,lamb)*prehodna(prejsnji[j],prejsnji[(j+1)%M],beta,M,lamb))
        if a==0:
            continue
        verj = prehodna(prejsnji[(j-1)%M],trenutni[j],beta,M,lamb)*prehodna(trenutni[j],prejsnji[(j+1)%M],beta,M,lamb)/a
        if verj > 1:
            #sprejeti+=1
            continue
        else:
            U = np.random.rand()
            if U < verj:
                continue
            else:
                trenutni=prejsnji
    #return 0
    return vrnjene

def energija(beta,podatki,M,lamb):
    n = len(podatki)
    energije = np.array([E(podatki[i],beta,M,lamb) for i in range(n)])
    return (np.sum(energije)/n,1/np.sqrt(n)*(1/n*np.sum(energije**2)-np.sum(1/n*energije)**2))

def potencialna(beta,podatki,M,lamb):
    n = len(podatki)
    energije = np.array([V2(podatki[i],beta,M,lamb) for i in range(n)])
    #energije = np.array([V(podatki[i][0],lamb) for i in range(n)])
    return (np.sum(energije)/n,1/np.sqrt(n)*(1/n*np.sum(energije**2)-np.sum(1/n*energije)**2))


def kineticna(beta,podatki,M,lamb):
    #return K(beta)
    n = len(podatki)
    energije = np.array([K2(podatki[i],beta,M,lamb) for i in range(n)])
    return (np.sum(energije)/n,1/np.sqrt(n)*(1/n*np.sum(energije**2)-np.sum(1/n*energije)**2))

"""
def epsil(beta):
    return 0.063*beta**(0.517)
"""
def epsil(beta):
    return 0.8*beta**0.488

def vseBeta(beta,lamb,M,prva,stevilo=10000000,cakaj=10000,povprecuj=50):
    eps = epsil(beta)
    prva = metropolisBrez(cakaj,beta,M,lamb,eps,genZac(M))
    podatki=metropolisKor(stevilo,beta,M,lamb,eps,prva)
    """
    for i in range(stevilo):
        prva = metropolisBrez(povprecuj,beta,M,lamb,eps,prva)
        podatki[i]=prva
    """
    en1,en2 = energija(beta,podatki,M,lamb)
    pot1,pot2 = potencialna(beta,podatki,M,lamb)
    kin1,kin2 = kineticna(beta,podatki,M,lamb)
    return [np.array([en1,en2,pot1,pot2,kin1,kin2]),podatki[-1]]

def analiticenH(N,lamb):
    qji = np.zeros((N,N))
    matrika = np.zeros((N,N))
    for i in range(N):
        matrika[i][i] = i + 1/2 
        for j in range(i):
            if abs(i-j)==1:
                qji[i][j]=0.5*np.sqrt(i+j+1)
                qji[j][i]=0.5*np.sqrt(i+j+1)
    return matrika + lamb*np.linalg.matrix_power(qji,4)


if 0:
    #en osnovnega stanja od lambde
    #M=600
    lambde=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    prejsnja = []
    M=400
    bete = [i*0.1 for i in range(10,101)]
    for lamb in lambde:
        H=lin.eigh(analiticenH(500,lamb))[0][:100]
        prejsnja.append(np.sum(H*np.exp(-5*H)/np.sum(np.exp(-5*H))))
    plt.plot(lambde,prejsnja,label="Diagonalizacija")
    rez = []
    for lamb in lambde:
        prva = genZac(M)
        print(lamb)
        for beta in bete:
            eps = epsil(beta)
            prva = metropolisBrez(10000,beta,M,lamb,eps,prva)
            if beta==1:
                break
        ostale = metropolisKor(10000000,100,M,lamb,eps,prva)[::50]
        rez.append(sum(ostale)/len(ostale))
    plt.plot(lambde,rez,label=r"Monte Carlo, $M={}$".format(M))
    plt.grid(True)
    plt.legend(loc="best")
    plt.xlabel(r"$\lambda$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$E$",fontsize=14)
    plt.title(r"Energija pri $\beta=10$",fontsize=13)
    plt.savefig("Figures/graf5.pdf")

if 0:
    #isto kot spodaj samo za vec lambda
    cakaj=500000
    n=20000
    M=100
    lambde=[0,0.25,0.5,0.75,1]
    bete = [i*0.5 for i in range(1,41)]
    for lamb in lambde:
        prva = genZac(M)
        energije=[]
        #ennapak=[]
        #potencialne=[]
        #potnapak=[]
        #kineticne=[]
        #kinnapak=[]
        rez=np.ones((n,3))
        for beta in bete:
            print(beta)
            eps = epsil(beta)
            prva = metropolisBrez(10000000,beta,M,lamb,eps,prva)
            ostale = metropolisKor(1000000,beta,M,lamb,eps,prva,rez)
            energije.append(np.sum(rez[:,0])/n)
            #ennapak.append(1/np.sqrt(n)*(1/n*np.sum(rez[:,0]**2)-energije[-1]**2))
            #potencialne.append(np.sum(rez[:,1])/n)
            #potnapak.append(1/np.sqrt(n)*(1/n*np.sum(rez[:,1]**2)-potencialne[-1]**2))
            #kineticne.append(np.sum(rez[:,2])/n)
            #kinnapak.append(1/np.sqrt(n)*(1/n*np.sum(rez[:,2]**2)-kineticne[-1]**2))
            rez = np.zeros((n,3))
        plt.plot(bete,energije,linestyle="--",label=r"$E, \lambda={}$".format(lamb))
        #plt.errorbar(bete,energije,yerr=ennapak,color="red",linestyle="--",label=r"$E$")
        #plt.errorbar(bete,potencialne,yerr=potnapak,color="orange",linestyle="--",label=r"$V$")
        #plt.errorbar(bete,kineticne,yerr=np.array(potnapak)+np.array(ennapak),color="magenta",linestyle="--",label=r"$T$")
        #plt.errorbar(bete,potencialne,yerr=potnapak,color="orange",linestyle="--",label=r"$V$")
        #plt.errorbar(bete,kineticne,yerr=kinnapak,linestyle="--",color="magenta",label=r"$T$")
    plt.xlim(0.5,20)
    plt.ylim(0,3)
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$\beta$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    #plt.ylabel(r"$C$",fontsize=14)
    plt.title(r"Odvisnost energije od $\beta$, $M=100$",fontsize=13)
    plt.savefig("Figures/graf2.pdf")

if 0:
    #E(beta)za vec M
    cakaj=500000
    n=20000
    M=100
    lamb = 0.5
    Mji = [100,200,300,400,500,600]
    bete = [i*0.5 for i in range(2,11)]
    for M in Mji:
        prva = genZac(M)
        energije=[]
        #ennapak=[]
        #potencialne=[]
        #potnapak=[]
        #kineticne=[]
        #kinnapak=[]
        rez=np.ones((n,3))
        for beta in bete:
            #print(beta)
            eps = epsil(beta)
            prva = metropolisBrez(10000000,beta,M,lamb,eps,prva)
            ostale = metropolisKor(1000000,beta,M,lamb,eps,prva,rez)
            energije.append(np.sum(rez[:,0])/n)
            #ennapak.append(1/np.sqrt(n)*(1/n*np.sum(rez[:,0]**2)-energije[-1]**2))
            #potencialne.append(np.sum(rez[:,1])/n)
            #potnapak.append(1/np.sqrt(n)*(1/n*np.sum(rez[:,1]**2)-potencialne[-1]**2))
            #kineticne.append(np.sum(rez[:,2])/n)
            #kinnapak.append(1/np.sqrt(n)*(1/n*np.sum(rez[:,2]**2)-kineticne[-1]**2))
            rez = np.zeros((n,3))
        print(energije[-1])
        plt.plot(bete,energije,linestyle="--",label=r"$E, M={}$".format(M))
        #plt.yscale("log")
        #plt.errorbar(bete,energije,yerr=ennapak,color="red",linestyle="--",label=r"$E$")
        #plt.errorbar(bete,potencialne,yerr=potnapak,color="orange",linestyle="--",label=r"$V$")
        #plt.errorbar(bete,kineticne,yerr=np.array(potnapak)+np.array(ennapak),color="magenta",linestyle="--",label=r"$T$")
        #plt.errorbar(bete,potencialne,yerr=potnapak,color="orange",linestyle="--",label=r"$V$")
        #plt.errorbar(bete,kineticne,yerr=kinnapak,linestyle="--",color="magenta",label=r"$T$")
    plt.xlim(1,5)
    #plt.ylim(0,4)
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$\beta$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    #plt.ylabel(r"$C$",fontsize=14)
    plt.title(r"Odvisnost energije od $\beta, \lambda=0.5$",fontsize=13)
    plt.savefig("Figures/graf4.pdf")
    
if 0:
    #vse stvari v odv od beta
    n=20000
    M=100
    lamb=0
    bete = [i*0.5 for i in range(1,61)]
    prva = genZac(M)
    energije=[]
    ennapak=[]
    potencialne=[]
    potnapak=[]
    kineticne=[]
    kinnapak=[]
    rez=np.ones((n,3))
    for beta in bete:
        print(beta)
        eps = epsil(beta)
        prva = metropolisBrez(10000000,beta,M,lamb,eps,prva)
        ostale = metropolisKor(1000000,beta,M,lamb,eps,prva,rez)
        energije.append(np.sum(rez[:,0])/n)
        ennapak.append(1/np.sqrt(n)*(1/n*np.sum(rez[:,0]**2)-energije[-1]**2))
        potencialne.append(np.sum(rez[:,1])/n)
        potnapak.append(1/np.sqrt(n)*(1/n*np.sum(rez[:,1]**2)-potencialne[-1]**2))
        kineticne.append(np.sum(rez[:,2])/n)
        kinnapak.append(1/np.sqrt(n)*(1/n*np.sum(rez[:,2]**2)-kineticne[-1]**2))
        rez = np.zeros((n,3))
    plt.xlim(0.5,30)
    plt.ylim(0,4)
    plt.errorbar(bete,energije,yerr=ennapak,color="red",linestyle="--",label=r"$E$")
    plt.errorbar(bete,potencialne,yerr=potnapak,color="orange",linestyle="--",label=r"$V$")
    plt.errorbar(bete,kineticne,yerr=np.array(potnapak)+np.array(ennapak),color="magenta",linestyle="--",label=r"$T$")
    #plt.errorbar(bete,potencialne,yerr=potnapak,color="orange",linestyle="--",label=r"$V$")
    #plt.errorbar(bete,kineticne,yerr=kinnapak,linestyle="--",color="magenta",label=r"$T$")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$\beta$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    #plt.ylabel(r"$C$",fontsize=14)
    plt.title(r"Odvisnost energije od $\beta$, $M=100$",fontsize=13)
    plt.savefig("Figures/graf1.pdf")


if 0:
    #korelacija izmerkov lol
    #avtokorelacija
    cakaj=1000000
    nsteps=100000
    beta=5
    lamb=0
    eps=epsil(beta)
    M=100
    zacetni=genZac(M)
    rez = metropolisBrez(cakaj,beta,M,lamb,eps,zacetni)
    rez = metropolisKor(nsteps,beta,M,lamb,eps,rez)
    rez -= np.average(rez)
    #plt.plot(list(range(len(rez))),rez)
    kor = np.abs(np.fft.ifft(np.fft.fft(rez)*np.conj(np.fft.fft(rez))))
    #kor = np.correlate(rez,np.concatenate((rez,rez)))
    #print(kor)
    kor = kor/max(kor)
    x = [-int(len(kor)/2) + i for i in range(len(kor))]
    plt.plot(x,np.concatenate((kor[int(len(kor)/2):],kor[:int(len(kor)/2)])))
    #plt.plot([i for i in range(len(rez))],rez,label=r"$\beta={}$".format(beta))
    #plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$i$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$C$",fontsize=14)
    plt.title(r"Avtokorelacija izmerkov energije, $\beta=5, M=100$",fontsize=13)
    plt.savefig("Figures/avtokor.pdf")

if 0:
    #q(t)
    beta=10
    lamb=0
    eps=epsil(beta)
    M=100
    zacetni=genZac(M)
    nsteps = 1000000
    cakaj = 100000
    stara = metropolisBrez(cakaj,beta,M,lamb,eps,zacetni)
    rez = metropolisKor(nsteps,beta,M,lamb,eps,stara)[:100]
    suma=0
    povprecja = []
    for i in range(len(rez)):
        suma += rez[i]
        povprecja.append(suma/(i+1))
    plt.plot(list(range(1,len(rez)+1)),rez,".")
    plt.plot(list(range(1,len(povprecja)+1)),povprecja,"--",color="k")
    #plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$i$",fontsize=14)
    plt.title(r"$q(t)$  $\beta=10,M=100$",fontsize=13)
    plt.savefig("Figures/qt.pdf") 

if 0:
    #koliko casa do povpre
    beta=30
    lamb=0
    eps=epsil(beta)
    M=100
    zacetni=genZac(M)
    nsteps = 1000000
    cakanja = [10000,100000,1000000,5000000,50000000,100000000]
    for cakaj in cakanja:
        stara = metropolisBrez(cakaj,beta,M,lamb,eps,zacetni)
        rez = metropolisKor(nsteps,beta,M,lamb,eps,stara)[::2500]
        suma=0
        povprecja = []
        for i in range(len(rez)):
            suma += rez[i]
            povprecja.append(suma/(i+1))
        plt.plot(list(range(1,len(povprecja)+1)),povprecja,label=r"$n={}$".format(cakaj))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$i$",fontsize=14)
    plt.ylabel(r"$\langle E \rangle$",fontsize=14)
    plt.title(r"Tekoča povprečja energije po relaksaciji $\beta=30,M=100$",fontsize=13)
    plt.savefig("Figures/povp.pdf")  

if 0:
    #iskanje eps(beta)
    Mji = [100,200,300,400,500,600]
    bete = list(range(1,26))
    epsiloni=[0.01*i for i in range(1,101)]
    for M in Mji:
        optimalni = []
        zacetni = genZac(M)
        for beta in bete:
            sprejeti=[]
            for eps in epsiloni:
                sprejeti.append(metropolisBrez(100000,beta,M,0,eps,zacetni))
            a = np.abs(np.array(sprejeti)-0.5*np.ones(100))
            arg = np.argmin(a)
            optimalni.append(epsiloni[arg])
            print(a[arg])
        def curva(x,a,b):
            return a*x**b
        plt.plot(bete,optimalni,"x",color="k")
        y1,y2 = curve_fit(curva,bete,optimalni)
        a,b = y1
        x=np.linspace(1,25,1000)
        plt.plot(x,a*x**b,label=r"$M={}$".format(M))
        if M==600:
            plt.text(10,0.1,r"$\epsilon=" + str(round(a,3)) + r"\cdot \beta^{" + str(round(b,3)) + "}$",fontsize=18)
        if M==100:
            plt.text(7,0.9,r"$\epsilon=" + str(round(a,3)) + r"\cdot \beta^{" + str(round(b,3)) + "}$",fontsize=18)
    #plt.text(10,0.2,r"$\epsilon=" + str(round(a,3)) + r"\cdot \beta^{" + str(round(b,3)) + "}$",fontsize=25)
    plt.xlabel(r"$\beta$",fontsize=14)
    plt.legend(loc="best")
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$\epsilon$",fontsize=14)
    plt.title(r"$\epsilon(\beta)$ za $0.5$ delež sprejema $n=10^5$",fontsize=13)
    plt.savefig("Figures/sprejeti6.pdf")            
        

if 0:
    #delez sprejetih v odv od epsilon za več M
    epsiloni = [0.05*i for i in range(21)]
    M=100
    #Mji = [25,50,75,100,125,150,175,200]
    lambde=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    beta = 10
    for lamb in lambde:
        zacetni = genZac(M)
        sprejeti=[]
        for eps in epsiloni:
            sprejeti.append(metropolisBrez(100000,beta,M,lamb,eps,zacetni))
        plt.plot(epsiloni,sprejeti,".:",label=r"$\lambda={}$".format(lamb))
    plt.legend(loc="best",ncol=3)
    plt.axhline(0.5,0,1,color="k")
    plt.grid(True)
    plt.xlim(0,1)
    plt.xlabel(r"$\epsilon$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"Delež",fontsize=14)
    plt.title(r"Delež sprejetih potez, $\beta=10, M=100, n=10^5$",fontsize=13)
    plt.savefig("Figures/sprejeti5.pdf")

if 0:
    #delez sprejetih v odv od epsilon za več beta
    epsiloni = [0.05*i for i in range(1,21)]
    bete = [1,2,3,4,5,10,15,20,25]
    M=100
    zacetni = genZac(M)
    for beta in bete:
        sprejeti=[]
        for eps in epsiloni:
            sprejeti.append(metropolisBrez(100000,beta,M,0,eps,zacetni))
        plt.plot(epsiloni,sprejeti,".:",label=r"$\beta={}$".format(beta))
    plt.legend(loc="best",ncol=3)
    plt.axhline(0.5,0,1,color="k")
    plt.grid(True)
    plt.xlim(0,1)
    plt.xlabel(r"$\epsilon$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"Delež",fontsize=14)
    plt.title(r"Delež sprejetih potez, $M=100, n=10^5$",fontsize=13)
    plt.savefig("Figures/sprejeti1.pdf")


 


#heisenberg =============
    
def genStanjaHeis(N):
    vmesni = [[0],[1]]
    while len(vmesni) != 2**N:
        novi = []
        for konf in vmesni:
            novi.append(konf + [0])
            novi.append(konf + [1])
        vmesni = [novi[i] for i in range(len(novi))]
    return np.array(vmesni)

def genZacHeis(M,stanja,N,n):
    rez = np.zeros((M,N))
    for i in range(M):
        rez[i] = stanja[np.random.randint(0,n)]
    return rez

@jit(nopython=True)
def toBinaryHeis(arej):
    rez = 0
    for i in range(1,len(arej)+1):
        rez += arej[-i]*(2**(i-1))
    return rez

@jit(nopython=True)
def potezaHeis(stara,M,n,stanja):
    nova = np.copy(stara)
    j = np.random.randint(0,M) #0 inclusive N exclusive
    i = np.random.randint(0,n)
    nova[j]=stanja[i]
    return (nova,j)

@jit(nopython=True)
def genU(z):
    U = np.zeros((4,4))
    U[0][0]=np.exp(2*z)
    U[1][1]=np.cosh(2*z)
    U[1][2]=np.sinh(2*z)
    U[2][1]=np.sinh(2*z)
    U[2][2]=np.cosh(2*z)
    U[3][3]=np.exp(2*z)
    return np.exp(-z)*U
    #return np.exp(-z)*np.array([[np.exp(2*z),0,0,0],[0,np.cosh(2*z),np.sinh(2*z),0],[0,np.sinh(2*z),np.cosh(2*z),0],[0,0,0,np.exp(2*z)]])

@jit(nopython=True)
def tensorConv(a,b):
    if a==0:
        if b==0:
            return 0
        if b==1:
            return 1
    if b==0:
        return 2
    if b==1:
        return 3
    print(1/0)

@jit(nopython=True)
def P(a,b,beta,M,N):
    U = genU(-beta/M)
    prod=1
    for k in range(int(N/2)):
        prod*= U[int(tensorConv(a[2*k-1],a[2*k]))][int(tensorConv(b[2*k-1],b[2*k]))]
    return prod

@jit(nopython=True)
def Q(a,b,beta,M,N):
    U = genU(-beta/M)
    prod=1
    for k in range(int(N/2)):
        prod*= U[int(tensorConv(a[2*k],a[(2*k+1)%N]))][int(tensorConv(b[2*k],b[(2*k+1)%N]))]
    return prod

@jit(nopython=True)
def prehodnaHeis(a,b,j,beta,M,N):
    if j%2==1:
        return P(a,b,beta,M,N)
        #P
    else:
        return Q(a,b,beta,M,N)
        #Q

@jit(nopython=True)
def ZHeis(trenutni,beta,M,N):
    Z=1
    for i in range(M):
        Z*= prehodnaHeis(trenutni[i],trenutni[(i+1)%M],i,beta,M,N)
    return Z


@jit(nopython=True)
def metropolisBrezHeis(nsteps,beta,M,N,n,stanja,zacetni):
    #sprejeti = 0
    prejsnji = np.copy(zacetni)
    trenutni = np.copy(prejsnji)
    for i in range(nsteps):
        prejsnji = trenutni
        trenutni, j = potezaHeis(prejsnji,M,n,stanja)
        a=prehodnaHeis(prejsnji[(j-1)%M],prejsnji[j],j,beta,M,N)*prehodnaHeis(prejsnji[j],prejsnji[(j+1)%M],j,beta,M,N)
        if a==0:
            continue
        verj = prehodnaHeis(prejsnji[(j-1)%M],trenutni[j],j,beta,M,N)*prehodnaHeis(trenutni[j],prejsnji[(j+1)%M],beta,j,M,N)/a
        if verj > 1:
            #sprejeti+=1
            continue
        else:
            U = np.random.rand()
            if U < verj:
                #sprejeti+=1
                continue
            else:
                trenutni=prejsnji
    #return sprejeti/nsteps
    return trenutni

  

@jit(nopython=True)
def metropolisKorHeis(nsteps,beta,M,N,n,stanja,zacetni):
    Z=0
    prejsnji = np.copy(zacetni)
    trenutni = np.copy(prejsnji)
    for i in range(nsteps):
        if i%100==0:
            print(Z)
            Z+= ZHeis(trenutni,beta,M,N)
        prejsnji = trenutni
        trenutni, j = potezaHeis(prejsnji,M,n,stanja)
        a=prehodnaHeis(prejsnji[(j-1)%M],prejsnji[j],j,beta,M,N)*prehodnaHeis(prejsnji[j],prejsnji[(j+1)%M],j,beta,M,N)
        if a==0:
            continue
        verj = prehodnaHeis(prejsnji[(j-1)%M],trenutni[j],j,beta,M,N)*prehodnaHeis(trenutni[j],prejsnji[(j+1)%M],beta,j,M,N)/a
        if verj > 1:
            #sprejeti+=1
            continue
        else:
            U = np.random.rand()
            if U < verj:
                #sprejeti+=1
                continue
            else:
                trenutni=prejsnji
    #return sprejeti/nsteps
    return Z
    return trenutni
"""
N=8
n=2**N
M=10
beta=20
stanja = genStanjaHeis(N)
zacetni = genZacHeis(M,stanja,N,n)
koncni = metropolisBrezHeis(1000000,beta,M,N,n,stanja,zacetni)
koncni = metropolisKorHeis(100000,beta,M,N,n,stanja,koncni)
Z = koncni/2000
print(1/(-beta)*np.log(Z))
"""