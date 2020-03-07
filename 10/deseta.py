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
from scipy.integrate import simps
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
pi = np.pi

epsil = 0

zacetnipribl = [0,0.1]


def k2(V,E): #Izracuna k kvadrat za dano schr. enacbo
    return 2*(-V+E)
def numerovkorak(u,x,h,V,E): #en korak numerova
    """y = [y_i-1,y_i], x = [x_i-1,x_i,x_i+1]"""
    faktor = 1/(1+h*h/12*k2(V[2],E))
    prvi = 2*(1-5*h*h/12*k2(V[1],E))
    drugi = -(1+h*h/12*k2(V[0],E))
    return faktor*(prvi*u[1]+drugi*u[0])

def numerov(x,E,h,maxt,velikost,V): #iz zacetnih pridemo do resitve shcr
    u = np.ones(velikost)
    u[0]=zacetnipribl[0]
    #print(zacetnipribl)
    u[1] = zacetnipribl[1] 
    for i in range(2,velikost):
        #print(i)
        u[i]= numerovkorak(u[i-2:i],x[i-2:i+1],h,V[i-2:i+1],E)
    return (x,u) #mogoce obiwan error?

def resiNumerov(h,V,maxt,eps=0.000000001): #bisekcija na e
    velikost= int(maxt/h)
    x = np.linspace(10**(-10),maxt,velikost)
    leva = -20
    desna = 0
    while (desna-leva) > eps:
        sredina = (desna+leva)/2
        #print(sredina)
        resitev = numerov(x,sredina,h,maxt,velikost,V)[1][-1]
        resitevleva = numerov(x,leva,h,maxt,velikost,V)[1][-1]
        if resitevleva*resitev < 0:
            desna = sredina
        else:
            leva = sredina
    pravaresitev = numerov(x,sredina,h,maxt,velikost,V)
    global epsil
    epsil=sredina
    norm = np.sum(pravaresitev[1]**2 * h)
    return (pravaresitev[0],pravaresitev[1]/np.sqrt(norm))

def dobiU(x,h,maxt,velikost,u):
    """
    V = np.ones(velikost)
    U = np.zeros(velikost)
    for i in range(1,velikost):
        U[i] = U[i-1]+h*V[i-1]
        V[i] = V[i-1]+h*(-u[i-1]**2/x[i-1])
    """
    V = np.ones(velikost)
    for i in range(1,velikost):
        V[i] = simps(-u[:i]**2/x[:i],x[:i])+1
    U = np.zeros(velikost)
    for i in range(1,velikost):
        U[i] = simps(V[:i],x[:i])
    k = (1-U[-1])/x[-1]
    U = U+k*x
    return U

def dobiV(U,x,u):
    hf = 2*U/x
    xc = -(1.5*(u/(pi*x))**2)**(1/3)
    return -2/x + hf +xc

def izracunE(U,x,u):
    hf = 2*U/x
    xc = -(1.5*(u/(pi*x))**2)**(1/3)
    return 2*epsil - simps(hf*u*u,x)-0.5*simps(xc*u*u,x)

def hartreeFock(x,h,maxt,velikost,zac,delta=0.01): #HF iteracija
    x = np.linspace(10**(-10),maxt,int(maxt/h))
    zacetna = zac
    prejsnja = np.ones(x.size)
    energije = []
    potenciali = []
    psiji = [zac]
    i = 1
    while np.sum(np.abs(prejsnja-zacetna))>delta:
        print(i)
        prejsnja = zacetna
        U = dobiU(x,h,maxt,velikost,zacetna)
        V = dobiV(U,x,zacetna)
        potenciali.append(U)
        energije.append(izracunE(U,x,prejsnja))
        temp = resiNumerov(h,V,maxt)
        zacetna = temp[1]
        psiji.append(zacetna)
        i+=1
    #return (U,V,zacetna,zacetna/x,energije[-1])
    return (energije,potenciali,psiji)


if 1:
    #U za vec zacentih u0
    h = 0.001
    maxt=4.5
    velikost = int(maxt/h)
    x = np.linspace(10**(-10),maxt,velikost)
    u = 2*x*np.exp(-x)
    plt.plot(x,u*u/x,label=r"$u_0 = 2re^{-r}$")
    u = 1-np.exp(-x) 
    plt.plot(x,u*u/x,label=r"$u_0 = 1-e^{-r}$")
    u = np.sin(x)
    plt.plot(x,u*u/x,label=r"$u_0 = \sin(r)$")
    u = np.sin(3*x)**2
    plt.plot(x,u*u/x,label=r"$u_0 = \sin^2(3r)$")
    plt.ylabel(r"$u_0^2/r$",fontsize=14)
    plt.xlabel(r"$r$",fontsize=14)
    plt.legend(loc="best")
    plt.title(r"$u_0^2(r)/r$",fontsize=13)
    plt.savefig("figures/zadnje3.pdf")

if 0:
    #U za vec zacentih u0
    h = 0.001
    maxt=4.5
    velikost = int(maxt/h)
    x = np.linspace(10**(-10),maxt,velikost)
    u = 2*x*np.exp(-x)
    energije, potenciali, psiji = hartreeFock(x,h,maxt,velikost,u)
    plt.plot(x,potenciali[1],label=r"$2re^{-r}$")    
    u = 1-np.exp(-x)
    energije, potenciali, psiji = hartreeFock(x,h,maxt,velikost,u)
    plt.plot(x,potenciali[1],label=r"$1-e^{-x}$") 
    u = np.sin(x)
    energije, potenciali, psiji = hartreeFock(x,h,maxt,velikost,u)
    plt.plot(x,potenciali[1],label=r"$\sin(r)$")
    u = np.sin(3*x)**2
    energije, potenciali, psiji = hartreeFock(x,h,maxt,velikost,u)
    plt.plot(x,potenciali[1],label=r"$\sin^2(3r)$") 
    plt.ylabel(r"$U(r)$",fontsize=14)
    plt.xlabel(r"$r$",fontsize=14)
    plt.legend(loc="best")
    plt.title(r"$U(r)$ po dveh iteracijah za več $u_0(r)$",fontsize=13)
    plt.savefig("figures/zadnje2.pdf")
if 0:
    #energija v  odv od delta
    hji= [0.0005 + 0.0001*i for i in range(11)]
    energije = []
    maxt=4.5
    for h in hji:
        print(h)
        velikost=int(maxt/h)
        x = np.linspace(10**(-10),maxt,velikost)
        uzac = 2*np.exp(-x)*x
        en, pot, psi = hartreeFock(x,h,maxt,velikost,uzac)
        energije.append(en[-1])
    plt.xlabel(r"$h$",fontsize=14)
    plt.ylabel(r"$\Delta E$",fontsize=14)
    plt.grid()
    #plt.yscale("log")
    plt.title(r"Odvisnost $\Delta E$ od $h$, $\delta=0.01, r_{max}=10$",fontsize=13)
    plt.plot(hji,np.abs(np.array(energije)+2.72),"-o")
    plt.savefig("figures/napake5.pdf")
if 0:
    #slike stvari skozi iteracijo
    h = 0.001
    maxt=4.5
    velikost = int(maxt/h)
    x = np.linspace(10**(-10),maxt,velikost)
    u = 2*x*np.exp(-x)
    energije, potenciali, psiji = hartreeFock(x,h,maxt,velikost,u)
    if 0: #vse koncno
        U,V,zac,psi,energija = hartreeFock(x,h,maxt,velikost,u)
        #ax.plot(x,zac,label=r"$u(r)$")
        #ax.plot(x[1:],psi[1:],label=r"$\psi(r)$")
        f, (ax1,ax2) = plt.subplots(1,2,gridspec_kw={"width_ratios" : [1,3]})
        ax1.plot(x[:1000],U[:1000])
        ax1.plot(x[1:1000],V[1:1000])
        ax1.set_xlabel(r"$r$",fontsize=14)
        #ax1.set_xlim(-0.01,5)
        plt.suptitle(r"Helijev atom, $r_{max}=10, h=0.001$",fontsize=13)
        ax2.plot(x,U,label=r"$U(r)$")
        ax2.plot(x,np.abs(V),label=r"$|V(r)|$")
        ax2.legend(loc="best")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$r$",fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("figures/Konec2.pdf")
        
    if 0: #napaka energije
        plt.plot(list(range(17)),np.abs(np.array(energije)+2.72))
        plt.ylabel(r"$\Delta E$",fontsize=14)
        plt.xlabel(r"$n$",fontsize=14)
        #plt.axhline(-2.72,0,16,color="red",ls="--")
        #plt.text(8,-2.65,r"$E=-2.72$")
        plt.yscale("log")
        #plt.legend(loc="best")
        plt.grid()
        plt.title(r"Napaka energije, $r_{max}=10, h=0.001$",fontsize=13)
        plt.savefig("figures/EE1.pdf")
        
    if 0: #E(n)
        plt.plot(list(range(15)),energije)
        plt.ylabel(r"$E$",fontsize=14)
        plt.xlabel(r"$n$",fontsize=14)
        plt.axhline(-2.72,0,16,color="red",ls="--")
        plt.text(6,-2.65,r"$E=-2.72$")
        plt.grid()
        #plt.legend(loc="best")
        plt.title(r"Energija na vsakem koraku algoritma $n$, $u_0(r) = \sin^2 (3r)$",fontsize=13)
        #plt.savefig("figures/evol3.pdf")
        
    if 0: #evolucija priblizkov
        plt.plot(x,psiji[0],label=r"Zacetni priblizek")
        plt.plot(x,psiji[1],label=r"1. Korak")
        plt.plot(x,psiji[2],label=r"2. Korak")
        plt.plot(x,psiji[3],label=r"3. Korak")
        plt.plot(x,psiji[-1],label=r"15. Korak")
        plt.ylabel(r"$u(r)$",fontsize=14)
        plt.xlabel(r"$r$",fontsize=14)
        plt.legend(loc="best")
        plt.title(r"Spreminjanje $u(r)$ med algoritmom, $u_0(r) = \sin^2 (3r)$",fontsize=13)
        plt.savefig("figures/evol3.pdf")
"""
#V = -1/x
#x,y = resiNumerov(h,V,maxt)
#U = dobiU(x,h,maxt,velikost,u)
#plt.plot(x,U)
#plt.plot(x,-(x+1)*np.exp(-2*x)+1)
"""













