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
pi = np.pi

T1=3
T2=1

@jit(nopython=True)
def noseHoover(t,y,N,lamb,tau):
    #q p ceta
    rez = np.zeros(2*N) #+2 CE NOSE HOOVER!
    rez[:N] = y[N:2*N]
    rez[N+1:2*N-1]= -3*y[1:N-1] - 4*lamb*y[1:N-1]**3 + y[2:N] + y[:N-2]
    rez[N] = y[1]-2*y[0]-4*lamb*y[0]**3
    rez[2*N-1] = y[N-2]-2*y[N-1]-4*lamb*y[N-1]**3
    #rez[N] = y[1]-2*y[0]-4*lamb*y[0]**3-y[-2]*y[N]
    #rez[2*N-1] = y[N-2]-2*y[N-1]-4*lamb*y[N-1]**3-y[-1]*y[2*N-1]
    #rez[-2] = 1/tau * (y[N]**2 - T1)
    #rez[-1]=1/tau * (y[2*N-1]**2 - T2)
    return rez



@jit(nopython=True)
def rKorak(y,h,N,lamb,tau):
    k1 = h*noseHoover(0,y,N,lamb,tau)
    k2 = h*noseHoover(0,y + k1*0.5,N,lamb,tau)
    k3 = h*noseHoover(0,y+ k2*0.5,N,lamb,tau)
    k4 = h*noseHoover(0,y+k3,N,lamb,tau)
    return y + 1/6*(k1+2*k2+2*k3+k4)

@jit(nopython=True)
def rPotek(y0,N,stevilo,h,lamb,tau):
    rez = np.zeros((stevilo+1,2*N+2))
    rez[0]=y0
    for i in range(1,stevilo+1):
        rez[i] = rKorak(rez[i-1],h,N,lamb,tau)
    return rez
"""
@jit(nopython=True)
def rPotek2(y0,N,stevilo,h,lamb,tau):
    rez2 = np.zeros(2*N+2)
    rez2 = y0
    for i in range(1,stevilo+1):
        rez2=rKorak(rez2,h,N,lamb,tau)
    return rez2

@jit(nopython=True)
def rTemp(y0,N,stevilo,h,lamb,tau):
    rez2 = np.zeros(2*N+2)
    rez2 = y0
    rez = y0[N:2*N]*y0[N:2*N]
    for i in range(1,stevilo+1):
        rez2=rKorak(rez2,h,N,lamb,tau)
        rez += rez2[N:2*N]*rez2[N:2*N]
    return rez/(stevilo+1)        

@jit(nopython=True)
def rTok(y0,N,stevilo,h,lamb,tau):
    rez2 = np.zeros(2*N+2)
    rez2 = y0
    rez = -0.5*(y0[2:N]-y0[:N-2])*y0[N+1:2*N-1]
    for i in range(1,stevilo+1):
        rez2=rKorak(rez2,h,N,lamb,tau)
        rez -= 0.5*(rez2[2:N]-rez2[:N-2])*rez2[N+1:2*N-1]
    return rez/(stevilo+1)
"""

@jit(nopython=True)
def rPotek2(y0,N,stevilo,h,lamb,tau):
    rez2 = np.zeros(2*N)
    rez2 = y0
    #counter=0
    for i in range(1,stevilo+1):
        if i%tau==0:
            rez2[N]=np.random.normal(0,np.sqrt(T1))
            rez2[2*N-1]=np.random.normal(0,np.sqrt(T2))
            #rez2[N]=rendom1[counter]
            #rez2[2*N-1]=rendom2[counter]
            #counter+=1
        rez2=rKorak(rez2,h,N,lamb,tau)
    return rez2


@jit(nopython=True)
def rTemp(y0,N,stevilo,h,lamb,tau):
    rez2 = np.zeros(2*N)
    rez2 = y0
    #counter=0
    rez = y0[N:2*N]*y0[N:2*N]
    for i in range(1,stevilo+1):
        if i%tau==0:
            rez2[N]=np.random.normal(0,np.sqrt(T1))
            rez2[2*N-1]=np.random.normal(0,np.sqrt(T2))
            #rez2[N]=rendom1[counter]
            #rez2[2*N-1]=rendom2[counter]
            #counter+=1
        rez2=rKorak(rez2,h,N,lamb,tau)
        rez += rez2[N:2*N]*rez2[N:2*N]
    return rez/(stevilo+1)        

@jit(nopython=True)
def rTok(y0,N,stevilo,h,lamb,tau):
    rez2 = np.zeros(2*N-2)
    rez2 = y0
    rez = -0.5*(y0[2:N]-y0[:N-2])*y0[N+1:2*N-1]
    for i in range(1,stevilo+1):
        if i%tau==0:
            rez2[N]=np.random.normal(0,np.sqrt(T1))
            rez2[2*N-1]=np.random.normal(0,np.sqrt(T2))
        rez2=rKorak(rez2,h,N,lamb,tau)
        rez -= 0.5*(rez2[2:N]-rez2[:N-2])*rez2[N+1:2*N-1]
    return rez/(stevilo+1)

def RK4(y0,N,stevilo,h,lamb,tau,keep=True,n=100):
    r = ode(noseHoover).set_integrator("dopri5").set_initial_value(y0,0).set_f_params(N,lamb,tau)
    if keep:
        resitve = np.zeros((stevilo+1,2*N+2))
        resitve[0] = y0
        for i in range(1,stevilo+1):
            resitve[i]=r.integrate(r.t+h)
        #print(r.t)
        #print(resitve[-1])
        return resitve
    else:
        resitve = np.zeros((n+1,2*N+2))
        resitve[0] = r.integrate(stevilo*h)
        #print(r.t)
        #print(resitve[0])
        for i in range(n):
            resitve[i+1] = r.integrate(r.t+h)
        return resitve
    
def temperature(y,N):
    return y[:,N:2*N]**2

def tokovi(y,N):
    rez = -0.5*(np.roll(y[:,:N],-1)-np.roll(y[:,:N],1))*y[:,N:2*N]
    return rez[:,1:-1]

def povpreci(y,n,N):
    return np.sum(y[N-n:N],axis=0)/n

def model(x,kapa,C):
    return (T1-T2)/x*kapa + C

if 0:
    #slika tiste gauss...
    x = np.linspace(-5,5,10000)
    plt.plot(x,1/np.sqrt(2*pi*T1)*np.exp(-x*x/(2*T1)),label=r"$T_1=3$")
    plt.plot(x,1/np.sqrt(2*pi*T2)*np.exp(-x*x/(2*T2)),label=r"$T_2=1$")
    plt.title("Porazdelitev po kateri žrebamo",fontsize=14)
    plt.xlabel("p",fontsize=14)
    plt.ylabel("dP/dp",fontsize=14)
    plt.legend(loc="best")
    plt.savefig("Figures/porazd.pdf")

if 0:
    #j(N) za par lambda + fit
    Nji = [i for i in range(5,32,2)]
    tau=25
    barve=["blue","red"]
    #barve=["cyan","orange"]
    lambde = [0.9,0.25]
    #lambde=[0.1,0.25]
    x = np.linspace(5,31,1000)
    h = 0.1
    tr=500000
    ta=1000000
    #J=[[],[],[],[]]
    J = [[],[]]
    for i in range(1):
        print(i)
        lamb = lambde[i]
        for N in Nji:
            y0 = np.random.rand(2*N)
            y0 = rPotek2(y0,N,tr,h,lamb,tau)
            rez = rTok(y0,N,ta,h,lamb,tau)
            J[i].append(np.sum(rez)/(N-2))
    #plt.plot(Nji,J[0],color="blue",label=r"$\lambda=0$")
    #n=np.polyfit(Nji,J[0],0)[0]
    #plt.plot(x,x**0*n,"--",color="blue")
    #plt.text(20,0.14,r"$J={}$".format(round(n,4)),backgroundcolor="white",fontsize=10)
    for i in range(1):
        n,C = curve_fit(model,Nji,J[i])[0]
        print(n)
        print(1/0)
        plt.plot(Nji,J[i],color=barve[i],label=r"$\lambda={}$".format(lambde[i],round(float(n),4)))
        plt.plot(x,n*(T1-T2)/x+C*x**0,"--",color=barve[i])
    #plt.text(7,0.15,r"$J=\kappa \cdot (T_L-T_R)/N + C$",backgroundcolor="white",fontsize=15)
    plt.text(5,0.08,r"$J={}*(T_L-T_R)/N + C$".format(round(n,4)),backgroundcolor="white",fontsize=10)
    plt.grid(True)
    plt.legend(loc="best")
    plt.xlabel("N",fontsize=14)
    plt.ylabel("J",fontsize=14)
    plt.title("Odvisnost povprečnega toplotnega toka od $N$")
    plt.savefig("Figures/maxfit2.pdf")
if 0:
    #tok od tau
    N = 20
    lamb=0.2
    h=0.1
    tr=1000000
    ta=1000000
    y0 = np.random.rand(2*N)
    resul=[]
    taui=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    for tau in taui:
        y00 = rPotek2(y0,N,tr,h,lamb,tau)
        rez = rTok(y0,N,ta,h,lamb,tau)
        resul.append(np.sum(rez)/(N-2))
    plt.plot(np.array(taui)*0.1,resul)
    plt.grid(True)
    plt.xlabel(r"$\tau$",fontsize=14)
    plt.ylabel("J",fontsize=14)
    plt.title(r"Odvisnost povprečnega toplotnega toka od $\tau$")
    plt.savefig("Figures/maxtau2.pdf")
if 0:
    #topl konst od lamb
    lambde = [0.1,0.25,0.5,0.75,0.9,1]
    x = np.linspace(0.1,1,1000)
    konst = [0.2723017714541392, 0.2532,0.201,0.146,0.132267136274865,0.1227]
    k,n= np.polyfit(lambde,konst,1)
    plt.text(0.54,0.065,r"$\kappa = {} \cdot \lambda + {}$".format(round(k,4),round(n,4)),backgroundcolor="white",fontsize=10)
    plt.plot(lambde,konst,"o-")
    plt.plot(x,k*x+n*x**0,"--")
    plt.grid(True)
    plt.xlabel(r"$\lambda$",fontsize=14)
    plt.ylabel(r"$\kappa$",fontsize=14)
    plt.title(r"Odvisnost sorazmernostnega koeficienta $\kappa ( \lambda )$")
    plt.savefig("Figures/maxtopl.pdf")    
if 0:
    #fiksna lamb  vec N
    lamb = 1
    tau=0.5
    h=0.1
    tr = 10000000
    ta = 10000000
    for N in [10,20,30,40,50]:
        kraj = [i for i in range(1,N-1)]
        y00=np.random.rand(2*N+2)
        y0 = rPotek2(y00,N,tr,h,lamb,tau)
        #rez = rPotek(y0,N,ta,h,lamb,tau)
        #tok = tokovi(rez,N)
        #temp = temperature(rez,N)
        #rez = povpreci(tok,ta,ta)
        rez = rTok(y0,N,ta,h,lamb,tau)
        plt.plot(kraj,rez,"-",label=r"$N={}$".format(N))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("i",fontsize=14)
    plt.xticks([i for i in range(0,50,10)])
    #plt.xlim(0,50-1)
    plt.ylabel("T",fontsize=14)
    plt.title(r"Krajevna odvisnost toka za več $N$, $\lambda=1$",fontsize=13)
    plt.savefig("Figures/vecN4.pdf")   

if 0:
    #fiksen N več lambda
    N=100
    tau=25
    h=0.1
    tr = 10000000
    ta = 10000000
    y00=np.random.rand(2*N)
    kraj = [i for i in range(N)]
    for lamb in [0,0.25,0.5,0.75,1]:
        #rendom1 = np.random.norm()
        y0 = rPotek2(y00,N,tr,h,lamb,tau)
        #rez = rPotek(y0,N,ta,h,lamb,tau)
        #tok = tokovi(rez,N)
        #temp = temperature(rez,N)
        #rez = povpreci(temp,ta,ta)
        rez = rTemp(y0,N,ta,h,lamb,tau)
        plt.plot(kraj,rez,"-",label=r"$\lambda={}$".format(lamb))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("i",fontsize=14)
    plt.xticks([i for i in range(0,N+1,10)])
    #plt.xlim(0,N-1)
    plt.ylabel("T",fontsize=14)
    plt.title(r"Temperaturni profil za več $\lambda$, $N=100$",fontsize=13)
    plt.savefig("Figures/maxprofil2.pdf")    

if 0:
    #odvisnost od tau
    lamb = 0
    h=0.1
    tr = 10000000
    ta = 1000000
    N = 20
    kraj = [i for i in range(N)]
    y00=np.random.rand(2*N+2)
    for tau in [0.25,0.5,0.75,1]:
        y0 = rPotek2(y00,N,tr,h,lamb,tau)
        rez = rPotek(y0,N,ta,h,lamb,tau)
        #tok = tokovi(rez,N)
        temp = temperature(rez,N)
        rez = povpreci(temp,ta,ta)
        plt.plot(kraj,rez,"-",label=r"$\tau={}$".format(tau))
    x = np.linspace(0,N-1,1000)
    plt.plot(x,2*x**0,"--",color="k")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("i",fontsize=14)
    #plt.xticks([i for i in range(0,21,10)])
    #plt.xlim(0,50-1)
    plt.ylabel("T",fontsize=14)
    plt.title(r"Temperaturni profil za več $\tau$",fontsize=13)
    plt.savefig("Figures/tau.pdf")   

if 0:
    #odvisnost od h
    N=20
    tau=1
    tk = 10
    lamb=0
    y00=np.random.rand(2*N+2)
    kraj = [i for i in range(N)]
    for h in [0.001,0.01,0.1]:
        ta = int(tk/h)
        y0 = rPotek2(y00,N,ta,h,lamb,tau)
        rez = rPotek(y0,N,ta,h,lamb,tau)
        #tok = tokovi(rez,N)
        temp = temperature(rez,N)
        rez = povpreci(temp,ta,ta)
        plt.plot(kraj,rez,"-o",label=r"$h={}$".format(h))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("i",fontsize=14)
    plt.xticks([i for i in range(0,N+1,10)])
    #plt.xlim(0,N-1)
    plt.ylabel("T",fontsize=14)
    plt.title(r"Temperaturni profil za več $h$",fontsize=13)
    plt.savefig("Figures/hji.pdf") 
   
if 0:
    #fiksen lambda vec tau
    N=50
    lamb=0.5
    h=0.1
    tr = 1000000
    ta = 1000000
    y00=np.random.rand(2*N+2)
    kraj = [i for i in range(N)]
    for tau in [0.4,0.5,0.6,0.8,1]:
        y0 = rPotek2(y00,N,tr,h,lamb,tau)
        rez = rPotek(y0,N,ta,h,lamb,tau)
        #tok = tokovi(rez,N)
        temp = temperature(rez,N)
        rez = povpreci(temp,ta,ta)
        plt.plot(kraj,rez,"-",label=r"$\lambda={}$".format(tau))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("i",fontsize=14)
    plt.xticks([i for i in range(0,N+1,10)])
    #plt.xlim(0,N-1)
    plt.ylabel("T",fontsize=14)
    plt.title(r"Temperaturni profil za več $\lambda$, $N=50$",fontsize=13)
    #plt.savefig("Figures/veclamb2.pdf")      

if 0:
    #odv od tau za maxa
    N=20
    lamb=0
    tr = 10000000
    h=0.1
    y00=np.random.rand(2*N)
    #y00 = np.ones(2*N+2)
    kraj = [i for i in range(N)]
    for tau in [5,10,15,20,25,30]:
        y0 = rPotek2(y00,N,tr,h,lamb,tau)
        rez = rTemp(y0,N,1000000,h,lamb,tau)
        plt.plot(kraj,rez,"-",label=r"$\tau={}$".format(round(tau*0.1,2)))
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xlabel("i",fontsize=14)
    plt.xticks([i for i in range(1,21)])
    plt.ylabel("T",fontsize=14)
    plt.title(r"Temperaturni profil od $\tau$. $\lambda=0$",fontsize=13)
    plt.savefig("Figures/maxtau3.pdf")
if 0:
    #test relaksacije
    N=20
    lamb=0
    tau=10
    h=0.1
    y00=np.random.rand(2*N)
    #y00 = np.ones(2*N+2)
    relaksacije = [10000,100000,1000000,10000000]
    kraj = [i for i in range(N)]
    for i in relaksacije:
        y0 = rPotek2(y00,N,i,h,lamb,tau)
        rez = rTemp(y0,N,1000000,h,lamb,tau)
        plt.plot(kraj,rez,"-o",label=r"$t_r={}$".format(int(i*h)))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("i",fontsize=14)
    plt.xticks([i for i in range(1,21)])
    plt.ylabel("T",fontsize=14)
    plt.title(r"Temp. profil v odv. od $t_r$. $t_a = 100000, \lambda=0, \tau = 1$",fontsize=13)
    plt.savefig("Figures/maxrel.pdf")
    
   
if 0:
    #Test povprecenja
    N=20
    lamb=0
    tau=10
    h=0.1
    y0=np.random.rand(2*N)
    y0 = rPotek2(y0,N,10000000,h,lamb,tau)
    povprecenja = [100,1000,10000,100000,1000000,2000000]
    kraj = [i for i in range(N)]
    print("lol")
    for i in povprecenja:
        rez = rTemp(y0,N,i,h,lamb,tau)
        plt.plot(kraj,rez,"-o",label=r"$t_a={}$".format(int(i*h)))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("i",fontsize=14)
    plt.xticks([i for i in range(1,21)])
    plt.ylabel("T",fontsize=14)
    plt.title(r"Temp. profil v odv. od $t_a$. $t_r = 1000000, \lambda=0, \tau = 1$",fontsize=13)
    plt.savefig("Figures/maxpovp.pdf")
if 1:
    #lepa slikca vec casov
    Z = [[0,0],[0,0]]
    levels = np.linspace(1,3/1,100)
    barve = plt.get_cmap("gist_rainbow")
    CS3 = plt.contourf(Z, levels, cmap=barve)
    plt.clf()
    casi = [0.1,0.2,0.3,0.4,0.5]
    N = 41
    tau=25
    tr = 1000000
    ta = 100000
    lambde = [0,0.25,0.5,0.75,1]
    h=0.1
    for i in range(len(lambde)):
        y0 = np.random.rand(2*N)
        y0 = rPotek2(y0,N,tr,h,lambde[i],tau)
        temp = rTemp(y0,N,ta,h,lambde[i],tau)
        tok = rTok(y0,N,ta,h,lambde[i],tau)
        plt.plot(np.linspace(0,N,300),np.ones(300)*lambde[i],"--",color="k",lw=0.5)
        for spin in range(1,N-1):
            plt.plot([spin],[lambde[i]],"o",markersize=10,color=barve(1*(temp[spin]-T2)/(T1-T2)))
        #plt.scatter([i for i in range(1,7)],[casi[cas] for i in  range(1,7)],s=100,c=sekvence,cmap=barve)
        plt.quiver([int(N/2)-1 for j in range(1,2)],[lambde[i]+0.05 for j in range(1,2)],[tok[j] for j in range(1)],[0 for j in range(1)])
    plt.vlines(0,-0.1,1.1,color=barve(1.0),lw=10)
    plt.vlines(N-1,-0.1,1.1,color=barve(0),lw=10)
    a = plt.colorbar(CS3)
    a.set_label(r"$T$")
    plt.xlim(0,N-1)
    plt.ylim(-0.1,1.1)
    plt.ylabel(r"$\lambda$")
    plt.title(r"Temperaturni profil za več $\lambda, N=40$")
    plt.savefig("Figures/lepa4.pdf")


if 0:
    #neka kolicina od t
    N=10
    n=50000
    h=0.1
    tau=1
    lamb=0
    stevilo=100000
    y0=np.random.rand(2*N+2)
    rez = rPotek(y0,N,stevilo,h,lamb,tau)
    plt.plot([i for i in range(stevilo+1)],rez[:,1])
    if 1:
        temp = temperature(rez,N)
        print(1/0)
        #temp = tokovi(rez,N)
        rez = temp
        rez = povpreci(temp,n,stevilo)
        cas = [i*h for i in range(stevilo-n)]
        povpodm = [np.sum(np.abs(rez[i]-rez[i-5000]))/N for i in range(10000,150000,10000)]
        plt.plot([i for i in range(1,15)],povpodm)
        print("done")
        #plt.plot(cas[::1000],rez[:,6][::1000])

if 0:
    #pri koncnem t neki od i
    N=5
    n=10000
    h=0.1
    tau=1
    lamb=0
    stevilo=100000
    y0=np.random.rand(2*N+2)
    y0 = rPotek2(y0,N,100000,h,lamb,tau)
    rez = rPotek(y0,N,1000000,h,lamb,tau)
    #temp = temperature(rez,N)
    #rez = povpreci(temp,n,N,stevilo)
    temp = tokovi(rez,N)
    kraj = [i for i in range(1,N-1)]
    rez = povpreci(temp,1000000,1000000)
    plt.plot(kraj,rez)
    plt.legend(loc="best")