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

"""
def RK4(y0,lamb,N,h):
    def hamilton(t,y):
        #y = (x,y,px,py)
        return np.array([y[1],-(y[0]+4*lamb*y[0]**3)])
        #return np.array([y[2],y[3],-y[0]-2*lamb*y[0]*y[1]*y[1],-y[1]-2*lamb*y[1]*y[0]*y[0]])
    r = ode(hamilton).set_integrator("dopri5",nsteps=20000).set_initial_value(y0,0)
    resitve = np.zeros((N,2))
    #resitve = np.zeros((N,4))
    resitve[0] = y0
    for i in range(1,N):
        resitve[i]=r.integrate(r.t+h)
    return resitve
"""

def RK4(y0,lamb,N,h):
    resitve = np.zeros((N,2))
    resitve[0]=y0
    for i in range(1,N):
        resitve[i]=rungeKutta(resitve[i-1],h,lamb)
    return resitve
def rkEval(y, lamb):
    return np.array([y[1], -(y[0]+4*lamb*y[0]**3)])

def rungeKutta(y, dt, lambd):
    k1 = rkEval(y, lambd)
    k2 = rkEval(y + k1 * dt/2, lambd)
    k3 = rkEval(y + k2 * dt/2, lambd)
    k4 = rkEval(y + k3 * dt, lambd)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def liouvT(x,tau,lamb):
    #x = x, y, px, py
    return np.array([x[0]+tau*x[1],x[1]])
    #return np.array([x[0]+tau*x[2], x[1]+tau*x[3],x[2],x[3]])

def liouvV(x,tau,lamb):
    #x = x, y, px, py
    return np.array([x[0],x[1]-tau*(x[0]+4*lamb*x[0]**3)])
    #return np.array([x[0], x[1],x[2] - tau*(x[0]+2*lamb*x[0]*x[1]*x[1]),x[3]-tau*(x[1]+2*lamb*x[1]*x[0]*x[0])])

def s2(x,tau,lamb):
    return liouvT(liouvV(liouvT(x,tau*0.5,lamb),tau,lamb),tau*0.5,lamb)

def s4(x,tau,lamb):
    x1 = 1/(2-2**(1/3))
    x0 = - 2**(1/3)*x1
    return s2(s2(s2(x,tau*x1,lamb),tau*x0,lamb),tau*x1,lamb)

def s3(x,tau,lamb):
    p1 = 0.25*(1+1j/np.sqrt(3))
    p2 = 2*p1
    p5 = np.conjugate(p1)
    p4 = 2*p5
    p3 = 0.5
    return liouvT(liouvV(liouvT(liouvV(liouvT(x,tau*p5,lamb),p4*tau,lamb),p3*tau,lamb),p2*tau,lamb),p1*tau,lamb)

def s3C(x, tau, lamb):
    p1 = 0.25*(1-1j/np.sqrt(3))
    p2 = 2*p1
    p5 = np.conjugate(p1)
    p4 = 2*p5
    p3 = 0.5
    return liouvT(liouvV(liouvT(liouvV(liouvT(x,tau*p5,lamb),p4*tau,lamb),p3*tau,lamb),p2*tau,lamb),p1*tau,lamb)
    
def disip(x,tau,lamb):
    return s3(s3C(x,tau/2,lamb),tau/2,lamb)
def trotter(y0,lamb,N,h,metoda=0):
    resitve = np.ones((N,2))+ 0j*np.ones((N,2))
    #resitve = np.ones((N,4)) + 0j*np.ones((N,4))
    resitve[0]=y0
    for i in range(1,N):
        if metoda==0:
            resitve[i]= s2(resitve[i-1],h,lamb)
        elif metoda==1:
            resitve[i]= s4(resitve[i-1],h,lamb)
        elif metoda==2:
            resitve[i]=disip(resitve[i-1],h,lamb)
    return resitve

def poincare(y0,lamb,N,h):
    if y0[1]==0:
        resitve1 = [y0[0]]
        resitve2 = [y0[2]]
    else:
        resitve1=[]
        resitve2=[]
    prev = y0
    for i in range(1,N):
        temp= s4(prev,h,lamb)
        if np.sign(prev[1]) != np.sign(temp[1]) and temp[3]>0:
            resitve1.append((temp[0]+prev[0])*0.5)
            resitve2.append((temp[2]+prev[2])*0.5)
        prev = temp
    return (resitve1, resitve2)
def energija(x,lamb):
    return 0.5*(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]+x[3]*x[3])+lamb*x[0]*x[0]*x[1]*x[1]

def lyapunov(y0,y1,lamb,N):
    delta0 = lin.norm(y1[:2]-y0[:2])
    delta = delta0
    K = 5
    Kji = 0
    prev1 = y0
    prev2 = y1
    for i in range(1,N):
        prev1= s4(prev1,0.1,lamb)
        prev2 =s4(prev2,0.1,lamb)
        delta = lin.norm(prev2[:2]-prev1[:2])
        if delta > K*delta0:
            Kji += 1
            prev2[:2] = prev2[:2]/K
            prev1[:2] = prev1[:2]/K
        
    return Kji*np.log(K)/(N*0.1)

def area(v):
    a = 0
    x0,y0 = v[0]
    for i in range(1,len(v)):
        dx = v[i][0]-x0
        dy = v[i][1]-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = v[i][0]
        y0 = v[i][1]
    return abs(a)

#clen predstavljen kot tuple (stevilka,[potence konstant],[10101 komutator])



if 0:
    #liouville
    if 0:
        lamb = 0.01
        #ohranjanje volumna
        radiji = [0.001,0.01,0.1]
        casi = np.linspace(0,1000,1000)
        for r in radiji:
            y0 = np.array([[1+r*np.cos(x),1+r*np.sin(x)] for x in np.linspace(0,2*pi,25)])
            rezultatis4 = []
            rezultatiRK = []
            for tocke in y0:
                rezultatis4.append(trotter(tocke,lamb,100000,0.01,1)[::100])
                rezultatiRK.append(RK4(tocke,lamb,100000,0.01)[::100])
            rezultatis4 = np.array(rezultatis4)
            rezultatiRK = np.array(rezultatiRK)
            ploscines4 = []
            ploscineRK = []
            for i in range(1000):
                print(i)
                ploscines4.append(area(rezultatis4[:,i]))
                ploscineRK.append(area(rezultatiRK[:,i]))
            if r==0.001:
                plt.plot(casi,ploscines4,"-.",color="r",label=r"$r=0.001$, $S_4$")
                plt.plot(casi,ploscineRK,"-",color="r",label=r"$r=0.001$, RK4")
            if r==0.01:
                plt.plot(casi,ploscines4,"-.",color="g",label=r"$r=0.01$, $S_4$")
                plt.plot(casi,ploscineRK,"-",color="g",label=r"$r=0.01$, RK4")
            if r==0.1:
                plt.plot(casi,ploscines4,"-.",color="blue",label=r"$r=0.1$, $S_4$")
                plt.plot(casi,ploscineRK,"-",color="blue",label=r"$r=0.1$, RK4")
        plt.xlabel(r"$t$",fontsize=16)
        plt.legend(loc="best")
        plt.ylabel(r"$V$",fontsize=16)
        plt.title(r"$V(t)$, $\lambda=0.01$")
        plt.savefig("Figures/zadnja22.pdf")
        
    tau = 0.01
    if 0:
        #S4
        barva = plt.get_cmap("inferno")
        barve = np.linspace(0,1,10)
        casovnikorak=10
        lamb=0.1
        y0 = np.array([[1+0.1*np.cos(x),1+0.1*np.sin(x)] for x in np.linspace(0,2*pi,25)])
        y00 = [np.array([1,1])]
        plt.fill(y0[:,0],y0[:,1],alpha=0.5,color=barva(barve[0]))
        for i in range(1,10):
            print(area(y0))
            if i!=9:
                y00.append(trotter(y00[i-1],lamb,int(casovnikorak/tau),tau,1)[-1])
            for j in range(25):
                y0[j] = trotter(y0[j],lamb,int(casovnikorak/tau),tau,1)[-1]
            plt.fill(y0[:,0],y0[:,1],alpha=0.5,color=barva(barve[i]))
        y00 = np.array(y00)
        plt.plot(y00[:,0],y00[:,1],"--",lw=0.5,color="green")       
        plt.xlabel(r"$x$")
        plt.ylabel(r"$p_x$")
        plt.title(r"Liouville, $S_4$, $\lambda=0.1$")
        plt.savefig("Figures/Liouv5.pdf")
    if 1:
        #RK4
        barva = plt.get_cmap("inferno")
        barve = np.linspace(0,1,10)
        lamb = 0.1
        casovnikorak=10
        y0 = np.array([[1+0.1*np.cos(x),1+0.1*np.sin(x)] for x in np.linspace(0,2*pi,25)])
        y00 = [np.array([1,1])]
        plt.fill(y0[:,0],y0[:,1],alpha=0.5,color=barva(barve[0]))
        for i in range(1,10):
            print(area(y0))
            if i!=9:
                y00.append(RK4(y00[i-1],lamb,int(casovnikorak/tau),tau)[-1])
            for j in range(25):
                y0[j] = RK4(y0[j],lamb,int(casovnikorak/tau),tau)[-1]
            plt.fill(y0[:,0],y0[:,1],alpha=0.5,color=barva(barve[i]))
        y00 = np.array(y00)
        plt.plot(y00[:,0],y00[:,1],"--",lw=0.5,color="green") 
        plt.xlabel(r"$x$")
        plt.ylabel(r"$p_x$")
        plt.title(r"Liouville, RK4, $\lambda=0.1$")
        #plt.savefig("Figures/Liouv6.pdf")
if 0:
    #equipartition
    y0 = np.array([0,0,0.5,np.sqrt(1-0.5*0.5)])
    lamb = 4
    N = 100000
    resitev = trotter(y0,lamb,N,0.1,1)
    if 1:
        Avgpx = []
        Avgpy = []
        for i in range(1,1000):
            Avgpx.append(np.sum(resitev[:,2][:i*100]**2)/(i*100))
            Avgpy.append(np.sum(resitev[:,3][:i*100]**2)/(i*100))
        casi = np.array(list(range(1,1000)))*10
        fig, ax1 = plt.subplots()
        ax1.plot(casi,Avgpx,color="red",label=r"$\langle p_x^2 \rangle$")
        ax1.plot(casi,Avgpy,color="blue",label=r"$\langle p_y^2 \rangle$")
        ax1.set_xlabel(r"$t$", fontsize=16)
        ax1.set_ylabel(r"$p^2$", fontsize=16)
        ax1.set_title(r"$\lambda={}$".format(round(lamb,3)),fontsize=18)
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$\delta p^2$",fontsize=16)
        ax2.set_yscale("log")
        ax2.plot(casi,np.abs(np.array(Avgpy)-np.array(Avgpx)),color="k",ls="--",alpha=0.5,label=r"$\langle |p_y^2 - p_x^2| \rangle$")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2 , labels2 = ax2.get_legend_handles_labels()
        ax1.legend([handles1[0],handles1[1],handles2[0]], [labels1[0],labels1[1],labels2[0]])
        plt.tight_layout()
        plt.savefig("Figures/ekvi8.pdf")    
    if 0:
        fig, ax1 = plt.subplots()
        Nk = 2000
        casi = np.array(list(range(1,int(Nk))))*0.1
        Avgpxx = []
        Avgpyy = []
        for i in range(1,2000):
            Avgpxx.append(np.sum(resitev[:,2][:i]**2)/(i))
            Avgpyy.append(np.sum(resitev[:,3][:i]**2)/(i))
        ax1.plot(casi,Avgpxx,color="red",label=r"$\langle p_x^2 \rangle$")
        ax1.plot(casi,Avgpyy,color="blue",label=r"$\langle p_y^2 \rangle$")
        ax1.set_xlabel(r"$t$", fontsize=16)
        ax1.set_ylabel(r"$p^2$", fontsize=16)
        ax1.set_title(r"$\lambda={}$".format(round(lamb,3)),fontsize=18)
        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$\delta p^2$",fontsize=16)
        ax2.set_yscale("log")
        ax2.plot(casi,np.abs(np.array(Avgpyy)-np.array(Avgpxx)),color="k",ls="--",alpha=0.5,label=r"$\langle |p_y^2 - p_x^2| \rangle$")
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2 , labels2 = ax2.get_legend_handles_labels()
        ax1.legend([handles1[0],handles1[1],handles2[0]], [labels1[0],labels1[1],labels2[0]])
        plt.tight_layout()
        plt.savefig("Figures/ekvi61.pdf")         
if 0:
    y0 = np.array([1,0,0,0.5])
    y1 = np.array([1-10**(-7),0,0,0.5])
    print(lyapunov(y0,y1,1,1000000))

if 0:
    #poincare
    N = 400
    lamb=0.01
    E = 1
    fig,ax=plt.subplots()
    barva = plt.get_cmap("gist_rainbow")
    barve = np.linspace(0,1,N)
    seznambarv = list(range(N))
    barvee = np.random.permutation(seznambarv)
    iksi = np.linspace(-1,1,N)
    for i in range(N):
        print(i)
        y0 = np.array([0,0,iksi[i],np.sqrt(E-iksi[i]**2)])
        resitev = poincare(y0,lamb,20000,0.1)
        ax.plot(resitev[0],resitev[1],"o",markersize=0.5,color=barva(barve[barvee[i]]))
        ax.set_aspect(1)
    plt.show()
    #plt.title(r"$E=0.5, \lambda = {}$".format(round(lamb,3)))
    #plt.xlabel(r"$x$",fontsize=18)
    #plt.ylabel(r"$p_x$",fontsize=18)
    #plt.savefig("Figures/poincare12.png")
        


if 0:
    #x, y(t) animacija
    lambde = [0.05,0.1,1,3]
    y0 = np.array([1,0,0,0.5])
    N = 10000
    h = 0.1
    resitev1 = trotter(y0,0.05,N,h,1)
    resitev2 = trotter(y0,0.1,N,h,1)
    resitev3 = trotter(y0,1,N,h,1)
    resitev4 = trotter(y0,3,N,h,1)
    resitve = [resitev1,resitev2,resitev3,resitev4]
    fig, ax = plt.subplots(2,2, figsize=(15,15))
    axi = [ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
    cmap = plt.get_cmap("viridis")
    barve = np.linspace(0,1,1000)
    for i in range(4):
        axi[i].set_title(r"$\lambda={}$".format(round(lambde[i],2)))
    def animiraj(t):
        print(t)
        for i in range(4):
            if t==0:
                axi[i].plot(resitve[i][:,0][t*10:(t+1)*10],resitve[i][:,1][t*10:(t+1)*10],color=cmap(barve[t]))
            elif t!=999:
                axi[i].plot(resitve[i][:,0][(t-1)*10:(t+1)*10],resitve[i][:,1][(t-1)*10:(t+1)*10],color=cmap(barve[t]))
        plt.suptitle(r"$t={}$".format(round(t*0.1*10,3)))
    ani = animation.FuncAnimation(fig,animiraj,range(1000),interval=50)   
    #plt.show()
    ani.save("dinamika.mp4")  
    print("evo")
    
if 0:
    #prehod v kaos 1 animacija
    lambde = np.linspace(3.85,3.86,50)
    y0 = np.array([1,0,0,0.5])
    N = 10000
    h = 0.1
    fig, ax = plt.subplots()
    def animiraj(t):
        print(t)
        ax.clear()
        casi = np.array([i*h for i in range(1,N)])
        resitev = trotter(y0,lambde[t],N,h,1)
        points = np.array([resitev[:,0], resitev[:,1]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0,(N)*h)
        lc = matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(casi)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        ax.set_xlim(resitev[:,0].min(),resitev[:,0].max())
        ax.set_ylim(resitev[:,1].min(),resitev[:,1].max())
        ax.set_title(r"$\lambda={}$".format(round(lambde[t],4)))
    ani = animation.FuncAnimation(fig,animiraj,range(50),interval=100)   
    #plt.show()
    ani.save("kaos2.mp4")  
    print("evo")
    
    
if 0:
    #x,y(t) 
    lamb = 4
    y0 = np.array([1,0,0,0.5])
    N = 10000
    h = 0.1
    casi = np.array([i*h for i in range(1,N)])
    resitev = trotter(y0,lamb,N,h,1)
    points = np.array([resitev[:,0], resitev[:,1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots()
    #ax.plot(resitev[:,0],resitev[:,1])
    norm = plt.Normalize(0,(N)*h)
    lc = matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(casi)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    ax.set_xlim(resitev[:,0].min(),resitev[:,0].max())
    ax.set_ylim(resitev[:,1].min(),resitev[:,1].max())
    ax.set_xlabel(r"$x$",fontsize=18)
    ax.set_ylabel(r"$y$",fontsize=18)
    cbar = fig.colorbar(line)
    cbar.set_label(r"$t$",fontsize=18)
    ax.set_title(r"$\lambda=4$")
    plt.tight_layout()
    plt.savefig("Figures/dinamika16.pdf")
    plt.show()
if 0:
    #test
    h = 0.01
    koncnicas=20
    lamb = 0.5
    y0 = np.array([0.5,0,0,1])
    E0 = energija(y0,lamb)
    N = int(koncnicas/h)
    hji = list(range(N))
    #rez = RK4(y0,lamb,N,h)
    #rezrk = [abs(energija(rez[i*50000],lamb)-E0)/E0 for i in range(200)]
    rez = trotter(y0,lamb,N,h,0)
    rezs2 = [abs(energija(rez[i],lamb)-E0)/E0 for i in range(N)]
    rez = trotter(y0,lamb,N,h,1)
    rezs4 = [abs(energija(rez[i],lamb)-E0)/E0 for i in range(N)]
    rez = trotter(y0,lamb,N,h,2)
    rezs3 = [abs(energija(rez[i],lamb)-E0)/E0 for i in range(N)]
    plt.title(r"$|E(t)-E(0)|/E(0)$  $\lambda=0.5, \tau=0.1$")
    plt.ylabel(r"$\delta E(t)$")
    plt.xlabel(r"$t$")
    plt.yscale("log")
    plt.grid(True)
    plt.plot(hji,rezs2,label="S2")
    plt.plot(hji,rezs4,label="S4")
    plt.plot(hji,rezs3,label="Complex")
    #plt.plot(hji,rezrk,label="RK4")
if 0:
    #fiksen lambda napake v energiji q
    hji = np.linspace(0.001,0.1,100)
    koncnicas=10
    lamb = 10
    rezs2 = []
    rezrk = []
    rezs4 = []
    rezs3 = []
    y0 = np.array([1,0,0,0.5])
    E0 = energija(y0,lamb)
    for h in hji:
        N = int(koncnicas/h)
        rez = energija(RK4(y0,lamb,2,10)[-1],lamb)
        rezrk.append(abs((rez-E0))/E0)
        rez = energija(trotter(y0,lamb,N,h,0)[-1],lamb)
        rezs2.append(abs((rez-E0))/E0)
        rez = energija(trotter(y0,lamb,N,h,1)[-1],lamb)
        rezs4.append(abs((rez-E0))/E0)
        rez = energija(trotter(y0,lamb,N,h,2)[-1],lamb)
        rezs3.append(abs((rez-E0))/E0)
    plt.title(r"$|E(10)-E(0)|/E(0)$  $\lambda=10$")
    plt.ylabel(r"$\delta E$",fontsize=16)
    plt.xlabel(r"$\tau$",fontsize=16)
    plt.yscale("log")
    plt.grid(True)
    plt.plot(hji,rezrk,label="RK4")
    plt.plot(hji,rezs2,label="S2")
    plt.plot(hji,rezs4,label="S4")
    plt.plot(hji,rezs3,label="Complex")
    plt.legend(loc="best")
    plt.savefig("Figures/napake3.pdf")

if 0:
    #fiksen lambda napake v energiji 2
    h = 0.1
    hji = [0.1*50000*i for i in range(200)]
    koncnicas=1000000
    lamb = 0.5
    y0 = np.array([1,0,0,0.5])
    E0 = energija(y0,lamb)
    N = int(koncnicas/h)
    rez = RK4(y0,lamb,N,h)
    rezrk = [abs(energija(rez[i*50000],lamb)-E0)/E0 for i in range(200)]
    rez = trotter(y0,lamb,N,h,0)
    rezs2 = [abs(energija(rez[i*50000],lamb)-E0)/E0 for i in range(200)]
    rez = trotter(y0,lamb,N,h,1)
    rezs4 = [abs(energija(rez[i*50000],lamb)-E0)/E0 for i in range(200)]
    rez = trotter(y0,lamb,N,h,2)
    rezs3 = [abs(energija(rez[i*50000],lamb)-E0)/E0 for i in range(200)]
    plt.title(r"$|E(t)-E(0)|/E(0)$  $\lambda=0.5, \tau=0.1$")
    plt.ylabel(r"$\delta E(t)$")
    plt.xlabel(r"$t$")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    plt.plot(hji,rezs2,label="S2")
    plt.plot(hji,rezs4,label="S4")
    plt.plot(hji,rezs3,label="Complex")
    plt.plot(hji,rezrk,label="RK4")
    plt.legend(loc="best")
    plt.savefig("Figures/napake2.pdf")

if 0:
    #fiksen lambda napake v energiji 2 samo neka proba
    h = 0.1
    hji = list(range(100))
    koncnicas=1000
    lamb = 1
    y0 = np.array([1,0,0,0.5])
    E0 = energija(y0,lamb)
    N = int(koncnicas/h)
    rez = RK4(y0,lamb,N,h)
    rezrk = [abs(energija(rez[i*100],lamb)-E0)/E0 for i in range(100)]
    rez = trotter(y0,lamb,N,h,0)
    rezs2 = [abs(energija(rez[i*100],lamb)-E0)/E0 for i in range(100)]
    rez = trotter(y0,lamb,N,h,1)
    rezs4 = [abs(energija(rez[i*100],lamb)-E0)/E0 for i in range(100)]
    rez = trotter(y0,lamb,N,h,2)
    rezs3 = [abs(energija(rez[i*100],lamb)-E0)/E0 for i in range(100)]
    plt.title(r"$|E(t)-E(0)|/E(0)$  $\lambda=0.5, \tau=0.1$")
    plt.ylabel(r"$\delta E(t)$")
    plt.xlabel(r"$t$")
    plt.yscale("log")
    #plt.xscale("log")
    plt.grid(True)
    plt.plot(hji,rezs2,label="S2")
    plt.plot(hji,rezs3,label="Complex")
    plt.plot(hji,rezs4,label="S4")

    #plt.plot(hji,rezrk,label="RK4")
    plt.legend(loc="best")
    plt.savefig("Figures/to.pdf")


if 0:
    #casovna zahtevnost
    tji = np.linspace(1000,100000,30)
    casirk = []
    casis2 = []
    casis4 = []
    casis3 = []
    y0 = np.array([1,0,0,0.5])
    lamb = 0.5
    for t in tji:
        print(t)
        temp = time.time()
        res = RK4(y0,lamb,2,t)
        casirk.append(time.time()-temp)
        temp = time.time()
        res = trotter(y0,lamb,int(t/0.1),0.1,0)
        casis2.append(time.time()-temp)
        temp = time.time()
        res = trotter(y0,lamb,int(t/0.1),0.1,1)
        casis4.append(time.time()-temp)                
        temp = time.time()
        res = trotter(y0,lamb,int(t/0.1),0.1,2)
        casis3.append(time.time()-temp)
        
    plt.title(r"Časovna zahtevnost - časovni razvoj do $T$   $\lambda=0.5, \tau=0.1$")
    plt.ylabel(r"$t[s]$")
    plt.xlabel(r"$T$")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    plt.plot(tji,casis2,"-.",label="S2")
    plt.plot(tji,casis4,"-.",label="S4")
    plt.plot(tji,casis3,"-.",label="Complex")
    plt.plot(tji,casirk,"-.",label="RK4")
    plt.legend(loc="best")
    plt.savefig("Figures/hitrost.pdf")    


