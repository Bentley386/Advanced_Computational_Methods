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

velikostLambde=0

#generira stanja
def genZac(N):
    vmesni = [[0],[1]]
    while len(vmesni) != 2**N:
        novi = []
        for konf in vmesni:
            novi.append(konf + [0])
            novi.append(konf + [1])
        vmesni = [novi[i] for i in range(len(novi))]
    return np.array(vmesni)

#converta arej v stevilko
@jit(nopython=True)
def toBinary(arej):
    rez = 0
    for i in range(1,len(arej)+1):
        rez += arej[-i]*(2**(i-1))
    return rez        
    
#baza tenz prod dveh
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

#zreba koeficiente
@jit(nopython=True)
def zrebaj(N):
    n = 2**N
    #sigma = np.sqrt(1/(2*n))
    sigma=1
    cifre = np.random.normal(0,sigma,n)
    #cifre = np.random.normal(0,sigma,n)+1j*np.random.normal(0,sigma,n)
    cifre = cifre/np.sqrt(np.sum(np.abs(cifre)**2))
    return cifre


def dobiKoef(Aji,n):
    stanja = genZac(n)
    koef = np.zeros(2**n)*1j
    for i in range(2**n):
        matrika = Aji[0][stanja[i][0]]
        for j in range(1,n):
            matrika = np.matmul(matrika,Aji[j][stanja[i][j]])
        koef[i]=matrika
    return koef

def prviKorak(psi,N,lambde=0):
    U,s,Vh = svd(np.reshape(psi,(2,N//2)),False)
    if lambde==1:
        return s
    if lambde==2:
        return [[U[0],U[1]],np.matmul(np.diag(s),Vh),-np.sum(np.abs(s)**2 * np.log(np.abs(s)**2))] #Aja , psi        
    return [[U[0],U[1]],np.matmul(np.diag(s),Vh),s] #Aja , psi

def splosniKorak(psi,N,k,lambde=0,odrezi=False):
    l = psi.size
    U, s, Vh = svd(np.reshape(psi,(2*k,l//(2*k))),False)
    if odrezi:
        indeks = np.argmin(s)
        #indeks=1
        global velikostLambde
        velikostLambde = s[indeks]
        np.delete(s,indeks)
        np.delete(U,indeks,axis=1)
        np.delete(Vh,indeks,axis=0)
        #s[indeks]=0
    if lambde==1:
        return s
    if lambde==2:
        return [[U[::2],U[1::2]],np.matmul(np.diag(s),Vh),len(s),-np.sum(np.abs(s)**2 * np.log(np.abs(s)**2))] #Aja, psi, k
    return [[U[::2],U[1::2]],np.matmul(np.diag(s),Vh),len(s),s] #Aja, psi, k

def zadnjiKorak(psi,N):
    temp = np.reshape(psi,(2,2))
    return [np.transpose(temp[:,0]),np.transpose(temp[:,1])]



#psi = [0,0,0,1,0,0,0,0]
def dobiA(psi,N,lambde=0,lambdee=0,odrezi=0):
    Aji = []
    sji=[]
    if lambdee>0:
        entr = 0
    for i in range(n):
        if i == 0:
            if lambde==1:
                return prviKorak(psi,N,True)
            if lambdee>0 and (i+1)%lambdee:
                temp = prviKorak(psi,N,2)
                entr+=temp[-1]
                Aji.append(temp[0])
                sji.append(temp[-1])
                continue
            temp = prviKorak(psi,N)
            Aji.append(temp[0])
            continue
        elif i==1:
            if lambde==2:
                return splosniKorak(temp[1],N,2,True)
            temp = splosniKorak(temp[1],N,2)
            Aji.append(temp[0])
            sji.append(temp[-1])
            continue
        if lambde==i+1:
            return splosniKorak(temp[1],N,temp[2],True)
        if i==n-1:
            temp = zadnjiKorak(temp[1],N)
            Aji.append(temp)
            break
        if i==odrezi:
            temp = splosniKorak(temp[1],N,temp[2],odrezi=True)
            Aji.append(temp[0])            
            continue
        temp = splosniKorak(temp[1],N,temp[2])
        Aji.append(temp[0])
        sji.append(temp[-1])
    return Aji


def hamiltonian(n):
    """
   Heisenberg chain of length n without BC.
   """
    N = 2**n
    binaryFactor = np.array([1, 2, 4])
    row = []
    col = []
    value = []
    for j in range(n-1):
        for i in range(N):
            spin1 = (i % binaryFactor[1]) // binaryFactor[0]
            spin2 = (i % binaryFactor[2]) // binaryFactor[1]
            if spin1 != spin2:
                row.append(i)
                col.append(i)
                value.append(-1)
               
                row.append(i)
                value.append(2)
                if spin1 == 1:
                    col.append((i + binaryFactor[0]) % N)
                else:
                    col.append((i - binaryFactor[0]) % N)
            else:
                row.append(i)
                col.append(i)
                value.append(1)
        binaryFactor *= 2
    return scipy.sparse.coo_matrix((value, (row, col)), shape = (N, N), dtype = np.float64)
       
def hamiltonianPeriodic(n):
    """
   Heisenberg chain of length n with periodic BC
   """
    N = 2**n
    binaryFactor = np.array([1, 2, 4])
    row = []
    col = []
    value = []
    for j in range(n):
        for i in range(N):
            if j == n-1:
                spin1 = i // binaryFactor[0]
                spin2 = (i % 2)
                cyclicCorrection = 1
            else:
                spin1 = (i % binaryFactor[1]) // binaryFactor[0]
                spin2 = (i % binaryFactor[2]) // binaryFactor[1]
                cyclicCorrection = 0
            if spin1 != spin2:
                row.append(i)
                col.append(i)
                value.append(-1)
               
                row.append(i)
                value.append(2)
                if spin1 == 1:
                    col.append((i + binaryFactor[0] + cyclicCorrection) % N)
                else:
                    col.append((i - binaryFactor[0] - cyclicCorrection) % N)
            else:
                row.append(i)
                col.append(i)
                value.append(1)
        binaryFactor *= 2
    return scipy.sparse.coo_matrix((value, (row, col)), shape = (N, N), dtype = np.float64)
 
def directDiagBase(hamiltonianMatrix):
    """
   Returns base state energy and coefficients
   """
    base = scipy.sparse.linalg.eigsh(hamiltonianMatrix, which = 'SA')
    return base[0][0], base[1].T[0]
def naredi(inp):

    rez = []
    counter=-1
    for i in range(len(inp)):
        counter+=1
        if counter==4:
            counter=0
        if counter<2:
            rez.append(inp[i])

    return np.array(rez)

def preuredi(psi,N):
    auxstanja = genZac(N//2)
    stanja=genZac(N)
    rez = np.zeros((2**(N//2),2**(N//2)))*1j
    indeksi = np.zeros(2**(N//2))*1j
    for i in range(2**N):
        for j in range(2**(N//2)):
            if np.array_equal(naredi(stanja[i]),auxstanja[j]):
                rez[j][int(indeksi[j])] = psi[i]
                indeksi[j] = indeksi[j] + 1
                break
    return rez

def preuredi2(psi,N):
    auxstanja = genZac(N//2)
    stanja=genZac(N)
    rez = np.zeros((2**(N//2),2**(N//2)))*1j
    indeksi = np.zeros(2**(N//2))*1j
    for i in range(2**N):
        for j in range(2**(N//2)):
            if np.array_equal(stanja[i][::2],auxstanja[j]):
                rez[j][int(indeksi[j])] = psi[i]
                indeksi[j] = indeksi[j] + 1
                break
    return rez

def zrebajj(n):
    psi = np.ones(2**n)
    return psi/(np.sqrt(2**n))

if 1:
    #entropija nekompaktnih blokov od n
    nji = [2,4,6,8,10,12,14,16]
    nakljucna = [zrebaj(n) for n in nji]
    entropije = []
    for n in nji:
        print(n)
        zac = zrebajj(n)
        #zac = nakljucna[int(n/2)-1]
        sji = np.abs(dobiA(zac,2**n,n//2))**2
        entropije.append(-np.sum(sji*np.log(sji)))
    plt.plot(nji,entropije,".:",label=r"Sim. Biparticija")
    entropije = []
    for n in nji:
        print(n)
        #zac = nakljucna[int(n/2)-1]
        zac = zrebajj(n)
        sji = np.abs(svd(preuredi2(zac,n),False)[1])**2
        entropije.append(-np.sum(sji*np.log(sji)))
    plt.plot(nji,entropije,".:",label=r"ABABAB..")
    nji = [4,8,12,16]
    entropije = []
    for n in nji:
        print(n)
        zac = zrebajj(n)
        sji = np.abs(svd(preuredi(zac,n),False)[1])**2
        entropije.append(-np.sum(sji*np.log(sji)))
        """
        zac = directDiagBase(hamiltonian(n))[1]
        sji = np.abs(svd(preuredi(zac,n),False)[1])**2
        entropije1.append(-np.sum(sji*np.log(sji)))
        zac = directDiagBase(hamiltonianPeriodic(n))[1]
        sji = np.abs(svd(preuredi(zac,n),False)[1])**2
        entropije2.append(-np.sum(sji*np.log(sji)))
        """
    #plt.plot(nji,entropije1,".:",label=r"OBC")
    #plt.plot(nji,entropije2,".:",label=r"PBC")
    plt.plot(nji,entropije,".:",label="AABBAA..")
    plt.grid(True)
    plt.xlabel(r"$n$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$S$",fontsize=14)
    plt.legend(loc="best")
    plt.title(r"Entropija prepletenosti za več biparticij",fontsize=13)
    plt.savefig("entropija8.pdf")


if 0:
    #preverjanje
    n=14
    zacetni = zrebaj(n)
    lambde = []
    sji = dobiA(zacetni,2**n)
    plt.plot([i for i in range(1,len(sji)+1)],[np.sum(x**2) for x in sji],".:")
    plt.grid(True)
    plt.xlabel(r"$j$")
    #plt.legend(loc="best")
    plt.title("Vsota kvadratov $|\lambda|^2$ na vsakem koraku")
    plt.savefig("zadnja.pdf")
if 0:
    #rezanje
    n=14
    zacetni = zrebaj(n)
    kji = [2,3,4,5,6,7,8,9,10,11]
    razlike = []
    lambde = []
    for k in kji:
        print(k)
        Aji = dobiA(zacetni,2**n,odrezi=k)
        lambde.append(np.abs(velikostLambde))
        novi = dobiKoef(Aji,n)
        razlike.append(np.sum(np.abs(zacetni-novi)))
    
    plt.plot(kji,razlike,".:",label=r"$|\psi - \psi'|$")
    plt.plot(kji,lambde,".:",label=r"$|\lambda|$")
    plt.grid(True)
    plt.xlabel(r"$j$")
    plt.legend(loc="best")
    plt.title("Rezanje ene izmed $\lambda_k$ na j-tem koraku")
    plt.savefig("zadnja2.pdf")
if 0:
    #samo test
    n = 14
    korak=5
    psi = zrebaj(n)
    Aji = dobiA(psi,2**n)
    dim = [x[0].shape for x in Aji]
    celotenx = 0
    celoteny = 0
    for i in range(len(dim)):
        if i==0:
            celotenx+= dim[i][0]+korak
            continue
        if i==len(dim)-1:
            celotenx+= korak
            continue
        celotenx += dim[i][1]+korak
        if dim[i][0] > celoteny:
            celoteny = dim[i][0]
    print(dim)
    fig,ax=plt.subplots()
    im = [[(0,0,0) for x in range(celotenx)] for y in range(int(2*celoteny))]
    xkoord = 0
    cmap = plt.get_cmap("hot")
    barve = np.linspace(0.1,1,len(dim))
    ax.imshow(im,origin="lower")
    for i in range(len(dim)):
        if i==0:
            rect=matplotlib.patches.Rectangle((xkoord,celoteny),dim[i][0],1,color=cmap(barve[i]))
            xkoord += dim[i][0]+korak
            ax.add_patch(rect)
            continue
        if i==len(dim)-1:
            rect=matplotlib.patches.Rectangle((xkoord,celoteny),1,dim[i][0],color=cmap(barve[i]))
            xkoord += dim[i][0]+korak
            ax.add_patch(rect)            
            continue
        rect=matplotlib.patches.Rectangle((xkoord,celoteny-int(dim[i][0]/2)),dim[i][1],dim[i][0],color=cmap(barve[i]))
        xkoord += dim[i][1]+korak
        ax.add_patch(rect)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.title(r"Oblika matrik $A$ za primer $n=14$")
    plt.savefig("matrike.png")
if 0:
    #napaka po rekonstrukciji
    nji = [4,6,8,10,12,14]
    razlike = []
    for n in nji:
        print(n)
        psi = zrebaj(n)
        Aji = dobiA(psi,2**n)
        koef = dobiKoef(Aji,n)
        razlike.append(np.sum(np.abs(np.abs(psi)-np.abs(koef))))
    plt.grid(True)
    plt.xlabel(r"$n$",fontsize=14)
    plt.plot(nji,razlike,".:")
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$\epsilon$",fontsize=14)
    plt.title(r"Napaka rekonstrukcije stanja",fontsize=13)
    plt.savefig("rekonstrukcija.pdf")

if 0:
    #entropija sim. particije od n/2
    nji = [4,6,8,10,12,14]
    entropijeOpen = []
    entropijePer = []
    for n in nji:
        print(n)
        lastno = directDiagBase(hamiltonian(n))[1]
        sji = np.abs(dobiA(lastno,2**n,2))**2
        entropijeOpen.append(-np.sum(sji*np.log(sji)))
        lastno = directDiagBase(hamiltonianPeriodic(n))[1]
        sji = np.abs(dobiA(lastno,2**n,2))**2
        entropijePer.append(-np.sum(sji*np.log(sji)))
               
    plt.grid(True)
    plt.xlabel(r"$n$",fontsize=14)
    plt.plot(nji,entropijeOpen,".:",label="OBC")
    plt.plot(nji,entropijePer,".:",label="PBC")
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$S$",fontsize=14)
    plt.legend(loc="best")
    plt.title(r"Entropija prepletenosti za biparticijo, kjer $|A|=2$",fontsize=13)
    plt.savefig("entropija4.pdf")        

if 0:
    #entropija nekompaktnih blokov od n
    nji = [4,6,8,10,12,14]
    entropijeOpen = []
    entropijePer = []
    for n in nji:
        print(n)
        lastno = directDiagBase(hamiltonian(n))[1]
        sji = np.abs(svd(preuredi(lastno,n),False)[1])**2
        entropijeOpen.append(-np.sum(sji*np.log(sji)))
        lastno = directDiagBase(hamiltonianPeriodic(n))[1]
        sji = np.abs(svd(preuredi(lastno,n),False)[1])**2
        entropijePer.append(-np.sum(sji*np.log(sji)))
               
    plt.grid(True)
    plt.xlabel(r"$n$",fontsize=14)
    plt.plot(nji,entropijeOpen,".:",label="OBC")
    plt.plot(nji,entropijePer,".:",label="PBC")
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$S$",fontsize=14)
    plt.legend(loc="best")
    plt.title(r"Entropija prepletenosti za alternirajočo biparticijo",fontsize=13)
    plt.savefig("entropija5.pdf")

if 0:
    #entropija od k za več n
    nji = [4,6,8,10,12,14]
    for n in nji:
        print(n)
        kji = list(range(1,n))
        entropije = []
        for k in kji:
            lastno = directDiagBase(hamiltonian(n))[1]
            sji = np.abs(dobiA(lastno,2**n,k))**2
            entropije.append(-np.sum(sji*np.log(sji)))
        plt.plot(kji,entropije,".:",label=r"$n={}$".format(n))
    plt.grid(True)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$S$")
    plt.legend(loc="best",ncol=3)
    plt.title("Entropija prepletenosti za več biparticij - OBC")
    plt.savefig("entropija3.pdf")

if 0:
    #entropija nekompaktne particije od n
    nji = [4,6,8,10,12,14]
    entropijeOpen = []
    entropijePer = []
    for n in nji:
        print(n)
        lastno = directDiagBase(hamiltonian(n))[1]
        sji = np.abs(dobiA(lastno,2**n,n//3))**2
        entropijeOpen.append(-np.sum(sji*np.log(sji)))
        lastno = directDiagBase(hamiltonianPeriodic(n))[1]
        sji = np.abs(dobiA(lastno,2**n,n//3))**2
        entropijePer.append(-np.sum(sji*np.log(sji)))
               
    plt.grid(True)
    plt.xlabel(r"$n$",fontsize=14)
    plt.plot(nji,entropijeOpen,".:",label="OBC")
    plt.plot(nji,entropijePer,".:",label="PBC")
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$S$",fontsize=14)
    plt.legend(loc="best")
    plt.title(r"Entropija prepletenosti za biparticijo $ABAB\dots$",fontsize=13)
    plt.savefig("entropija5.pdf")    
