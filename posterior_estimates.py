"""
Python script accompanying the paper "Bayesian parameter estimation with guarantees via interval analysis and 
simulation", for the estimation of posterior quantities: Cumulative Distribution Functions (CDFs) 
and their expectations.
  
  
The main functions are:
    
    - boxBackReach(f,X0,Y,delta): computes a covering of the parametric space.
        
    - buildSampler(descr): builds a sampler for each parameter.    
        
    - def estimateCDF(f,fIntv,X0,j,S,P,sl,enclR,res,B,delta,Xin): estimates the univariate CDF 
                                                                  for the parameter of interest.
        
    - cdfLU(hi,eps): computes an over- and an under-approximation of the true CDF.
    
    - mump(cl,cu,Pn): computes the confidence interval for expected value of the posterior density.   

    
Example. Consider the case of simple ball equation with two dependent variables: z_1,z_2,
         and one parameter: \theta: z':=(z_2,-\theta). We have: 
 
time_horizon=0.4
x,y=var('x y')
var('th1 th2 gamma')
FN= [y,-th1]
fFN=lambdify([x,y,th1], FN)
gamma=0.05 # dimension of S*
param=0 # index of the parameter we want to estimate.
tau=.1 # set Euler step
X0FN=[ [[7,12]] ] # define initial range for parameters
SFN=[[a-gamma-1,a+gamma+1] for a in adv([9.8],[0,-4],0,time_horizon,[time_horizon],tau, fFN)] # definition of the observation set S
fAdv=lambda th1: adv([th1],[0,-4],0,time_horizon,[time_horizon],tau,fFN) # advection function
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:1]),[0,-4],0,time_horizon,[time_horizon],tau,fFN)))  #interval extension of the advection function
slDescrFN=[('uniform', (7,12)), ('trunc_gauss',(0,.1,-1,1)), ('trunc_gauss',(0,.1,-1,1))] # defining the measures
slFN=buildSampler(slDescrFN) # build samplers one for each parameter
P=list(np.arange(7,12,0.7)) # nodes points for the approximate CDF
rX0FN=refine(X0FN,P,param) #refinement step
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True) # convering computation
ns=ceiling(Nsample(sum([e[1] for e in Xout]),.001,.001/(len(P)+1))) # number of samples for epsilon=0.001 and delta=0.001
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True) #cdf estimation
cl,cu=cdfLU(hi,.001) # lower and upper  CDFs
mump(cl,cu,Pn) # confidence interval for expected value of the posterior density

   
Additional examples of usage and experiments are at the end of the script.                            
"""


import scipy
from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import time
from mpmath import iv
from sortedcontainers import SortedList
import copy



def widthint(I):
    '''
    It returns the width of an hyperinterval as the maximum width of intervals composing it.
    '''
    Diff=[ii[1]-ii[0] for ii in I]
    j= np.argmax(Diff)
    return Diff[j], j

def int2iv(I):
    '''
    It creates intervals using mpmath Python library.
    '''
    return [iv.mpf(ii) for ii in I]

def bisect(I,w=None,j=None):
    '''
    It bisects hyperinterval I on dimension j.
    '''
    if w==None or j==None:
        w,j=widthint(I)
    a=I[j][0]
    b=I[j][1]
    I1=I.copy()
    I2=I.copy()
    I1[j]=[a,a+w/2]
    I2[j]=[a+w/2,b]
    return I1, I2


def refine(X,P,j):
    '''
    It refines X on dimension j, according to points in P.
    
    Args:
        - X: initial set to be refined.
        - P: set of points used to refine the initial set.
        - j: index of the parameter for which we are refining.

    Returns:
        - newX: refined X.
    '''

    newX=[]
    for R in X:
        Rlist=[]
        a=R[j][0]
        b=R[j][1]
        cutpoints=sorted([p for p in P if R[j][0]<p and R[j][1]>p])
        if cutpoints!=[]:
            left=a
            for p in cutpoints:
                Rnew=copy.deepcopy(R)
                Rnew[j][0]=left
                Rnew[j][1]=p
                Rlist=Rlist+[Rnew]
                left=p
            Rnew=copy.deepcopy(R)
            Rnew[j][0]=left
            Rnew[j][1]=b
            Rlist=Rlist+[Rnew]
            newX=newX+Rlist
        else:
            newX=newX+[R]
    return newX

def boxBackReach(f,X0,Y,delta=.1,intvf=True):   
    '''
    Implementatiation of the SIVIA algorithm. It performs backward reachability via Interval Arithmetic.
        
    Args:
        - f: interval extension of the function to be inverted.
        - X0: initial box.
        - Y: image of the set of interest.
        - delta: resolution threshold 

    Returns:
        - X: boxes on the boundary of  f^{-1}(Y).
    '''
    start_time=time.time()
    if not intvf:
        NN=lambda I: [f(*int2iv(I))]
    else:
        NN=f
    B=boundingBoxIntv(X0)
    delta=delta*widthint(B)[0]
    L=X0.copy()
    X=[]
    nbisec=0
    while L!=[]:
        I=L.pop()
        w,j=widthint(I)
        J=iv2intv(NN(I))
        if notEmptyIntersection(J,Y):
            if w<=delta:
                X.append(I)
            else:
                nbisec+=1            
                I1,I2=bisect(I,w,j)
                L.append(I1)
                L.append(I2)
    print("Elapsed total time: %s seconds ---" % (time.time() - start_time))
    print("N. of bisections:", nbisec)
    print("N. of rectangles:", len(X))
    return X


def included(J,Y):
    '''
    It checks if one set is included into another.
    '''
    for R,S in zip(J,Y):
        if (R[0]<S[0]) or (R[1]>S[1]):
            return False
    return True

def pointInclusion(v,X):
    '''
    It checks if a point is included into a set.
    '''
    n=len(v)
    for I in X:
        if all([(v[i]>= I[i][0]) * (v[i]<= I[i][1]) for i in range(n)]):
            return True
    return False

def boundingBoxIntv(X):  
    '''
    It generates a hyper-rectangle that over-approximates a set of hyper-rectangles.  
    '''
    if X==[]:
        return []
    n=len(X[0])
    a=np.array(X).reshape(len(X),2*n)
    return [ [min(a[:,2*i]),max(a[:,2*i+1])] for i in range(n)  ]

def boxIntersect(R1,R2): 
    '''
    It checks if there is an intersection between two hyper-rectangles.
    '''
    n=len(R1)
    return all([ ( (R1[j][0]<=R2[j][1]) and  (R1[j][1]>=R2[j][0])) or ( (R2[j][0]<=R1[j][1]) and  (R2[j][1]>=R1[j][0])) for j in range(n) ])

def notEmptyIntersection(R,X):
    '''
    It checks if there is an intersection between one hyper-rectangles and a set of hyper-rectangles.
    '''
    for J in X:
        if boxIntersect(R,J):
            return True
    return False

def iv2intv(Iv):
    '''
    It represents mpmath intervals using Python lists.
    '''
    i=0
    for z in Iv:
        if type(z)==float: 
            z=iv.mpf(z)
            Iv[i]=z
        i=i+1
    return [ [float(z.a),float(z.b)] for z in Iv ]

def measure(R,samplerList):
    '''
    It counts the fraction of samples falling in each interval taken as input.

    '''
    return np.prod([sampler(I,op='cdf') for I,sampler in zip(R,samplerList)])

def boxBackReachAlt(f,X0,Y,sl,delta=.1,intvf=True,N=None):  
    '''
    Implementation of the SIVIA algorithm. it represents an enhanced version of boxBackReach.
    
    Args:
        - f: interval extension of the function to be inverted.
        - X0: initial box.
        - Y:  n-dimensional hyper-rectangle representing the image of the set of interest.
        - delta: resolution threshold 

    Returns:
        - Xin: boxes included  in f^{-1}(Y).        
        - Xout: boxes on the boundary of  f^{-1}(Y).
    '''
    global sivia_time
    start_time=time.time()
    if not intvf:
        NN=lambda I: [f(*int2iv(I))]
    else:
        NN=f
    B=boundingBoxIntv(X0)
    delta=delta*widthint(B)[0]
    L=[(R,measure(R,sl)) for R in X0.copy()]
    L=SortedList(L,key=lambda p: p[1])
    Xin=[]
    Xout=[]
    nbisec=0
    k=0
    vol=sum([e[1] for e in L])
    print(vol)
    while L!=[]:
        k+=1
        I,v=L.pop()
        w,j=widthint(I)
        J=iv2intv(NN(I))
        if included(J,Y):
            Xin.append((I,v))
        elif notEmptyIntersection(J,[Y]):
            if w<=delta:
                Xout.append((I,v))
            else:
                nbisec+=1            
                I1,I2=bisect(I,w,j)
                v1=measure(I1,sl)
                L.add((I1,v1))
                v2=measure(I2,sl)
                L.add((I2,v2))
        else:
            vol=vol-v
            if N!=None:
                print(k,')',vol,v,.5*vol/(N-k))
                if (k==N) or (v<=.5*vol/(N-k)):
                    break
    sivia_time=time.time() - start_time
    print("Elapsed total time: %s seconds ---" % (sivia_time))
    print("N. of bisections:", nbisec)
    print("N. of inner rectangles:", len(Xin))
    print("N. of outer rectangles:", len(Xout)+len(L))
    if N!=None:
        return Xin,Xout+L,vol,N-k
    return Xin,Xout+L,vol



def buildSampler(descr):    
    '''
    It builds a list of samplers.
     Args:
        - descr: string describing the probability distributions from which to sample.

     Returns:
        - SamplerList: list of samplers.
    '''
    global sl
    sl=[]
    SamplerList=[]
    for name,pars in descr:
        if name=='uniform':
            c=pars[1]-pars[0]
            sl.append(lambda t,t0=pars[0]: min(1,max(0,t-t0)/c))
            def sampler(i=None,N=1,op=None,c=c):  
                if type(i)==type(None):
                    i=[pars[0],pars[0]]

                if op==None:
                    return np.random.uniform(low=i[0],high=i[1],size=N)
                return (i[1]-i[0])/c
            
        if name=='gauss':
            mu,sigma=pars
            sl.append(scipy.stats.norm(loc=mu, scale=sigma).cdf)
            def sampler(i=None,N=1,op=None,mu=mu,sigma=sigma): 
                if type(i)!=type(None):
                    distr=scipy.stats.truncnorm((i[0] - mu)/sigma, (i[1] - mu)/sigma, loc=mu, scale=sigma)
                distrN=scipy.stats.norm(loc=mu, scale=sigma)
                #if 
                if op==None:
                    if i==None:
                        return distrN.rvs(N)
                    return distr.rvs(N)
                return distrN.cdf(i[1])-distrN.cdf(i[0])
        if name=='trunc_gauss':
            mu,sigma,a,b=pars
            sl.append(scipy.stats.truncnorm((a-mu)/sigma,(b-mu)/sigma,loc=mu, scale=sigma).cdf)
            def sampler(i=None,N=1,op=None,mu=mu,sigma=sigma,a=a,b=b):  
                if type(i)!=type(None):
                    distr=scipy.stats.truncnorm((i[0] - mu)/sigma, (i[1] - mu)/sigma, loc=mu, scale=sigma)
                distrT=scipy.stats.truncnorm((a-mu)/sigma,(b-mu)/sigma,loc=mu, scale=sigma)
                if op==None:
                    if i==None:
                        return distrT.rvs(N)
                    return distr.rvs(N)
                return distrT.cdf(i[1])-distrT.cdf(i[0])
        if name=='trunc_exp':
            rate,a,b=pars
            scale=1/rate
            sl.append(scipy.stats.truncexpon(b=(b-a)/scale, loc=a, scale=scale).cdf)
            def sampler(i=None,N=1,op=None,rate=rate,a=a,b=b):  
                if type(i)!=type(None):
                    distr=scipy.stats.truncexpon(b=(b-a)/scale, loc=a, scale=scale)
                distrT=scipy.stats.truncexpon(b=(b-a)/scale, loc=a, scale=scale)
                if op==None:
                    if i==None:
                        return distrT.rvs(N)
                    return distr.rvs(N)
                return distrT.cdf(i[1])-distrT.cdf(i[0])
        SamplerList=SamplerList+[sampler]
    return SamplerList
       

def volumefIMC(f,R,Y,samplerList,N): 
    '''
    It is used for the MC estimation of of Pr(f(Theta)+h(Psi) in Y | Theta in R).  
    It samples from the random variables \Theta and \Psi with support given in R 
    and then counts how many samplings lead to values contained in the set Y taken as input
    
    
      Args:
          - f: function to be inverted.
          - R: list indicating the initial ranges of each parameter.
          - Y:  n-dimensional hyper-rectangle representing the image of the set of interest.
          - samplerList: list of samplers.
          - N: number of samplings.

      Returns:
          - estimation of of Pr(f(Theta)+h(Psi) in Y | Theta in R).       
    '''
    n=len(R)                        
    theta=np.array([ sampler(I,N=N) for I,sampler in zip(R,samplerList[:n])])
    psi=np.array([ sampler(N=N) for sampler in samplerList[n:]])
    if len(psi)>0:
        y=[f(*thetai)+psii for thetai,psii in zip(theta.T,psi.T)]#y=f(*v)
    else:
        y=[f(*thetai) for thetai  in  theta.T ]
    incl=np.array([pointInclusion(yj,Y) for yj in y])
    count=len(incl[incl==True])
    return count/N

def estimateCDF(f,fIntv,X0,j,S,P,sl,enclR=None,res=None,B=1,delta=.1,Xin=None,intvf=True):
    '''
      It estimates the univariate CDF of parameter theta_j, Pr(theta_j<=t_i | f(Theta,Psi) in S) for t_i in P.
      
      Args:
          - f: function to be inverted.
          - fIntv: interval extension of the function to be inverted.
          - X0: initial box.
          - j: index of the parameter to be estimated.
          - S: set of observations we are considering.
          - P: nodes for CDF computation.
          - sl: list of samplers.
          - enclR: enclosure for the values that the parameters can take.
          - res: rectangles on the boundary of the parametric space.
          - B: number of samplings.
          - delta: resolution threshold.
          - Xin: rectangles included in the parametric space.

      Returns:
          - mu: estimate of g^-1(S).
          - hi: array s.t. X_t=hi[:j].
          - vol: fraction of sampling in the rectangles at the border of the parametric space.
          - covering of f^-1(S)
          
    '''

    global estimate_time
    start_time=time.time()
    if type(enclR)==type(None):
        enclR=X0[0]
    maxp=max([R[j][1] for R in X0])
    if type(res)==type(None):
        print('  Computing set inversion...')
        res=boxBackReach(fIntv,X0,S,delta=delta,intvf=intvf)
        print('  Done. Elapsed time: ',time.time() - start_time)
    #vol=0
    if type(Xin)==type(None):
        resV=[(R,measure(R,sl)) for R in res]
        Xin=[]
    else:
        resV=res
    vol=sum([e[1] for e in resV])
    P=P+[maxp]
    histogram=[0]*len(P)
    meas=0
    BoVol=np.float64(B)/vol

    for R,volR in resV:  
        measR=volumefIMC(f,R,S,sl,ceiling(BoVol*volR))*volR
        meas+=measR
        p=R[j][1]
        idx=P.index(min(x for x in P if x >= p))
        histogram[idx]+=measR
    for R,volR in Xin:  
        meas+=volR
        p=R[j][1]
        idx=P.index(min(x for x in P if x >= p))
        histogram[idx]+=volR    
    estimate_time=(time.time() - start_time)
    print("Elapsed total time: %s seconds ---" % (estimate_time))
    return meas,np.array(histogram), P, vol, res+Xin    


def Nsample(v,epsilon,delta):
    '''
    It returns the number of samplings to obtain the confidence taken as input.
    '''
    return np.log(2/delta) * v**2 / epsilon**2 / 2


def cdfLU(hi,eps):
    '''
    It returns an over- and under-approximation of the true CDF.
    '''
    v=sum(hi)
    cu=[ min(1,np.float64((sum(hi[:j+1])+eps))/max(0,v-eps)) for j in range(len(hi)-1)]+[1]
    cl=[ max(0,np.float64((sum(hi[:j])-eps))/(v+eps)) for j in range(len(hi))]
    return cl, cu 

def mump(cl,cu,Pn):     
    '''
    It returns the confidence interval for expected value of the posterior density.
    '''
    mumin=Pn[0]+sum([(1-cu[j])*(Pn[j+1]-Pn[j]) for j in range(len(Pn)-1)])
    muplu=Pn[0]+sum([(1-cl[j])*(Pn[j+1]-Pn[j]) for j in range(len(Pn)-1)])
    return mumin,muplu

def adv(theta,x0,a,b,out,tau,fn,j=-1):
    '''
    It computes the advection function.
    '''
    select=[ceiling((c-a)/tau) for c in out]
    sol=eulerInt(a,b,x0,theta,tau,fn)
    if j!=-1:
        sol=sol[:,j]
    return flatten([sol[i] for i in select])


'''
Support functions for plotting:
'''  
def plotCDF(cl,cu,Pn):
    plt.step(Pn,cl,where='post')
    plt.step(Pn,cu,where='post')

fig, ax = plt.subplots()                
def plotboxes(boundslist,col='b',newplot=False,lw=1):
    for bb in boundslist:
        bound=flatten(bb)
        bottomleftx=bound[0]
        bottomlefty=bound[2]
        width=bound[1]-bottomleftx
        height=bound[3]-bottomlefty
        rect=plt.Rectangle((bottomleftx, bottomlefty), width, height, color=col, linewidth=lw, fill=None) 
        currentAxis = plt.gca()
        currentAxis.add_patch(rect)
        ax.autoscale_view()
    ax.relim()
    plt.show()

'''
Functions implementing Euler method:
'''

def eulerStep(tau,x0,theta,fn):
    return x0+tau*np.array(fn(*(x0+theta)))

def eulerInt(a,b,x0,theta,tau,fn):
    niter=ceiling((b-a)/tau)
    sol=[x0]
    for _ in range(niter):
        x0=list(eulerStep(tau,x0,theta,fn))
        sol.append(x0)
    return np.array(sol).reshape(niter+1,len(x0))


##########################################################################
######################### EXPERIMENTS ####################################
##########################################################################


################## Code for Example 2 (simple ball/2): ##################
'''
gamma=0.5
time_horizon=1
x,y=var('x y')
var('th1 th2 gamma')
FN= [y,-th1]
fFN=lambdify([x,y,th1], FN)
param=0
tau=.1
X0FN=[ [[7,12]] ]
SFN=[[a-gamma-1,a+gamma+1] for a in adv([9.8],[0,-4],0,time_horizon,[time_horizon],tau, fFN)]
fAdv=lambda th1: adv([th1],[0,-4],0,time_horizon,[time_horizon],tau,fFN)
fAdvI=lambda pars :flatten(np.array(adv(int2iv(pars[:1]),[0,-4],0,time_horizon,[time_horizon],tau,fFN)))
slDescrFN=[('uniform', (7,12)), ('trunc_gauss',(0,.1,-1,1)),('trunc_gauss',(0,.1,-1,1))] # defining the measures
slFN=buildSampler(slDescrFN)
P=[7,8.8]  
Xin,Xout,vol = boxBackReachAlt(fAdvI,X0FN,SFN,sl=slFN,delta=.2,intvf=True)
rX0FN=refine(X0FN,P,param)
vol_out=sum([e[1] for e in Xout+Xin]) 
Ns=ceiling(Nsample(vol_out,.01,.001)) 
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout+Xin,B=25600,delta=.1,Xin=[],intvf=True)
'''


################## Code for Example 3 (simple ball/3): ##################
'''    
sivia_time=0
estimate_time=0
time_horizon=0.4
x,y=var('x y')
var('th1 th2 gamma')
FN= [y,-th1]
fFN=lambdify([x,y,th1], FN)
gamma=0.05
param=0
tau=.1
X0FN=[ [[7,12]] ]
SFN=[[a-gamma-1,a+gamma+1] for a in adv([9.8],[0,-4],0,time_horizon,[time_horizon],tau, fFN)]
fAdv=lambda th1: adv([th1],[0,-4],0,time_horizon,[time_horizon],tau,fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:1]),[0,-4],0,time_horizon,[time_horizon],tau,fFN)))
slDescrFN=[('uniform', (7,12)), ('trunc_gauss',(0,.1,-1,1)), ('trunc_gauss',(0,.1,-1,1))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(7,12,0.2))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001)
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''

#################################################################
###Fitz-Nagumo ODE-2 parameters (Chou, Sankaranarayanan 2019)###
#################################################################
'''
sivia_time=0
estimate_time=0
time_horizon=1
x,y=var('x y')
var('th1 th2 gamma')
FN= [.5*(x-x**3/3+y),2*(x-th1+th2*y)]
fFN=lambdify([x,y,th1,th2], FN)
fFNt=lambda   u, t, th : [.5*(u[0]-u[0]**3/3+u[1]) , 2*(u[0]-th[0]+th[1]*u[1])]
gamma=0.05
param=1
tau=.1
X0FN=[ [[0,.5],[0,.3] ]]
SFN=[[a-gamma-0.1,a+gamma+0.1] for a in adv([.3,.15],[-1,1],0,time_horizon,[time_horizon],tau, fFN)]
fAdv=lambda th1,th2: adv([th1,th2],[-1,1],0,time_horizon,[time_horizon],tau,fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:2]),[-1,1],0,time_horizon,[time_horizon],tau,fFN)))
slDescrFN=[('uniform',(0,.5)),('uniform',(0,.3)), ('trunc_gauss',(0,.01,-0.1,0.1)), ('trunc_gauss',(0,.01,-0.1,0.1))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(0,.3,0.01))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001) 
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''

#############################################################################
####### Fitz-Nagumo ODE 3 parameters  (Chou, Sankaranarayanan 2019) ##############
#############################################################################
'''
sivia_time=0
estimate_time=0
time_horizon=.1
x,y=var('x y')
var('th1 th2 th3')
FN3= [th3*(x-x**3/3+y),-(1/th3)*(x-th1+th2*y)]
fFN=lambdify([x,y,th1,th2, th3], FN3)
fFNt=lambda   u, t, th : [th[2]*(u[0]-u[0]**3/3+u[1]) , -(1/th[2])*(u[0]-th[0]+th[1]*u[1])]
gamma=0.01
trunc=0.5
param=2
tau=.01
X0FN=[ [[0,.5],[0,.3],[0.1,.9] ]]
SFN=[[a-gamma-trunc,a+gamma+trunc] for a in adv([.3,.15,0.5],[-1,1],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th1,th2,th3: adv([th1,th2,th3],[-1,1],0,time_horizon,[time_horizon],tau,fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:3]),[-1,1],0,time_horizon,[time_horizon],tau,fFN)))
slDescrFN=[('uniform',(0,.5)),('uniform',(0,.3)),('uniform',(0.1,.9)), ('trunc_gauss',(0,.1,-trunc,trunc)), ('trunc_gauss',(0,.1,-trunc,trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(0.1,.9,0.05))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001) 
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''
#################################################################
############### P53  (Chou, Sankaranarayanan 2019) #############
#################################################################

'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6=var('x1 x2 x3 x4 x5 x6')
var('th1 th2')
FN= [0.5-9.97*10**(-6)*x1*x5-1.93*10**(-5)*x1,1.5*10**(-3)+1.5*10**(-2)*(x1**2/(740**2+x1**2))-8*10**(-4)*x2,8*10**(-4)*x2-1.4*10**(-4)*x3, 1.7*10**(-2)*x3-th1*x4, th1*x4-1.7*10**(-7)*x4*x4-th2*x5*x6, 0.5-3.2*10**(-5)*x6-th2*x5*x6]
fFN=lambdify([x1, x2, x3, x4, x5, x6,th1,th2], FN)
param=1
tau=.01
gamma=0.04 
dev= 14.14 
a_trunc=-14.14*3 
b_trunc=14.14*3 
time_horizon=.5
X0FN=[ [[7.5*10**(-4),10.5*10**(-4)],[8*10**(-6),11*10**(-6)] ]]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([9*10**(-4),9.963*10**(-6)],[20,20,20,20,20,20],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th1,th2: adv([th1,th2],[20,20,20,20,20,20],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:2]),[20,20,20,20,20,20],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(7.5*10**(-4),10.5*10**(-4))),('uniform',(8*10**(-6),11*10**(-6))), ('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(8*10**(-6),11*10**(-6),.0000001))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001) 
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
kde=scipy.stats.gaussian_kde(Pn, bw_method='silverman', weights=np.array(hi)/mu)
plt.plot(Pn,kde(Pn))
'''
#################################################################
########### LAUB-LOOMIS_1  (Chou, Sankaranarayanan 2019) ########
#################################################################
'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6, x7=var('x1 x2 x3 x4 x5 x6 x7')
var('th1 th2 th3 th4 th5 th6 th7 th8 th9 th10 th11 th12 th13')
FN= [1.4*x3-0.9*x1, 2.5*x5-1.5*x2, 0.6*x7-th2*x3*x2, 2-1.3*x4*x3, 0.7*x1-1*x4*x5, th3*x1-3.1*x6, th6*x6-1.5*x7*x2]
fFN=lambdify([x1, x2, x3, x4, x5, x6, x7, th2, th3, th6], FN)
param=0
tau=0.01
gamma=0.05
dev= 0.022 
a_trunc=-0.55
b_trunc=0.55
time_horizon=.5
X0FN=[ [[0.5,3],[0.5,1],[0,0.5] ]]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([1.8,0.8,0.3],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th2,th3,th6: adv([th2,th3,th6],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:3]),[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(0.5,3)),('uniform',(0.5,1)),('uniform',(0,0.5)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(0.5,3,.05))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001)
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''
#################################################################
########### LAUB-LOOMIS_2  (Chou, Sankaranarayanan 2019) ########
#################################################################
'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6, x7=var('x1 x2 x3 x4 x5 x6 x7')
var('th1 th2 th3 th4 th5 th6 th7 th8 th9 th10 th11 th12 th13')
FN= [1.4*x3-th1*x1, 2.5*x5-1.5*x2, 0.6*x7-th2*x3*x2, 2-1.3*x4*x3, 0.7*x1-1.0*x4*x5, th3*x1-3.1*x6, 1.8*x6-1.5*x7*x2]
fFN=lambdify([x1, x2, x3, x4, x5, x6, x7, th1, th2, th3], FN)
param=1
tau=.05
gamma=0.05 
dev= 0.022 
a_trunc=-0.4 
b_trunc=0.4
time_horizon=1
X0FN=[ [[0.6,1.2],[0.5,1],[0,0.5] ]]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([0.9,0.8,0.3],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th1,th2,th3: adv([th1,th2,th3],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:3]),[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(0.6,1.2)),('uniform',(0.5,1)),('uniform',(0,0.5)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(0.5,1,.05))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001)
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''
#################################################################
########### LAUB-LOOMIS_3  (Chou, Sankaranarayanan 2019) ########
#################################################################
'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6, x7=var('x1 x2 x3 x4 x5 x6 x7')
var('th1 th2 th3 th4 th5 th6 th7 th8 th9 th10 th11 th12 th13')
FN= [1.4*x3-0.9*x1, th4*x5-1.5*x2, 0.6*x7-th2*x3*x2, 2-1.3*x4*x3, 0.7*x1-1.0*x4*x5, th3*x1-3.1*x6, th6*x6-1.5*x7*x2]
fFN=lambdify([x1, x2, x3, x4, x5, x6, x7, th2, th3, th4, th6], FN)
param=0
tau=.05
gamma=0.07
dev= 0.022 
a_trunc=-0.45 
b_trunc=0.45 
time_horizon=.5
X0FN=[ [[0.5,3],[0.5,1],[0,0.5],[2,3] ]]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([1.8,0.8,0.3,2.5],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau, fFN)]
fAdv=lambda th2,th3,th4,th6: adv([th2,th3,th4,th6],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:4]),[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(0.5,3)),('uniform',(0.5,1)),('uniform',(0,0.5)),('uniform',(2,3)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(0.5,3,0.05))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001) 
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''
#################################################################
########### LAUB-LOOMIS_4  (Chou, Sankaranarayanan 2019) ########
#################################################################
'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6, x7=var('x1 x2 x3 x4 x5 x6 x7')
var('th1 th2 th3 th4 th5 th6 th7 th8 th9 th10 th11 th12 th13')
FN= [th7*x3-th1*x1, 2.5*x5-1.5*x2, 0.6*x7-th2*x3*x2, 2-1.3*x4*x3, 0.7*x1-1.0*x4*x5, th3*x1-3.1*x6, 1.8*x6-1.5*x7*x2]
fFN=lambdify([x1, x2, x3, x4, x5, x6, x7, th1, th2, th3, th7], FN)
param=3
tau=.1
gamma=0.05 
dev= 0.022 
a_trunc=-0.22 
b_trunc=0.22 
time_horizon=.4
X0FN=[ [[0.6,1.2],[0.5,1],[0,0.5],[1.2,1.7] ]]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([0.9,0.8,0.3,1.4],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau, fFN)]
fAdv=lambda th1,th2,th3,th7: adv([th1,th2,th3,th7],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:4]),[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(0.6,1.2)),('uniform',(0.5,1)),('uniform',(0,0.5)),('uniform',(1.2,1.7)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(1.2,1.7,0.01)) 
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001)
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''
#################################################################
########### LAUB-LOOMIS_5  (Chou, Sankaranarayanan 2019) ########
#################################################################
'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6, x7=var('x1 x2 x3 x4 x5 x6 x7')
var('th1 th2 th3 th4 th5 th6 th7 th8 th9 th10 th11 th12 th13')
FN= [1.4*x3-th1*x1, th4*x5-1.5*x2, 0.6*x7-th2*x3*x2, 2-th5*x4*x3, 0.7*x1-1.0*x4*x5, th3*x1-3.1*x6, 1.8*x6-1.5*x7*x2]
fFN=lambdify([x1, x2, x3, x4, x5, x6, x7, th1, th2, th3, th4, th5], FN)
param=4
tau=.1
gamma=0.05 
dev= 0.022 
a_trunc=-0.22 
b_trunc=0.22 
time_horizon=.5
X0FN=[ [[0.6,1.2],[0.5,1],[0,0.5],[2,3],[1,1.5] ]]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([0.9,0.8,0.3,2.5,1.3],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th1,th2,th3,th4,th5: adv([th1,th2,th3,th4,th5],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:5]),[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(0.6,1.2)),('uniform',(0.5,1)),('uniform',(0,0.5)),('uniform',(2,3)),('uniform',(1.0,1.5)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(1,1.5,.05))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001)
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''

#################################################################
########### LAUB-LOOMIS_6  (Chou, Sankaranarayanan 2019) ########
#################################################################
'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6, x7=var('x1 x2 x3 x4 x5 x6 x7')
var('th1 th2 th3 th4 th5 th6 th7 th8 th9 th10 th11 th12 th13')
FN= [1.4*x3-0.9*x1, th4*x5-1.5*x2, 0.6*x7-th2*x3*x2, 2-th5*x4*x3, 0.7*x1-1.0*x4*x5, th3*x1-3.1*x6, th6*x6-1.5*x7*x2]
fFN=lambdify([x1, x2, x3, x4, x5, x6, x7, th2, th3, th4, th5, th6], FN)
param=4
tau=.1
gamma=0.05 
dev= 0.022 
a_trunc=-0.22
b_trunc=0.22 
time_horizon=.2
X0FN=[ [[0.5,3],[0.5,1],[0,0.5],[2,3],[1,1.5] ]]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([1.8,0.8,0.3,2.5,1.3],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th2,th3,th4,th5,th6: adv([th2,th3,th4,th5,th6],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:5]),[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(0.5,3)),('uniform',(0.5,1)),('uniform',(0,0.5)),('uniform',(2,3)),('uniform',(1.0,1.5)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(1,1.5,.05)) 
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001) # lower and upper CDF
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''

#################################################################
########### LAUB-LOOMIS_7  (Chou, Sankaranarayanan 2019) ########
#################################################################
'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6, x7=var('x1 x2 x3 x4 x5 x6 x7')
var('th1 th2 th3 th4 th5 th6 th7 th8 th9 th10 th11 th12 th13')
FN= [1.4*x3-th1*x1, th4*x5-1.5*x2, 0.6*x7-th2*x3*x2, 2-th5*x4*x3, 0.7*x1-1.0*x4*x5, th3*x1-3.1*x6, th6*x6-1.5*x7*x2]
fFN=lambdify([x1, x2, x3, x4, x5, x6, x7, th1, th2, th3, th4, th5, th6], FN)
param=4
tau=.1
gamma=0.05 
dev= 0.022 
a_trunc=-0.22 
b_trunc=0.22 
time_horizon=.5
X0FN=[ [[0.6,1.2],[0.5,1],[0,0.5],[2,3],[1,1.5],[0.5,3] ]]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([0.9,0.8,0.3,2.5,1.3,1.8],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th1,th2,th3,th4,th5,th6: adv([th1,th2,th3,th4,th5,th6],[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:6]),[1.2,1.0,1.5,2.4,1.0,0.1,0.45],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(0.6,1.2)),('uniform',(0.5,1)),('uniform',(0,0.5)),('uniform',(2,3)),('uniform',(1.2,1.7)),('uniform',(0.5,3)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(1,1.5,0.05))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001) 
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''
#################################################################
######### Rossler (Chou, Sankaranarayanan 2019) #########
#################################################################

'''
sivia_time=0
estimate_time=0
x1, x2, x3=var('x1 x2 x3 ')
var('th1 th2')
FN=[-x2-x3,x1+th1*x2,th2+x3*(x1-1)]
fFN=lambdify([x1, x2, x3, th1, th2], FN)
param=0
tau=0.05
gamma=0.05
var=0.09
a_trunc=-0.1
b_trunc=0.1
time_horizon=.5
X0FN=[ [[0,0.5],[0,0.5]] ]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([0.1,0.1],[0,3,1],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th1,th2: adv([th1,th2],[0,3,1],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:2]),[0,3,1],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(0,0.5)),('uniform',(0,0.5)), ('trunc_gauss',(0, 0.3,a_trunc,b_trunc)), ('trunc_gauss',(0, 0.3,a_trunc,b_trunc)),('trunc_gauss',(0, 0.3,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(0,0.5,.05))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001)
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''

#################################################################
######### Genetic (Chou, Sankaranarayanan 2019) #########
#################################################################
'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6, x7, x8, x9 =var('x1 x2 x3 x4 x5 x6 x7 x8 x9')
var('th1 th2')
FN=[th1*x3-0.1*x1*x6,100*x4-x1*x2,0.1*x1*x6-th1*x3,x2*x6-100*x4,5*x3+0.5*x1-10*x5,th2*x5+th1*x3+100*x4-x6*(0.1*x1+x2+2*x8+1),50*x4+0.01*x2-0.5*x7,0.5*x7-2*x6*x8+x9-0.2*x8,2*x6*x8-x9]
fFN=lambdify([x1, x2, x3, x4, x5, x6, x7, x8, x9, th1, th2], FN)
param=0
tau=0.1
gamma=0.05
dev= 10**(-5)
a_trunc=-5*10**(-5)
b_trunc=5*10**(-5)
time_horizon=0.3
X0FN=[ [[48.5,51.5],[47,52]] ]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([50,50],[1.0, 1.3, 0.1, 0.1, 0.1, 1.3, 2.5, 0.6, 1.3],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th1,th2: adv([th1,th2],[1.0, 1.3, 0.1, 0.1, 0.1, 1.3, 2.5, 0.6, 1.3],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:2]),[1.0, 1.3, 0.1, 0.1, 0.1, 1.3, 2.5, 0.6, 1.3],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(48.5,51.5)),('uniform',(47,52)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
P=list(np.arange(48.5,51.5,0.01)) 
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,0.0001) 
mump(cl,cu,Pn)
'''

#################################################################
############### Dalla-Man (Chou, Sankaranarayanan 2019) #############
#################################################################
'''
sivia_time=0
estimate_time=0
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11=var('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11')
var('th1 th2')
FN=[0.1*(0.5521*x6-x1),
    -0.0278*x2+0.0278*(18.2129*x8-100.25),
    0.0142*x3-0.0078*x4 + 0.488758553275,
     0.0152*x3-0.0078*x4,
     -0.0039*(3.2267+0.0313*x2)*x5*(1-0.0026*x5+2.5097*10**(-6)*x5**2)+th1*x6-th2*x5,
     3.7314-0.0047*x6-0.0121*x10-th1*x6+th2*x5+50*((1.141*10**(-4))*x11**2+(6.134*10**(-5))*x11),
     -0.4219*x7+0.225*x8,
      -0.315*x8+0.1545*x7+1.9*10**(-3)*x3+7.8*10**(-3)*x4,
      -0.0046*(x9-18.2129*x8),-0.0046*(x10-x9),
      0
      ]
fFN=lambdify([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, th1, th2], FN)
param=0
tau=.3
gamma=0.05
dev= 0.5
a_trunc=-1.5
b_trunc=1.5 
time_horizon=.1
X0FN=[ [[0.04,0.08],[0.05,0.12] ]]
SFN=[[a-gamma-b_trunc,a+gamma+b_trunc] for a in adv([0.0581,0.0871],[140, 72.43, 141.15, 162.45, 268, 3.2, 5.5, 100.25, 100.25, 0, 0],0,time_horizon,[time_horizon],tau,fFN)]
fAdv=lambda th1,th2: adv([th1,th2],[140,72.43,141.14,162.45,268,3.2,5.5,100.25,100.25,0,0],0,time_horizon,[time_horizon],tau,fn=fFN)
fAdvI=lambda pars : flatten(np.array(adv(int2iv(pars[:2]),[140,72.43,141.14,162.45,268,3.2,5.5,100.25,100.25,0,0],0,time_horizon,[time_horizon],tau,fn=fFN)))
slDescrFN=[('uniform',(0.04,0.08)),('uniform',(0.05,0.12)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)), ('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc)),('trunc_gauss',(0,dev,a_trunc,b_trunc))]
slFN=buildSampler(slDescrFN)
refinement=.001
P=list(np.arange(0.04,0.08,refinement))
rX0FN=refine(X0FN,P,param)
Xin,Xout,vol = boxBackReachAlt(fAdvI,rX0FN,SFN,sl=slFN,delta=.1,intvf=True)
vol_out=sum([e[1] for e in Xout])
ns=ceiling(Nsample(vol_out,.001,.001/(len(P)+1)))
mu,hi,Pn,volN,_=estimateCDF(fAdv,fAdvI,rX0FN,param,[SFN],P,slFN,enclR=None,res=Xout,B=ns,delta=.1,Xin=Xin,intvf=True)
plt.step(Pn,hi/mu,where='post')
plt.show()
cl,cu=cdfLU(hi,.001)
mump(cl,cu,Pn)
plotCDF(cl,cu,Pn)
plt.show()
'''

################################################################################################################################
################## Estimate Feature Relevance in  MNIST images (Stavros P. Adam, Aristidis C. Likas, 2022)  ####################
###############################################################################################################################
'''

####### to handle trained NN ##############
    
def ReLU(x):
    return max(x,0)

def eLU(x):
    if x>=0:
        return x
    return 0.2*(np.exp(x)-1)

def applyFFN(x,WBAL,selectout=None,pre=None,post=None):
    if type(pre)!=type(None):
        x=pre(x)
    k=len(WBAL)
    j=0
    cv=x
    for W,b,act in WBAL:
        j+=1
        if type(selectout)!=type(None) and (j==k):
            y=(W[selectout]@cv+b[selectout])
        else:
            y=(W@cv+b)
        cv=np.array([act(yi) for yi in y],dtype=np.float64)
    if type(post)!=type(None):
        cv=post(cv)
    return cv

def applyFFNIntv(I,WBAL,selectout=None,pre=None,post=None):  # interval version of the above
    if type(pre)!=type(None):
        I=[ [pre(J[0]),pre(J[1])] for pre,J in zip(pre,I)]
    k=len(WBAL)
    j=0
    cv=np.array(int2iv(I))
    for W,b,act in WBAL:
        j+=1
        if type(selectout)!=type(None) and (j==k):
            y=(W[selectout]@cv+b[selectout])
        else:
            y=(W@cv+b)
        cv=np.array(int2iv([[act(float(yi.a)),act(float(yi.b))] for yi in y]))            
    if type(post)!=type(None):
        cv=np.array(int2iv([[post(float(yi.a)),post(float(yi.b))] for post,yi in zip(post,cv)]))# outputf(cv)    
    return cv

def generateFFN(WBAL,selectout=None,pre=None,post=None):
    return lambda *x, WBAL=WBAL, selectout=selectout, pre=pre, post=post: applyFFN(x,WBAL,selectout,pre,post)

def generateFFNIntv(WBAL,selectout=None,pre=None,post=None):
    return lambda I, WBAL=WBAL, selectout=selectout, pre=pre, post=post : applyFFNIntv(I,WBAL,selectout,pre,post)


######### import MNIST images  ##############
from PIL import Image
test_images = scipy.io.loadmat(r'test_images.mat')['images']
labels_test = scipy.io.loadmat(r'labels_test.mat')['labels']

############ utilities for MNIST images ####################
def searchMin(Nim,thresh=0,N=10,changeClass=False,display=False): # modifies a set of max N pixels of original image to obtain a change in classification
    global test_images, labels_test
    label=labels_test[Nim][0]
    im0=test_images[:,Nim].copy()
    minv=np.inf
    pos=None
    val=None
    modList=[]
    nmod=0
    while(minv>thresh) and (nmod<N):
        for i in range(len(im0)):
            p=im0[i]
            if p<=.5:
                v=1
            else:
                v=0
            im0[i]=v
            res=applyFFN(im0,NNdescr)
            argmax=np.argmax(res)
            if changeClass & (argmax!=label):
                modList.append([pos,v,res[label]])
                modList.append(['*','*',res[argmax],argmax])
                return modList,im0
            im0[i]=p
            if res[label]<minv:
                minv=res[label]
                pos=i
                val=v
        modList.append([pos,val,minv])
        im0[pos]=val     
        nmod+=1    
    if display:
        show(test_images[:,Nim])
        show(im0)
    return modList,im0


def show(im):
    z0= (im * 255).astype(np.uint8).reshape(28,28)
    img = Image.fromarray(z0)
    img.show()
    
######### import   NN classifier ##############
W2im = scipy.io.loadmat(r'wtwo.mat')['w12']    # weights matrices
W3im = scipy.io.loadmat(r'wthree.mat')['w23']
W4im = scipy.io.loadmat(r'wfour.mat')['w34']
b2im = scipy.io.loadmat(r'btwo.mat')['b12'][:,0]   # bias vectors
b3im = scipy.io.loadmat(r'bthree.mat')['b23'][:,0]
b4im = scipy.io.loadmat(r'bfour.mat')['b34'][:,0]
NNdescr=[[W2im,b2im,ReLU],[W3im,b3im,ReLU],[W4im,b4im,ReLU]]

class_out=[max(applyFFN(test_images[:,j],NNdescr)) for j in range(len(test_images))] # classifier output (max value) for all MNIST images
reshigh=[(i,v) for i,v in zip(range(len(class_out)),class_out) if v>=0.8]   # images for which classifier returns >=0.8
Nim=1  # change 1 to wanted image's number, taken from reshigh
im0=np.array(test_images[:,Nim].copy(),dtype=object)    # copy of selected image
a,_=searchMin(Nim,N=1)   # search most 'relevant' pixel (feature) in selected image. NB: actually *any* npixel in the range 0..783 can be investigated
npixel=a[0][0]           # npixel represents the "feature" whose relevance we want to quantify 
vpixel=im0[npixel]
label=labels_test[Nim][0]

def fMNIST(x):
    global im0,npixel,label
    im0[npixel]=x
    res=applyFFN(im0,NNdescr)[label]
    return  [res]

def fMNISTI(I):      # interval version of the above: I must be an 1-dimensional rectangle I= [ [a,b] ]
    global im0,pixel,label
    im0[npixel]=I[0]
    res=np.array([applyFFNIntv(im0,NNdescr)[label]])
    return res
   
X0MNIST=[ [[0,1]] ]   
SMNIST=[[.8,1.5]]
slDescrMNIST=[('uniform',(0,1))]
slMNIST=buildSampler(slDescrMNIST)    
res = boxBackReach(fMNISTI,X0MNIST,[SMNIST],delta=.1,intvf=True)
slDescrFN=[('uniform',(0,.5)),('uniform',(0,.3)), ('trunc_gauss',(0,.01,-.1,.1)), ('trunc_gauss',(0,.01,-.1,.1))]
slFN=buildSampler(slDescrFN)
vol=sum([measure(R,slFN) for R in res])
Ns=ceiling(Nsample(vol,.01,.001))
mu,_,_,_,_=estimateCDF(fMNIST,fMNISTI,X0MNIST,0,[SMNIST],[],slMNIST,res=res,B=Ns)
# Here 1-mu represents an estimate (+/-0.01, with a confidence of 0.001) of the feature's relevance
# this can be repeated for all the pixels of the image
'''
