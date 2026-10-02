from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad

@dataclass
class CellConstants:
    t: float
    dmax_deg: float
    radius_deg: float
    mu: float
    c2: float
    c3: float
    Pq: np.ndarray
    edge_dual: float

def local_cell_constants(problem,theta_center,radius_deg=.02,delta_bar_deg=84.0,t=0.0):
    B=problem.Inc; k=problem.Kedge; E=problem.E; Mhi=np.diag(1/np.sqrt(problem.M))
    dc=B.T@np.asarray(theta_center); db=math.radians(float(delta_bar_deg)); r=math.radians(float(radius_deg))
    if np.max(np.abs(dc))+r>=math.pi/2:
        r=max(0.0,math.pi/2-1e-6-np.max(np.abs(dc)))
    Lc=B@np.diag(k*np.cos(dc))@B.T; Pq=E.T@Lc@E
    w,V=eigh((Pq+Pq.T)/2)
    if w.min()<=1e-12: raise RuntimeError('local stiffness metric not positive definite')
    Pinv=(V*(1/w))@V.T
    Csum=0.0; c2=0.0; c3=0.0; edge_dual=0.0
    for e in range(len(k)):
        b=B[:,e]; br=E.T@b; u=Mhi@b
        re=float(br@Pinv@br); sre=math.sqrt(max(re,0.0)); edge_dual=max(edge_dual,sre)
        lo=max(-db,float(dc[e]-r)); hi=min(db,float(dc[e]+r))
        dcos=max(abs(math.cos(lo)-math.cos(float(dc[e]))),abs(math.cos(hi)-math.cos(float(dc[e]))))
        Csum += k[e]*dcos*np.linalg.norm(u)*sre
        sinmax=max(abs(math.sin(lo)),abs(math.sin(hi)))
        cosmax=max(abs(math.cos(lo)),abs(math.cos(hi)))
        c2 += k[e]*sinmax*re*np.linalg.norm(u)
        c3 += k[e]*cosmax*(re**1.5)*np.linalg.norm(u)
    return CellConstants(float(t),float(np.degrees(np.max(np.abs(dc)))),float(np.degrees(r)),.5*Csum,float(c2),float(c3),Pq,float(edge_dual))

def flow_majorants(mu,c2,c3,horizon):
    t=float(horizon)
    if abs(mu)<1e-14:
        return 1.0,c2*t,1.5*c2*c2*t*t+c3*t
    e=math.exp(mu*t)
    j=e
    h=(c2/mu)*e*(e-1)
    k=(c3/(2*mu))*e*(e*e-1)+(3*c2*c2/(2*mu*mu))*e*(e-1)**2
    return float(j),float(h),float(k)

def _h_derivs(g):
    c=math.cos(g);t=math.tan(g);sec2=1/(c*c)
    return -math.log(c),t,sec2,2*sec2*t,4*sec2*t*t+2*sec2*sec2

def _extended_h_derivative_scalar(x,order,guard):
    h0,h1,h2,h3,h4=_h_derivs(guard); a=abs(float(x)); sg=1.0 if x>=0 else -1.0
    if a<=guard:
        t=math.tan(a); sec2=1/(math.cos(a)**2)
        if order==1:return sg*t
        if order==2:return sec2
        if order==3:return sg*2*sec2*t
    z=a-guard
    if order==1:return sg*(h1+h2*z+.5*h3*z*z+(h4/6)*z**3)
    if order==2:return h2+h3*z+.5*h4*z*z
    if order==3:return sg*(h3+h4*z)
    raise ValueError(order)

def global_compact_cost_derivative_bounds(problem,delta_bar_deg=84.,omega_bar=6.5,running_energy=.08,running_barrier=.02,terminal_energy=.20,guard_deg=82.):
    db=math.radians(delta_bar_deg); c=math.cos(db); alpha=.5*(1+c); k=problem.Kedge; m=problem.Inc.shape[1]; ds=problem.delta_star
    md=[]
    for x in ds:
        md.append(max(abs(math.sin(-db)-math.sin(x)),abs(math.sin(db)-math.sin(x)),abs(math.sin(x))))
    md=np.asarray(md)
    E1a=math.sqrt(np.sum(k*md*md)/alpha); E1w=omega_bar*math.sqrt(np.sum(problem.M)); E1=math.sqrt(E1a*E1a+E1w*E1w)
    E2=max(1/alpha,1.); E3=1/(alpha**1.5*math.sqrt(np.min(k)))
    guard=math.radians(guard_deg)
    grid=np.linspace(-db,db,4001)
    h2max=max(abs(_extended_h_derivative_scalar(x,2,guard)) for x in grid)*1.001
    h3max=max(abs(_extended_h_derivative_scalar(x,3,guard)) for x in grid)*1.001
    bcoeff=[]
    for x in ds:
        hpstar=math.tan(float(x)); bcoeff.append(max(abs(_extended_h_derivative_scalar(y,1,guard)-hpstar) for y in grid)/m)
    bcoeff=np.asarray(bcoeff)
    B1=math.sqrt(np.sum(bcoeff*bcoeff/(alpha*k))); B2=h2max/(m*alpha*np.min(k)); B3=h3max/(m*alpha**1.5*np.min(k)**1.5)
    return dict(running_first=running_energy*E1+running_barrier*B1,
                running_second=running_energy*E2+running_barrier*B2,
                running_third=running_energy*E3+running_barrier*B3,
                terminal_first=terminal_energy*E1,
                terminal_second=terminal_energy*E2,
                terminal_third=terminal_energy*E3)

def action_derivative_majorants(mu,c2,c3,cost,horizon):
    H=float(horizon)
    def vals(s): return flow_majorants(mu,c2,c3,s)
    C1=quad(lambda s:cost['running_first']*vals(s)[0],0,H,epsabs=1e-9,limit=100)[0]
    C2=quad(lambda s:cost['running_second']*vals(s)[0]**2+cost['running_first']*vals(s)[1],0,H,epsabs=1e-9,limit=100)[0]
    C3=quad(lambda s:cost['running_third']*vals(s)[0]**3+3*cost['running_second']*vals(s)[1]*vals(s)[0]+cost['running_first']*vals(s)[2],0,H,epsabs=1e-8,limit=100)[0]
    j,h,k=vals(H)
    C1+=cost['terminal_first']*j
    C2+=cost['terminal_second']*j*j+cost['terminal_first']*h
    C3+=cost['terminal_third']*j**3+3*cost['terminal_second']*h*j+cost['terminal_first']*k
    return float(C1),float(C2),float(C3)

def value_third_bound(C1,C2,C3,lam):
    return float(C3+6*C1*C2/lam+8*C1**3/(lam*lam))
