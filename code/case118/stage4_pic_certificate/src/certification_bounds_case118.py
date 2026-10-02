from __future__ import annotations
import math
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq,minimize_scalar

# Generalized from the case39 cycle/energy terminal enclosure in the frozen
# reinforcement-planning code base. The only structural change is that the
# physical injection-control Gramian is architecture-specific (H0--H3).

def _hp(x,g):
    a=abs(float(x)); sg=1.0 if x>=0 else -1.0
    t=math.tan(g); sec2=1/(math.cos(g)**2); h3=2*sec2*t; h4=4*sec2*t*t+2*sec2*sec2
    if a<=g: return math.tan(float(x))
    z=a-g; return sg*(t+sec2*z+.5*h3*z*z+(h4/6)*z**3)

def _h2(x,g):
    a=abs(float(x)); sec2=1/(math.cos(g)**2); t=math.tan(g); h3=2*sec2*t; h4=4*sec2*t*t+2*sec2*sec2
    if a<=g:return 1/(math.cos(float(x))**2)
    z=a-g; return sec2+h3*z+.5*h4*z*z

def scalar_edge_energy_intervals(problem,Vcap,delta_bar_deg=84.0):
    db=math.radians(float(delta_bar_deg)); ds=problem.delta_star; k=problem.Kedge
    lo=np.empty(len(k)); hi=np.empty(len(k))
    for e,(s,ke) in enumerate(zip(ds,k)):
        def ve(x): return ke*(math.cos(s)-math.cos(x)-math.sin(s)*(x-s))
        l=-db; h=db
        if ve(l)>Vcap:l=brentq(lambda x:ve(x)-Vcap,-db,float(s),xtol=1e-13)
        if ve(h)>Vcap:h=brentq(lambda x:ve(x)-Vcap,float(s),db,xtol=1e-13)
        lo[e]=l;hi[e]=h
    return lo,hi

def cycle_energy_edge_enclosure(problem,Vcap,delta_bar_deg=84.0,max_iter=100,tol=1e-11):
    lo,hi=scalar_edge_energy_intervals(problem,Vcap,delta_bar_deg)
    B=problem.Inc;k=problem.Kedge;ds=problem.delta_star;hist=[]
    for _ in range(max_iter):
        c=np.cos(np.maximum(np.abs(lo),np.abs(hi)));Lmin=B@np.diag(k*c)@B.T
        ev=np.linalg.eigvalsh((Lmin+Lmin.T)/2)
        if len(ev)<2 or ev[1]<=1e-12: raise RuntimeError('cycle-energy lower stiffness lost connectivity')
        Lpinv=np.linalg.pinv(Lmin,rcond=1e-12)
        rad=np.array([math.sqrt(max(0.,2*Vcap*(B[:,e]@Lpinv@B[:,e]))) for e in range(len(k))])
        nlo=np.maximum(lo,ds-rad);nhi=np.minimum(hi,ds+rad)
        diff=max(float(np.max(np.abs(nlo-lo))),float(np.max(np.abs(nhi-hi))))
        lo,hi=nlo,nhi;hist.append(diff)
        if diff<tol:break
    pad=2e-10;db=math.radians(float(delta_bar_deg));lo=np.maximum(-db,lo-pad);hi=np.minimum(db,hi+pad)
    return dict(lo=lo,hi=hi,iterations=len(hist),last_change=hist[-1] if hist else 0.,max_abs_deg=np.degrees(np.maximum(np.abs(lo),np.abs(hi))))

def _constants(problem,geom,enclosure,running_energy=.08,running_barrier=.02,terminal_energy=.20,guard_deg=82.0):
    n=problem.n;m=problem.Inc.shape[1];d0=terminal_energy/n
    M=np.diag(problem.M);Minv=np.diag(1/problem.M);D=np.diag(problem.damp);Pg=np.asarray(geom.injection_gain_physical,float)
    C=(running_energy/n)*M-2*d0*D-d0*d0*Pg;Ac=Minv@(D+d0*Pg);G=Minv@Pg@Minv;Q=.5*(Ac.T@C+C@Ac)
    Mh=np.diag(np.sqrt(problem.M));Mhi=np.diag(1/np.sqrt(problem.M));Cy=Mhi@C@Mhi;Acy=Mh@Ac@Mhi;Gh=Mh@G@Mh;Qy=Mhi@Q@Mhi
    B=problem.Inc;k=problem.Kedge;L0y=Mhi@B@np.diag(k)@B.T@Mhi;omega02=float(np.linalg.eigvalsh((L0y+L0y.T)/2).max());omega0=math.sqrt(omega02)
    lo=enclosure['lo'];hi=enclosure['hi'];ds=problem.delta_star;g=math.radians(float(guard_deg))
    beta=np.array([max(abs(_hp(a,g)-math.tan(float(s))),abs(_hp(b,g)-math.tan(float(s)))) for a,b,s in zip(lo,hi,ds)])
    barrier_grad=float((running_barrier/m)*np.linalg.norm(Mhi@B,2)*np.linalg.norm(beta))
    A=geom.omega_deleted;rho0=float(d0*np.trace(A@M));rho1=float(np.trace(A@(2*d0*D-(running_energy/n)*M+d0*d0*Pg)));rho2=float(-np.trace(A@(Ac.T@C+C@Ac)))
    Dtilde=.5*geom.lam*G
    h2max=np.array([max(float(_h2(a,g)),float(_h2(b,g))) for a,b in zip(lo,hi)])
    Hbar=Mhi@B@np.diag((running_barrier/m)*h2max)@B.T@Mhi;Hnorm=float((running_energy/n)*omega02+np.linalg.norm(Hbar,2));Jell=float(Hnorm+omega02*np.linalg.norm(Cy,2))
    return dict(Pg=Pg,Cy=Cy,Acy=Acy,Gh=Gh,Qy=Qy,omega0=omega0,omega02=omega02,barrier_grad=barrier_grad,rho0=rho0,rho1=rho1,rho2=rho2,tr_Dtilde_Q=float(np.trace(Dtilde@Q)),Hnorm=Hnorm,Jell=Jell)

def terminal_cycle_energy_residual_enclosure(problem,geom,enclosure=None,energy_per_bus_bar=.34,omega_inf_bar=6.5,delta_bar_deg=84.0,running_energy=.08,running_barrier=.02,terminal_energy=.20,guard_deg=82.0,max_layer=.02):
    Ecap=float(energy_per_bus_bar)*problem.n
    enc=cycle_energy_edge_enclosure(problem,Ecap,delta_bar_deg) if enclosure is None else enclosure
    c=_constants(problem,geom,enc,running_energy,running_barrier,terminal_energy,guard_deg)
    Cyn=float(np.linalg.norm(c['Cy'],2));Acyn=float(np.linalg.norm(c['Acy'],2));Ghn=float(np.linalg.norm(c['Gh'],2));Qyn=float(np.linalg.norm(c['Qy'],2));omega0=c['omega0'];Bgrad=c['barrier_grad'];trdq=c['tr_Dtilde_Q'];Jell=c['Jell']
    masssum=float(np.sum(problem.M))
    def pieces(V,s):
        V=min(max(float(V),0.),Ecap);K=Ecap-V;gbar=omega0*math.sqrt(max(0.,2*V));Y=min(math.sqrt(max(0.,2*K)),omega_inf_bar*math.sqrt(masssum))
        rbar=(running_energy/problem.n)*gbar+Bgrad;jbar=rbar+Cyn*gbar;acbar=gbar+Acyn*Y;pH=Cyn*Y;p2=jbar+2*Qyn*Y
        Amax=Ghn*pH*pH;DHabs=Jell*Y*Y+acbar*p2+2*abs(trdq);w3=Amax+DHabs;cross=Ghn*pH*p2;bquad=Ghn*p2*p2
        neg=.5*s*s*w3+.5*s**3*cross+.125*s**4*bquad
        return neg,dict(V=V,K=K,gbar=gbar,Y=Y,jbar=jbar,w3=w3,cross=cross,bquad=bquad)
    def worst(s):
        grid=np.linspace(0,Ecap,801);vals=np.array([pieces(v,s)[0] for v in grid]);j=int(np.argmax(vals));best=float(vals[j]);vb=float(grid[j]);lo=grid[max(0,j-2)];hi=grid[min(len(grid)-1,j+2)]
        if hi>lo:
            rr=minimize_scalar(lambda v:-pieces(v,s)[0],bounds=(lo,hi),method='bounded',options={'xatol':1e-11})
            if -rr.fun>best:best=float(-rr.fun);vb=float(rr.x)
        return best,pieces(vb,s)[1]
    def rq(s):return c['rho0']-c['rho1']*s+.5*c['rho2']*s*s
    def lower(s):return rq(s)-worst(float(s))[0]
    hi=float(max_layer)
    if lower(hi)>=0:safe=hi
    else:
        lo=0.
        for _ in range(80):
            mid=.5*(lo+hi)
            if lower(mid)>=0:lo=mid
            else:hi=mid
        safe=lo
    samples=[]
    for s in [0.,.0001,.00025,.0005,.001,.002,.005,.01]:
        neg,where=worst(s);samples.append(dict(s=s,rho_quadratic=rq(s),negative_residual_bound=neg,lower=rq(s)-neg,worst_split=where))
    return dict(energy_cap=Ecap,edge_enclosure_iterations=enc['iterations'],edge_max_abs_deg=float(np.max(enc['max_abs_deg'])),rho_T=c['rho0'],rho_t_T=c['rho1'],rho_tt_T=c['rho2'],mass_constants=dict(omega0=c['omega0'],barrier_grad=Bgrad,Cy_norm=Cyn,Acy_norm=Acyn,Gh_norm=Ghn,Qy_norm=Qyn,Htheta_norm=c['Hnorm'],Jell_norm=Jell),analytic_safe_layer_s=float(safe),samples=samples)
