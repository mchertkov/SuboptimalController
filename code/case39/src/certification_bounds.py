"""Current analytic enclosures for the case39 curvature certificate.

Two complementary constructions are implemented.

1. A cycle-consistent compact-energy enclosure for the terminal HJB Taylor
   layer.  Scalar edge-energy restrictions are intersected with an iterative
   network (cut-space) quadratic enclosure.  The terminal residual is then
   bounded in mass-scaled coordinates and optimized over the potential/kinetic
   energy split.

2. Cellwise first/second/third stochastic-flow sensitivity majorants.  Each
   cell uses the *local* cohesive stiffness as its mechanical metric.  Frozen
   swing dynamics are dissipative in this metric; only the stiffness variation
   inside the cell contributes a positive logarithmic-norm defect.

The first construction is a deterministic analytic enclosure up to ordinary
floating-point/eigensolver tolerances (reported with outward safety factors).
The second is a tube-wise sensitivity majorant.  It is not, by itself, a cover
of the full 78-dimensional compact domain; that distinction is kept explicit
in the manuscript and result files.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import math
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq,minimize_scalar
from smooth_barrier import h_extended_prime,h_extended_second


def _terminal_matrices(problem,control_weight=.2,running_energy=.08,terminal_energy=.20):
    n=problem.n; r=float(control_weight); d0=terminal_energy/n
    M=np.diag(problem.M); Minv=np.diag(1/problem.M); D=np.diag(problem.damp)
    Pg=problem.Sg@(np.eye(problem.ng)/r)@problem.Sg.T
    C=(running_energy/n)*M-2*d0*D-d0*d0*Pg
    Ac=Minv@(D+d0*Pg)
    G=Minv@Pg@Minv
    Q=.5*(Ac.T@C+C@Ac)
    return d0,M,Minv,D,Pg,C,Ac,G,Q


def scalar_edge_energy_intervals(problem,Vcap,delta_bar_deg=84.0):
    """Necessary edge intervals from positivity of every potential-energy term."""
    db=math.radians(float(delta_bar_deg)); ds=problem.delta_star; k=problem.Kedge
    lo=np.empty(len(k)); hi=np.empty(len(k))
    for e,(s,ke) in enumerate(zip(ds,k)):
        def ve(x): return ke*(math.cos(s)-math.cos(x)-math.sin(s)*(x-s))
        l=-db; h=db
        if ve(l)>Vcap:
            l=brentq(lambda x:ve(x)-Vcap,-db,float(s),xtol=1e-13)
        if ve(h)>Vcap:
            h=brentq(lambda x:ve(x)-Vcap,float(s),db,xtol=1e-13)
        lo[e]=l; hi[e]=h
    return lo,hi


def cycle_energy_edge_enclosure(problem,Vcap,delta_bar_deg=84.0,max_iter=100,tol=1e-11):
    """Tighten edge intervals using cut-space consistency and the energy cap.

    If q=theta-theta* (with one gauge fixed), then along the line segment from
    theta* to theta,
        V(theta) >= .5 q^T L_min q,
    where L_min=B diag(k c_e) B^T and c_e is the minimum cosine over the current
    interval for edge e.  Hence
        |b_e^T q| <= sqrt(2 Vcap b_e^T L_min^+ b_e).
    Intersecting this with the current edge intervals gives a monotone outer
    enclosure.  Starting from the scalar energy intervals also uses the fact
    that each nonnegative edge Bregman energy cannot exceed the total Vcap.
    """
    lo,hi=scalar_edge_energy_intervals(problem,Vcap,delta_bar_deg)
    B=problem.Inc; k=problem.Kedge; ds=problem.delta_star
    hist=[]
    for it in range(max_iter):
        c=np.cos(np.maximum(np.abs(lo),np.abs(hi)))
        Lmin=B@np.diag(k*c)@B.T
        ev=np.linalg.eigvalsh((Lmin+Lmin.T)/2)
        if len(ev)<2 or ev[1]<=1e-12:
            raise RuntimeError('cycle-energy lower stiffness lost connectivity')
        Lpinv=np.linalg.pinv(Lmin,rcond=1e-12)
        rad=np.array([math.sqrt(max(0.0,2*Vcap*(B[:,e]@Lpinv@B[:,e]))) for e in range(len(k))])
        nlo=np.maximum(lo,ds-rad); nhi=np.minimum(hi,ds+rad)
        diff=max(float(np.max(np.abs(nlo-lo))),float(np.max(np.abs(nhi-hi))))
        lo,hi=nlo,nhi
        hist.append(diff)
        if diff<tol: break
    # small outward padding avoids presenting floating-point roots as exact interval arithmetic
    pad=2e-10
    db=math.radians(float(delta_bar_deg))
    lo=np.maximum(-db,lo-pad);hi=np.minimum(db,hi+pad)
    return dict(lo=lo,hi=hi,iterations=len(hist),last_change=hist[-1] if hist else 0.0,
                max_abs_deg=np.degrees(np.maximum(np.abs(lo),np.abs(hi))))


def _mass_scaled_terminal_constants(problem,geom,enclosure,control_weight=.2,
                                    running_energy=.08,running_barrier=.02,
                                    terminal_energy=.20,guard_deg=82.0):
    n=problem.n;m=problem.Inc.shape[1]
    d0,M,Minv,D,Pg,C,Ac,G,Q=_terminal_matrices(problem,control_weight,running_energy,terminal_energy)
    Mh=np.diag(np.sqrt(problem.M)); Mhi=np.diag(1/np.sqrt(problem.M))
    Cy=Mhi@C@Mhi; Acy=Mh@Ac@Mhi; Gh=Mh@G@Mh; Qy=Mhi@Q@Mhi
    B=problem.Inc;k=problem.Kedge
    L0y=Mhi@B@np.diag(k)@B.T@Mhi
    omega02=float(np.linalg.eigvalsh((L0y+L0y.T)/2).max());omega0=math.sqrt(omega02)
    lo=enclosure['lo'];hi=enclosure['hi'];ds=problem.delta_star;g=math.radians(float(guard_deg))
    beta=np.array([max(abs(float(h_extended_prime(a,g))-math.tan(float(s))),
                           abs(float(h_extended_prime(b,g))-math.tan(float(s))))
                   for a,b,s in zip(lo,hi,ds)])
    # independent signs are only used for the small barrier-gradient part;
    # the dominant network force is bounded from the *total convex energy* below.
    barrier_grad=float((running_barrier/m)*np.linalg.norm(Mhi@B,2)*np.linalg.norm(beta))
    # exact terminal curvature coefficients
    A=geom.omega_deleted
    rho0=float(d0*np.trace(A@M))
    rho1=float(np.trace(A@(2*d0*D-(running_energy/n)*M+d0*d0*Pg)))
    rho2=float(-np.trace(A@(Ac.T@C+C@Ac)))
    Dtilde=.5*geom.lam*G
    # useful matrix norms in mass coordinates
    return dict(d0=d0,M=M,Minv=Minv,D=D,Pg=Pg,C=C,Ac=Ac,G=G,Q=Q,
                Cy=Cy,Acy=Acy,Gh=Gh,Qy=Qy,omega02=omega02,omega0=omega0,
                barrier_grad=barrier_grad,rho0=rho0,rho1=rho1,rho2=rho2,
                tr_Dtilde_Q=float(np.trace(Dtilde@Q)))


def terminal_cycle_energy_residual_enclosure(problem,geom,energy_per_bus_bar=.34,
                                              omega_inf_bar=6.5,delta_bar_deg=84.0,
                                              control_weight=.2,running_energy=.08,
                                              running_barrier=.02,terminal_energy=.20,
                                              guard_deg=82.0,max_layer=.02):
    """Cycle/energy-aware lower enclosure for the second-order terminal Taylor candidate.

    The key improvement over the old independent-edge force bound is the convex
    energy inequality
        ||M^{-1/2} grad V|| <= omega_0 sqrt(2 V),
    valid because the cohesive potential is convex and its mass-scaled Hessian
    is bounded above by omega_0^2 I.  Potential and kinetic energy cannot both
    take their worst values simultaneously, so the final scalar majorant is
    maximized over V in [0,Ecap], with K=Ecap-V.
    """
    Ecap=float(energy_per_bus_bar)*problem.n
    enc=cycle_energy_edge_enclosure(problem,Ecap,delta_bar_deg)
    c=_mass_scaled_terminal_constants(problem,geom,enc,control_weight,running_energy,
                                      running_barrier,terminal_energy,guard_deg)
    Cy=c['Cy'];Acy=c['Acy'];Gh=c['Gh'];Qy=c['Qy']
    Cyn=float(np.linalg.norm(Cy,2));Acyn=float(np.linalg.norm(Acy,2));
    Ghn=float(np.linalg.norm(Gh,2));Qyn=float(np.linalg.norm(Qy,2))
    omega0=c['omega0']; Bgrad=c['barrier_grad']; trdq=c['tr_Dtilde_Q']
    # The full |J_ell| bound is retained for rigor here.  It is evaluated in
    # mass coordinates, avoiding the catastrophic 1/M_min Euclidean factor.
    B=problem.Inc;k=problem.Kedge;Mhi=np.diag(1/np.sqrt(problem.M));m=len(k);n=problem.n
    h2max=np.array([max(float(h_extended_second(a,math.radians(guard_deg))),
                            float(h_extended_second(b,math.radians(guard_deg))))
                    for a,b in zip(enc['lo'],enc['hi'])])
    Hbar=Mhi@B@np.diag((running_barrier/m)*h2max)@B.T@Mhi
    Hnorm=float((running_energy/n)*c['omega02']+np.linalg.norm(Hbar,2))
    Jell=float(Hnorm+c['omega02']*Cyn)

    def pieces(V,s):
        V=min(max(float(V),0.0),Ecap);K=Ecap-V
        # energy-consistent force bound; cycle consistency is already built into V
        gbar=omega0*math.sqrt(max(0.0,2*V))
        Y=min(math.sqrt(max(0.0,2*K)),omega_inf_bar*math.sqrt(float(np.sum(problem.M))))
        rbar=(running_energy/n)*gbar+Bgrad
        jbar=rbar+Cyn*gbar
        acbar=gbar+Acyn*Y
        pH=Cyn*Y
        p2=jbar+2*Qyn*Y
        Amax=Ghn*pH*pH
        DHabs=Jell*Y*Y+acbar*p2+2*abs(trdq)
        w3=Amax+DHabs
        cross=Ghn*pH*p2
        bquad=Ghn*p2*p2
        neg=.5*s*s*w3+.5*s**3*cross+.125*s**4*bquad
        return neg,dict(V=V,K=K,gbar=gbar,Y=Y,jbar=jbar,w3=w3,cross=cross,bquad=bquad)

    def worst_negative(s):
        grid=np.linspace(0,Ecap,1601)
        vals=np.array([pieces(v,s)[0] for v in grid])
        j=int(np.argmax(vals)); best=float(vals[j]); vb=float(grid[j])
        lo=grid[max(0,j-2)];hi=grid[min(len(grid)-1,j+2)]
        if hi>lo:
            rr=minimize_scalar(lambda v:-pieces(v,s)[0],bounds=(lo,hi),method='bounded',options={'xatol':1e-12})
            if -rr.fun>best: best=float(-rr.fun);vb=float(rr.x)
        return best,pieces(vb,s)[1]

    def rho_quad(s): return c['rho0']-c['rho1']*s+.5*c['rho2']*s*s
    def lower(s):
        neg,_=worst_negative(float(s));return rho_quad(float(s))-neg
    hi=float(max_layer)
    if lower(hi)>=0: safe=hi
    else:
        lo=0.0
        for _ in range(100):
            mid=.5*(lo+hi)
            if lower(mid)>=0:lo=mid
            else:hi=mid
        safe=lo
    sample_s=np.array([0,.00025,.0005,.001,.0015,.002,.003,.005,.01])
    rows=[]
    for s in sample_s:
        neg,where=worst_negative(float(s));rows.append(dict(s=float(s),rho_quadratic=float(rho_quad(s)),negative_residual_bound=float(neg),lower=float(rho_quad(s)-neg),worst_split=where))
    return dict(energy_cap=float(Ecap),edge_lo_deg=np.degrees(enc['lo']).tolist(),edge_hi_deg=np.degrees(enc['hi']).tolist(),
                edge_enclosure_iterations=enc['iterations'],rho_T=c['rho0'],rho_t_T=c['rho1'],rho_tt_T=c['rho2'],
                mass_constants=dict(omega0=c['omega0'],barrier_grad=Bgrad,Cy_norm=Cyn,Acy_norm=Acyn,Gh_norm=Ghn,Qy_norm=Qyn,Htheta_norm=Hnorm,Jell_norm=Jell),
                analytic_safe_layer_s=float(safe),samples=rows)


@dataclass
class CellConstants:
    t: float
    dmax_deg: float
    radius_deg: float
    mu: float
    c2: float
    c3: float
    Pq: np.ndarray


def local_cell_constants(problem,theta_center,radius_deg=.5,delta_bar_deg=84.0,t=0.0):
    """Local mechanical-metric constants around one cohesive angle center."""
    B=problem.Inc;k=problem.Kedge;E=problem.E;Mhi=np.diag(1/np.sqrt(problem.M))
    dc=B.T@np.asarray(theta_center);db=math.radians(float(delta_bar_deg));r=math.radians(float(radius_deg))
    if np.max(np.abs(dc))+r>=math.pi/2:
        r=max(0.0,math.pi/2-1e-6-np.max(np.abs(dc)))
    Lc=B@np.diag(k*np.cos(dc))@B.T;Pq=E.T@Lc@E
    w,V=eigh((Pq+Pq.T)/2)
    if w.min()<=1e-12: raise RuntimeError('local stiffness metric not positive definite')
    Pinv=(V*(1/w))@V.T; Pinvsqrt=(V*(1/np.sqrt(w)))@V.T
    # logarithmic norm: frozen Hamiltonian part has zero symmetric part in this metric.
    # Only the allowed stiffness perturbation contributes positively.
    Csum=0.0;c2=0.0;c3=0.0
    for e in range(len(k)):
        b=B[:,e];br=E.T@b;u=Mhi@b
        re=float(br@Pinv@br); sre=math.sqrt(max(re,0.0))
        lo=max(-db,float(dc[e]-r));hi=min(db,float(dc[e]+r))
        dcos=max(abs(math.cos(lo)-math.cos(float(dc[e]))),abs(math.cos(hi)-math.cos(float(dc[e]))))
        Csum += k[e]*dcos*np.linalg.norm(u)*sre
        # sin maximum and cos maximum on the interval
        cand=[abs(math.sin(lo)),abs(math.sin(hi))]
        sinmax=max(cand)
        cosmax=max(abs(math.cos(lo)),abs(math.cos(hi)))
        c2 += k[e]*sinmax*re*np.linalg.norm(u)
        c3 += k[e]*cosmax*(re**1.5)*np.linalg.norm(u)
    mu=.5*Csum
    return CellConstants(float(t),float(np.degrees(np.max(np.abs(dc)))),float(np.degrees(r)),float(mu),float(c2),float(c3),Pq)


def metric_transition(Pold,Pnew):
    """Worst ratio ||q||_{Pnew}/||q||_{Pold}; omega metric is unchanged."""
    vals=eigh((Pnew+Pnew.T)/2,(Pold+Pold.T)/2,eigvals_only=True)
    return float(max(1.0,math.sqrt(max(float(vals.max()),0.0))))


def propagate_cellwise_variations(cells,dt_cell,substeps=20):
    """Propagate scalar first/second/third directional-flow majorants through cells."""
    J=1.0;H=0.0;K=0.0;rows=[]
    for i,c in enumerate(cells):
        h=float(dt_cell)/substeps
        for _ in range(substeps):
            # positive majorant ODEs
            dJ=c.mu*J
            dH=c.mu*H+c.c2*J*J
            dK=c.mu*K+3*c.c2*H*J+c.c3*J**3
            J+=h*dJ;H+=h*dH;K+=h*dK
        kappa=1.0
        if i+1<len(cells):
            kappa=metric_transition(c.Pq,cells[i+1].Pq)
            J*=kappa;H*=kappa;K*=kappa
        rows.append(dict(t=float(c.t+dt_cell),mu=c.mu,c2=c.c2,c3=c.c3,kappa_to_next=kappa,J=float(J),H=float(H),K=float(K)))
    return rows
