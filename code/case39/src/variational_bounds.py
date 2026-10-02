"""Variational equations and conservative curvature bounds.

For the additive-noise stochastic swing equation, spatial derivatives of the
stochastic flow satisfy deterministic random-coefficient ODEs along every
Brownian path.  This module implements the first, second, and third directional
variations, mechanical-energy norm majorants on cohesive angle sectors,
compact-tube cost derivative bounds, and exact terminal HJB coefficients for
``rho = tr((D-Dtilde) Hess_omega W)``.

The uniform Feynman--Kac derivative majorants are rigorous for paths restricted
to the specified compact tube (or for a stopped/cutoff surrogate carrying the
same derivative bounds).  For the globally smooth unstopped performance cost,
they are interior/tube bounds until a stochastic tail estimate is supplied.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

from smooth_barrier import _h_derivatives_at


@dataclass
class MechanicalMajorant:
    delta_bar_deg: float
    omega_bar: float
    alpha: float
    cos_delta_bar: float
    omega0: float
    damping_rate: float
    kappa: float
    mu: float
    drift_second: float
    drift_third: float
    metric_conversion: float


@dataclass
class CostDerivativeBounds:
    energy_first: float
    energy_second: float
    energy_third: float
    barrier_first: float
    barrier_second: float
    barrier_third: float
    running_first: float
    running_second: float
    running_third: float
    terminal_first: float
    terminal_second: float
    terminal_third: float


def swing_jacobian(problem, theta):
    """DF(theta,omega); independent of omega for the swing drift."""
    B=problem.Inc; d=B.T@np.asarray(theta)
    L=B@np.diag(problem.Kedge*np.cos(d))@B.T
    Minv=np.diag(1.0/problem.M)
    n=problem.n
    return np.block([[np.zeros((n,n)),np.eye(n)],
                     [-Minv@L,-Minv@np.diag(problem.damp)]])


def swing_second_direction(problem, theta, z1, z2):
    """D^2F[z1,z2] for z=(theta,omega)."""
    B=problem.Inc; d=B.T@np.asarray(theta)
    a=B.T@np.asarray(z1[:problem.n]); b=B.T@np.asarray(z2[:problem.n])
    acc=(B@(problem.Kedge*np.sin(d)*a*b))/problem.M
    return np.r_[np.zeros(problem.n),acc]


def swing_third_direction(problem, theta, z1, z2, z3):
    """D^3F[z1,z2,z3] for z=(theta,omega)."""
    B=problem.Inc; d=B.T@np.asarray(theta)
    a=B.T@np.asarray(z1[:problem.n]); b=B.T@np.asarray(z2[:problem.n]); c=B.T@np.asarray(z3[:problem.n])
    acc=(B@(problem.Kedge*np.cos(d)*a*b*c))/problem.M
    return np.r_[np.zeros(problem.n),acc]


def directional_variation_rhs(problem, theta, J1, J2=None, H12=None,
                              J3=None, H13=None, H23=None, K123=None):
    """Right sides for directional first/second/third flow variations.

    J_i = DX a_i,
    H_ij = D^2X[a_i,a_j],
    K_123 = D^3X[a_1,a_2,a_3].
    """
    A=swing_jacobian(problem,theta)
    out={'J1':A@J1}
    if J2 is not None:
        out['J2']=A@J2
    if H12 is not None and J2 is not None:
        out['H12']=A@H12+swing_second_direction(problem,theta,J1,J2)
    if J3 is not None:
        out['J3']=A@J3
    if H13 is not None and J3 is not None:
        out['H13']=A@H13+swing_second_direction(problem,theta,J1,J3)
    if H23 is not None and J2 is not None and J3 is not None:
        out['H23']=A@H23+swing_second_direction(problem,theta,J2,J3)
    if K123 is not None and all(v is not None for v in [J2,J3,H12,H13,H23]):
        out['K123']=(A@K123
            +swing_second_direction(problem,theta,H12,J3)
            +swing_second_direction(problem,theta,H13,J2)
            +swing_second_direction(problem,theta,H23,J1)
            +swing_third_direction(problem,theta,J1,J2,J3))
    return out


def mechanical_norm(problem,z,alpha):
    """Gauge-invariant mechanical norm; uniform angle shifts have zero norm."""
    n=problem.n; th=np.asarray(z[:n]); om=np.asarray(z[n:])
    L0=problem.Inc@np.diag(problem.Kedge)@problem.Inc.T
    return float(np.sqrt(max(alpha*th@(L0@th)+np.sum(problem.M*om*om),0.0)))


def mechanical_majorant(problem,delta_bar_deg=84.0,omega_bar=6.5):
    """Uniform first/second/third drift derivative constants on a cohesive sector.

    The angle weight alpha=(1+cos(delta_bar))/2 minimizes the elementary
    stiffness-sector mismatch max(1-alpha,alpha-cos(delta_bar))/sqrt(alpha).
    """
    db=math.radians(float(delta_bar_deg)); c=math.cos(db)
    if c<=0: raise ValueError('delta_bar must be below 90 degrees')
    alpha=.5*(1.0+c)
    B=problem.Inc; k=problem.Kedge
    L0=B@np.diag(k)@B.T
    Mh=np.diag(1/np.sqrt(problem.M))
    omega0=float(np.sqrt(np.linalg.eigvalsh(Mh@L0@Mh).max()))
    eps=max(1-alpha,alpha-c)
    kappa=float(eps/math.sqrt(alpha)*omega0)
    dstar=float(np.min(problem.damp/problem.M))
    mu=float(.5*(-dstar+math.sqrt(dstar*dstar+kappa*kappa)))
    CB=float(np.linalg.norm(Mh@B,2))
    c2=float(CB/alpha)
    c3=float(CB/(alpha**1.5*math.sqrt(np.min(k))))
    metric=float(math.sqrt(alpha*db*db*np.sum(k)+omega_bar*omega_bar*np.sum(problem.M)))
    return MechanicalMajorant(delta_bar_deg=float(delta_bar_deg),omega_bar=float(omega_bar),
        alpha=alpha,cos_delta_bar=c,omega0=omega0,damping_rate=dstar,kappa=kappa,
        mu=mu,drift_second=c2,drift_third=c3,metric_conversion=metric)


def _extended_h_derivative_scalar(x,order,guard):
    h0,h1,h2,h3,h4=_h_derivatives_at(guard)
    a=abs(float(x)); sg=1.0 if x>=0 else -1.0
    if a<=guard:
        t=math.tan(a); sec2=1.0/(math.cos(a)**2)
        if order==1: return sg*t
        if order==2: return sec2
        if order==3: return sg*2*sec2*t
    z=a-guard
    if order==1: return sg*(h1+h2*z+.5*h3*z*z+(h4/6.)*z**3)
    if order==2: return h2+h3*z+.5*h4*z*z
    if order==3: return sg*(h3+h4*z)
    raise ValueError(order)


def compact_cost_derivative_bounds(problem,majorant,running_energy=.08,
                                    running_barrier=.02,terminal_energy=.20,
                                    guard_deg=82.0):
    """Derivative bounds in the mechanical norm on the compact angle/frequency box."""
    alpha=majorant.alpha; db=math.radians(majorant.delta_bar_deg); wb=majorant.omega_bar
    k=problem.Kedge; m=problem.Inc.shape[1]
    # exact elementary maxima for sin(delta)-sin(delta_star) on [-db,db]
    ds=problem.delta_star
    md=[]
    for x in ds:
        vals=[abs(math.sin(-db)-math.sin(x)),abs(math.sin(db)-math.sin(x)),abs(math.sin(x))]
        md.append(max(vals))
    md=np.asarray(md)
    E1a=math.sqrt(np.sum(k*md*md)/alpha)
    E1w=wb*math.sqrt(np.sum(problem.M))
    E1=math.sqrt(E1a*E1a+E1w*E1w)
    E2=max(1.0/alpha,1.0)
    E3=1.0/(alpha**1.5*math.sqrt(np.min(k)))

    guard=math.radians(float(guard_deg))
    # one-dimensional maxima; a 0.1% outward inflation absorbs optimizer/roundoff error
    def absmax(order):
        grid=np.linspace(-db,db,4001)
        vals=np.array([abs(_extended_h_derivative_scalar(x,order,guard)) for x in grid])
        return float(vals.max()*1.001)
    h2max=absmax(2); h3max=absmax(3)
    bcoeff=[]
    for x in ds:
        hpstar=math.tan(float(x))
        grid=np.linspace(-db,db,4001)
        bcoeff.append(max(abs(_extended_h_derivative_scalar(y,1,guard)-hpstar) for y in grid)/m)
    bcoeff=np.asarray(bcoeff)
    B1=math.sqrt(np.sum(bcoeff*bcoeff/(alpha*k)))
    B2=h2max/(m*alpha*np.min(k))
    B3=h3max/(m*alpha**1.5*np.min(k)**1.5)
    q1=running_energy*E1+running_barrier*B1
    q2=running_energy*E2+running_barrier*B2
    q3=running_energy*E3+running_barrier*B3
    p1=terminal_energy*E1; p2=terminal_energy*E2; p3=terminal_energy*E3
    return CostDerivativeBounds(E1,E2,E3,B1,B2,B3,q1,q2,q3,p1,p2,p3)


def flow_majorants(majorant,horizon):
    """Bounds j,h,k for first/second/third directional flow derivatives."""
    mu=majorant.mu; c2=majorant.drift_second; c3=majorant.drift_third; t=float(horizon)
    if abs(mu)<1e-14:
        j=1.0; h=c2*t; k=1.5*c2*c2*t*t+c3*t
    else:
        e=math.exp(mu*t)
        j=e
        h=(c2/mu)*e*(e-1)
        k=(c3/(2*mu))*e*(e*e-1)+(3*c2*c2/(2*mu*mu))*e*(e-1)**2
    return j,h,k


def action_derivative_majorants(majorant,cost,horizon):
    """C1,C2,C3 bounds for the Feynman--Kac action on a compact tube."""
    mu=majorant.mu; c2=majorant.drift_second; c3=majorant.drift_third; H=float(horizon)
    def vals(s):
        if abs(mu)<1e-14:
            j=1.; h=c2*s; kk=1.5*c2*c2*s*s+c3*s
        else:
            e=math.exp(mu*s); j=e
            h=(c2/mu)*e*(e-1)
            kk=(c3/(2*mu))*e*(e*e-1)+(3*c2*c2/(2*mu*mu))*e*(e-1)**2
        return j,h,kk
    def f1(s):
        j,_,_=vals(s); return cost.running_first*j
    def f2(s):
        j,h,_=vals(s); return cost.running_second*j*j+cost.running_first*h
    def f3(s):
        j,h,kk=vals(s); return cost.running_third*j**3+3*cost.running_second*h*j+cost.running_first*kk
    # scipy quad is enough for horizons used in executable reports before overflow.
    C1=quad(f1,0,H,epsabs=1e-10,limit=200)[0]
    C2=quad(f2,0,H,epsabs=1e-10,limit=200)[0]
    C3=quad(f3,0,H,epsabs=1e-10,limit=200)[0]
    j,h,kk=vals(H)
    C1+=cost.terminal_first*j
    C2+=cost.terminal_second*j*j+cost.terminal_first*h
    C3+=cost.terminal_third*j**3+3*cost.terminal_second*h*j+cost.terminal_first*kk
    return float(C1),float(C2),float(C3)


def value_third_bound(C1,C2,C3,lam):
    """Sup-norm tilted-measure bound for ||D^3 W||."""
    lam=float(lam)
    return float(C3+6*C1*C2/lam+8*C1**3/(lam*lam))


def spatial_lrho_bound(problem,geom,majorant,W3):
    """Convert ||D^3 W|| in mechanical norm to the Stage-4 normalized metric."""
    M=np.diag(problem.M)
    trace_hat=float(np.trace(geom.omega_deleted@M))
    L_energy=trace_hat*float(W3)
    return dict(trace_hat=trace_hat,L_energy=L_energy,
                L_normalized_metric=majorant.metric_conversion*L_energy)


def _terminal_matrices(problem,control_weight=.2,running_energy=.08,
                       running_barrier=.02,terminal_energy=.20):
    n=problem.n; r=float(control_weight); d0=terminal_energy/n
    M=np.diag(problem.M); Minv=np.diag(1/problem.M); D=np.diag(problem.damp)
    Pg=problem.Sg@(np.eye(problem.ng)/r)@problem.Sg.T
    C=(running_energy/n)*M-2*d0*D-d0*d0*Pg
    Ac=Minv@(D+d0*Pg)
    G=Minv@Pg@Minv
    Q=.5*(Ac.T@C+C@Ac)
    return d0,M,Minv,D,Pg,C,Ac,G,Q


def terminal_rho_coefficients(problem,geom,control_weight=.2,running_energy=.08,
                              running_barrier=.02,terminal_energy=.20,
                              delta_bar_deg=84.0,guard_deg=82.0):
    """Exact rho(T), rho_t(T), rho_tt(T), and an angle-box interval for rho_ttt(T).

    The rho_ttt interval relaxes cycle consistency by treating edge angles
    independently; this makes it conservative for the cohesive angle box.
    One-dimensional scalar extrema are solved numerically and the returned
    interval is inflated by 1% for safe reporting (not machine interval arithmetic).
    """
    d0,M,Minv,D,Pg,C,Ac,G,Q=_terminal_matrices(problem,control_weight,running_energy,running_barrier,terminal_energy)
    A=geom.omega_deleted
    rho0=float(d0*np.trace(A@M))
    rho1=float(np.trace(A@(2*d0*D-(running_energy/problem.n)*M+d0*d0*Pg)))
    rho2=float(-np.trace(A@(Ac.T@C+C@Ac)))
    B=problem.Inc; k=problem.Kedge; me=B.shape[1]
    const=float(np.trace(A@(2*C@G@C-2*(Ac.T@Q+Q@Ac))))
    db=math.radians(float(delta_bar_deg)); guard=math.radians(float(guard_deg))
    mins=[]; maxs=[]
    for e in range(me):
        be=B[:,e]
        ae=float(be@A@be)
        ce=float(be@A@C@Minv@be)
        coef=float(k[e]*((running_energy/problem.n)*ae-ce))
        bcoef=float((running_barrier/me)*ae)
        def fun(x):
            hpp=_extended_h_derivative_scalar(x,2,guard)
            return coef*math.cos(x)+bcoef*hpp
        cand=[fun(0.0),fun(guard),fun(db)]
        for lo,hi in [(0.0,min(guard,db)),(min(guard,db),db)]:
            if hi-lo>1e-12:
                rr=minimize_scalar(fun,bounds=(lo,hi),method='bounded',options={'xatol':1e-13});cand.append(float(rr.fun))
                rr=minimize_scalar(lambda x:-fun(x),bounds=(lo,hi),method='bounded',options={'xatol':1e-13});cand.append(float(-rr.fun))
        mins.append(min(cand)); maxs.append(max(cand))
    raw_lo=const-2*sum(maxs); raw_hi=const-2*sum(mins)
    # 1% outward inflation plus tiny absolute pad.
    pad=.01*max(abs(raw_lo),abs(raw_hi),1.0)+1e-9
    lo=raw_lo-pad; hi=raw_hi+pad
    disc=rho1*rho1-2*rho0*rho2
    smin=rho1/rho2 if rho2>0 else float('nan')
    qmin=rho0-rho1*rho1/(2*rho2) if rho2>0 else float('nan')
    return dict(rho_T=rho0,rho_t_T=rho1,rho_tt_T=rho2,
                rho_ttt_T_raw_interval=[float(raw_lo),float(raw_hi)],
                rho_ttt_T_inflated_interval=[float(lo),float(hi)],
                quadratic_discriminant=float(disc),quadratic_s_min=float(smin),
                quadratic_min=float(qmin))

def terminal_taylor_residual_bound(problem,geom,coeffs,control_weight=.2,
                                   running_energy=.08,running_barrier=.02,
                                   terminal_energy=.20,delta_bar_deg=84.0,
                                   omega_bar=6.5,guard_deg=82.0):
    """Coarse pointwise residual bound for the explicit second-order terminal HJB Taylor candidate.

    W2(x,T-s)=phi+s H(phi)+.5 s^2 W_tt(T).  Its deleted-noise curvature is
        rho2(s)=rho_T-rho_t(T)s+.5 rho_tt(T)s^2.
    The surrogate-HJB residual is exactly
        -.5 s^2 w3 -.5 s^3 <grad H0,G grad w2> -.125 s^4 <grad w2,G grad w2>.
    This routine bounds the three scalar terms on the compact angle/frequency box
    by elementary operator norms.  The resulting layer is deliberately very
    conservative but pointwise (no Monte Carlo and no stochastic-flow tail issue).
    """
    n=problem.n; B=problem.Inc; k=problem.Kedge; me=B.shape[1]
    db=math.radians(float(delta_bar_deg)); wb=float(omega_bar); guard=math.radians(float(guard_deg))
    d0,M,Minv,D,Pg,C,Ac,G,Q=_terminal_matrices(problem,control_weight,running_energy,running_barrier,terminal_energy)
    Wnorm=math.sqrt(n)*wb
    # monotonicity on [-db,db] because db<pi/2 and the Stage-4 extension remains convex.
    hppmax=max(abs(_extended_h_derivative_scalar(-db,2,guard)),abs(_extended_h_derivative_scalar(db,2,guard)))
    # edgewise elementary differences
    d_sin=[]; d_hp=[]
    for ds in problem.delta_star:
        d_sin.append(max(abs(math.sin(-db)-math.sin(float(ds))),abs(math.sin(db)-math.sin(float(ds)))))
        hpstar=math.tan(float(ds))
        d_hp.append(max(abs(_extended_h_derivative_scalar(-db,1,guard)-hpstar),
                        abs(_extended_h_derivative_scalar(db,1,guard)-hpstar)))
    d_sin=np.asarray(d_sin); d_hp=np.asarray(d_hp)
    a0max=float(np.linalg.norm(Minv@B@np.diag(np.sqrt(k)),2)*np.linalg.norm(np.sqrt(k)*d_sin))
    edge_grad=(running_energy/n)*k*d_sin+(running_barrier/me)*d_hp
    rmax=float(np.linalg.norm(B,2)*np.linalg.norm(edge_grad))
    L0=B@np.diag(k)@B.T
    Lmax=float(np.linalg.norm(L0,2))
    Hthmax=float((running_energy/n)*Lmax+(running_barrier/me)*hppmax*np.linalg.norm(B@B.T,2))
    Jlmax=float(Hthmax+np.linalg.norm(C@Minv,2)*Lmax)
    lmax=float(rmax+np.linalg.norm(C,2)*a0max)
    gradw2=float(lmax+2*np.linalg.norm(Q,2)*Wnorm)
    gom=float(a0max+np.linalg.norm(Ac,2)*Wnorm)
    gradH0w=float(np.linalg.norm(C,2)*Wnorm)
    Amax=float(np.linalg.norm(G,2)*gradH0w**2)
    Dtilde=.5*geom.lam*G
    DHmax=float(Wnorm*Jlmax*Wnorm+gom*gradw2+2*abs(np.trace(Dtilde@Q)))
    w3max=float(Amax+DHmax)
    crossmax=float(np.linalg.norm(G,2)*gradH0w*gradw2)
    gradw2G=float(np.linalg.norm(G,2)*gradw2**2)
    def rhoquad(s):
        return coeffs['rho_T']-coeffs['rho_t_T']*s+.5*coeffs['rho_tt_T']*s*s
    def lower(s):
        s=float(s)
        neg=.5*s*s*w3max+.5*s**3*crossmax+.125*s**4*gradw2G
        return rhoquad(s)-neg
    # Find the first positive root / safe interval endpoint by bisection.
    lo=0.0; hi=.1
    if lower(hi)>=0:
        safe=hi
    else:
        for _ in range(100):
            mid=.5*(lo+hi)
            if lower(mid)>=0: lo=mid
            else: hi=mid
        safe=lo
    return dict(a0_max=a0max,rtheta_max=rmax,L_max=Lmax,Htheta_max=Hthmax,
                Jell_max=Jlmax,l_max=lmax,grad_w2_max=gradw2,gomega_max=gom,
                grad_H0_omega_max=gradH0w,w3_abs_max=w3max,cross_abs_max=crossmax,
                gradw2_G_gradw2_max=gradw2G,analytic_safe_layer_s=float(safe),
                residual_lower_at_safe=float(lower(safe)),
                residual_lower_samples={str(x):float(lower(x)) for x in [1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3]})
