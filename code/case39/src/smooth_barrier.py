"""Smooth cohesion barrier used by the current PIC formulation.

The physical barrier h(delta)=-log cos(delta) is used up to a guard angle.
Beyond the guard its even C^4 Taylor continuation is used.  The continuation
removes the artificial absorbing-boundary singularity from the performance
PDE; first-exit probability is treated separately as a safety quantity.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def _h_derivatives_at(g: float):
    """Return h,h',h'',h''',h'''' for h=-log cos at positive g."""
    c=math.cos(g); t=math.tan(g); sec2=1.0/(c*c)
    h=-math.log(c)
    h1=t
    h2=sec2
    h3=2.0*sec2*t
    h4=4.0*sec2*t*t+2.0*sec2*sec2
    return h,h1,h2,h3,h4


def h_extended(x, guard=math.radians(82.0)):
    """Even C^4 extension of -log cos outside |x|<=guard."""
    a=np.abs(np.asarray(x,float)); sg=np.sign(np.asarray(x,float))
    h0,h1,h2,h3,h4=_h_derivatives_at(float(guard))
    inside=a<=guard
    out=np.empty_like(a,dtype=float)
    out[inside]=-np.log(np.cos(a[inside]))
    z=a[~inside]-guard
    out[~inside]=h0+h1*z+.5*h2*z*z+(h3/6.0)*z**3+(h4/24.0)*z**4
    return out


def h_extended_prime(x, guard=math.radians(82.0)):
    a=np.abs(np.asarray(x,float)); sg=np.sign(np.asarray(x,float))
    _,h1,h2,h3,h4=_h_derivatives_at(float(guard))
    inside=a<=guard
    out=np.empty_like(a,dtype=float)
    out[inside]=np.tan(a[inside])
    z=a[~inside]-guard
    out[~inside]=h1+h2*z+.5*h3*z*z+(h4/6.0)*z**3
    return sg*out


def h_extended_second(x, guard=math.radians(82.0)):
    a=np.abs(np.asarray(x,float))
    _,_,h2,h3,h4=_h_derivatives_at(float(guard))
    inside=a<=guard
    out=np.empty_like(a,dtype=float)
    out[inside]=1.0/np.cos(a[inside])**2
    z=a[~inside]-guard
    out[~inside]=h2+h3*z+.5*h4*z*z
    return out


def h_extended_third(x, guard=math.radians(82.0)):
    a=np.abs(np.asarray(x,float)); sg=np.sign(np.asarray(x,float))
    _,_,_,h3,h4=_h_derivatives_at(float(guard))
    inside=a<=guard
    out=np.empty_like(a,dtype=float)
    sec2=1.0/np.cos(a[inside])**2
    out[inside]=2.0*sec2*np.tan(a[inside])
    z=a[~inside]-guard
    out[~inside]=h3+h4*z
    return sg*out


def barrier_bregman(problem, theta, guard_deg=82.0):
    d=np.asarray(theta)@problem.Inc
    ds=problem.delta_star
    g=math.radians(float(guard_deg))
    return np.mean(h_extended(d,g)-h_extended(ds,g)-h_extended_prime(ds,g)*(d-ds),axis=-1)


@dataclass
class SmoothPICCost:
    running_energy: float = 0.08
    running_barrier: float = 0.02
    terminal_energy: float = 0.20
    guard_deg: float = 82.0

    def running(self,problem,theta,omega):
        e=np.maximum(problem.energy(theta,omega),0.0)/problem.n
        b=barrier_bregman(problem,theta,self.guard_deg)
        return self.running_energy*e+self.running_barrier*b

    def terminal(self,problem,theta,omega):
        return self.terminal_energy*np.maximum(problem.energy(theta,omega),0.0)/problem.n
