"""Smoke/regression tests for the current stochastic-synchronization package."""
from pathlib import Path
import sys, math, unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pic_case39 import build_case39_problem
from actuation_hierarchy import build_hierarchy, general_pic_geometry
from stochastic_swing import noise_profile
from certification_bounds import cycle_energy_edge_enclosure, terminal_cycle_energy_residual_enclosure, local_cell_constants
from smooth_barrier import h_extended, h_extended_prime, h_extended_second
from control_inflation import control_inflation_geometry

class CurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.problem,meta=build_case39_problem(alpha=4.0,damping=0.05)
        except FileNotFoundError as exc:
            raise unittest.SkipTest('MATPOWER case39.m is not bundled; place it in cyclespace_base/data for heavy tests') from exc
        cls.meta=meta
        cls.hier=build_hierarchy(cls.problem,meta['bus_ids'])

    def test_case39_event(self):
        self.assertEqual(self.problem.n,39)
        self.assertEqual(self.problem.Inc.shape[1],46)
        self.assertEqual(self.problem.ng,10)
        self.assertAlmostEqual(self.problem.dP*self.problem.baseMVA,2724.476,places=3)

    def test_cycle_energy_enclosure_contains_equilibrium(self):
        enc=cycle_energy_edge_enclosure(self.problem,.34*self.problem.n,84.0)
        ds=self.problem.delta_star
        self.assertTrue(np.all(ds>=enc['lo']-1e-9))
        self.assertTrue(np.all(ds<=enc['hi']+1e-9))
        self.assertTrue(np.all(enc['lo']>=-math.radians(84)-1e-8))
        self.assertTrue(np.all(enc['hi']<= math.radians(84)+1e-8))

    def test_positive_terminal_layer(self):
        Gamma=noise_profile(self.problem,.1,'homogeneous')
        geom=general_pic_geometry(self.problem,self.hier[-1],Gamma,.2)
        out=terminal_cycle_energy_residual_enclosure(self.problem,geom)
        self.assertGreater(out['analytic_safe_layer_s'],1e-3)
        self.assertGreater(out['rho_T'],0)

    def test_local_variational_constants(self):
        c=local_cell_constants(self.problem,self.problem.theta_star,radius_deg=.5)
        self.assertGreaterEqual(c.mu,0)
        self.assertGreater(c.c2,0)
        self.assertGreater(c.c3,0)


    def test_control_inflation_homogeneous_h3_is_load_only(self):
        Gamma=noise_profile(self.problem,.1,'homogeneous')
        geom=control_inflation_geometry(self.problem,self.hier[-1],Gamma,.2)
        self.assertAlmostEqual(geom.lam,.002,places=12)
        # H3 physical generator directions are already matched; virtual authority lives on loads.
        D=geom.injection_gain_virtual
        gen=self.problem.gen_idx
        load=np.array([i for i in range(self.problem.n) if i not in set(gen)])
        self.assertLess(np.linalg.norm(D[np.ix_(gen,gen)]),1e-10)
        self.assertAlmostEqual(np.mean(np.diag(D)[load]),5.0,places=10)
        self.assertEqual(geom.virtual_rank,len(load))

    def test_barrier_extension_c2_at_guard(self):
        g=math.radians(82.0); eps=1e-8
        for fn in (h_extended,h_extended_prime,h_extended_second):
            l=float(fn(g-eps,g)); r=float(fn(g+eps,g))
            self.assertLess(abs(l-r),1e-4)

if __name__=='__main__': unittest.main()
