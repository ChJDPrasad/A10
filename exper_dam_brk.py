import numpy as np

from pysph.solver.application import Application
from pysph.base.utils import get_particle_array_wcsph
from pysph.sph.wc.basic import TaitEOS,MomentumEquation
from pysph.sph.basic_equations import XSPHCorrection,ContinuityEquation
from pysph.sph.wc.transport_velocity import MomentumEquationViscosity
from pysph.sph.equation import Group
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.base.kernels import CubicSpline


class EllipticalDrop(Application):


    def create_equations(self):
        self.dx = 0.03
        self.hdx = 1.3
        self.rho0 = 1000.
        self.rho0_solid = 1000.
        self.c0 = 60.
        self.c0_solid = 60.
        self.gamma = 7.
        self.gamma_solid = 7.
        self.alpha = 0.1
        self.beta = 0.
        self.gx, self.gy, self.gz = 0., -9.81, 0.
        self.nu = 0.
        self.dim = 2
        self.tensile_correction = False
        self.hg_correction = False
        self.update_h = True
        equations = self.get_equations()
        return equations


    def create_particles(self):
        x_f,y_f = np.mgrid[0:5:self.dx,0:5:self.dx]
        x_s,y_s = np.mgrid[0-self.dx:10+self.dx:self.dx/10.,0-self.dx:10+self.dx:self.dx/10.]
        x_s1, y_s1 = np.mgrid[6:8:self.dx, 1:3:self.dx]
        c = np.arange(0-self.dx,10+self.dx,self.dx)
        x_s = np.concatenate((np.concatenate(((0-self.dx)*np.ones_like(c),c)),(10+self.dx)*np.ones_like(c)))
        y_s = np.concatenate((np.concatenate((c[::-1],(0-self.dx)*np.ones_like(c))),c))
        np.arange(0-self.dx,10+self.dx,self.dx)

        pa_fluid = get_particle_array_wcsph(name = 'fluid', x = x_f, y = y_f, m = self.rho0*self.dx*self.dx, rho = np.ones_like(x_f)*self.rho0 , h = self.dx*self.hdx, u = 0*x_f, v = 0*y_f)
        pa_solid = get_particle_array_wcsph(name = 'solid', x = x_s, y = y_s, m = self.rho0_solid*self.dx*self.dx, rho = np.ones_like(x_s)*self.rho0_solid , h = self.dx*self.hdx, u = 0*x_s, v = 0*y_s)
        pa_cube = get_particle_array_wcsph(name = 'cube', x = x_s1, y = y_s1, m = self.rho0_solid*self.dx*self.dx, rho = np.ones_like(x_s1)*self.rho0_solid , h = self.dx*self.hdx, u = 0*x_s1, v = 0*y_s1)

        return [pa_fluid ,pa_solid, pa_cube]
    
    # def create_equations(self):
    #     equations = [Group(equations = [TaitEOS(dest = 'fluid',sources= None , rho0 = 1.0,c0 = 1400,gamma = 7.0),TaitEOS(dest = 'solid',sources= None , rho0 = 1.0,c0 = 1400,gamma = 7.0)],real = False),
    #                     Group(equations = [ContinuityEquation(dest = 'fluid',sources = ['fluid','solid']),
    #                                         ContinuityEquation(dest = 'solid',sources = ['fluid']),
    #                                         MomentumEquationViscosity(dest = 'fluid',sources = ['fluid','solid'],alpha = 1.0,beta = 0.0,gy = -9.81,c0 = 1400),
    #                                         XSPHCorrection(dest = 'fluid',sources = ['fluid'])]),]

    #     return equations
    

  

    def get_equations(self):
        from pysph.sph.equation import Group
        from pysph.sph.wc.basic import MomentumEquation, TaitEOS, TaitEOSHGCorrection,UpdateSmoothingLengthFerrari
        from pysph.sph.wc.basic import ContinuityEquationDeltaSPH, MomentumEquationDeltaSPH
        from pysph.sph.basic_equations import ContinuityEquation, SummationDensity, XSPHCorrection
        from pysph.sph.wc.viscosity import LaminarViscosity

        self.fluids = ['fluid']
        self.solids = ['solid', 'cube']
        equations = []
        g1 = []
        all = self.fluids + self.solids

        # if self.summation_density:
        #     g0 = []
        #     for name in self.fluids:
        #         g0.append(SummationDensity(dest=name, sources=all))
        #     equations.append(Group(equations=g0, real=False))

        for name in self.fluids:
            g1.append(TaitEOS(
                dest=name, sources=None, rho0=self.rho0, c0=self.c0,
                gamma=self.gamma
            ))

        for name in self.solids:
            g1.append(TaitEOS(
                dest=name, sources=None, rho0=self.rho0_solid, c0=self.c0_solid,
                gamma=self.gamma_solid
            ))


        if self.hg_correction:
            # This correction applies only to solids.
            for name in self.solids:
                g1.append(TaitEOSHGCorrection(
                    dest=name, sources=None, rho0=self.rho0_solid, c0=self.c0_solid,
                    gamma=self.gamma_solid
                ))

        equations.append(Group(equations=g1, real=False))

        g2 = []
        for name in self.solids:
            g2.append(ContinuityEquation(dest=name, sources=self.fluids))

        for name in self.fluids:
            # if self.delta_sph:
            #     other = all[:]
            #     other.remove(name)
            #     g2.append(
            #         ContinuityEquationDeltaSPH(
            #             dest=name, sources=[name], c0=self.c0,
            #             delta=self.delta
            #         )
            #     )
            #     if len(other) > 0:
            #         g2.append(ContinuityEquation(dest=name, sources=other))
            #     g2.append(
            #         MomentumEquationDeltaSPH(
            #             dest=name, sources=[name], rho0=self.rho0, c0=self.c0,
            #             alpha=self.alpha, gx=self.gx, gy=self.gy, gz=self.gz,
            #         )
            #     )
            #     if len(other) > 0:
            #         g2.append(
            #             MomentumEquation(
            #                 dest=name, sources=other, c0=self.c0,
            #                 alpha=self.alpha, beta=self.beta,
            #                 gx=self.gx, gy=self.gy, gz=self.gz,
            #                 tensile_correction=self.tensile_correction
            #             )
            #         )

            #     g2.append(XSPHCorrection(dest=name, sources=[name]))
            # else:
            # if not self.summation_density:
            g2.append(ContinuityEquation(dest=name, sources=all))
            
            g2.extend([
                MomentumEquation(
                    dest=name, sources=all, alpha=self.alpha,
                    beta=self.beta, gx=self.gx, gy=self.gy, gz=self.gz,
                    c0=self.c0, tensile_correction=self.tensile_correction
                ),
                XSPHCorrection(dest=name, sources=[name])
            ])


            if abs(self.nu) > 1e-14:
                eq = LaminarViscosity(
                    dest=name, sources=self.fluids, nu=self.nu
                )
                g2.insert(-1, eq)
        equations.append(Group(equations=g2))

        if self.update_h:
            g3 = [
                UpdateSmoothingLengthFerrari(
                    dest=x, sources=None, dim=self.dim, hdx=self.hdx
                ) for x in self.fluids
            ]
            equations.append(Group(equations=g3, real=False))

        return equations


    def create_solver(self):
        kernel = CubicSpline(dim = 2)
        
        integrator = PECIntegrator(fluid = WCSPHStep(),solid = WCSPHStep())
        
        dt = .1
        tf = 5.0
        solver = Solver(kernel = kernel,dim = 2,integrator = integrator,dt = dt,tf = tf, adaptive_timestep=True)
        
        return solver

    
if __name__ == "__main__":
    app = EllipticalDrop()
    app.run()