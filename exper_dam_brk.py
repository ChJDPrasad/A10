import numpy as np

from pysph.solver.application import Application
from pysph.base.utils import get_particle_array_wcsph
from pysph.sph.wc.basic import TaitEOS,MomentumEquation
from pysph.sph.basic_equations import XSPHCorrection,ContinuityEquation

from pysph.sph.equation import Group
from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.base.kernels import CubicSpline


class EllipticalDrop(Application):

    def create_particles(self):

        self.dx = 0.025
        self.hdx = 1.3
        self.rho = 1.0
        x_f,y_f = np.mgrid[0:5:self.dx,0:5:self.dx]
        x_s,y_s = np.mgrid[0-self.dx:10+self.dx:self.dx,0-self.dx:10+self.dx:self.dx]
        c = np.arange(0-self.dx,10+self.dx,self.dx)
        x_s = np.concatenate((np.concatenate(((0-self.dx)*np.ones_like(c),c)),(10+self.dx)*np.ones_like(c)))
        y_s = np.concatenate((np.concatenate((c[::-1],(0-self.dx)*np.ones_like(c))),c))
        np.arange(0-self.dx,10+self.dx,self.dx)
        # mask = x**2 + y**2 < 1.0

        pa_fuild = get_particle_array_wcsph(name = 'fluid', x = x_f, y = y_f, m = self.rho*self.dx*self.dx, rho = np.ones_like(x_f)*self.rho , h = self.dx*self.hdx, u = 0*x_f, v = 0*y_f)
        pa_solid = get_particle_array_wcsph(name = 'solid', x = x_s, y = y_s, m = self.rho*self.dx*self.dx, rho = np.ones_like(x_s)*self.rho , h = self.dx*self.hdx, u = 0*x_s, v = 0*y_s)
        
        return [pa_fuild ,pa_solid]
    
    def create_equations(self):
        equations = [Group(equations = [TaitEOS(dest = 'fluid',sources= None , rho0 = 1.0,c0 = 1400,gamma = 7.0),TaitEOS(dest = 'solid',sources= None , rho0 = 1.0,c0 = 1400,gamma = 7.0)],real = False),
                        Group(equations = [ContinuityEquation(dest = 'fluid',sources = ['fluid','solid']),
                                            ContinuityEquation(dest = 'solid',sources = ['fluid']),
                                            MomentumEquation(dest = 'fluid',sources = ['fluid','solid'],alpha = 1.0,beta = 0.0,gy = -9.81,c0 = 1400),
                                            XSPHCorrection(dest = 'fluid',sources = ['fluid'])]),]

        return equations
    
    def create_solver(self):
        kernel = CubicSpline(dim = 2)
        
        integrator = PECIntegrator(fluid = WCSPHStep(),solid = WCSPHStep())
        
        dt = .01
        tf = 1.0
        solver = Solver(kernel = kernel,dim = 2,integrator = integrator,dt = dt,tf = tf)
        
        return solver

    
if __name__ == "__main__":
    app = EllipticalDrop()
    app.run()