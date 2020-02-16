#!/usr/bin/env python
import numpy as np
from numpy import sqrt
from numba import jit, float64, int64
import os
import sys


class DPDSim():
    """Class to perform a dissipative particle dynamics simulation"""
    def __init__(self,
                 N=375,
                 L=5,
                 dt=0.05,
                 steps=100,
                 implementation="numba",
                 equilibration_steps=0,
                 thermo=10,
                 dump_freq=100,
                 gamma=4.5,
                 seed=1234,
                 integration_style="verlet",
                 dump_vel=False,
                 dump_for=False):
        self.seed = seed
        np.random.seed(seed)

        self.N = N
        self.box = parse_box(L)
        self.volume = np.prod(np.diag(self.box))
        self.inv_box = np.diag(1.0 / np.diag(self.box))
        self.rho = self.N / self.volume # number density
        
        self.gama = gamma
        self.kT = 1.0
        
        self.dt = dt
        self.Nt = steps
        self.Neq = equilibration_steps
        self.thermo = thermo
        self.style = integration_style.lower()
        styles = ["euler", "verlet"]
        if self.style not in styles:
            assert self.style in styles, "Choose style from %s." % styles
        self.imp = implementation
        imps = ["numba", "fortran"]
        if self.imp not in imps:
            assert self.imp in imps, "Choose implementation from %s." % imps
        
        # dumping frames options
        self.df = dump_freq
        self.dumpdir = "Dump"
        if not os.path.exists(self.dumpdir):
            os.makedirs(self.dumpdir)
        self.dump_vel = dump_vel
        if self.dump_vel and not os.path.exists("Dump_Vel"):
            os.makedirs("Dump_Vel")
        self.dump_for = dump_for
        if self.dump_for and not os.path.exists("Dump_For"):
            os.makedirs("Dump_For")
 
        # empty particle inputs
        self.X = None
        self.V = None
        self.bl = None
        self.ip = None
 
        # intermediates
        self.F = None
        self.vir = None
        self.sigma = None
 
        # measured quantities
        self.T = np.zeros(self.Nt+1)
        self.KE = np.zeros(self.Nt+1)
        self.PE = np.zeros(self.Nt+1)
        self.P = np.zeros(self.Nt+1)
        self.Pxx = np.zeros(self.Nt+1)
        self.Pyy = np.zeros(self.Nt+1)
        self.Pzz = np.zeros(self.Nt+1)
        

    def create_particle_inputs(self, kind="pure", a=25.0, f=0.5, da=10.0):
        """Randomly generate coordinates and velocities"""
        self.X = np.random.rand(self.N, 3) * np.diag(self.box)
        self.V = np.random.randn(self.N, 3) * self.kT
        self.V = self.V - np.sum(self.V, 0) / self.N
        
        a = 25.0
        kinds = ["pure", "binary"]
        assert kind in kinds, "Choose kind from %s" % kinds
        
        if kind == "pure":
            self.bl = np.ones(self.N).astype(int)
            self.ip = np.zeros((2, 2))
            self.ip[1, 1] = a
        
        elif kind == "binary":
            N1 = int(f * self.N)
            N2 = N - N1
            self.bl = np.r_[np.ones(self.N1), np.ones(self.N2)].astype(int)
            self.ip = np.zeros((3, 3))
            self.ip[1:, 1:] = np.array([[a, a+da], [a+da, a]])
            

    def read_particle_inputs(X, V, bl, ip):
        """Read pre-created particle inputs"""
        self.X = X
        self.V = V
        self.bl = bl
        self.ip = ip
        assert self.X.shape == (self.N, 3), \
            "Length of X not same as number of particles."
        assert self.V.shape == (self.N, 3), \
            "Length of V not same as number of particles."
        assert len(set(self.bl)) == len(ip), \
            "Number of interaction parameters not same as number of species."


    # =====
    # Physics
    # =====
    @jit#(nopython=True)
    def compute_ke(self):
#        return (V * V).sum() / 2.0
        KE = 0.0
        for i in range(len(self.V)):
            for j in range(3):
                KE += self.V[i, j]**2 / 2.0
        return KE


    @jit #(nopython=True)
    def compute_pe(self):
        rij = np.zeros(3)
        g = np.zeros(3)
        a = 0.0
        PE = 0.0

        for i in range(self.N):
            for j in range(3):
                rij = self.X[i] - self.X[j]
                g = matvecmul(self.inv_box, rij)
                g = g - round_numba(g)
                rij = matvecmul(self.box, g)
                a = self.ip[self.bl[i], self.bl[j]]
                PE += a * wr(norm_numba(rij))**2 / 2.0
        return PE


    @jit #(nopython=True)
    def compute_temperature(self):
        return self.compute_ke() / ((3 * self.N - 3) / 2.0)


    @jit #(nopython=True)
    def compute_force(self):
        """Compute force and stress tensor"""
        Fm = np.zeros((self.N, self.N, 3))
        g = np.zeros(3)
        rij = np.zeros(3)
        vij = np.zeros(3)
        a = 0.0
        nr = 0.0
        fpair = 0.0

        self.vir = 0.0
        sigma_kin = np.zeros((3, 3))
        sigma_pot = np.zeros((3, 3))

        for i in range(self.N):
            for j in range(i):
                rij = self.X[i] - self.X[j]
                g = matvecmul(self.inv_box, rij)
                g = g - np.round_(g, 0, np.empty_like(g))
                rij = matvecmul(self.box, g)
                vij = self.V[i] - self.V[j]
             
                a = self.ip[self.bl[i], self.bl[j]]
                nr = norm_numba(rij)
             
                fc = a * wr(nr)
                fpair = fc \
                    - self.gama * wr(nr)**2 * dot_numba(rij, vij) / nr \
                    + sqrt(2.0 * self.gama * self.kT) * wr(nr) * \
                        np.random.randn() / sqrt(self.dt)
                Fm[i, j, :] = fpair / nr * rij
                Fm[j, i, :] = -fpair / nr * rij

                sigma_pot += np.outer(Fm[i, j, :], rij)
                self.vir += Fm[i, j, :] @ rij
    
        for i in range(self.N):
            sigma_kin -= np.outer(self.V[i], self.V[i])

        self.sigma = (sigma_kin + sigma_pot) / self.volume
        self.F = np.sum(Fm, 1)


#    def compute_stress_tensor(self):
#        pass
#
#
#    def compute_pressure(self):
#        pass

    
    # =====
    # Integration
    # =====
    def _euler_step(self):
        self.compute_force()
        self.V += self.F * self.dt
        self.X += self.V * self.dt


    def _verlet_step(self):
        V2 = self.V + 0.5 * self.F * self.dt
        self.X += V2 * self.dt
        self.compute_force()
        self.V = V2 + 0.5 * self.F * self.dt

    
    def save_frames(self, it):
        """In the future, modify so that no external function is called"""
        save_xyzfile("Dump/dump_%05i.xyz" % it, self.bl, self.X)
        if self.dump_vel:
            save_xyzfile("Dump/dump_%05i.vel" % it, self.bl, self.V)
        if self.dump_for:
            save_xyzfile("Dump/dump_%05i.for" % it, self.bl, self.F)
   

    def _integrate_numba(self):
        """Integrate with Numba code"""
        self.compute_force()
        self.save_frames(0)

        print("step temp ke pe p pxx pyy pzz")
        for it in range(1, self.Nt+1):
            if self.style == "euler":
                self._euler_step()
            elif self.style == "verlet":
                self._verlet_step()
 
            # enforce PBC
            self.X = self.X % np.diag(self.box)

            ke = self.compute_ke()
            pe = self.compute_pe()
            temp = ke / ((3*self.N - 3) / 2.0)
            pxx, pyy, pzz = np.diag(self.sigma)
            p = (pxx + pyy + pzz) / 3.0
#            p = self.rho * self.kT + self.vir / (3 * self.volume)

            self.KE[it], self.PE[it], self.T[it], self.P[it], \
                self.Pxx[it], self.Pyy[it], self.Pzz[it] = \
                ke, pe, temp, p, pxx, pyy, pzz
 
            if it % self.thermo == 0:
                print("%3.i %.5f %.3e %.3e %.3e %.3e %.3e %.3e" % \
                    (it, temp, ke, pe, p, pxx, pyy, pzz))

            if it >= self.Neq and it % self.df == 0:
                self.save_frames(it)


    def _integrate_fortran(self):
        from Fdpd.dpd_f import dpd_f

        self.F = dpd_f.force_mat(self.X, self.V, self.bl, self.ip, self.box, \
            self.gama, self.kT, self.dt)

        # save first frame
        self.save_frames(0)
        
        print("step temp ke pe")
        for it in range(1, self.Nt+1):
            if self.style == "euler":
                _ = dpd_f.euler_step(self.X, self.V, self.bl, self.ip, \
                    self.box, self.gama, self.kT, self.dt)
            elif self.style == "verlet":
                _ = dpd_f.verlet_step(self.X, self.V, self.F, self.bl, \
                    self.ip, self.box, self.gama, self.kT, self.dt)

            # enforce PBC
            self.X = self.X % np.diag(self.box)
 
            self.KE[it] = self.compute_ke()
            self.PE[it] = self.compute_pe()
            self.T[it] = self.KE[it] / ((3 * self.N - 3) / 2.0)
 
            if it % self.thermo == 0:
                print("%3.i %.5f %.3e %.3e" % (it, self.T[it], self.KE[it], self.PE[it]))

            if it >= self.Neq and it % self.df == 0:
                self.save_frames(it)


    def run(self):
        self._verify_integrity()

        if self.imp == "numba":
            print("Integrating with Numba...")
            self._integrate_numba()

        elif self.imp == "fortran":
            self.X = np.asfortranarray(self.X)
            self.V = np.asfortranarray(self.V)
            print("Integrating with Fortran...")
            self._integrate_fortran()


    def _verify_integrity(self):
        """Check if position and velocity frames are not empty"""
        assert self.X is not None, "Coordinates not defined."
        assert self.V is not None, "Velocities not defined."


# =====
# Helper functions
# =====
def parse_box(L):
    """Input: a float or a vector of length 3"""
    if type(L) == list and len(L) == 3:
        box = np.diag(L)
    elif type(L) == float or type(L) == int:
        box = np.eye(3) * float(L)
    else:
        sys.exit("L should be a float or a vector of length 3.")
    return box


def save_xyzfile(fname, names, mat):
    """Save a coordinate matrix in a xyz format"""
    M = len(mat)
    s = "%i\nbla\n" % M
    for i in range(M):
        s += "%i\t%.10f\t%.10f\t%.10f\n" % \
                (names[i], mat[i, 0], mat[i, 1], mat[i, 2])
    open(fname, "w").write(s)


def read_xyzfile(fname):
    try:
        A = open(fname, "r").readlines()[2:]
    except FileNotFoundError:
        sys.exit("File %s not found." % fname)
    A = [line.split() for line in A]
    A = np.array(A).astype(float)
    if A.shape[1] != 4:
        sys.exit("Incorrect number of columns in %s." % fname)
    nm, xyz = A[:, 0], A[:, 1:]
    return nm, xyz


@jit(float64(float64[:]), nopython=True)
def norm_numba(r):
    rn = 0.0
    for ri in r:
        rn += ri * ri
    return sqrt(rn)


@jit(float64[:](float64[:, :], float64[:]), nopython=True)
def matvecmul(A, b):
    N, M = A.shape
    c = np.zeros(N)
    for i in range(N):
        for j in range(M):
            c[i] += A[i, j] * b[j]
    return c


@jit(float64[:](float64[:]))
def round_numba(g):
    """Does not work with nopython"""
    N = len(g)
    gr = np.zeros(N)
    for i in range(N):
        gr[i] = round(g[i])
    return gr


@jit(float64(float64[:], float64[:]), nopython=True)
def dot_numba(r, v):
    rv = 0.0
    for i in range(len(r)):
        rv += r[i] * v[i]
    return rv

@jit(float64[:, :](float64[:, :, :]), nopython=True)
def sum_numba(A):
    """Sum by 1st index"""
    N = A.shape
    B = np.zeros((N[0], N[2]))
    for i in range(N[0]):
        for j in range(N[2]):
            for k in range(N[1]):
                B[i, j] += A[i, k, j]
    return B


@jit(float64(float64), nopython=True)
def wr(nr):
    """Weight function, r>0 is mutual distance"""
    return (1 - nr) if nr < 1.0 else 0.0




