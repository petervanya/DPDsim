#!/usr/bin/env python
import numpy as np
from numpy import sqrt, pi
from numba import jit, float64, int64
import os
import sys
import time
from .Fdpd.mdpd_f import mdpd_f


class MDPDSim():
    """Class to perform a dissipative particle dynamics simulation"""
    def __init__(self,
                 N=375,
                 L=5,
                 dt=0.01,
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
        self.rho2 = None
        self.V = None
        self.bl = None
        self.ip_A = None
        self.ip_B = None
        self.rd = None
        self.Nbt = None  # number of bead types
 
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
        

    def create_particle_inputs(self, kind="pure", A=-40.0, B=25.0, rd=0.75, \
        f=0.5, dA=10.0):
        """Randomly generate coordinates and velocities"""
        self.X = np.random.rand(self.N, 3) * np.diag(self.box)
        self.V = np.random.randn(self.N, 3) * self.kT
        self.V = self.V - np.sum(self.V, 0) / self.N
        
        kinds = ["pure", "binary"]
        assert kind in kinds, "Choose kind from %s" % kinds
        
        if kind == "pure":
            self.bl = np.ones(self.N).astype(int)
            self.Nbt = len(set(self.bl))
            self.rho2 = np.zeros((len(self.X), self.Nbt+1))
            self.ip_A = np.zeros((2, 2))
            self.ip_A[1, 1] = A
            self.ip_B = B
            self.rd = rd
        
        elif kind == "binary":
            N1 = int(f * self.N)
            N2 = self.N - N1
            self.bl = np.r_[np.ones(N1), 2 * np.ones(N2)].astype(int)

            self.ip_A = np.zeros((3, 3))
            self.ip_A[1:, 1:] = np.array([[A, A + dA], [A + dA, A]])
            self.ip_B = B
            self.rd = rd
            

    def read_particle_inputs(self, X, bl, ip_A, ip_B, rd, V=None):
        """Read pre-created particle inputs"""
        self.X = X
        if V is None:
            self.V = np.random.randn(self.N, 3) * sqrt(self.kT)
            self.V -= np.sum(self.V, 0) / self.N
        else:
            self.V = V

        self.bl = bl
        self.Nbt = len(set(self.bl))
        self.rho2 = np.zeros((len(self.X), self.Nbt+1))
        self.ip_A = ip_A
        self.ip_B = ip_B
        self.rd = rd

        assert self.X.shape == (self.N, 3), \
            "Length of X not same as number of particles."
        assert self.V.shape == (self.N, 3), \
            "Length of V not same as number of particles."
        assert len(set(self.bl)) == len(ip_A) - 1, \
            "Number of interaction parameters not same as number of species."


    # =====
    # Physics
    # =====
    @jit #(nopython=True)
    def compute_ke(self):
        self._verify_integrity()
        KE = 0.0
        for i in range(len(self.V)):
            for j in range(3):
                KE += self.V[i, j]**2 / 2.0
        return KE


    def compute_pe(self):
        self._verify_integrity()

        if self.imp == "numba":
            return pe_numba(self.X, self.rho2, self.bl, \
                self.ip_A, self.ip_B, self.rd, self.box)
        elif self.imp == "fortran":
            return mdpd_f.compute_pe(self.X, self.rho2, self.bl, \
                self.ip_A[1:, 1:], self.ip_B, self.rd, self.box)


    def compute_temperature(self):
        return self.compute_ke() / ((3 * self.N - 3) / 2.0)


    def compute_local_density(self):
        self._verify_integrity()
        if self.imp == "numba":
            self.rho2 = local_density_numba(\
                self.X, self.bl, self.Nbt, self.rd, self.box)

        elif self.imp == "fortran":
            if self.rho2 is None:
                self.rho2 = np.zeros((N, len(set(self.bl))+1), order="F")
            self.rho2 = mdpd_f.local_density(self.X, self.bl, self.Nbt, \
                self.rd, self.box)


    @jit
    def compute_force(self):
        """Compute force and stress tensor"""
        self.compute_local_density()

        if self.imp == "numba":
            self.F, self.vir, self.sigma = \
                force_numba(self.X, self.V, self.rho2, self.bl, \
                    self.ip_A, self.ip_B, self.rd, \
                    self.box, self.gama, self.kT, self.dt)

        elif self.imp == "fortran":
            if self.F is None:
                self.F = np.zeros_like(self.X, order="F")
            if self.vir is None:
                self.vir = 0.0
            if self.sigma is None:
                self.sigma = np.zeros(3, order="F")

            mdpd_f.compute_force(self.F, self.vir, self.sigma, \
                self.X, self.V, self.rho2, self.bl, \
                self.ip_A[1:, 1:], self.ip_B, self.rd, \
                self.box, self.gama, self.kT, self.dt)


    def compute_force_cube(self):
        if self.imp == "numba":
            self.Fcube = \
                force_cube_numba(self.X, self.V, self.rho2, self.bl, \
                self.ip_A, self.ip_B, self.rd, \
                self.box, self.gama, self.kT, self.dt)
        elif self.imp == "fortran":
            self.Fcube = mdpd_f.compute_force_cube(\
                self.X, self.V, self.rho2, self.bl, \
                self.ip_A[1:, 1:], self.ip_B, self.rd, \
                self.box, self.gama, self.kT, self.dt)
    

#    def compute_stress_tensor(self):
#        assert self.Fcube is not None, "Need to compute force cube first."
#        pass
#
#
#    def compute_pressure(self):
#        assert self.Fcube is not None, "Need to compute force cube first."
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

    
    def run(self):
        self._verify_integrity()

        ti = time.time()
        self._integrate_numba()
        tf = time.time()

        print("Done. Simulation time: %.2f s." % (tf - ti))


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
            pxx, pyy, pzz = self.sigma
            p = (pxx + pyy + pzz) / 3.0

            self.KE[it], self.PE[it], self.T[it], self.P[it], \
                self.Pxx[it], self.Pyy[it], self.Pzz[it] = \
                ke, pe, temp, p, pxx, pyy, pzz
 
            if it % self.thermo == 0:
                print("%3.i %.5f %.3e %.3e %.3e %.3e %.3e %.3e" % \
                    (it, temp, ke, pe, p, pxx, pyy, pzz))

            if it >= self.Neq and it % self.df == 0:
                self.save_frames(it)


    def _verify_integrity(self):
        """Check if position and velocity frames are not empty"""
        assert self.X is not None, "Coordinates not defined."
        assert self.V is not None, "Velocities not defined."


    def _initialise_vars(self):
        if self.F is None:
            self.F = np.zeros_like(self.X, order="F")
        if self.vir is None:
            self.vir = 0.0
        if self.sigma is None:
            self.sigma = np.zeros(3, order="F")


    def save_frames(self, it):
        """In the future, modify so that no external function is called"""
        save_xyzfile("Dump/dump_%05i.xyz" % it, self.bl, self.X)
        if self.dump_vel:
            save_xyzfile("Dump/dump_%05i.vel" % it, self.bl, self.V)
        if self.dump_for:
            save_xyzfile("Dump/dump_%05i.for" % it, self.bl, self.F)
   

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


@jit(nopython=True)
def force_numba(X, V, rho2, bl, ip_A, ip_B, rd, box, gamma, kT, dt):
    N = len(X)
    F = np.zeros((N, 3))
    Fcube = np.zeros((N, N, 3))
    inv_box = np.zeros((3, 3))
    for i in range(3): inv_box[i, i] = 1.0 / box[i, i]
    g = np.zeros(3)
    rij = np.zeros(3)
    vij = np.zeros(3)
    A = 0.0
    nr = 0.0
    fpair = 0.0
    rhoi = 0.0
    rhoj = 0.0

    vir = 0.0
    sigma = np.zeros(3)
    volume = np.linalg.det(box)

    for i in range(N):
        for j in range(i):
            rij = X[i] - X[j]
            g = matvecmul(inv_box, rij)
            g = g - np.round_(g, 0, np.empty_like(g))
            rij = matvecmul(box, g)
            vij = V[i] - V[j]

            nr = norm_numba(rij)
            A = ip_A[bl[i], bl[j]]
            rhoi = rho2[i, bl[j]]
            rhoj = rho2[j, bl[i]]

            fpair = A * wr(nr) \
                + ip_B * (rhoi + rhoj) * wr(nr / rd) \
                - gamma * wr(nr)**2 * dot_numba(rij, vij) / nr \
                + sqrt(2.0*gamma*kT) * wr(nr) * np.random.randn() / sqrt(dt)
            Fcube[i, j, :] = fpair / nr * rij
            Fcube[j, i, :] = -Fcube[i, j, :]

            vir += Fcube[i, j, :] @ rij
            sigma += Fcube[i, j, :] * rij

    for i in range(N):
        sigma += V[i] * V[i]

    sigma = sigma / volume
    F = np.sum(Fcube, 1)
    return F, vir, sigma


@jit(nopython=True)
def force_cube_numba(X, V, rho2, bl, ip_A, ip_B, rd, box, gamma, kT, dt):
    N = len(X)
    Fcube = np.zeros((N, N, 3))
    inv_box = np.zeros((3, 3))
    for i in range(3): inv_box[i, i] = 1.0 / box[i, i]
    g = np.zeros(3)
    rij = np.zeros(3)
    vij = np.zeros(3)
    A = 0.0
    nr = 0.0
    fpair = 0.0
    rhoi = 0.0
    rhoj = 0.0

    for i in range(N):
        for j in range(i):
            rij = X[i] - X[j]
            g = matvecmul(inv_box, rij)
            g = g - np.round_(g, 0, np.empty_like(g))
            rij = matvecmul(box, g)
            vij = V[i] - V[j]

            nr = norm_numba(rij)
            A = ip_A[bl[i], bl[j]]
            rhoi = rho2[i, bl[j]]
            rhoj = rho2[j, bl[i]]

            fpair = A * wr(nr) \
                + ip_B * (rhoi + rhoj) * wr(nr / rd) \
                - gamma * wr(nr)**2 * dot_numba(rij, vij) / nr \
                + sqrt(2.0*gamma*kT) * wr(nr) * np.random.randn() / sqrt(dt)
            Fcube[i, j, :] = fpair / nr * rij
            Fcube[j, i, :] = -Fcube[i, j, :]

    return Fcube


@jit(nopython=True)
def pe_numba(X, rho2, bl, ip_A, ip_B, rd, box):
        N = len(X)
        inv_box = np.zeros((3, 3))
        for i in range(3): inv_box[i, i] = 1.0 / box[i, i]
        rij = np.zeros(3)
        g = np.zeros(3)
        nr = 0.0
        pe = 0.0

        # standard part
        for i in range(N):
            for j in range(i):
                rij = X[i] - X[j]
                g = matvecmul(inv_box, rij)
                g = g - round_numba(g)
                nr = norm_numba(matvecmul(box, g))
                pe += ip_A[bl[i], bl[j]] * wr(nr)**2 / 2.0

        # many-body part
        for i in range(N):
            pe += pi*rd**4 / 30.0 * ip_B * np.sum(rho2[i])**2
        return pe


def local_density_numba(X, bl, Nbt, rd, box):
        N = len(X)
        inv_box = np.zeros((3, 3))
        for i in range(3): inv_box[i, i] = 1.0 / box[i, i]
        rij = np.zeros(3)
        g = np.zeros(3)
        nr = 0.0
        d_rho = 0.0
        rho = np.zeros((N, Nbt+1))

        for i in range(N):
            for j in range(i):
                rij = X[i] - X[j]
                g = matvecmul(inv_box, rij)
                g = g - round_numba(g)
                nr = norm_numba(matvecmul(box, g))

                d_rho = wr(nr / rd)**2 * 15.0 / (2.0*pi*rd**3)
                rho[i, bl[j]] += d_rho
                rho[j, bl[i]] += d_rho
        return rho




