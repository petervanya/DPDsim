#!/usr/bin/env python3
"""Testing pipeline for all simulation versions.

Created: 2023-08-16
"""


def test_dpd_sim_fortran():
    from dpdsim import DPDSim

    print("Testing DPD with Fortran...")

    sim = DPDSim(implementation="fortran", steps=10, thermo=100)
    sim.create_particle_inputs(kind="pure")
    sim.run()

    sim.compute_pe()
    sim.compute_ke()
    sim.compute_force()


def test_dpd_sim_numba():
    from dpdsim import DPDSim

    print("Testing DPD with Numba...")

    sim = DPDSim(implementation="numba", steps=10, thermo=100)
    sim.create_particle_inputs(kind="pure")
    sim.run()

    sim.compute_pe()
    sim.compute_ke()
    sim.compute_force()


def test_mdpd_sim_fortran():
    from dpdsim import MDPDSim

    print("Testing MDPDsim with Fortran...")

    sim = MDPDSim(implementation="fortran", steps=10, thermo=100)
    sim.create_particle_inputs(kind="pure")
    sim.run()

    sim.compute_pe()
    sim.compute_ke()
    sim.compute_local_density()
    sim.compute_force()


def test_gmdpd_sim_fortran():
    from dpdsim import GMDPDSim

    print("Testing GMDPDsim with Fortran...")

    sim = GMDPDSim(implementation="fortran", steps=10, thermo=100)
    sim.create_particle_inputs(kind="pure")
    sim.run()

    sim.compute_pe()
    sim.compute_ke()
    sim.compute_local_density()
    sim.compute_force()


# def test_processing():
#     from dpdsim.utils import compute_profile, compute_rdf
#     pass


if __name__ == "__main__":
    # test_dpd_sim_numba()
    test_dpd_sim_fortran()
    test_mdpd_sim_fortran()
    test_gmdpd_sim_fortran()
