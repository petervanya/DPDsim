#!/usr/bin/julia
doc = """Usage:
    dpd_sim.py [--L <L> --N <N> --steps <ns> --dt <dt> --thermo <th> --gamma <g>]
               [--read <xyz> --config <config>]

Options:
    --N <N>            Number of atoms [default: 500]
    --L <L>            Box size [default: 5]
    --steps <ns>       Number of steps [default: 300]
    --dt <dt>          Timestep [default: 0.01]
    --thermo <th>      Frequency of printing to screen and file [default: 10]
    --gamma <g>        Friction [default: 0.0]
    --read <xyz>       Read the xyz file
    --config <config>  Read the DL_MESO CONFIG file

08/07/17
"""
using DocOpt
include("dpd_functions.jl")
include("dpd_io.jl")
using dpd_functions.init_pos, dpd_functions.init_vel,
    dpd_functions.integrate_verlet, dpd_functions.integrate_euler

args = docopt(doc)
#for (k, v) in args; println(k, "  ", v); end
L = parse(Float64, args["--L"])
N = parse(Int, args["--N"])
dt = parse(Float64, args["--dt"])
Ns = parse(Int, args["--steps"])
thermo = parse(Int, args["--thermo"])
gamma = parse(Float64, args["--gamma"])
kT = 1.0
rc = 1.0

rho = N / L^3
seed = 1234
srand(seed)

@printf "===== DPD simulations =====\n"
@printf "Beads: %i | Box: %.2f | rho: %.2f | gamma: %.1f\n" N L rho gamma
@printf "Steps: %i | dt: %.3f | thermo: %i\n" Ns dt thermo

A = 25.0 * ones(2, 2)

if args["--read"] != nothing
    fname = args["--read"]
    println("Reading initial xyz from $fname...")
    blist, X = read_xyz(fname)
    N = size(X, 1)
    V = init_vel(N, kT)
elseif args["--config"] != nothing
    fname = args["--config"]
    println("Reading initial positions and velocities from $fname...")
    X, V = read_config(fname)
    N = size(X, 1)
    blist = ones(Int, N)

else
    println("Random initialisation of positions and velocities.")
    blist = ones(Int, N)
    X = init_pos(N, L)
    V = init_vel(N, kT)
end

ti = time()
#T, KE, PE = integrate_euler(X, V, A, blist, L, gamma, kT, dt, rc, Ns, thermo)
T, KE, PE = integrate_verlet(X, V, A, blist, L, gamma, kT, dt, rc, Ns, thermo)

println("Simulation finished.Time: $(time() - ti) s.")
writedlm("data.out", [1:Ns+1 T KE PE])


