#!/usr/bin/julia
module dpd_functions
include("dpd_io.jl")

function wR(r, rc=1.0)
    nr = norm(r)
    wr = 0.0
    if nr < rc
        wr = 1 - nr / rc
    end
    wr
end


function tot_PE(X, A, bl, L, rc=1.0)
    N = size(X, 1)
    box = L * eye(3)
    inv_box = pinv(box)
    g = zeros(3); gn = zeros(3)
    dr = zeros(3); drn = zeros(3)
    E = 0.0

    for i in 1:N
        for j in 1:i-1
            dr = X[i, :] - X[j, :]
            g = inv_box * dr
            gn = g - round(g)
            drn = box * gn
            nr = norm(drn)
            if nr < rc
                E += A[bl[i], bl[j]] * rc / 2 * (1 - nr / rc)^2
            end
        end
    end
    E
end


function tot_KE(V)
    KE = sum(V .* V) / 2.0
end


function avg_speed(V)
    mean(sqrt(sum(V .* V, 2)))
end


function init_pos(N, L)
    X = rand(N, 3) .* L
end


function init_vel(N, kT)
    V = randn(N, 3) * kT
    V - ones(N, 3) .* mean(V, 1)
end


function F_tot(r, v, a, gamma, kT, dt, rc)
    nr = norm(r)
    FC = a * wR(r, rc) * r / nr
    FD = gamma * wR(r, rc)^2 * dot(r, v) * r / nr^2
    FR = sqrt(2 * gamma * kT) * wR(r, rc) * randn() / sqrt(dt) * r / nr
    ftot = FC + FD + FR
    ftot
end


function force_mat(X, V, A, bl, L, gamma, kT, dt, rc)
    """Force matrix. Output: (N, N, 3) matrix"""
    N = size(X, 1)
    fcube = zeros(N, N, 3)
    box = L * eye(3)
    inv_box = pinv(box)
    g = zeros(3); gn = zeros(3)
    dr = zeros(3); drn = zeros(3)
    vij = zeros(3)
    F = zeros(N, 3)

    for i in 1:N
        for j in 1:i-1
            dr = X[j, :] - X[i, :]   # previously X[i, :] - X[j, :], WRONG
            g = inv_box * dr
            gn = g - round(g)
            drn = box * gn
            vij = V[i, :] - V[j, :]
            fcube[i, j, :] = F_tot(drn, vij, A[bl[i], bl[j]], gamma, kT, dt, rc)
            fcube[j, i, :] = -fcube[i, j, :]
        end
    end
    F = -sum(fcube, 2)[:, 1, :]
end


function integrate_euler(X, V, A, bl, L, gamma, kT, dt, rc, Nt, thermo)
    N = size(X, 1)
    T = zeros(Nt+1)
    KE = zeros(Nt+1)
    PE = zeros(Nt+1)
    F = zeros(size(X))

    ti = time()
    for i in 2:Nt+1
        X = X + V * dt
        V = V + F * dt
        F = force_mat(X, V, A, bl, L, gamma, kT, dt, rc)
        X = X % L
        KE[i] = tot_KE(V)
        PE[i] = tot_PE(X, A, bl, L, rc)
        T[i] = KE[i] / (3 * N / 2)
        tf = time()
        if i % thermo == 0
            writedlm("Dump/dump_$(i).xyz", [bl X])   # rewrite
#            write_xyz("Dump/dump_$(i).xyz", bl, X)
            @printf "Step %i | t: %.3f | T: %.3f | PE: %.3e | Time: %.2f\n" i  i * dt  T[i]  PE[i]  tf - ti
        end
    end
    T, KE, PE
end


function integrate_verlet(X, V, A, bl, L, gamma, kT, dt, rc, Nt, thermo)
    N = size(X, 1)
    T = zeros(Nt+1)
    KE = zeros(Nt+1)
    PE = zeros(Nt+1)
    Fnew = zeros(size(X))

    F = force_mat(X, V, A, bl, L, gamma, kT, dt, rc)
    KE[1] = tot_KE(V)
    T[1] = KE[1] / (3.0 * N / 2)
    PE[1] = tot_PE(X, A, bl, L, rc)
#    writedlm("Dump/dump_1.xyz", [bl X])   # rewrite
    write_xyz("Dump/dump_1.xyz", bl, X)

    ti = time()
    for i in 2:Nt+1
        X = X + V * dt + 0.5 * F * dt^2
        Fnew = force_mat(X, V, A, bl, L, gamma, kT, dt, rc)
        V = V + 0.5 * (F + Fnew) * dt
        F = Fnew

#        Vnew = V + F * dt / 2
#        X = X + Vnew * dt
#        F = force_mat(X, Vnew, A, bl, L, gamma, kT, dt, rc)
#        V = Vnew + F * dt / 2

        X = mod(X, L)
        KE[i] = tot_KE(V)
        PE[i] = tot_PE(X, A, bl, L, rc)
        T[i] = KE[i] / (3 * N / 2)
        tf = time()
        if i % thermo == 0
#            writedlm("Dump/dump_$(i).xyz", [bl X])   # rewrite
            write_xyz("Dump/dump_$(i).xyz", bl, X)
            @printf "Step: %i | t: %.3f | T: %.3f | PE: %.3e | Time: %.2f\n" i  i * dt  T[i]  PE[i]  tf - ti
        end
    end
    T, KE, PE
end


end
