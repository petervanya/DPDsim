#!/usr/bin/julia
"""
Collection of input/output functions.
"""

function write_xyz(fname, bl, xyz)
    """Input:
    * bl: (N, 1) vector of names
    * xyz: (N, 3) matrix of positions
    """
    N = size(bl, 1)
    open(fname, "w") do f
        write(f, "$(N)\nbla\n")
        for i in 1:N
            write(f, "$(bl[i])\t$(xyz[i, 1])\t$(xyz[i, 2])\t$(xyz[i, 3])\n")
        end
    end
end


function read_xyz(fname)
    s = readlines(open(fname))[3:end]
    N = length(s)
    bl = zeros(Int, N)
    xyz = zeros(N, 3)
    for i in 1:N
        tmp = split(s[i])
        bl[i] = parse(Int, tmp[1])
        xyz[i, :] = map(x->parse(Float64, x), tmp[2:4])
    end
    bl, xyz
end


function read_config(fname)
    """Returns only positions and velocities
    not yet atom names."""
    s = readlines(open(fname))
    Ns = length(s)
    println("Number of lines: $Ns")
    vel_flag = parse(Int, split(s[2])[2])
    println("Velocity flag: $(vel_flag)")
    if vel_flag == 0
        N = Int((Ns - 2) / 2)
        println("Number of atoms: $N")
        mask_X = collect(4:2:Ns)
        X = zeros(N, 3)
        for i in 1:N
            tmp = split(s[mask_X[i]])
            X[i, :] = map(x->parse(Float64, x), tmp[1:3])
        end
        X
    elseif vel_flag == 1
        N = Int((Ns - 2) / 3)
        println("Number of atoms: $N")
        mask_X = collect(4:3:Ns)
        mask_V = collect(5:3:Ns)
        X = zeros(N, 3)
        V = zeros(N, 3)
        for i in 1:N
            tmp = split(s[mask_X[i]])
            X[i, :] = map(x->parse(Float64, x), tmp[1:3])
            tmp = split(s[mask_V[i]])
            V[i, :] = map(x->parse(Float64, x), tmp[1:3])
        end
        X, V
    end
end


