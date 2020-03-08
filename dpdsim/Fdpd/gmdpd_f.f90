module gmdpd_f

implicit none
contains


real(8) function randd()
    call random_number(randd)
end function


pure function wr(nr)
    real(8), intent(in) :: nr
    real(8) :: wr
    wr = 0.0
    if (nr <= 1.0) then
        wr = 1.0 - nr
    endif
end function


pure function wdd(nr, rd)
    real(8), intent(in) :: nr, rd
    real(8) :: wdd
    wdd = 0.0
    if (nr <= rd) then
        wdd = 1.0 - nr/rd
    endif
end function


function gaussnum() result (y)
    real(8) :: x1, x2, w, y
    w = 2.0
    do while (w >= 1.0)
        x1 = 2.0 * randd() - 1.0
        x2 = 2.0 * randd() - 1.0
        w = x1**2 + x2**2
    enddo

    w = sqrt((-2.0 * log(w)) / w)
    y = x1 * w
end function


function local_density(X, bl, nbt, ip_rd, n_rho, box) result (rho2)
    ! -----
    ! X: position matrix
    ! bl: vector of bead types
    ! nbt: number of bead types
    ! ip_rd: manybody cutoff, constant w.r.t. bead type
    ! -----
    integer, intent(in) :: bl(:), nbt
    real(8), intent(in) :: X(:, :), ip_rd(:, :), n_rho(:, :), box(3, 3)
    integer :: i, j, N
    real(8) :: rij(3), g(3), inv_box(3, 3), d_rho, pi, rd, nrm, nri
    real(8) :: rho2(size(X, 1), nbt)
    N = size(X, 1)
    g = 0.0
    inv_box = 0.0
    rho2 = 0.0
    pi = 4.0 * atan(1.0)
    rd = 0.0
    nrm = 0.0
    nri = 0.0

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)

            rd = ip_rd(bl(i), bl(j))
            nri = n_rho(bl(i), bl(j))
            nrm = (nri+1.0)*(nri+2.0)*(nri+3.0) / (8.0*pi*rd*rd*rd)

            d_rho = nrm * wdd(norm2(rij), rd)**nri
            rho2(i, bl(j)) = rho2(i, bl(j)) + d_rho
            rho2(j, bl(i)) = rho2(j, bl(i)) + d_rho
        enddo
    enddo
end function


function compute_force_fun(X, V, rho2, bl, &
        ip_a, ip_b, ip_rd, n_wrap, n_rho, box, gama, kT, dt) result (F)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: X(:, :), V(:, :), rho2(:, :)
    real(8), intent(in) :: ip_a(:, :), ip_b, ip_rd(:, :), n_wrap(:), n_rho(:, :)
    real(8), intent(in) :: box(3, 3), gama, kT, dt
    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3), fpair, a, nr
    real(8) :: rhoi, rhoj, rd, nrm, pi, nwi, nwj, nri
    integer :: i, j, N
    real(8) :: fcube(size(X, 1), size(X, 1), 3), F(size(X, 1), 3)
    N = size(X, 1)
    fcube = 0.0
    g = 0.0
    fpair = 0.0
    inv_box = 0.0
    rd = 0.0
    nrm = 0.0
    pi = 4.0 * atan(1.0)

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            vij = V(i, :) - V(j, :)

            nr = norm2(rij)
            a = ip_a(bl(i), bl(j))
            rd = ip_rd(bl(i), bl(j))
            nwi = n_wrap(bl(i))
            nwj = n_wrap(bl(j))
            nri = n_rho(bl(i), bl(j))
            nrm = (nri+1.0)*(nri+2.0)*(nri+3.0) / (8.0*pi*rd*rd*rd)
            rhoi = rho2(i, bl(j))
            rhoj = rho2(j, bl(j))

            fpair = a * wr(nr) &
                + ip_b * (rhoi**(nwi-1.0) + rhoj**(nwj-1.0)) &
                    * nrm * wdd(nr, rd)**(nri-1.0) * nri / rd &
                - gama * wr(nr)**2 * dot_product(rij, vij) / nr &
                + sqrt(2*gama*kT) * wr(nr) * gaussnum() / sqrt(dt)
            fcube(i, j, :) = fpair / nr * rij(:)
            fcube(j, i, :) = -fcube(i, j, :)
        enddo
    enddo
    F = sum(fcube, 2)
end function


subroutine compute_force(F, vir, sigma, &
        X, V, rho2, bl, ip_a, ip_b, ip_rd, n_wrap, n_rho, &
        box, gama, kT, dt)
    real(8), intent(inout) :: F(:, :), vir, sigma(3)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: X(:, :), V(:, :), rho2(:, :)
    real(8), intent(in) :: ip_a(:, :), ip_b, ip_rd(:, :), n_wrap(:), n_rho(:, :)
    real(8), intent(in) :: box(3, 3), gama, kT, dt
    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3), volume, fpair, a, nr
    real(8) :: rhoi, rhoj, rd, nrm, pi, nwi, nwj, nri
    integer :: i, j, N
    real(8) :: fcube(size(X, 1), size(X, 1), 3)
    N = size(X, 1)
    fcube = 0.0
    g = 0.0
    fpair = 0.0
    inv_box = 0.0
    rd = 0.0
    nwi = 0.0
    nwj = 0.0
    nri = 0.0
    nrm = 0.0
    volume = 1.0
    pi = 4.0 * atan(1.0)

    vir = 0.0
    sigma = 0.0
    F = 0.0

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)
    do i = 1, 3
        volume = volume * box(i, i)
    enddo

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            vij = V(i, :) - V(j, :)

            nr = norm2(rij)
            a = ip_a(bl(i), bl(j))
            rd = ip_rd(bl(i), bl(j))
            nwi = n_wrap(bl(i))
            nwj = n_wrap(bl(j))
            nri = n_rho(bl(i), bl(j))
            nrm = (nri+1.0)*(nri+2.0)*(nri+3.0) / (8.0*pi*rd*rd*rd)
            rhoi = rho2(i, bl(j))
            rhoj = rho2(j, bl(j))

            fpair = a * wr(nr) &
                + ip_b * (rhoi**(nwi-1.0) + rhoj**(nwj-1.0)) &
                    * nrm * wdd(nr, rd)**(nri-1.0) * nri / rd &
                - gama * wr(nr)**2 * dot_product(rij, vij) / nr &
                + sqrt(2*gama*kT) * wr(nr) * gaussnum() / sqrt(dt)
            fcube(i, j, :) = fpair / nr * rij(:)
            fcube(j, i, :) = -fcube(i, j, :)

            vir = vir + dot_product(fcube(i, j, :), rij(:))
            sigma(:) = sigma(:) + fcube(i, j, :) * rij(:)
        enddo
    enddo

    do i = 1, N  ! kinetic term of the stress tensor
        sigma(:) = sigma(:) + V(i, :) * V(i, :)
    enddo

    sigma = sigma / volume
    F = sum(fcube, 2)
end subroutine


function compute_force_cube(X, V, rho2, bl, &
        ip_a, ip_b, ip_rd, n_wrap, n_rho, &
        box, gama, kT, dt) result (fcube)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: X(:, :), V(:, :), rho2(:, :)
    real(8), intent(in) :: ip_a(:, :), ip_b, ip_rd(:, :), n_wrap(:), n_rho(:, :)
    real(8), intent(in) :: box(3, 3), gama, kT, dt
    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3), fpair, a, nr
    real(8) :: rhoi, rhoj, rd, nrm, pi, nwi, nwj, nri
    integer :: i, j, N
    real(8) :: fcube(size(X, 1), size(X, 1), 3)
    N = size(X, 1)
    fcube = 0.0
    g = 0.0
    fpair = 0.0
    inv_box = 0.0
    rd = 0.0
    nwi = 0.0
    nwj = 0.0
    nri = 0.0
    nrm = 0.0
    nrm = 0.0
    pi = 4.0 * atan(1.0)

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            vij = V(i, :) - V(j, :)

            nr = norm2(rij)
            a = ip_a(bl(i), bl(j))
            rd = ip_rd(bl(i), bl(j))
            nwi = n_wrap(bl(i))
            nwj = n_wrap(bl(j))
            nri = n_rho(bl(i), bl(j))
            nrm = (nri+1.0)*(nri+2.0)*(nri+3.0) / (8.0*pi*rd*rd*rd)
            rhoi = rho2(i, bl(j))
            rhoj = rho2(j, bl(j))

            fpair = a * wr(nr) &
                + ip_b * (rhoi**(nwi-1.0) + rhoj**(nwj-1.0)) &
                    * nrm * wdd(nr, rd)**(nri-1.0) * nri / rd &
                - gama * wr(nr)**2 * dot_product(rij, vij) / nr &
                + sqrt(2*gama*kT) * wr(nr) * gaussnum() / sqrt(dt)
            fcube(i, j, :) = fpair / nr * rij(:)
            fcube(j, i, :) = -fcube(i, j, :)
        enddo
    enddo
end function


function compute_pe(X, rho2, bl, ip_a, ip_b, n_wrap, box) result (pe)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: X(:, :), rho2(:, :)
    real(8), intent(in) :: ip_a(:, :), ip_b, n_wrap(:), box(3, 3)
    integer :: i, j, N
    real(8) :: pe, rij(3), g(3), inv_box(3, 3), pi, nwi
    pe = 0.0
    N = size(X, 1)
    inv_box = 0.0
    pi = 4.0 * atan(1.0)
    nwi = 0.0

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)
    
    do i = 1, n
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            pe = pe + ip_a(bl(i), bl(j)) * wr(norm2(rij))**2 / 2.0
        enddo
    enddo

    do i = 1, n
        nwi = n_wrap(bl(i))
        pe = pe + ip_b * sum(rho2(i, :), 1)**nwi / nwi
    enddo
end function


function compute_ke(V) result (ke)
    real(8), intent(in) :: V(:, :)
    real(8) :: ke
    integer :: i, j, N
    ke = 0.0
    N = size(V, 1)

    do i = 1, N
        do j = 1, 3
            ke = ke + V(i, j) * V(i, j) / 2.0
        enddo
    enddo
end function


function init_pos(N, box) result(X)
    integer, intent(in) :: N
    real(8), intent(in) :: box(3, 3)
    real(8) :: X(N, 3)
    integer :: i, j

    do i = 1, N
        do j = 1, 3
            X(i, j) = randd() * box(j, j)
        enddo
    enddo
end function


function init_vel(N, kT) result(V)
    integer, intent(in) :: N
    real(8), intent(in) :: kT
    real(8) :: V(N, 3), Vcom(3)
    integer :: i, j

    do i = 1, N
        do j = 1, 3
            V(i, j) = gaussnum() * kT
        enddo
    enddo
    Vcom = sum(V, 1) / N
    do i = 1, N
        V(i, :) = V(i, :) - Vcom(:)
    enddo
end function


subroutine euler_step(X, V, rho2, bl, nbt, ip_a, ip_b, ip_rd, n_wrap, n_rho, &
    box, gama, kT, dt)
    real(8), intent(inout) :: X(:, :), V(:, :), rho2(:, :)
    real(8), intent(in) :: ip_a(:, :), ip_b, ip_rd(:, :), n_wrap(:), n_rho(:, :)
    integer, intent(in) :: bl(:), nbt
    real(8), intent(in) :: box(3, 3), gama, kT, dt
    real(8) :: F(size(X, 1), 3)
    F = 0.0

    rho2 = local_density(X, bl, nbt, ip_rd, n_rho, box)
    F = compute_force_fun(X, V, rho2, bl, ip_a, ip_b, ip_rd, n_wrap, n_rho, &
        box, gama, kT, dt)
    V(:, :) = V(:, :) + F(:, :) * dt
    X(:, :) = X(:, :) + V(:, :) * dt
end subroutine


subroutine verlet_step(X, V, F, rho2, bl, nbt, ip_a, ip_b, ip_rd, &
    n_wrap, n_rho, box, gama, kT, dt)
    real(8), intent(inout) :: X(:, :), V(:, :), F(:, :), rho2(:, :)
    real(8), intent(in) :: ip_a(:, :), ip_b, ip_rd(:, :), n_wrap(:), n_rho(:, :)
    integer, intent(in) :: bl(:), nbt
    real(8), intent(in) :: box(3, 3), gama, kT, dt
    real(8) :: V2(size(X, 1), 3)
    V2 = 0.0

    V2(:, :) = V(:, :) + 0.5 * F(:, :) * dt
    X(:, :) = X(:, :) + V2(:, :) * dt
    rho2 = local_density(X, bl, nbt, ip_rd, n_rho, box)
    F = compute_force_fun(X, V2, rho2, bl, ip_a, ip_b, ip_rd, n_wrap, n_rho, &
        box, gama, kT, dt)
    V(:, :) = V2(:, :) + 0.5 * F(:, :) * dt
end subroutine


subroutine print_mat(A)
    real(8), intent(in) :: A(:, :)
    integer :: M, N, i
    M = size(A, 1)
    N = size(A, 2)
    do i = 1, M
        print '(f5.3, " ", f5.3, " ", f5.3)', A(i, 1), A(i, 2), A(i, 3)
    enddo
end subroutine


end module


