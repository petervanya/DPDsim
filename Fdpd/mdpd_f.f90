module mdpd_f

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


function local_density(X, bl, ntp, rd, box) result (rho)
    ! -----
    ! X: position matrix
    ! bl: vector of bead types
    ! ntp: number of bead types
    ! rd: manybody cutoff, constant w.r.t. bead type
    ! -----
    integer, intent(in) :: bl(:), ntp
    real(8), intent(in) :: X(:, :), rd, box(3, 3)
    integer :: i, j, N
    real(8) :: rij(3), g(3), inv_box(3, 3), rho_incr, pi
    real(8) :: rho(size(X, 1), ntp)
    N = size(X, 1)
    g = 0.0
    inv_box = 0.0  ! do not forget this, else will spend hours debugging
    rho = 0.0
    pi = 4*atan(1.0)

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            rho_incr = wdd(norm2(rij), rd)**2 * 15.0 / (2.0*pi*rd*rd*rd)
            rho(i, bl(j)) = rho(i, bl(j)) + rho_incr
            rho(j, bl(i)) = rho(j, bl(i)) + rho_incr
        enddo
    enddo
end function


function force_mat(X, V, bl, ntp, ip, box, gama, kT, dt, ipb, rd) result (F)
    integer, intent(in) :: bl(:), ntp
    real(8), intent(in) :: X(:, :), V(:, :), ip(:, :), ipb(:, :), box(3, 3)
    real(8), intent(in) :: gama, kT, dt, rd
    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3), fpair, a, b, nr
    real(8) :: rhoi, rhoj
    integer :: i, j, N
    real(8) :: fm(size(X, 1), size(X, 1), 3), F(size(X, 1), 3)
    real(8) :: rho(size(X, 1), ntp)
    N = size(X, 1)
    fm = 0.0
    g = 0.0
    fpair = 0.0
    inv_box = 0.0  ! do not forget this, else will spend hours debugging

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)

    rho = local_density(X, bl, ntp, rd, box)

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            vij = V(i, :) - V(j, :)

            a = ip(bl(i), bl(j))
            b = ipb(bl(i), bl(j))
            nr = norm2(rij)
            rhoi = rho(i, bl(j))
            rhoj = rho(j, bl(j))

            fpair = a * wr(nr) &
                + b * (rhoi + rhoj) * wdd(nr, rd) &
                - gama * wr(nr)**2 * dot_product(rij, vij) / nr &
                + sqrt(2*gama*kT) * wr(nr) * gaussnum() / sqrt(dt)
            fm(i, j, :) = fpair / nr * rij(:)
            fm(j, i, :) = -fpair / nr * rij(:)
        enddo
    enddo
    F = sum(fm, 2)
end function


function tot_pe(X, bl, ip, box) result (pe)
    integer, intent(in) :: bl(:)
    real(8), intent(in) :: X(:, :), ip(:, :), box(3, 3)
    integer :: i, j, N
    real(8) :: pe, rij(3), g(3), inv_box(3, 3)
    pe = 0.0
    N = size(X, 1)
    inv_box = 0.0  ! do not forget this, else will spend hours debugging

    forall (i = 1:3) inv_box(i, i) = 1.0 / box(i, i)
    
    do i = 1, n
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            pe = pe + ip(bl(i), bl(j)) * wr(norm2(rij))**2 / 2.0
        enddo
    enddo
end function


function tot_ke(V) result (ke)
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


subroutine euler_step(X, V, bl, ntp, ip, box, gama, kT, dt, ipb, rd)
    real(8), intent(inout) :: X(:, :), V(:, :)
    real(8), intent(in) :: ip(:, :), box(3, 3), ipb(:, :)
    integer, intent(in) :: bl(:), ntp
    real(8), intent(in) :: gama, kT, dt, rd
    real(8) :: F(size(X, 1), 3)
    F = 0.0

    F = force_mat(X, V, bl, ntp, ip, box, gama, kT, dt, ipb, rd)
    V(:, :) = V(:, :) + F(:, :) * dt
    X(:, :) = X(:, :) + V(:, :) * dt
end subroutine


subroutine verlet_step(X, V, F, bl, ntp, ip, box, gama, kT, dt, ipb, rd)
    real(8), intent(inout) :: X(:, :), V(:, :), F(:, :)
    real(8), intent(in) :: ip(:, :), box(3, 3), ipb(:, :)
    integer, intent(in) :: bl(:), ntp
    real(8), intent(in) :: gama, kT, dt, rd
    real(8) :: V2(size(X, 1), 3)
    V2 = 0.0

    V2(:, :) = V(:, :) + 0.5 * F(:, :) * dt
    X(:, :) = X(:, :) + V2(:, :) * dt
    F = force_mat(X, V2, bl, ntp, ip, box, gama, kT, dt, ipb, rd)
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


