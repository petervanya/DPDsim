module dpd_f

implicit none
contains

pure function wr(nr)
    real(8), intent(in) :: nr
    real(8) :: wr
    if (nr <= 1.0) then
        wr = 1 - nr
    else
        wr = 0.0
    endif
end function


function gnum() result (y)
    real(8) :: x1, x2, w, y
    w = 2.0
    do while (w >= 1.0)
        x1 = 2.0 * rand() - 1.0
        x2 = 2.0 * rand() - 1.0
        w = x1 * x1 + x2 * x2
    enddo

    w = sqrt((-2.0 * log(w)) / w)
    y = x1 * w
end function


function f_tot(r, v, a, gama, kT, dt) result (ft)
    real(8), intent(in) :: r(3), v(3), a, gama, kT, dt
    real(8) :: ft(3), nr
    nr = norm2(r)
    ft = a * wr(nr) * r / nr &
        - gama * wr(nr)**2 * dot_product(r, v) * r / nr**2 &
        + sqrt(2 * gama * kT) * wr(nr) * gnum() / sqrt(dt) * r / nr
end function


function tot_pe(X, ip, blist, box) result (pe)
    integer, intent(in) :: blist(:)
    real(8), intent(in) :: X(:, :), ip(:, :), box(3, 3)
    integer :: i, j, N, Np
    real(8) :: pe, rij(3), g(3), inv_box(3, 3)
    pe = 0.0
    N = size(X, 1)
    Np = size(ip, 1)

    do i = 1, 3
        inv_box(i, i) = 1.0 / box(i, i)
    enddo
    
    do i = 1, n
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            pe = pe + ip(blist(i), blist(j)) * wr(norm2(rij))**2 / 2.0
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


function force_list(X, V, ip, blist, box, gama, kT, dt) result (F)
    integer, intent(in) :: blist(:)
    real(8), intent(in) :: X(:, :), V(:, :), ip(:, :), box(3, 3)
    real(8), intent(in) :: gama, kT, dt
    real(8) :: g(3), rij(3), vij(3), inv_box(3, 3)
    integer :: i, j, N, Np
    real(8) :: fm(size(X, 1), size(X, 1), 3), F(size(X, 1), 3)
    N = size(X, 1)
    Np = size(ip, 1)

    do i = 1, 3
        inv_box(i, i) = 1.0 / box(i, i)
    enddo

    do i = 1, N
        do j = 1, i-1
            rij = X(i, :) - X(j, :)
            g = matmul(inv_box, rij)
            g = g - nint(g)
            rij = matmul(box, g)
            vij = V(i, :) - V(j, :)
            fm(i, j, :) = f_tot(rij, vij, ip(blist(i), blist(j)), gama, kT, dt)
            fm(j, i, :) = fm(i, j, :)
        enddo
    enddo
    F = sum(fm, 1)
end function


function init_pos(N, box) result(X)
    integer, intent(in) :: N
    real(8), intent(in) :: box(3, 3)
    real :: X(N, 3)
    integer :: i, j

    do i = 1, N
        do j = 1, 3
            X(i, j) = rand() * box(j, j)
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
            V(i, j) = gnum() * kT
        enddo
    enddo
    Vcom = sum(V, 1) / N
    do i = 1, N
        V(i, :) = V(i, :) - Vcom(:)
    enddo
end function


end module


