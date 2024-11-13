module distance_m
    use, intrinsic :: iso_fortran_env, only : int32, real64
    implicit none
    
contains

subroutine calc_distance_pbc(N, L, r, mat)
    integer(int32),intent(in) :: N
    real(real64),intent(in) :: L(3)
    real(real64),intent(in) :: r(3,N)
    real(real64),intent(inout) :: mat(N,N)

    integer(int32) i, j
    real(real64) ri(3), rij(3), dij

    !initialization
    mat(:,:) = 0.0d0

    do i = 1, N-1
        ri(:) = r(:,i)
        do j = i, N
            rij(:) = abs(ri(:) - r(:,j))
            rij(:) = merge(rij - L, rij, mask=rij > L/2)
            dij = norm2(rij)
            mat(i,j) = dij
            mat(j,i) = dij
        end do    
    end do

end subroutine

pure subroutine calc_distance_two_paricle_pbc(N, L, r1, r2, dist)
    integer(int32),intent(in) :: N
    real(real64),intent(in) :: L(3)
    real(real64),intent(in) :: r1(3)
    real(real64),intent(in) :: r2(3,N)
    real(real64),intent(inout) :: dist(N)

    integer(int32) i
    real(real64) r1j(3), d1j_sq

    do i = 1, N
        r1j(:) = abs(r1(:) - r2(:,i))
        r1j(:) = merge(r1j - L, r1j, mask=r1j > L/2)
        d1j_sq = sum(r1j*r1j)
        dist(i) = sqrt(d1j_sq)
    end do

end subroutine

end module distance_m