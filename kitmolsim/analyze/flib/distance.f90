module distance_m
    use, intrinsic :: iso_fortran_env, only : int32, real64
    use, intrinsic :: iso_c_binding
    implicit none
    
contains

subroutine calc_distance_pbc_interface(N, L, r, mat) bind(c, name="fort_calc_distance_pbc")
    integer(c_int),intent(in) :: N
    real(c_double),intent(in) :: L(3)
    real(c_double),intent(in) :: r(3,N)
    real(c_double),intent(inout) :: mat(N,N)

    call calc_distance_pbc(N, L, r, mat)

end subroutine

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


end module distance_m