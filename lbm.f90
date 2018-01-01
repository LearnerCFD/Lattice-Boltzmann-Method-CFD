program lbm

implicit none
include 'mpif.h'

integer :: height, width, i, j, k, N
real :: omega, u0

! array declaration
real, dimension (9) :: w
real, dimension (200, 400, 9) :: f, f_eq, f_copy
real, dimension (200, 400) :: rho, ux, uy, u, zeros
real, dimension (9, 200, 400) :: eu

height = 200
width = 400
u0 = 0.18
omega = 1.6
N = 3000 ! number of steps
zeros(height,width) = 0

! initializing number density
w(1:9) = (/ 1/36., 1/9., 1/36., 1/9., 4/9., 1/9., 1/36., 1/9., 1/36. /)

do i=1,height
    do j=1, width
        f(i,j,1) = w(1) * (1. - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
        f(i,j,2) = w(2) * (1. - 1.5*u0**2)
        f(i,j,3) = w(3) * (1. + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
        f(i,j,4) = w(4) * (1. - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
        f(i,j,5) = w(5) * (1. - 1.5*u0**2)
        f(i,j,6) = w(6) * (1. + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
        f(i,j,7) = w(7) * (1. - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
        f(i,j,8) = w(8) * (1. - 1.5*u0**2)
        f(i,j,9) = w(9) * (1. + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    end do
end do


! initializing macroscopic quantities
rho(:,:) = 1.
ux(:,:) = f(:,:,3) + f(:,:,6) + f(:,:,9) -(f(:,:,1) + f(:,:,4) + f(:,:,7)) / rho(:,:)
uy(:,:) = f(:,:,1) + f(:,:,2) + f(:,:,3) -(f(:,:,7) + f(:,:,8) + f(:,:,9)) / rho(:,:)
u(:,:) = SQRT(ux(:,:)**2 + uy(:,:)**2)

! starting data file
open(unit=1, file='values.dat')
write(1,*) N
write(1,*) height
write(1,*) width

do i=1, N
    rho(:,:) = 0.
    
    do j=1, 9
        rho(:,:) = rho(:,:) + f(:,:,j)
    end do
    
    ux(:,:) = (f(:,:,3) + f(:,:,6) + f(:,:,9) - (f(:,:,1) + f(:,:,4) + f(:,:,7))) / rho(:,:)
    uy(:,:) = (f(:,:,1) + f(:,:,2) + f(:,:,3) - (f(:,:,7) + f(:,:,8) + f(:,:,9))) / rho(:,:)
    u(:,:) = SQRT(ux(:,:)**2 + uy(:,:)**2)
    
    eu(1, :, :) = uy-ux
    eu(2, :, :) = uy
    eu(3, :, :) = ux+uy
    eu(4, :, :) = -ux
    eu(5, :, :) = zeros
    eu(6, :, :) = ux
    eu(7, :, :) = -uy-ux
    eu(8, :, :) = -uy
    eu(9, :, :) = -uy+ux
    
    do j=1, 9 
        f_eq(:,:,j) = rho(:,:) * w(j) * (1 + 3*eu(j,:,:) + 4.5*eu(j,:,:)**2 - 1.5*u(:,:)**2) 
    end do
    
    f(:,:,:) = f(:,:,:) + omega *(f_eq(:,:,:) - f(:,:,:))
    
    ! flow to the left 
    f(:,width, 3) = w(3) * (1. + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    f(:,width, 6) = w(6) * (1. + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    f(:,width, 9) = w(9) * (1. + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    f(:,width, 1) = w(1) * (1. - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    f(:,width, 4) = w(4) * (1. - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    f(:,width, 7) = w(7) * (1. - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
    
    f_copy(:,:,:) = f(:,:,:)
    
    ! streaming step
    
    !$OMP PARALLEL DO PRIVATE(j,k)
        do j=1,height
            do k=2,width-1
                f(j,k,1) = f_copy(MIN(j+1, height), k-1, 1)
                f(j,k,2) = f_copy(MIN(j+1, height), k, 2)
                f(j,k,3) = f_copy(MIN(j+1, height), k+1, 3)
                f(j,k,4) = f_copy(j, k-1, 4)
                f(j,k,5) = f_copy(j, k, 5)
                f(j,k,6) = f_copy(j, k+1, 6)
                f(j,k,7) = f_copy(MAX(j-1, 1), k-1, 7)
                f(j,k,8) = f_copy(MAX(j-1, 1), k, 8)
                f(j,k,9) = f_copy(MAX(j-1, 1), k+1, 9)               
            end do
            
            f(j,1,1) = f_copy(MIN(j+1, height), width, 1)
            f(j,1,2) = f_copy(MIN(j+1, height), 1, 2)
            f(j,1,3) = f_copy(MIN(j+1, height), 2, 3)
            f(j,1,4) = f_copy(j, width, 4)
            f(j,1,5) = f_copy(j, 1, 5)
            f(j,1,6) = f_copy(j, 2, 6)
            f(j,1,7) = f_copy(MAX(j-1, 1), width, 7)
            f(j,1,8) = f_copy(MAX(j-1, 1), 1, 8)
            f(j,1,9) = f_copy(MAX(j-1, 1), 2, 9)
        
            f(j,width,1) = f_copy(MIN(j+1, height), width-1, 1)
            f(j,width,2) = f_copy(MIN(j+1, height), width, 2)
            f(j,width,3) = f_copy(MIN(j+1, height), 1, 3)
            f(j,width,4) = f_copy(j, width-1, 4)
            f(j,width,5) = f_copy(j, width, 5)
            f(j,width,6) = f_copy(j, 1, 6)
            f(j,width,7) = f_copy(MAX(j-1, 1), width-1, 7)
            f(j,width,8) = f_copy(MAX(j-1, 1), width, 8)
            f(j,width,9) = f_copy(MAX(j-1, 1), 1, 9)
        end do
    !$OMP END PARALLEL DO
    
    f_copy(:,:,:) = f(:,:,:)
    
    f(1,:, 7) = f(1,:, 1)
    f(1,:, 8) = f(1,:, 2)
    f(1,:, 9) = f(1,:, 3)
    
    f(height,:, 3) = f_copy(height,:, 9)
    f(height,:, 2) = f_copy(height,:, 8)
    f(height,:, 1) = f_copy(height,:, 7)
    
    f(40:60, 150:152, 1) = f(40:60, 150:152, 9)
    f(40:60, 150:152, 4) = f(40:60, 150:152, 6)
    f(40:60, 150:152, 7) = f(40:60, 150:152, 3)
    f(40:60, 150:152, 2) = f(40:60, 150:152, 8)
    
    f(40:60, 150:152, 3) = f_copy(40:60, 150:152, 7)
    f(40:60, 150:152, 6) = f_copy(40:60, 150:152, 4)
    f(40:60, 150:152, 9) = f_copy(40:60, 150:152, 1)
    f(40:60, 150:152, 8) = f_copy(40:60, 150:152, 2)
    
    
    
        
    do j=1,height
        write(1,*) ux(j,:)
    end do
    write(1,*) ''

end do    

end program lbm
