import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from numba import jit

height = 100 # lattice y length 
width = 200  # lattice x length
dx = 0.10   # distance step
dt = 0.012  # timestep

u0 = 0.18   # driven velocity
l0 = 1.     # lattice step length
v0 = 3e-6   # kinetic viscosity 

U0 = dt / dx     # characteristic velocity
L0 = width * dx  # characteristic length 

L_D = l0 / L0        # lattice length
u_D = u0 / U0        # lattice velocity
v_D = v0 / L0 / U0   # lattice viscosity 

Re =u_D * L_D / v_D  # Reynold's number

cs = U0 / np.sqrt(3)       # lattice speed of sound
tau = v0 / cs**2 + dt/2.   # relaxation constant 
omega = dt / tau           

# ------ final sim params -----
# Re=60,000
#     barrier: dx = 0.10, dt = 0.012, u0 = 0.18, v0 = 3e-6,
#      sphere: dx = 0.15, dt = 0.012, u0 = 0.18, v0 = 3e-6, 
#
# Re=30,000
#     barrier: dx = 0.10, dt = 0.012, u0 = 0.18, v0 = 6e-6, 
#      sphere: dx = 0.15, dt = 0.012, u0 = 0.18, v0 = 6e-6, 
# -----------------------------

# ----- intializing number density ----- 
w =  [1/36., 1/9., 1/36., 1/9., 4/9., 1/9., 1/36., 1/9., 1/36.] # probability weights
f = np.array(w * width * height).reshape(height, width, 9) # number densities

f[:,:,0] *= (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)  # updating for flow speed 
f[:,:,1] *= (1 - 1.5*u0**2)
f[:,:,2] *= (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
f[:,:,3] *= (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
f[:,:,4] *= (1 - 1.5*u0**2)
f[:,:,5] *= (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)
f[:,:,6] *= (1 - 3*u0 + 4.5*u0**2 - 1.5*u0**2)
f[:,:,7] *= (1 - 1.5*u0**2)
f[:,:,8] *= (1 + 3*u0 + 4.5*u0**2 - 1.5*u0**2)

f_eq = np.ones((height, width, 9)) # equilibrium number density

#----- initalizing macroscopic quantities
rho = np.ones((height, width))
ux = (f[:,:,2] + f[:,:,5] + f[:,:,8] - (f[:,:,0] + f[:,:,3] + f[:,:,6])) / rho # eq.(7)
uy = (f[:,:,0] + f[:,:,1] + f[:,:,2] - (f[:,:,6] + f[:,:,7] + f[:,:,8])) / rho
u = np.sqrt(ux**2 + uy**2)

#------ initialize borders ------
walltop = np.zeros((height,width), bool)
walltop[0,:] = True

wallbot = np.zeros((height,width), bool)
wallbot[-1,:] = True

barrier = np.zeros((height,width), bool)
barrier_outside = np.zeros((height,width), bool)

# ----- barrier initialization -----
#barrier[int(height/2.-7):int(height/2.+7), int(3*width/4.)] = True
#barrier_outside[int(height/2.-7):int(height/2.+7), int(3*width/4.+1)] = True
#barrier_outside[int(height/2.-7):int(height/2.+7), int(3*width/4.-1)] = True
#barrier_outside[int(height/2.-8),int(3*width/4.)] = True
#barrier_outside[int(height/2.+8),int(3*width/4.)] = True
# ----------------------------------

# ----- sphere initialization -----
for y in range(-height, height) :
    for x in range(-width, width) :
        if np.sqrt(x**2 + y**2) < 20. : 
            barrier[int(y+height/2), int(x+3*width/4)] = True
        if 20. < np.sqrt(x**2 + y**2) < 21. :
            barrier_outside[int(y+height/2), int(x+3*width/4)] = True
barrier_outside[int(height/2-20), int(3*width/4)] = True
barrier_outside[int(height/2+20), int(3*width/4)] = True
barrier_outside[int(height/2), int(3*width/4-20)] = True
barrier_outside[int(height/2), int(3*width/4+20)] = True
# ---------------------------------

@jit(nopython=True) 
def stream(x, y, height, width, f_copy)  :           
    return np.array([f_copy[min(y+1,height-1), x-1, 0],     
                    f_copy[min(y+1,height-1), x, 1],
                    f_copy[min(y+1,height-1), (x+1)%width, 2],
                    f_copy[y, x-1, 3], 
                    f_copy[y, x, 4], 
                    f_copy[y, (x+1)%width, 5],
                    f_copy[max(y-1,0), x-1, 6], 
                    f_copy[max(y-1,0), x, 7], 
                    f_copy[max(y-1,0), (x+1)%width, 8]])

global rho_tot, uxs, px_tot, py_tot
rho_tot = []
px_tot = []
py_tot = []

def step() :
    global f, f_eq
    # ----- collision -----
    rho = np.zeros((height, width))
    for j in range(9) :
        rho[:,:] += f[:,:,j]  # eq.(6)
    rho_tot.append(sum(rho.reshape(-1)))
    
    ux = (f[:,:,2] + f[:,:,5] + f[:,:,8] - (f[:,:,0] + f[:,:,3] + f[:,:,6])) / rho
    uy = (f[:,:,0] + f[:,:,1] + f[:,:,2] - (f[:,:,6] + f[:,:,7] + f[:,:,8])) / rho
    u = np.sqrt(ux**2 + uy**2)
    eu = np.array([uy-ux, uy, ux+uy, -ux, 0, ux, -uy-ux, -uy, -uy+ux]) # velocity vectors
    
    px = f[barrier_outside][:,0] * -ux[barrier_outside] + f[barrier_outside][:,3] * -ux[barrier_outside] + f[barrier_outside][:,6] * -ux[barrier_outside] # eq.(7)
    py = f[barrier_outside][:,0] * uy[barrier_outside] + f[barrier_outside][:,6] * -uy[barrier_outside] # eq.(7)
    px_tot.append(sum(px))
    py_tot.append(sum(py))
    
    for e in range(9) : # e == 0-8 direction
        f_eq[:,:,e] = rho * w[e] * (1 + 3*eu[e] + 4.5*eu[e]**2 - 1.5*u**2)  # eq.(5)
    f = f + omega * (f_eq - f)
    
    # flow to the left
    f[:,-1,2] = w[2] * (1 + 3*u0 - 1.5*u0**2 + 4.5*u0**2) # eq.(5)
    f[:,-1,5] = w[5] * (1 + 3*u0 - 1.5*u0**2 + 4.5*u0**2)
    f[:,-1,8] = w[8] * (1 + 3*u0 - 1.5*u0**2 + 4.5*u0**2)
    f[:,-1,0] = w[0] * (1 - 3*u0 - 1.5*u0**2 + 4.5*u0**2)
    f[:,-1,3] = w[3] * (1 - 3*u0 - 1.5*u0**2 + 4.5*u0**2)
    f[:,-1,6] = w[6] * (1 - 3*u0 - 1.5*u0**2 + 4.5*u0**2)
    f_copy = f.copy()  
    
    # ----- streaming -----
    for y in range(height) : 
        for x in range(width) :  
            f[y,x] = stream(x, y, height, width, f_copy)    
    
    f_copy = f.copy()
    # walls             
    f[walltop, 6] = f[walltop, 0]
    f[walltop, 7] = f[walltop, 1]      # bounceback top
    f[walltop, 8] = f[walltop, 2]    
    
    f[wallbot, 2] = f_copy[wallbot, 8]
    f[wallbot, 1] = f_copy[wallbot, 7]      # bounceback bottom
    f[wallbot, 0] = f_copy[wallbot, 6]
    
    f[barrier_outside, 3] = f[barrier_outside, 5]      # barrier object bounceback
    f[barrier_outside, 0] = f[barrier_outside, 8]
    f[barrier_outside, 6] = f[barrier_outside, 2]
    
    f[barrier_outside, 2] = f_copy[barrier_outside, 6]
    f[barrier_outside, 5] = f_copy[barrier_outside, 3]
    f[barrier_outside, 8] = f_copy[barrier_outside, 0]
    
    f[barrier_outside, 1] = f[barrier_outside, 7]
    f[barrier_outside, 7] = f_copy[barrier_outside, 1]
    
    return rho, ux, uy, u

def animation() :
    X,Y = np.meshgrid(range(width), range(height))
    rho, ux, uy, u = step()
    z = ux
    fig = plt.figure()
    ax = fig.gca()
    surf = ax.imshow(z, cmap = 'jet', interpolation='none')#, vmin=-.18 , vmax=.4081)
    plt.xticks([])
    plt.yticks([])
    
    def update_data(i, z, surf) : 
        rho, ux, uy, u = step()
        z = ux
        z[barrier] = -.35
        ax.clear()
        ax.text(0,height-1,('frame {}'.format(i)))
        plt.xticks([])
        plt.yticks([])
        plt.title('$Re = {}$'.format(Re))
        surf = ax.imshow(z, cmap = 'jet', interpolation='none')#, vmin=-.18 , vmax=.4081)
        return surf,

    anim = manimation.FuncAnimation(fig, update_data, fargs = (z,surf), interval = 1, blit = False, repeat = True)
    plt.show()
       
animation()
