import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = np.loadtxt('values.dat', skiprows=3)
steps, height, width = np.genfromtxt('values.dat',skip_footer=len(data), unpack=True, dtype=int)
data = data.reshape(steps,height,width)

fig = plt.figure(figsize=(10,5))
ax = fig.gca()
z = data[0]
surf = ax.imshow(z, cmap='jet', interpolation='none')
plt.xticks([])
plt.yticks([])

def update(i, z, surf) :
    z = data[i]
    z[39:59, 149:151] = -0.3
    ax.clear()
    ax.text(0,height-1,('frame {}'.format(i)))
    plt.xticks([])
    plt.yticks([])
    surf = ax.imshow(z, cmap='jet',interpolation='none')
    return surf,
    
anim = FuncAnimation(fig, update, fargs=(z,surf), interval=1, blit = True, repeat = False)
plt.show()