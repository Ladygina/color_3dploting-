import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap

def polygon_under_graph(xlist, ylist):
    return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]

def polynom_parallel():
    zs = np.linspace(-1, 1,40)
    verts=[]

    for i in zs:
        r = np.linspace(i, 1.25+i, 50)
        p = np.linspace(i, 2 * np.pi+i, 50)
        X, Y = r * np.cos(p), r * np.sin(p)
        verts.append(polygon_under_graph(X,Y))

    facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))
    poly = PolyCollection(verts,facecolors=facecolors)
    poly.set_alpha(0.7)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # The zdir keyword makes it plot the "z" vertex dimension (radius)
    # along the y axis. The zs keyword sets each polygon at the
    # correct radius value.
    ax.add_collection3d(poly, zs=zs, zdir='z')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    plt.show()

r =[0.00392157, 0.00326797, 0.00261438, 0.00196078]*255
g =[4.80253647e-19, 3.20169098e-19, 1.60084549e-19, 0.00000000e+00] *255
b = [2.40126823e-19, 1.30718954e-03, 2.61437908e-03, 3.92156863e-03]*255

print(r,g, b)
def rgb2hex(r, g, b):
    return '#%02x%02x%02x' % (r, g, b)
x =np.linspace(0,1,10)
y =x
color = rgb2hex( round(r[3]), round(g[3]), round(b[3]))
plt.plot(y, color=color)
plt.show()