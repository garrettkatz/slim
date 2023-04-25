import matplotlib.pyplot as pt
import numpy as np

ax = pt.figure().add_subplot(projection='3d')

# make circle patch
npts = 16
rad = 1
x = rad * np.cos(np.linspace(0, 2*np.pi, npts))
y = rad * np.sin(np.linspace(0, 2*np.pi, npts))
x, y = np.insert(x, 0, 0), np.insert(y, 0, 0)
z = np.zeros(npts+1)

tri = np.arange(npts)
tri = np.hstack((tri[:-1], tri[1:], np.zeros(npts)))

a = np.pi/3
M = np.array([[np.cos(a), 0, np.sin(a)],[0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
x, y, z = M @ np.vstack((x, y, z))
ax.plot_trisurf(x, y, z, triangles=tri, color=(.5, .5, 1), linewidth=0., antialiased=False)

a = np.pi/3
M = np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
x, y, z = M @ np.vstack((x, y, z))
ax.plot_trisurf(x, y, z, triangles=tri, color=(.5, .5, 1), linewidth=0., antialiased=False)

# n_radii = 8
# n_angles = 36

# # Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
# radii = np.linspace(0.125, 1.0, n_radii)
# angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]

# # Convert polar (radii, angles) coords to cartesian (x, y) coords.
# # (0, 0) is manually added at this stage,  so there will be no duplicate
# # points in the (x, y) plane.
# x = np.append(0, (radii*np.cos(angles)).flatten())
# y = np.append(0, (radii*np.sin(angles)).flatten())

# # Compute z to make the pringle surface.
# z = np.sin(-x*y)

# ax = pt.figure().add_subplot(projection='3d')

# # ax.plot_trisurf(x, y, z, edgecolor=np.zeros((x.size, 3)), facecolor=np.zeros((x.size, 3)), linewidth=0., antialiased=True)
# ax.plot_trisurf(x, y, z, color=(.5, .5, 1), linewidth=0., antialiased=False)

pt.show()


# # # Create the data.
# # from numpy import pi, sin, cos, mgrid
# # dphi, dtheta = pi/250.0, pi/250.0
# # [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
# # m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
# # r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
# # x = r*sin(phi)*cos(theta)
# # y = r*cos(phi)
# # z = r*sin(phi)*sin(theta)

# # # View it.
# # from mayavi import mlab
# # s = mlab.mesh(x, y, z)
# # mlab.show()

# from mayavi import mlab
# import numpy as np
# import matplotlib.pyplot as pt

# def arrow(x, y, z, u, v, w):
#     a = mlab.plot3d([x, u], [y, v], [z, w], color=(0,0,0), tube_radius=0.2)
#     return a

# a = arrow(0, 0, 0, 1, 0, 0)
# # x = y = z = np.zeros(3)
# # u, v, w = np.eye(3)
# # mlab.quiver3d(x, y, z, u, v, w, color=(0,0,0))
# mlab.savefig('tmp.png')
# mlab.show()

# # fig = mlab.figure(size=(480, 340))
# # mlab.quiver3d(x, y, z, u, v, w, color=(0,0,0))
# # img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
# # img = mlab.screenshot()
# # mlab.close(fig)

# # pt.imshow(img)
# # pt.show()

