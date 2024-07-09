import numpy as np
import pyfftw
import matplotlib.pyplot as plt
import math


def fps(f):
    aa = -2.0 / (dx * dx) - 2.0 / (dy * dy)
    bb = 2.0 / (dx * dx)
    cc = 2.0 / (dy * dy)
    data = np.empty((nx, ny), dtype="complex128")
    data1 = np.empty((nx, ny), dtype="complex128")
    for i in range(nx):
        for j in range(ny):
            data[i, j] = complex(f[i + 1, j + 1], 0.0)

    a = pyfftw.empty_aligned((nx, ny), dtype="complex128")
    b = pyfftw.empty_aligned((nx, ny), dtype="complex128")

    fft_object = pyfftw.FFTW(a, b, axes=(0, 1), direction="FFTW_FORWARD")
    fft_object_inv = pyfftw.FFTW(a, b, axes=(0, 1), direction="FFTW_BACKWARD")

    e = fft_object(data)
    e[0, 0] = 0.0
    for i in range(nx):
        for j in range(ny):
            data1[i,
                  j] = e[i, j] / (aa + bb * np.cos(kx[i]) + cc * np.cos(ky[j]))

    ut = np.real(fft_object_inv(data1))
    u = np.empty((nx + 3, ny + 3))
    u[1:nx + 1, 1:ny + 1] = ut
    u[:, ny + 1] = u[:, 1]
    u[nx + 1, :] = u[1, :]
    u[nx + 1, ny + 1] = u[1, 1]
    return u


def bc(u):
    u[:, 0] = u[:, ny]
    u[:, ny + 2] = u[:, 2]
    u[0, :] = u[nx, :]
    u[nx + 2, :] = u[2, :]

def rhs(nx, ny, dx, dy, re, w, s):
    aa = 1.0 / (re * dx * dx)
    bb = 1.0 / (re * dy * dy)
    gg = 1.0 / (4.0 * dx * dy)
    hh = 1.0 / 3.0
    f = np.empty((nx + 3, ny + 3))
    for i in range(1, nx + 2):
        for j in range(1, ny + 2):
            j1 = gg * ((w[i + 1, j] - w[i - 1, j]) *
                       (s[i, j + 1] - s[i, j - 1]) -
                       (w[i, j + 1] - w[i, j - 1]) *
                       (s[i + 1, j] - s[i - 1, j]))
            j2 = gg * (w[i + 1, j] *
                       (s[i + 1, j + 1] - s[i + 1, j - 1]) - w[i - 1, j] *
                       (s[i - 1, j + 1] - s[i - 1, j - 1]) - w[i, j + 1] *
                       (s[i + 1, j + 1] - s[i - 1, j + 1]) + w[i, j - 1] *
                       (s[i + 1, j - 1] - s[i - 1, j - 1]))
            j3 = gg * (w[i + 1, j + 1] *
                       (s[i, j + 1] - s[i + 1, j]) - w[i - 1, j - 1] *
                       (s[i - 1, j] - s[i, j - 1]) - w[i - 1, j + 1] *
                       (s[i, j + 1] - s[i - 1, j]) + w[i + 1, j - 1] *
                       (s[i + 1, j] - s[i, j - 1]))

            jac = (j1 + j2 + j3) * hh
            f[i, j] = -jac + aa * (w[i + 1, j] - 2.0 * w[i, j] + w[
                i - 1, j]) + bb * (w[i, j + 1] - 2.0 * w[i, j] + w[i, j - 1])

    return f


nt = 50
re = 1e3
dt = 1e-2
ns = 10
freq = int(nt / ns)
nx = ny = 32
lx = 2 * math.pi
ly = 2 * math.pi
dx = lx / np.float64(nx)
dy = ly / np.float64(ny)
time = 0.0
x = np.linspace(0.0, 2.0 * math.pi, nx + 1)
y = np.linspace(0.0, 2.0 * math.pi, ny + 1)
w = np.empty((nx + 3, ny + 3))
s = np.empty((nx + 3, ny + 3))
t = np.empty((nx + 3, ny + 3))

w = np.empty((nx + 3, ny + 3))
xc1 = math.pi - math.pi / 4.0
yc1 = math.pi
xc2 = math.pi + math.pi / 4.0
yc2 = math.pi
for i in range(1, nx + 2):
    for j in range(1, ny + 2):
        w[i, j] = np.exp(-math.pi *
                         ((x[i - 1] - xc1)**2 +
                          (y[j - 1] - yc1)**2)) + np.exp(-math.pi *
                                                         ((x[i - 1] - xc2)**2 +
                                                          (y[j - 1] - yc2)**2))
kx = np.empty(nx)
ky = np.empty(ny)
epsilon = 1.0e-6
hx = 2.0 * math.pi / np.float64(nx)
hy = 2.0 * math.pi / np.float64(ny)
for i in range(nx):
    kx[i] = hx * np.float64(i)
for i in range(ny):
    ky[i] = hy * np.float64(i)
kx[0] = epsilon
ky[0] = epsilon
        
bc(w)
w0 = np.copy(w)
s = fps(-w)
bc(s)
aa = 1.0 / 3.0
bb = 2.0 / 3.0
for k in range(nt):
    print(k)
    time += dt
    r = rhs(nx, ny, dx, dy, re, w, s)
    for i in range(1, nx + 2):
        for j in range(1, ny + 2):
            t[i, j] = w[i, j] + dt * r[i, j]
    bc(t)
    s = fps(-t)
    bc(s)
    r = rhs(nx, ny, dx, dy, re, t, s)
    for i in range(1, nx + 2):
        for j in range(1, ny + 2):
            t[i, j] = 0.75 * w[i, j] + 0.25 * t[i, j] + 0.25 * dt * r[i, j]
    bc(t)
    s = fps(-t)
    bc(s)
    r = rhs(nx, ny, dx, dy, re, t, s)
    for i in range(1, nx + 2):
        for j in range(1, ny + 2):
            w[i, j] = aa * w[i, j] + bb * t[i, j] + bb * dt * r[i, j]
    bc(w)
    s = fps(-w)
    bc(s)

for f, name in (w0, "0.png"), (w, "1.png"):
    plt.axis("equal")
    plt.imshow(f[1:nx + 2, 1:ny + 2].T, origin='lower', cmap='jet')
    plt.savefig(name, bbox_inches="tight")
    plt.close()
