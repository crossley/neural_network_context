import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

T = 1000
tau = 0.1
t = np.arange(0, T, tau)
N = t.shape[0]

n_cells = 1

w = np.zeros((n_cells, n_cells))

w[0, 0] = 0

iz_params = np.array([
    # TAN
    [100, -75, -45, 35, 1, 5, -55, 150, 1.2]
])

base = np.zeros((n_cells, ))
base[0] = 950

r = np.zeros((n_cells, N))
r_amp = 100
r_decay = 10

v = np.zeros((n_cells, N))
u = np.zeros((n_cells, N))
g = np.zeros((n_cells, N))
spike = np.zeros((n_cells, N))

v[:, 0] = -70
u[:, 0] = -15

psp_amp = 1e3
psp_decay = 100

I_external = np.zeros((n_cells, N))
I_external[0, int(N / 3):int(2 * N / 3)] = 1000

for i in range(1, N):

    dt = t[i] - t[i - 1]

    for j in range(n_cells):

        I = I_external[j, i - 1] + np.sum(w[:, j] * g[:, i - 1])

        C = iz_params[j, 0]
        vr = iz_params[j, 1]
        vt = iz_params[j, 2]
        vpeak = iz_params[j, 3]
        a = iz_params[j, 4]
        b = iz_params[j, 5]
        c = iz_params[j, 6]
        d = iz_params[j, 7]
        k = iz_params[j, 8]

        dvdt = (k * (v[j, i - 1] - vr) *
                (v[j, i - 1] - vt) - u[j, i - 1] + I + base[j]) / C
        dudt = a * (b * (v[j, i - 1] - vr) - u[j, i - 1] + 2.7 * r[j, i - 1])
        dgdt = (-g[j, i - 1] + psp_amp * spike[j, i - 1]) / psp_decay
        drdt = (-r[j, i - 1] + r_amp * I) / r_decay

        v[j, i] = v[j, i - 1] + dvdt * dt
        u[j, i] = u[j, i - 1] + dudt * dt
        g[j, i] = g[j, i - 1] + dgdt * dt
        r[j, i] = r[j, i - 1] + drdt * dt

        if v[j, i] >= vpeak:
            v[j, i - 1] = vpeak
            v[j, i] = c
            u[j, i] = u[j, i] + d
            spike[j, i] = 1

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(t, I_external[0, :], label='I_external')
ax[1].plot(t, r[0, :], label='r')
ax[2].plot(t, v[0, :], label='v')
[x.legend() for x in ax]
plt.show()

