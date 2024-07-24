import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

tau = 0.1
T = 3000
t = np.arange(0, T, tau)
n_steps = t.shape[0]

n_trials_acquisition = 20
n_trials_extinction = 20
n_trials_reacquisition = 20
n_trials = n_trials_acquisition + n_trials_extinction + n_trials_reacquisition

n_simulations = 1

alpha_d1 = 1e-15
beta_d1 = 1e-15

alpha_d2 = 1e-14
beta_d2 = 2e-14

alpha_pr = 0.05

psp_amp = 5e5
psp_decay = 400

resp_thresh = 5e5
motor_act_rec = np.zeros((n_simulations, n_trials))

# # striatal projection neuron
# C = 50; vr = -80; vt = -25; vpeak = 40;
# a = 0.01; b = -20; c = -55; d = 150; k = 1;

# # regular spiking neuron
# C = 100; vr = -60; vt = -40; vpeak = 35;
# a = 0.03; b = -2; c = -50; d = 100; k = 0.7;

iz_params = np.array([
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # regular spiking (ctx)
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # regular spiking (d1)
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # regular spiking (d2)
])

n_cells = iz_params.shape[0]

I_net = np.zeros((n_cells, n_steps))

w = np.zeros((n_cells, n_cells))

w_min = 0.1
w_max = 1.0

w[0, 1] = w_min  # ctx -> d1
w[0, 2] = w_min  # ctx -> d2
w[2, 1] = -0.375  # d2 -> d1

I_in = np.zeros(n_steps)
I_in[n_steps // 3:2 * n_steps // 3] = 3e2
w_in = np.zeros(n_cells)
w_in[0] = 0.6

v = np.zeros((n_cells, n_steps))
u = np.zeros((n_cells, n_steps))
g = np.zeros((n_cells, n_steps))
spike = np.zeros((n_cells, n_steps))
v[:, 0] = iz_params[:, 1]

obtained_reward = np.zeros((n_simulations, n_trials))
predicted_reward = np.zeros((n_simulations, n_trials))
delta = np.zeros((n_simulations, n_trials))
response = np.zeros((n_simulations, n_trials))
reward = np.zeros((n_simulations, n_trials))

w_rec_d1 = np.zeros((n_simulations, n_trials))
w_rec_d2 = np.zeros((n_simulations, n_trials))
w_rec_d1[0] = w[0, 1]
w_rec_d2[0] = w[0, 2]

for sim in range(n_simulations):

    for trl in range(1, n_trials):

        print(sim, trl)

        v[:] = 0
        g[:] = 0
        spike[:] = 0
        v[:, 0] = iz_params[:, 1]

        for i in range(1, n_steps):

            dt = t[i] - t[i - 1]

            I_net[:] = 0
            for jj in range(n_cells):
                for kk in range(n_cells):
                    if jj != kk:
                        I_net[jj, i - 1] += w[kk, jj] * g[kk, i - 1]

                I_net[jj, i - 1] += w_in[jj] * I_in[i - 1]

                C = iz_params[jj, 0]
                vr = iz_params[jj, 1]
                vt = iz_params[jj, 2]
                vpeak = iz_params[jj, 3]
                a = iz_params[jj, 4]
                b = iz_params[jj, 5]
                c = iz_params[jj, 6]
                d = iz_params[jj, 7]
                k = iz_params[jj, 8]

                dvdt = (k * (v[jj, i - 1] - vr) * (v[jj, i - 1] - vt) -
                        u[jj, i - 1] + I_net[jj, i - 1]) / C
                dudt = a * (b * (v[jj, i - 1] - vr) - u[jj, i - 1])
                dgdt = (-g[jj, i - 1] + psp_amp * spike[jj, i - 1]) / psp_decay

                v[jj, i] = v[jj, i - 1] + dvdt * dt
                u[jj, i] = u[jj, i - 1] + dudt * dt
                g[jj, i] = g[jj, i - 1] + dgdt * dt

                if v[jj, i] >= vpeak:
                    v[jj, i - 1] = vpeak
                    v[jj, i] = c
                    u[jj, i] = u[jj, i] + d
                    spike[jj, i] = 1

        motor_act_rec[sim, trl] = g[1, :].sum()

        if g[1, :].sum() > resp_thresh:
            response[sim, trl] = 1
        elif np.random.rand() < 0.3:
            response[sim, trl] = 1
        else:
            response[sim, trl] = 0

        if trl < n_trials_acquisition:
            if response[sim, trl] == 1:
                if np.random.rand() < 1:
                    obtained_reward[sim, trl] = 1

        elif trl >= n_trials_acquisition and trl < n_trials_acquisition + n_trials_extinction:
            obtained_reward[sim, trl] = 0

        elif trl >= n_trials_acquisition + n_trials_extinction:
            if response[sim, trl] == 1:
                if np.random.rand() < 1:
                    obtained_reward[sim, trl] = 1

        predicted_reward[sim, trl] = predicted_reward[
            sim, trl - 1] + alpha_pr * delta[sim, trl - 1]
        delta[sim,
              trl] = obtained_reward[sim, trl] - predicted_reward[sim, trl]

        pre = g[0, :].sum()
        post_d1 = g[1, :].sum()
        post_d2 = g[2, :].sum()
        if delta[sim, trl] > 0:
            w[0,
              1] += alpha_d1 * pre * post_d1 * delta[sim,
                                                     trl] * (w_max - w[0, 1])
            w[0, 2] -= beta_d2 * pre * post_d2 * delta[sim, trl] * w_min
        else:
            w[0, 1] += beta_d1 * pre * post_d1 * delta[sim, trl] * w_min
            w[0,
              2] -= alpha_d2 * pre * post_d2 * delta[sim,
                                                     trl] * (w_max - w[0, 2])

        w[0, 1] = np.clip(w[0, 1], w_min, w_max)
        w[0, 2] = np.clip(w[0, 2], w_min, w_max)

        w_rec_d1[sim, trl] = w[0, 1]
        w_rec_d2[sim, trl] = w[0, 2]

# NOTE: plot the results
fig, ax = plt.subplots(4, 2, squeeze=False, figsize=(12, 7))

ax[0, 0].plot(t, I_in)
ax[0, 0].set_title('Input signal')

ax1 = ax[1, 0]
ax2 = ax1.twinx()
ax1.plot(t, v[0, :], 'C0')
ax2.plot(t, g[0, :], 'C1')
ax1.set_title('Regular spiking neuron')

ax1 = ax[2, 0]
ax2 = ax1.twinx()
ax1.plot(t, v[1, :], 'C0')
ax2.plot(t, g[1, :], 'C1')
ax1.set_title('Striatal projection neuron (d1)')

ax1 = ax[3, 0]
ax2 = ax1.twinx()
ax1.plot(t, v[2, :], 'C0')
ax2.plot(t, g[2, :], 'C1')
ax1.set_title('Striatal projection neuron (d2)')

tt = np.arange(0, n_trials - 1, 1)

ax[0, 1].plot(tt, obtained_reward.mean(0)[tt], label='obtained reward')
ax[0, 1].plot(tt, predicted_reward.mean(0)[tt], label='predicted reward')
ax[0, 1].plot(tt, delta.mean(0)[tt], label='delta')
ax[0, 1].axvline(n_trials_acquisition - 1, color='k', linestyle='--')
ax[0, 1].axvline(n_trials_acquisition + n_trials_extinction - 1,
                 color='k',
                 linestyle='--')
ax[0, 1].set_ylim(-1.1, 1.1)
ax[0, 1].legend()

ax[1, 1].plot(tt, w_rec_d1.mean(0)[:-1], label='d1')
ax[1, 1].plot(tt, w_rec_d2.mean(0)[:-1], label='d2')
ax[1, 1].set_ylim(-0.1, 1.1)
ax[1, 1].set_title('Connection weight')
ax[1, 1].axvline(n_trials // 3, color='k', linestyle='--')
ax[1, 1].axvline(n_trials_acquisition + n_trials_extinction - 1,
                 color='k',
                 linestyle='--')
ax[1, 1].legend()

ax[2, 1].plot(tt, response.mean(0)[:-1])
ax[2, 1].axvline(n_trials_acquisition, color='k', linestyle='--')
ax[2, 1].axvline(n_trials_acquisition + n_trials_extinction - 1,
                 color='k',
                 linestyle='--')
ax[2, 1].set_title('Response')

ax[3, 1].plot(tt, motor_act_rec.mean(0)[:-1])
ax[3, 1].axhline(resp_thresh, color='k', linestyle='--')
ax[3, 1].axvline(n_trials_acquisition, color='k', linestyle='--')
ax[3, 1].axvline(n_trials_acquisition + n_trials_extinction - 1,
                 color='k',
                 linestyle='--')
ax[3, 1].set_title('Motor thresh')

[x.set_xticks(tt[::2]) for x in ax[:, 1].flatten()]
[x.set_xticklabels(tt[::2]) for x in ax[:, 1].flatten()]
plt.tight_layout()
plt.show()
