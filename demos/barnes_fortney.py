import emcee
import pocky
import numpy as np
import greenlantern
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nordplotlib.png import install, fgcolor; install()

G = 6.67e-8
RJup = 6.995e9
MSun = 1.988e33
RSun = 6.956e10
hr = 3600.
day = 24 * hr

Ms = 1.1 * MSun
Rs = 1.146 * RSun
Rp = 1.347 * RJup
inc = 86.68
u1_plus_u2 = 0.640
u1_minus_u2 = 0 #-0.055

Porb = 3.52474 * day
Omega = 2 * np.pi / Porb
d = np.cbrt(G * Ms / Omega**2)
f = 0.1
sigma = 1e-4

ctx = pocky.Context.default()
ctx1 = greenlantern.Context(ctx)

nt = 1000
time = np.linspace(-2 * hr, 2 * hr, nt)
time = pocky.BufferPair(ctx, time.astype(np.float32))
time.copy_to_device()
time.dirty = False

height_ratios = [1] * 3 + [0.33]
fig1, axs = plt.subplots(nrows=4, ncols=1, figsize=(7, 8),
    height_ratios=height_ratios, layout='constrained')
ax1, ax2, ax3, ax7 = axs

height_ratios = [1] * 3 + [0.25]
fig2, axs = plt.subplots(nrows=4, ncols=1, figsize=(7, 7),
    height_ratios=height_ratios, layout='constrained')
ax4, ax5, ax6, ax8 = axs

def spherical_model(theta, time):
    r, zs, beta, c1 = theta

    u1 = 0.5 * (c1 + u1_minus_u2)
    u2 = 0.5 * (c1 - u1_minus_u2)

    q1 = (u1 + u2)**2
    q2 = u1 / (2 * (u1 + u2)) if u1 > 0 else 0.

    params = np.array([[r, r, r, zs, 0., beta, 0., q1, q2, Porb]], dtype=np.float32)
    params = pocky.BufferPair(ctx, params)

    return ctx1.ellipsoid_transit_flux(time, params)

def log_prior(theta):
    r, zs, beta, q1 = theta
    if r < 0 or r > 1 or zs < 1 or beta < -0.5 * np.pi or beta > 0.5 * np.pi \
            or q1 < 0 or q1 > 1:
        return -np.inf
    return 0.

def log_likelihood(theta, time, flux, sigma):
    model = spherical_model(theta, time)
    return -0.5 * np.sum((flux.host - model.host)**2 / sigma**2) + log_prior(theta)

for ax, impact, gam, color, fit in zip([ax1, ax1, ax1, ax2, ax3, ax3, ax3, ax4, ax4, ax4, ax4, ax5, ax5, ax5, ax5, ax6, ax6, ax6, ax6],
                                [0., 0.3, 0.6, 0.7, 0.8, 0.9, 1., 0.35, 0.35, 0.35, 0.35, 0.7, 0.7, 0.7, 0.7, 0.83, 0.83, 0.83, 0.83],
                                [0., 0., 0., 0., 0., 0., 0., 0., 15., 30., 45., 0., 15., 30., 45., 0., 15., 30., 45.],
                                ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C0', 'C2', 'C4', 'C6', 'C0', 'C2', 'C4', 'C6', 'C0', 'C2', 'C4', 'C6'],
                                [False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True]):
    inc = np.arccos(impact / (d / Rs))
    inc = np.rad2deg(inc)

    r = Rp / Rs
    a = r * np.sqrt(1 - f)
    b = c = r / np.sqrt(1 - f)
    zs = d / Rs
    t0 = 0.
    beta = np.deg2rad(90 - inc)
    gamma = np.deg2rad(gam)
    u1 = 0.5 * (u1_plus_u2 + u1_minus_u2)
    u2 = 0.5 * (u1_plus_u2 - u1_minus_u2)

    q1 = (u1 + u2)**2
    q2 = u1 / (2 * (u1 + u2)) if u1 > 0 else 0.

    if fit:
        params = np.array([[a, b, c, zs, t0, beta, gamma, q1, q2, Porb]], dtype=np.float32)
        params = pocky.BufferPair(ctx, params)

        flux = np.empty((params.host.shape[0], nt), dtype=np.float32)
        flux = pocky.BufferPair(ctx, flux)

        ctx1.ellipsoid_transit_flux(time, params, output=flux)

        p0 = [r, zs, beta, u1_plus_u2]
        nwalkers, ndim = 50, len(p0)
        p0 += 1e-3 * np.random.normal(size=(nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim,
            log_likelihood, args=(time, flux, sigma))
        sampler.run_mcmc(p0, 1000, progress=True, progress_kwargs=dict(ascii=True))

        logprob = sampler.get_log_prob(flat=True)
        chain = sampler.get_chain(flat=True)
        popt = chain[np.argmax(logprob),:]
        best_flux = spherical_model(popt, time)

        ax.plot(time.host / hr, flux.host[0] - best_flux.host[0], c=color)
    else:
        params = np.array([[a, b, c, zs, t0, beta, gamma, q1, q2, Porb],
                           [r, r, r, zs, t0, beta, gamma, q1, q2, Porb]], dtype=np.float32)
        params = pocky.BufferPair(ctx, params)

        flux = np.empty((params.host.shape[0], nt), dtype=np.float32)
        flux = pocky.BufferPair(ctx, flux)

        ctx1.ellipsoid_transit_flux(time, params, output=flux)

        ax.plot(time.host / hr, flux.host[0] - flux.host[1], c=color)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2e-4, 2e-4)

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.axhline(0, ls='dotted', lw=1, c=fgcolor)
    ax.axvline(0, ls='dotted', lw=1, c=fgcolor)

for ax in [ax1, ax2, ax4, ax5]:
    ax.set_xticklabels([])

for ax in [ax3, ax6]:
    ax.set_xlabel('Time (hr)')

fig1.supylabel(r'$f_\mathrm{oblate} - f_\mathrm{spherical}$', fontsize=20)
fig2.supylabel(r'$f_\mathrm{oblate} - f_\mathrm{best~fit}$', fontsize=20)

fig1_handles = [Line2D([0], [0], c='C0', lw=1.5, label=r'$b = 0$'),
                Line2D([0], [0], c='C1', lw=1.5, label=r'$b = 0.3$'),
                Line2D([0], [0], c='C2', lw=1.5, label=r'$b = 0.6$'),
                Line2D([0], [0], c='C3', lw=1.5, label=r'$b = 0.7$')]
fig1.legend(handles=fig1_handles, bbox_transform=ax7.transAxes,
    bbox_to_anchor=(0, 0.5, 1, 1), loc='lower center', ncols=4, frameon=False,
    columnspacing=1, borderpad=-0.5, handlelength=1, handletextpad=0.3)

fig1_handles = [Line2D([0], [0], c='C4', lw=1.5, label=r'$b = 0.8$'),
                Line2D([0], [0], c='C5', lw=1.5, label=r'$b = 0.9$'),
                Line2D([0], [0], c='C6', lw=1.5, label=r'$b = 1$')]
fig1.legend(handles=fig1_handles, bbox_transform=ax7.transAxes,
    bbox_to_anchor=(0, 0, 1, 0.5), loc='lower center', ncols=3, frameon=False,
    columnspacing=1, borderpad=-0.5, handlelength=1, handletextpad=0.3)

fig2_handles = [Line2D([0], [0], c='C0', lw=1.5, label=r'$\gamma = 0^\circ$'),
                Line2D([0], [0], c='C2', lw=1.5, label=r'$\gamma = 15^\circ$'),
                Line2D([0], [0], c='C4', lw=1.5, label=r'$\gamma = 30^\circ$'),
                Line2D([0], [0], c='C6', lw=1.5, label=r'$\gamma = 45^\circ$')]
fig2.legend(handles=fig2_handles, bbox_transform=ax8.transAxes,
    bbox_to_anchor=(0, 0, 1, 1), loc='lower center', ncols=4, frameon=False,
    columnspacing=1, borderpad=-0.5, handlelength=1, handletextpad=0.3)

ax7.set_axis_off()
ax8.set_axis_off()

fig1.savefig('assets/barnes_fortney_fig4.png')
fig2.savefig('assets/barnes_fortney_fig9.png')
plt.show()

# vim: set ft=python:
