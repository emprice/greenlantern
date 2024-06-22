import time
import pocky
import batman
import numpy as np
import greenlantern
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nordplotlib.png import install; install()

Porb = 2 * np.pi
Omega = 2 * np.pi / Porb
alpha0, gamma = 0., 0.

ctx = pocky.Context.default()
ctx1 = greenlantern.Context(ctx)

nalpha = 10000
alpha = np.linspace(-0.3, 0.3, nalpha)
alpha_dev = pocky.BufferPair(ctx, alpha.astype(np.float32))
alpha_dev.copy_to_device()
alpha_dev.dirty = False

flux = np.empty((1, nalpha), dtype=np.float32)
flux = pocky.BufferPair(ctx, flux)

fig, axs = plt.subplots(nrows=3, figsize=(8, 8.5),
    layout='constrained', height_ratios=[1.5, 1, 0.5])

handles = list()

for i, (rr, zs, beta_deg, u1, u2) in \
        enumerate(zip([0.1, 0.05, 0.07, 0.09], [5., 10., 20., 3.], [10., 5., -2., -15.],
                      [0.1, 0.3, 0.4, 0.05], [-0.02, 0.05, 0.2, -0.04])):
    beta = np.deg2rad(beta_deg)

    q1 = (u1 + u2)**2
    q2 = u1 / (2 * (u1 + u2)) if u1 > 0 else 0.

    batp = batman.TransitParams()
    batp.t0 = alpha0
    batp.per = Porb
    batp.rp = rr
    batp.a = zs
    batp.inc = 90. - beta_deg
    batp.ecc = 0.
    batp.w = 90.
    batp.u = [u1, u2]
    batp.limb_dark = 'quadratic'

    model = batman.TransitModel(batp, alpha)
    model_flux = model.light_curve(batp)

    params = np.array([[rr, rr, rr, zs, alpha0, beta, gamma, q1, q2]], dtype=np.float32)
    params = pocky.BufferPair(ctx, params)
    ctx1.ellipsoid_transit_flux(alpha_dev, params, output=flux)

    axs[0].plot(alpha, model_flux, lw=2, ls='solid', alpha=0.6, c=f'C{i}')
    axs[0].plot(alpha, flux.host[0], lw=2, ls='dashed', alpha=1, c=f'C{i}')

    axs[1].scatter(alpha, 1e6 * (flux.host[0] - model_flux),
        alpha=0.05, s=8, c=f'C{i}', rasterized=True)

    label = rf'$R_p / R_\star = {rr:.2f}, d_\star = {zs:.0f}, i = {90.-beta_deg:.0f}^\circ, u_1 = {u1:.2f}, u_2 = {u2:.2f}$'
    handles.append(Line2D([0], [0], c=f'C{i}', lw=2, label=label))

axs[1].set_xlabel(r'Mean anomaly $\alpha$ (rad)')
axs[0].set_xticklabels([])
axs[0].set_xlim(axs[1].get_xlim())

axs[0].set_ylabel('Relative flux')
axs[1].set_ylabel('Residual (ppm)')

fig.legend(handles=handles, bbox_transform=axs[2].transAxes,
    bbox_to_anchor=(0, 0, 1, 1), loc='lower left', ncols=1, frameon=False,
    columnspacing=1, borderpad=-0.5, handlelength=1.5, handletextpad=0.5, fontsize=16)
axs[2].set_axis_off()

fig.align_ylabels()
plt.show()
