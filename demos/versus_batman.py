import time
import pocky
import batman
import numpy as np
import greenlantern
import scipy.optimize as opt
import matplotlib.pyplot as plt
from nordplotlib.png import install; install()

a, b, c, zs, alpha0, beta, gamma, u1, u2 = \
    0.1, 0.1, 0.1, 5., 0.01, -0.02, 0., 0.1, -0.02

Porb = 2 * np.pi
Omega = 2 * np.pi / Porb

q1 = (u1 + u2)**2
q2 = u1 / (2 * (u1 + u2)) if u1 > 0 else 0.

batp = batman.TransitParams()
batp.t0 = alpha0
batp.per = Porb
batp.rp = a
batp.a = zs
batp.inc = 90. - np.rad2deg(beta)
batp.ecc = 0.
batp.w = 90.
batp.u = [u1, u2]
batp.limb_dark = 'quadratic'

ctx = pocky.Context.default()
ctx1 = greenlantern.Context(ctx)

ntries = 10

def generate_timings():
    for log2_nalpha in np.linspace(10, 18, 50):
        nalpha = (2.**log2_nalpha).astype(int)
        alpha = np.linspace(-0.3, 0.3, nalpha)

        model = batman.TransitModel(batp, alpha)

        dt_batman = 0
        for _ in range(ntries):
            t0 = time.time()
            model_flux = model.light_curve(batp)
            t1 = time.time()
            dt_batman += (t1 - t0) / model_flux.size
        dt_batman /= ntries

        alpha = pocky.BufferPair(ctx, alpha.astype(np.float32))
        alpha.copy_to_device()
        alpha.dirty = False

        params = np.array([[a, a, a, zs, alpha0, beta, gamma, q1, q2]], dtype=np.float32)
        params = pocky.BufferPair(ctx, params)

        flux = np.empty((params.host.shape[0], nalpha), dtype=np.float32)
        flux = pocky.BufferPair(ctx, flux)

        dt_greenlantern = 0
        for _ in range(ntries):
            t0 = time.time()
            ctx1.ellipsoid_transit_flux(alpha, params, output=flux)
            t1 = time.time()
            dt_greenlantern += (t1 - t0) / flux.host.size
        dt_greenlantern /= ntries

        abserr = np.absolute(np.amax(flux.host[0] - model_flux))

        yield nalpha, dt_batman, dt_greenlantern, abserr

timings = np.array(list(generate_timings()))

fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
ax.scatter(timings[:,0], timings[:,1] * 1e6, label=r'\texttt{batman}', color='C2')
ax.scatter(timings[:,0], timings[:,2] * 1e6, label=r'\texttt{greenlantern}', color='C6')
ax.legend(loc='upper right')
ax.set_xscale('log')
ax.set_xlabel('Number of samples')
ax.set_ylabel(r'Time per sample (\textmu s)')
plt.show()

print('Maximum absolute error:', np.amax(timings[:,3]))
