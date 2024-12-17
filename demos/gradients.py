import time
import pocky
import numpy as np
import greenlantern
import matplotlib.pyplot as plt
from nordplotlib.png import install; install()

a, b, c, ds, tc, beta, zeta, eta, xi, u1, u2, Porb = \
    0.1, 0.2, 0.3, 5., 0., 0.01, 0.3, 0.2, 0.1, 0.1, -0.02, 2 * np.pi

q1 = (u1 + u2)**2
q2 = u1 / (2 * (u1 + u2)) if u1 > 0 else 0.

ctx = pocky.Context.default()
ctx1 = greenlantern.Context(ctx)

nt = 100
time = np.linspace(-0.3, 0.3, nt)
time = pocky.BufferPair(ctx, time.astype(np.float32))
time.copy_to_device()
time.dirty = False

params = np.array([[a, b, c, ds, tc, beta, zeta, eta, xi, q1, q2, Porb]], dtype=np.float32)
params = pocky.BufferPair(ctx, params)

flux = np.empty((nt,), dtype=np.float32)
flux = pocky.BufferPair(ctx, flux)

dflux = np.empty((12, nt), dtype=np.float32)
dflux = pocky.BufferPair(ctx, dflux)

ctx1.ellipsoid_transit_flux_dual(time, params, flux=flux, dflux=dflux, binsize=0.01*Porb)
plt.plot(time.host, dflux.host.T, lw=2)

ctx1.ellipsoid_transit_flux_dual(time, params, flux=flux, dflux=dflux)
plt.plot(time.host, dflux.host.T, lw=1, ls='dashed')

plt.show()
