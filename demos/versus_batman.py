import time
import pocky
import batman
import numpy as np
import greenlantern
import scipy.optimize as opt
import matplotlib.pyplot as plt
from nordplotlib.png import install; install()

rr, semimajor, tc, beta, zeta, eta, xi, u1, u2, Porb = \
    0.1, 5., 0.01, -0.02, 0., 0., 0., 0.1, -0.03, 2 * np.pi
test_eccentric = False

if test_eccentric:
    ecc, omega = 0.3, 0.2
else:
    ecc, omega = 0., -0.5 * np.pi

q1 = (u1 + u2)**2
q2 = u1 / (2 * (u1 + u2)) if u1 > 0 else 0.

batp = batman.TransitParams()
batp.t0 = tc
batp.per = Porb
batp.rp = rr
batp.a = semimajor
batp.inc = 90. - np.rad2deg(beta)
batp.ecc = ecc
batp.w = np.rad2deg(omega)
batp.u = [u1, u2]
batp.limb_dark = 'quadratic'

ctx = pocky.Context.default()
ctx1 = greenlantern.Context(ctx)

ntries = 10

def generate_timings():
    for log2_nt in np.linspace(10, 18, 50):
        nt = (2.**log2_nt).astype(int)
        time_host = np.linspace(-0.3, 0.3, nt)

        model = batman.TransitModel(batp, time_host, fac=1e-4)

        dt_batman = 0
        for _ in range(ntries):
            t0 = time.time()
            model_flux = model.light_curve(batp)
            t1 = time.time()
            dt_batman += (t1 - t0) / model_flux.size
        dt_batman /= ntries

        time_dev = pocky.BufferPair(ctx, time_host.astype(np.float32))
        time_dev.copy_to_device()
        time_dev.dirty = False

        flux = np.empty((1, nt), dtype=np.float32)
        flux = pocky.BufferPair(ctx, flux)

        if test_eccentric:
            params = np.array([[rr, rr, rr, semimajor, tc, beta,
                zeta, eta, xi, q1, q2, Porb, ecc, omega]], dtype=np.float32)
            params = pocky.BufferPair(ctx, params)
        else:
            params = np.array([[rr, rr, rr, semimajor, tc, beta,
                zeta, eta, xi, q1, q2, Porb]], dtype=np.float32)
            params = pocky.BufferPair(ctx, params)

        dt_greenlantern = 0
        for _ in range(ntries):
            t0 = time.time()
            ctx1.ellipsoid_transit_flux(time_dev, params,
                flux=flux, eccentric=test_eccentric)
            t1 = time.time()
            dt_greenlantern += (t1 - t0) / flux.host.size
        dt_greenlantern /= ntries

        abserr = np.absolute(np.amax(flux.host[0] - model_flux))

        yield nt, dt_batman, dt_greenlantern, abserr

timings = np.array(list(generate_timings()))

fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')

ax.scatter(timings[:,0], timings[:,1] * 1e6, label=r'\texttt{batman}', color='C2')
ax.scatter(timings[:,0], timings[:,2] * 1e6, label=r'\texttt{greenlantern}', color='C6')

ax.legend(loc='upper right')
ax.set_xscale('log')
ax.set_xlabel('Number of samples')
ax.set_ylabel(r'Time per sample (\textmu s)')

fig.savefig('assets/timing_versus_batman.png')
plt.show()

print('Maximum absolute error:', np.amax(timings[:,3]))
