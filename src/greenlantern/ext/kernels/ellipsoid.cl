/// Number of integration intervals
#define NMAX        (2048)

#define DTHETA      (2 * M_PI / NMAX)

float3 compute_integrand(float2 theta, float x0, float dx,
    float y0, float dy1, float dy2)
{
    float x = x0 + dx * theta.x;
    float y = y0 + dy1 * theta.x + dy2 * theta.y;
    float x2 = x * x, y2 = y * y;

    float dx_dtheta = -dx * theta.y;
    float dy_dtheta = -dy1 * theta.y + dy2 * theta.x;

    if (x2 + y2 > 1)
    {
        float dxi_dtheta = (-y * dx_dtheta + x * dy_dtheta) / (x2 + y2);

        float xi = atan2(y, x);
        y = sincos(xi, &x);

        dx_dtheta = -y * dxi_dtheta;
        dy_dtheta =  x * dxi_dtheta;

        x2 = x * x;
        y2 = y * y;
    }

    float flat = 0.5 * (x * dy_dtheta - y * dx_dtheta);
    float linear = 2 * (1 - pow(fmax(0, 1 - x2 - y2), 1.5)) /
        (3 * (x2 + y2)) * flat;
    float poly = 0.5 * (x * (y2 + x2 / 3) * dy_dtheta +
        -y * (x2 + y2 / 3) * dx_dtheta);
    return (float3)(flat, linear, poly);
}

float ellipsoid_transit_flux_workgroup_body(int gid, int gsz,
    float2 alpha, float2 beta, float2 zeta, float2 eta, float2 xi,
    float3 ax, float ds, float u1, float u2)
{
    float r1 = eta.x * xi.x;
    float r2 = -eta.x * xi.y;
    float r3 = eta.y;

    float r4 = zeta.y * eta.y * xi.x + zeta.x * xi.y;
    float r5 = zeta.x * xi.x - zeta.y * eta.y * xi.y;
    float r6 = -zeta.y * eta.x;

    float r7 = -zeta.x * eta.y * xi.x + zeta.y * xi.y;
    float r8 = zeta.y * xi.x + zeta.x * eta.y * xi.y;
    float r9 = zeta.x * eta.x;

    float r5748 = -eta.y;
    float r6749 = -eta.x * xi.y;
    float r6859 = -eta.x * xi.x;

    float r2718 = -zeta.y * eta.x;
    float r3719 = -zeta.x * xi.x + zeta.y * eta.y * xi.y;
    float r3829 = zeta.y * eta.y * xi.x + zeta.x * xi.y;

    float r2415 = -zeta.x * eta.x;
    float r3416 = zeta.y * xi.x + zeta.x * eta.y * xi.y;
    float r3526 = zeta.x * eta.y * xi.x - zeta.y * xi.y;

    float afac1 = ax.x * (beta.y * (alpha.x * r5748 + alpha.y * r6749) - beta.x * r6859);
    float bfac1 = ax.y * (beta.y * (alpha.x * r2718 + alpha.y * r3719) - beta.x * r3829);
    float cfac1 = ax.z * (beta.y * (alpha.x * r2415 + alpha.y * r3416) - beta.x * r3526);

    float afac2 = ax.x * (alpha.x * r6749 - alpha.y * r5748);
    float bfac2 = ax.y * (alpha.x * r3719 - alpha.y * r2718);
    float cfac2 = ax.z * (alpha.x * r3416 - alpha.y * r2415);

    float abfac = ax.x * ax.y * (beta.y * r7 + beta.x * (alpha.x * r9 - alpha.y * r8));
    float acfac = ax.x * ax.z * (beta.y * r4 + beta.x * (alpha.x * r6 - alpha.y * r5));
    float bcfac = ax.y * ax.z * (beta.y * r1 + beta.x * (alpha.x * r3 - alpha.y * r2));

    float dx = sqrt(afac1 * afac1 + bfac1 * bfac1 + cfac1 * cfac1);

    float dy1 = (afac1 * afac2 + bfac1 * bfac2 + cfac1 * cfac2) / dx;
    float dy2 = sqrt(abfac * abfac + acfac * acfac + bcfac * bcfac) / dx;

    float x0 = -ds * alpha.x * beta.y;
    float y0 = ds * alpha.y;

    float3 Ival = 0;

    for (int i = gid; i <= NMAX; i += gsz)
    {
        float theta_ang = i * DTHETA;

        float cos_theta, sin_theta;
        sin_theta = sincos(theta_ang, &cos_theta);
        float2 theta = (float2)(cos_theta, sin_theta);

        float3 fval = compute_integrand(theta, x0, dx, y0, dy1, dy2);
        fval *= ((i == 0) || (i == NMAX)) ? 1 : ((i & 0x1) ? 4 : 2);
        Ival += fval;
    }

    float integrand = ((1 - u1 - 2 * u2) * Ival.x +
        (u1 + 2 * u2) * Ival.y + u2 * Ival.z) * DTHETA / 3;
    float norm = 6. / (6 - 2 * u1 - u2);
    return integrand * norm;
}

#define LDA     (12)

kernel void ellipsoid_transit_flux_vector(global const float *time,
    global const float *params, global float *flux)
{
    int pid = get_global_id(0);      /* parameters index */
    int psz = get_global_size(0);    /* number of parameter sets */

    int sid = get_global_id(1);      /* sample index */
    int ssz = get_global_size(1);    /* number of samples */

    int gid = get_local_id(0);       /* summation group index */
    int gsz = get_local_size(0);     /* number of summation groups */

    pid /= gsz;

    local float local_params[LDA];
    local float u1, u2, ds;
    local float2 alpha, beta, zeta, eta, xi;
    local float3 ax;

    /* pre-load the parameters for this work group */
    for (int off = gid; off < LDA; off += gsz)
        local_params[off] = params[pid*LDA+off];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (gid == 0)
    {
        float a = local_params[0];   /* semiaxis along x */
        float b = local_params[1];   /* semiaxis along y */
        float c = local_params[2];   /* semiaxis along z */
        ax = (float3)(a, b, c);

        ds = local_params[3];           /* distance from ellipsoid to disk */
        float porb = local_params[11];  /* orbital period */
        float t0 = local_params[4];     /* midtransit offset in time */

        float t = time[sid];
        float this_alpha = 2 * M_PI * (t / porb);
        float alpha0 = 2 * M_PI * (t0 / porb);

        {
            /* precompute trig for alpha */
            float alpha_ang = this_alpha - alpha0;
            float sin_alpha, cos_alpha;
            sin_alpha = sincos(alpha_ang, &cos_alpha);
            alpha = (float2)(cos_alpha, sin_alpha);
        }

        {
            /* precompute trig for beta */
            float beta_ang = local_params[5];   /* complement of inclination */
            float sin_beta, cos_beta;
            sin_beta = sincos(beta_ang, &cos_beta);
            beta = (float2)(cos_beta, sin_beta);
        }

        {
            /* precompute trig for zeta */
            float zeta_ang = local_params[6];   /* orientation angle 1 */
            float sin_zeta, cos_zeta;
            sin_zeta = sincos(zeta_ang, &cos_zeta);
            zeta = (float2)(cos_zeta, sin_zeta);
        }

        {
            /* precompute trig for eta */
            float eta_ang = local_params[7];    /* orientation angle 2 */
            float sin_eta, cos_eta;
            sin_eta = sincos(eta_ang, &cos_eta);
            eta = (float2)(cos_eta, sin_eta);
        }

        {
            /* precompute trig for xi */
            float xi_ang = local_params[8];     /* orientation angle 3 */
            float sin_xi, cos_xi;
            sin_xi = sincos(xi_ang, &cos_xi);
            xi = (float2)(cos_xi, sin_xi);
        }

        /* convert q limb darkening to u limb darkening */
        float q1 = local_params[9];     /* limb darkening q1 */
        float q2 = local_params[10];    /* limb darkening q2 */
        float sqrt_q1 = sqrt(q1);
        u1 = 2 * sqrt_q1 * q2;
        u2 = sqrt_q1 * (1 - 2 * q2);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    float Ival = ellipsoid_transit_flux_workgroup_body(gid, gsz,
        alpha, beta, zeta, eta, xi, ax, ds, u1, u2);
    float Ival_tot = work_group_reduce_add(Ival);

    if (gid == 0) flux[pid*ssz+sid] = (alpha.x > 0) ? (1. - M_1_PI * Ival_tot) : 1.;
}

kernel void ellipsoid_transit_flux_binned_vector(global const float *time,
    global const float *params, float dt_bin, global float *flux)
{
    int pid = get_global_id(0);     /* parameters index */
    int psz = get_global_size(0);   /* number of parameter sets */

    int sid = get_global_id(1);     /* sample index */
    int ssz = get_global_size(1);   /* number of samples */

    int gid1 = get_local_id(0);     /* workgroup dim 0, summation */
    int gsz1 = get_local_size(0);

    int gid2 = get_local_id(1);     /* workgroup dim 1, binning */
    int gsz2 = get_local_size(1);

    pid /= gsz1; psz /= gsz1;
    sid /= gsz2; ssz /= gsz2;

    local float local_params[LDA];
    local float u1, u2, ds, alpha_ang_mid, dalpha_bin;
    local float2 beta, zeta, eta, xi;
    local float3 ax;

    /* pre-load the parameters for this work group */
    for (int off = gid1; (gid2 == 0) && (off < LDA); off += gsz1)
        local_params[off] = params[pid*LDA+off];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if ((gid1 == 0) && (gid2 == 0))
    {
        float a = local_params[0];   /* semiaxis along x */
        float b = local_params[1];   /* semiaxis along y */
        float c = local_params[2];   /* semiaxis along z */
        ax = (float3)(a, b, c);

        ds = local_params[3];           /* distance from ellipsoid to disk */
        float porb = local_params[11];  /* orbital period */
        float t0 = local_params[4];     /* midtransit offset in time */

        float t = time[sid];
        float this_alpha = 2 * M_PI * (t / porb);
        float alpha0 = 2 * M_PI * (t0 / porb);
        dalpha_bin = 2 * M_PI * (dt_bin / porb);

        alpha_ang_mid = this_alpha - alpha0;

        {
            /* precompute trig for beta */
            float beta_ang = local_params[5];   /* complement of inclination */
            float sin_beta, cos_beta;
            sin_beta = sincos(beta_ang, &cos_beta);
            beta = (float2)(cos_beta, sin_beta);
        }

        {
            /* precompute trig for zeta */
            float zeta_ang = local_params[6];   /* orientation angle 1 */
            float sin_zeta, cos_zeta;
            sin_zeta = sincos(zeta_ang, &cos_zeta);
            zeta = (float2)(cos_zeta, sin_zeta);
        }

        {
            /* precompute trig for eta */
            float eta_ang = local_params[7];    /* orientation angle 2 */
            float sin_eta, cos_eta;
            sin_eta = sincos(eta_ang, &cos_eta);
            eta = (float2)(cos_eta, sin_eta);
        }

        {
            /* precompute trig for xi */
            float xi_ang = local_params[8];    /* orientation angle 3 */
            float sin_xi, cos_xi;
            sin_xi = sincos(xi_ang, &cos_xi);
            xi = (float2)(cos_xi, sin_xi);
        }

        /* convert q limb darkening to u limb darkening */
        float q1 = local_params[9];     /* limb darkening q1 */
        float q2 = local_params[10];    /* limb darkening q2 */
        float sqrt_q1 = sqrt(q1);
        u1 = 2 * sqrt_q1 * q2;
        u2 = sqrt_q1 * (1 - 2 * q2);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    float Ival = 0;
    float h_alpha = dalpha_bin / (gsz2 - 1);
    float alpha_ang = alpha_ang_mid +
        (gid2 - (gsz2 - 1) / 2) * h_alpha;

    /* precompute trig for alpha */
    float sin_alpha, cos_alpha;
    sin_alpha = sincos(alpha_ang, &cos_alpha);
    float2 alpha = (float2)(cos_alpha, sin_alpha);

    if (alpha.x > 0)
    {
        Ival = (((gid2 == 0) || (gid2 == gsz2 - 1)) ? 1 : ((gid2 & 0x1) ? 4 : 2)) *
            ellipsoid_transit_flux_workgroup_body(gid1, gsz1,
                alpha, beta, zeta, eta, xi, ax, ds, u1, u2);
    }

    float Ival_tot = work_group_reduce_add(Ival);
    Ival_tot /= 3 * (gsz2 - 1);

    if ((gid1 == 0) && (gid2 == 0)) flux[pid*ssz+sid] = 1. - M_1_PI * Ival_tot;
}

#undef LDA
#define LDA     (14)

kernel void ellipsoid_eccentric_transit_flux_vector(global const float *time,
    global const float *params, global float *flux)
{
    int pid = get_global_id(0);      /* parameters index */
    int psz = get_global_size(0);    /* number of parameter sets */

    int sid = get_global_id(1);      /* sample index */
    int ssz = get_global_size(1);    /* number of samples */

    int gid = get_local_id(0);       /* summation group index */
    int gsz = get_local_size(0);     /* number of summation groups */

    pid /= gsz;

    local float local_params[LDA];
    local float u1, u2, ds, tmp;
    local float2 alpha, beta, zeta, eta, xi;
    local float3 ax;

    /* pre-load the parameters for this work group */
    for (int off = gid; off < LDA; off += gsz)
        local_params[off] = params[pid*LDA+off];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (gid == 0)
    {
        float a = local_params[0];   /* semiaxis along x */
        float b = local_params[1];   /* semiaxis along y */
        float c = local_params[2];   /* semiaxis along z */
        ax = (float3)(a, b, c);

        float semimajor = local_params[3];  /* orbit semimajor axis */
        float porb = local_params[11];      /* orbital period */

        /* we have to compute the time of periastron first */
        float t0 = local_params[4];         /* midtransit offset in time */
        float ecc = local_params[12];       /* eccentricity */
        float omega = local_params[13];     /* argument of periastron */
        float tp = time_of_periastron_calc(ecc, omega, t0, 0.5 * M_1_PI * porb);

        /* compute the true anomaly and the orbital distance */
        float t = time[sid];
        float this_M = 2 * M_PI * ((t - tp) / porb);
        float alpha_ang = trueanom_calc(this_M, ecc);
        ds = semimajor * (1 - ecc) * (1 + ecc) / (1 + ecc * cos(alpha_ang));
        alpha_ang -= M_PI_2 - omega;

        {
            /* precompute trig for alpha */
            float sin_alpha, cos_alpha;
            sin_alpha = sincos(alpha_ang, &cos_alpha);
            alpha = (float2)(cos_alpha, sin_alpha);
        }

        {
            /* precompute trig for beta */
            float beta_ang = local_params[5];   /* complement of inclination */
            float sin_beta, cos_beta;
            sin_beta = sincos(beta_ang, &cos_beta);
            beta = (float2)(cos_beta, sin_beta);
        }

        {
            /* precompute trig for zeta */
            float zeta_ang = local_params[6];   /* orientation angle 1 */
            float sin_zeta, cos_zeta;
            sin_zeta = sincos(zeta_ang, &cos_zeta);
            zeta = (float2)(cos_zeta, sin_zeta);
        }

        {
            /* precompute trig for eta */
            float eta_ang = local_params[7];    /* orientation angle 2 */
            float sin_eta, cos_eta;
            sin_eta = sincos(eta_ang, &cos_eta);
            eta = (float2)(cos_eta, sin_eta);
        }

        {
            /* precompute trig for xi */
            float xi_ang = local_params[8];     /* orientation angle 1 */
            float sin_xi, cos_xi;
            sin_xi = sincos(xi_ang, &cos_xi);
            xi = (float2)(cos_xi, sin_xi);
        }

        /* convert q limb darkening to u limb darkening */
        float q1 = local_params[9];     /* limb darkening q1 */
        float q2 = local_params[10];    /* limb darkening q2 */
        float sqrt_q1 = sqrt(q1);
        u1 = 2 * sqrt_q1 * q2;
        u2 = sqrt_q1 * (1 - 2 * q2);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    float Ival = ellipsoid_transit_flux_workgroup_body(gid, gsz,
        alpha, beta, zeta, eta, xi, ax, ds, u1, u2);
    float Ival_tot = work_group_reduce_add(Ival);

    if (gid == 0) flux[pid*ssz+sid] = (alpha.x > 0) ? (1. - M_1_PI * Ival_tot) : 1.;
}

kernel void ellipsoid_eccentric_transit_flux_binned_vector(global const float *time,
    global const float *params, float dt_bin, global float *flux)
{
    int pid = get_global_id(0);     /* parameters index */
    int psz = get_global_size(0);   /* number of parameter sets */

    int sid = get_global_id(1);     /* sample index */
    int ssz = get_global_size(1);   /* number of samples */

    int gid1 = get_local_id(0);     /* workgroup dim 0, summation */
    int gsz1 = get_local_size(0);

    int gid2 = get_local_id(1);     /* workgroup dim 1, binning */
    int gsz2 = get_local_size(1);

    pid /= gsz1; psz /= gsz1;
    sid /= gsz2; ssz /= gsz2;

    local float local_params[LDA];
    local float u1, u2, mid_M, semimajor, porb, ecc, omega, dalpha_bin;
    local float2 beta, zeta, eta, xi;
    local float3 ax;

    /* pre-load the parameters for this work group */
    for (int off = gid1; (gid2 == 0) && (off < LDA); off += gsz1)
        local_params[off] = params[pid*LDA+off];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if ((gid1 == 0) && (gid2 == 0))
    {
        float a = local_params[0];   /* semiaxis along x */
        float b = local_params[1];   /* semiaxis along y */
        float c = local_params[2];   /* semiaxis along z */
        ax = (float3)(a, b, c);

        semimajor = local_params[3];  /* orbit semimajor axis */
        porb = local_params[11];      /* orbital period */

        /* we have to compute the time of periastron first */
        float t0 = local_params[4];     /* midtransit offset in time */
        ecc = local_params[12];         /* eccentricity */
        omega = local_params[13];       /* argument of periastron */
        float tp = time_of_periastron_calc(ecc, omega, t0, 0.5 * M_1_PI * porb);

        /* compute the mean anomaly */
        float t = time[sid];
        mid_M = 2 * M_PI * ((t - tp) / porb);

        {
            /* precompute trig for beta */
            float beta_ang = local_params[5];   /* complement of inclination */
            float sin_beta, cos_beta;
            sin_beta = sincos(beta_ang, &cos_beta);
            beta = (float2)(cos_beta, sin_beta);
        }

        {
            /* precompute trig for zeta */
            float zeta_ang = local_params[6];   /* orientation angle 1 */
            float sin_zeta, cos_zeta;
            sin_zeta = sincos(zeta_ang, &cos_zeta);
            zeta = (float2)(cos_zeta, sin_zeta);
        }

        {
            /* precompute trig for eta */
            float eta_ang = local_params[7];    /* orientation angle 2 */
            float sin_eta, cos_eta;
            sin_eta = sincos(eta_ang, &cos_eta);
            eta = (float2)(cos_eta, sin_eta);
        }

        {
            /* precompute trig for xi */
            float xi_ang = local_params[8];     /* orientation angle 3 */
            float sin_xi, cos_xi;
            sin_xi = sincos(xi_ang, &cos_xi);
            xi = (float2)(cos_xi, sin_xi);
        }

        /* convert q limb darkening to u limb darkening */
        float q1 = local_params[9];     /* limb darkening q1 */
        float q2 = local_params[10];    /* limb darkening q2 */
        float sqrt_q1 = sqrt(q1);
        u1 = 2 * sqrt_q1 * q2;
        u2 = sqrt_q1 * (1 - 2 * q2);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    float Ival = 0;

    /* integrating over mean anomaly, which is linear in time */
    float h_M = 2 * M_PI * (dt_bin / porb) / (gsz2 - 1);
    float this_M = mid_M + (gid2 - (gsz2 - 1) / 2) * h_M;

    float alpha_ang = trueanom_calc(this_M, ecc);
    float ds = semimajor * (1 - ecc) * (1 + ecc) / (1 + ecc * cos(alpha_ang));
    alpha_ang -= M_PI_2 - omega;

    /* precompute trig for alpha */
    float sin_alpha, cos_alpha;
    sin_alpha = sincos(alpha_ang, &cos_alpha);
    float2 alpha = (float2)(cos_alpha, sin_alpha);

    if (alpha.x > 0)
    {
        Ival = (((gid2 == 0) || (gid2 == gsz2 - 1)) ? 1 : ((gid2 & 0x1) ? 4 : 2)) *
            ellipsoid_transit_flux_workgroup_body(gid1, gsz1,
                alpha, beta, zeta, eta, xi, ax, ds, u1, u2);
    }

    float Ival_tot = work_group_reduce_add(Ival);
    Ival_tot /= 3 * (gsz2 - 1);

    if ((gid1 == 0) && (gid2 == 0)) flux[pid*ssz+sid] = 1. - M_1_PI * Ival_tot;
}

#undef LDA

/* vim: set ft=opencl.doxygen: */
