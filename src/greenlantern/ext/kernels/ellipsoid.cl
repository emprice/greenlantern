/// Number of integration intervals
#define NMAX        (2048)

#define DTHETA      (2 * M_PI / NMAX)

float3 compute_integrand(float2 theta, float x0, float dx,
    float y0, float y1, float dy)
{
    float x = x0 + dx * theta.x;
    float y = y0 + y1 * theta.x + dy * theta.y;
    float x2 = x * x, y2 = y * y;

    float dx_dtheta = -dx * theta.y;
    float dy_dtheta = -y1 * theta.y + dy * theta.x;

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
    float2 alpha, float2 beta, float2 gamma, float3 ax, float zs, float u1, float u2)
{
    float3 ax2 = ax * ax;
    float2 alpha2 = alpha * alpha;
    float2 beta2 = beta * beta;
    float2 gamma2 = gamma * gamma;

    float q = beta.x * gamma.x - alpha.y * beta.y * gamma.y;
    float u = alpha.y * beta.y * gamma.x + beta.x * gamma.y;

    float r = beta.y * gamma.x + alpha.y * beta.x * gamma.y;
    float s = alpha.y * beta.x * gamma.x - beta.y * gamma.y;

    float x0 = -zs * alpha.x * beta.y;
    float dx2 = ax2.y + (ax.x - ax.y) * (ax.x + ax.y) * q * q +
        -(ax.y - ax.z) * (ax.y + ax.z) * alpha2.x * beta2.y;
    float dx = sqrt(dx2);

    float y0 = zs * alpha.y * (ax2.x - (ax.x - ax.z) * (ax.x + ax.z) *
        alpha2.x * beta2.y - (ax.x - ax.y) * (ax.x + ax.y) * u * u) / dx2;
    float y1 = alpha.x * ((ax.x - ax.z) * (ax.x + ax.z) * alpha.y * beta.y +
        -(ax.x - ax.y) * (ax.x + ax.y) * gamma.x * u) / dx;
    float dy = sqrt(ax2.x * ax2.y * alpha2.x * beta2.x +
        ax2.y * ax2.z * r * r + ax2.x * ax2.z * s * s) / dx;

    float3 Ival = 0;

    for (int i = gid; i <= NMAX; i += gsz)
    {
        float theta_ang = i * DTHETA;

        float cos_theta, sin_theta;
        sin_theta = sincos(theta_ang, &cos_theta);
        float2 theta = (float2)(cos_theta, sin_theta);

        float3 fval = compute_integrand(theta, x0, dx, y0, y1, dy);
        fval *= ((i == 0) || (i == NMAX)) ? 1 : ((i & 0x1) ? 4 : 2);
        Ival += fval;
    }

    float integrand = ((1 - u1 - 2 * u2) * Ival.x +
        (u1 + 2 * u2) * Ival.y + u2 * Ival.z) * DTHETA / 3;
    float norm = 6. / (6 - 2 * u1 - u2);
    return integrand * norm;
}

#define LDA     (10)

__kernel void ellipsoid_transit_flux_vector(__global const float *time,
    __global const float *params, __global float *flux)
{
    int pid = get_global_id(0);      /* parameters index */
    int sid = get_global_id(1);      /* sample index */

    int np = get_global_size(0);     /* number of parameter sets */
    int ns = get_global_size(1);     /* number of samples */

    int gid = get_local_id(0);       /* summation group index */
    int gsz = get_local_size(0);     /* number of summation groups */

    pid /= gsz;

    local float local_params[LDA];
    local float u1, u2, zs;
    local float2 alpha, beta, gamma;
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

        zs = local_params[3];           /* distance from ellipsoid to disk */
        float porb = local_params[9];   /* orbital period */
        float t0 = local_params[4];     /* midtransit offset in time */

        float t = time[sid];
        float this_alpha = 2 * M_PI * (t / porb);
        float alpha0 = 2 * M_PI * (t0 / porb);

        /* precompute trig for alpha */
        float alpha_ang = this_alpha - alpha0;
        float sin_alpha, cos_alpha;
        sin_alpha = sincos(alpha_ang, &cos_alpha);
        alpha = (float2)(cos_alpha, sin_alpha);

        /* precompute trig for beta */
        float beta_ang = local_params[5];   /* complement of inclination */
        float sin_beta, cos_beta;
        sin_beta = sincos(beta_ang, &cos_beta);
        beta = (float2)(cos_beta, sin_beta);

        /* precompute trig for gamma */
        float gamma_ang = local_params[6];  /* obliquity */
        float sin_gamma, cos_gamma;
        sin_gamma = sincos(gamma_ang, &cos_gamma);
        gamma = (float2)(cos_gamma, sin_gamma);

        /* convert q limb darkening to u limb darkening */
        float q1 = local_params[7];   /* limb darkening q1 */
        float q2 = local_params[8];   /* limb darkening q2 */
        float sqrt_q1 = sqrt(q1);
        u1 = 2 * sqrt_q1 * q2;
        u2 = sqrt_q1 * (1 - 2 * q2);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    float Ival = ellipsoid_transit_flux_workgroup_body(gid, gsz,
        alpha, beta, gamma, ax, zs, u1, u2);
    float Ival_tot = work_group_reduce_add(Ival);

    if (gid == 0) flux[pid*ns+sid] = (alpha.x > 0) ? (1. - M_1_PI * Ival_tot) : 1.;
}

__kernel void ellipsoid_transit_flux_binned_vector(__global const float *time,
    __global const float *params, float dt_bin, __global float *flux)
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
    local float u1, u2, zs, alpha_ang_mid, dalpha_bin;
    local float2 beta, gamma;
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

        zs = local_params[3];           /* distance from ellipsoid to disk */
        float porb = local_params[9];   /* orbital period */
        float t0 = local_params[4];     /* midtransit offset in time */

        float t = time[sid];
        float this_alpha = 2 * M_PI * (t / porb);
        float alpha0 = 2 * M_PI * (t0 / porb);
        dalpha_bin = 2 * M_PI * (dt_bin / porb);

        alpha_ang_mid = this_alpha - alpha0;

        /* precompute trig for beta */
        float beta_ang = local_params[5];   /* complement of inclination */
        float sin_beta, cos_beta;
        sin_beta = sincos(beta_ang, &cos_beta);
        beta = (float2)(cos_beta, sin_beta);

        /* precompute trig for gamma */
        float gamma_ang = local_params[6];  /* obliquity */
        float sin_gamma, cos_gamma;
        sin_gamma = sincos(gamma_ang, &cos_gamma);
        gamma = (float2)(cos_gamma, sin_gamma);

        /* convert q limb darkening to u limb darkening */
        float q1 = local_params[7];   /* limb darkening q1 */
        float q2 = local_params[8];   /* limb darkening q2 */
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
                alpha, beta, gamma, ax, zs, u1, u2);
    }

    float Ival_tot = work_group_reduce_add(Ival);
    Ival_tot /= 3 * (gsz2 - 1);

    if ((gid1 == 0) && (gid2 == 0)) flux[pid*ssz+sid] = 1. - M_1_PI * Ival_tot;
}

/* vim: set ft=opencl.doxygen: */
