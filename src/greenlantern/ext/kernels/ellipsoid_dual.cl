/// Number of integration intervals
#define NMAX        (2048)

#define DTHETA      (2 * M_PI / NMAX)

#define NP1         (9)

typedef struct
{
    float x0, y0;
    float dx, dy1, dy2;

    float diff_x0[NP1], diff_y0[NP1];
    float diff_dx[NP1], diff_dy1[NP1], diff_dy2[NP1];
}
dual_params_t;

typedef struct
{
    float3 val;
    float3 deriv[NP1];
}
dual_integrand_t;

dual_integrand_t compute_dual_integrand(float2 theta, dual_params_t p)
{
    float diff_x[NP1], diff_y[NP1];
    float diff_dx_dtheta[NP1], diff_dy_dtheta[NP1];

    float x = p.x0 + p.dx * theta.x;
    float y = p.y0 + p.dy1 * theta.x + p.dy2 * theta.y;

    float x2 = x * x, y2 = y * y;
    float x2_plus_y2 = x2 + y2;

    float dx_dtheta = -p.dx * theta.y;
    float dy_dtheta = -p.dy1 * theta.y + p.dy2 * theta.x;

    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < NP1; ++i)
    {
        diff_x[i] = p.diff_x0[i] + p.diff_dx[i] * theta.x;
        diff_y[i] = p.diff_y0[i] + p.diff_dy1[i] * theta.x + p.diff_dy2[i] * theta.y;

        diff_dx_dtheta[i] = -p.diff_dx[i] * theta.y;
        diff_dy_dtheta[i] = -p.diff_dy1[i] * theta.y + p.diff_dy2[i] * theta.x;
    }

    if (x2_plus_y2 > 1)
    {
        float diff_dlam_dtheta[NP1], diff_lam[NP1];

        float dlam_dtheta = (-y * dx_dtheta + x * dy_dtheta) / x2_plus_y2;

        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < NP1; ++i)
        {
            diff_dlam_dtheta[i] = (-(diff_y[i] * dx_dtheta + y * diff_dx_dtheta[i]) +
                (diff_x[i] * dy_dtheta + x * diff_dy_dtheta[i])) / x2_plus_y2 +
                -2 * (x * diff_x[i] + y * diff_y[i]) * dlam_dtheta / x2_plus_y2;
        }

        float lam = atan2(y, x);

        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < NP1; ++i)
        {
            diff_lam[i] = (-y * diff_x[i] + x * diff_y[i]) / x2_plus_y2;
        }

        y = sincos(lam, &x);
        x2 = x * x; y2 = y * y;

        dx_dtheta = -y * dlam_dtheta;
        dy_dtheta =  x * dlam_dtheta;

        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < NP1; ++i)
        {
            diff_x[i] = -y * diff_lam[i];
            diff_y[i] =  x * diff_lam[i];

            diff_dx_dtheta[i] = -(diff_y[i] * dlam_dtheta + y * diff_dlam_dtheta[i]);
            diff_dy_dtheta[i] =  (diff_x[i] * dlam_dtheta + x * diff_dlam_dtheta[i]);
        }
    }

    // HACK: mathematically, x^2 + y^2 <= 1, but numerical error can
    // sometimes push the value slightly higher
    x2_plus_y2 = fmin(1, x2 + y2);

    dual_integrand_t Ival;

    Ival.val.x = 0.5 * (x * dy_dtheta - y * dx_dtheta);
    Ival.val.y = 2 * (1 - pow(1 - x2_plus_y2, 1.5)) / (3 * x2_plus_y2) * Ival.val.x;
    Ival.val.z = 0.5 * (x * (y2 + x2 / 3) * dy_dtheta +
        -y * (x2 + y2 / 3) * dx_dtheta);

    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < NP1; ++i)
    {
        float diff_x2_plus_y2 = 2 * (x * diff_x[i] + y * diff_y[i]);

        Ival.deriv[i].x =
            0.5 * ( (diff_x[i] * dy_dtheta + x * diff_dy_dtheta[i]) +
                   -(diff_y[i] * dx_dtheta + y * diff_dx_dtheta[i]));
        Ival.deriv[i].y =
            (pow(1 - x2_plus_y2, 0.5) * diff_x2_plus_y2 / x2_plus_y2 +
             -2 * (1 - pow(1 - x2_plus_y2, 1.5)) * diff_x2_plus_y2 /
                (3 * x2_plus_y2 * x2_plus_y2)) * Ival.val.x +
            2 * (1 - pow(1 - x2_plus_y2, 1.5)) / (3 * x2_plus_y2) * Ival.deriv[i].x;
        Ival.deriv[i].z =
            0.5 * (( diff_x[i] * (y2 + x2 / 3) * dy_dtheta +
                    -diff_y[i] * (x2 + y2 / 3) * dx_dtheta) +
            2 * ( x * (y * diff_y[i] + x * diff_x[i] / 3) * dy_dtheta +
                 -y * (x * diff_x[i] + y * diff_y[i] / 3) * dx_dtheta) +
            ( x * (y2 + x2 / 3) * diff_dy_dtheta[i] +
             -y * (x2 + y2 / 3) * diff_dx_dtheta[i]));
    }

    return Ival;
}

typedef struct
{
    float r1, r2, r3;
    float r4, r5, r6;
    float r7, r8, r9;

    float dr1[3], dr2[3], dr3[3];
    float dr4[3], dr5[3], dr6[3];
    float dr7[3], dr8[3], dr9[3];
}
dual_ang_params_t;

dual_ang_params_t fill_dual_angular_parameters(float2 zeta, float2 eta, float2 xi)
{
    dual_ang_params_t p;

    p.r1 = eta.x * xi.x;
    p.dr1[0] = 0;                                        /* dr1_dzeta */
    p.dr1[1] = -eta.y * xi.x;                            /* dr1_deta */
    p.dr1[2] = -eta.x * xi.y;                            /* dr1_dxi */

    p.r2 = -eta.x * xi.y;
    p.dr2[0] = 0;                                        /* dr2_dzeta */
    p.dr2[1] = eta.y * xi.y;                             /* dr2_deta */
    p.dr2[2] = -eta.x * xi.x;                            /* dr2_dxi */

    p.r3 = eta.y;
    p.dr3[0] = 0;                                        /* dr3_dzeta */
    p.dr3[1] = eta.x;                                    /* dr3_deta */
    p.dr3[2] = 0;                                        /* dr3_dxi */

    p.r4 = zeta.y * eta.y * xi.x + zeta.x * xi.y;
    p.dr4[0] = zeta.x * eta.y * xi.x - zeta.y * xi.y;    /* dr4_dzeta */
    p.dr4[1] = zeta.y * eta.x * xi.x;                    /* dr4_deta */
    p.dr4[2] = -zeta.y * eta.y * xi.y + zeta.x * xi.x;   /* dr4_dxi */

    p.r5 = zeta.x * xi.x - zeta.y * eta.y * xi.y;
    p.dr5[0] = -zeta.y * xi.x - zeta.x * eta.y * xi.y;   /* dr5_dzeta */
    p.dr5[1] = -zeta.y * eta.x * xi.y;                   /* dr5_deta */
    p.dr5[2] = -zeta.x * xi.y - zeta.y * eta.y * xi.x;   /* dr5_dxi */

    p.r6 = -zeta.y * eta.x;
    p.dr6[0] = -zeta.x * eta.x;                          /* dr6_dzeta */
    p.dr6[1] = zeta.y * eta.y;                           /* dr6_deta */
    p.dr6[2] = 0;                                        /* dr6_dxi */

    p.r7 = -zeta.x * eta.y * xi.x + zeta.y * xi.y;
    p.dr7[0] = zeta.y * eta.y * xi.x + zeta.x * xi.y;    /* dr7_dzeta */
    p.dr7[1] = -zeta.x * eta.x * xi.x;                   /* dr7_deta */
    p.dr7[2] = zeta.x * eta.y * xi.y + zeta.y * xi.x;    /* dr7_dxi */

    p.r8 = zeta.y * xi.x + zeta.x * eta.y * xi.y;
    p.dr8[0] = zeta.x * xi.x - zeta.y * eta.y * xi.y;    /* dr8_dzeta */
    p.dr8[1] = zeta.x * eta.x * xi.y;                    /* dr8_deta */
    p.dr8[2] = -zeta.y * xi.y + zeta.x * eta.y * xi.x;   /* dr8_dxi */

    p.r9 = zeta.x * eta.x;
    p.dr9[0] = -zeta.y * eta.x;                          /* dr9_dzeta */
    p.dr9[1] = -zeta.x * eta.y;                          /* dr9_deta */
    p.dr9[2] = 0;                                        /* dr9_dxi */

    return p;
}

typedef struct
{
    float val;
    float deriv[NP1+2];
}
dual_flux_t;

dual_flux_t ellipsoid_transit_flux_dual_locked_workgroup_body(int gid,
    int gsz, float2 alpha, float2 beta, dual_ang_params_t p,
    float3 ax, float ds, float u1, float u2)
{
    dual_params_t q;
    dual_flux_t ret;

    float diff_afac1[NP1], diff_bfac1[NP1], diff_cfac1[NP1];
    float diff_afac2[NP1], diff_bfac2[NP1], diff_cfac2[NP1];
    float diff_abfac[NP1], diff_acfac[NP1], diff_bcfac[NP1];

    /* derivatives w.r.t. a, b, c */
    diff_afac1[0] = beta.y * (alpha.x * p.r3 + alpha.y * p.r2) - beta.x * p.r1;
    diff_bfac1[1] = beta.y * (alpha.x * p.r6 + alpha.y * p.r5) - beta.x * p.r4;
    diff_cfac1[2] = beta.y * (alpha.x * p.r9 + alpha.y * p.r8) - beta.x * p.r7;

    diff_afac1[1] = diff_afac1[2] = 0;
    diff_bfac1[0] = diff_bfac1[2] = 0;
    diff_cfac1[0] = diff_cfac1[1] = 0;

    diff_afac2[0] = alpha.y * p.r3 - alpha.x * p.r2;
    diff_bfac2[1] = alpha.y * p.r6 - alpha.x * p.r5;
    diff_cfac2[2] = alpha.y * p.r9 - alpha.x * p.r8;

    diff_afac2[1] = diff_afac2[2] = 0;
    diff_bfac2[0] = diff_bfac2[2] = 0;
    diff_cfac2[0] = diff_cfac2[1] = 0;

    diff_abfac[0] = ax.y * (beta.y * p.r7 + beta.x * (alpha.x * p.r9 + alpha.y * p.r8));
    diff_abfac[1] = ax.x * (beta.y * p.r7 + beta.x * (alpha.x * p.r9 + alpha.y * p.r8));
    diff_acfac[0] = ax.z * (beta.y * p.r4 + beta.x * (alpha.x * p.r6 + alpha.y * p.r5));
    diff_acfac[2] = ax.x * (beta.y * p.r4 + beta.x * (alpha.x * p.r6 + alpha.y * p.r5));
    diff_bcfac[1] = ax.z * (beta.y * p.r1 + beta.x * (alpha.x * p.r3 + alpha.y * p.r2));
    diff_bcfac[2] = ax.y * (beta.y * p.r1 + beta.x * (alpha.x * p.r3 + alpha.y * p.r2));

    diff_abfac[2] = diff_acfac[1] = diff_bcfac[0] = 0;

    /* values */
    float afac1 = ax.x * diff_afac1[0];
    float bfac1 = ax.y * diff_bfac1[1];
    float cfac1 = ax.z * diff_cfac1[2];

    float afac2 = ax.x * diff_afac2[0];
    float bfac2 = ax.y * diff_bfac2[1];
    float cfac2 = ax.z * diff_cfac2[2];

    float abfac = ax.x * diff_abfac[0];
    float acfac = ax.z * diff_acfac[2];
    float bcfac = ax.y * diff_bcfac[1];

    /* derivatives w.r.t. ds */
    diff_afac1[3] = diff_bfac1[3] = diff_cfac1[3] = 0;
    diff_afac2[3] = diff_bfac2[3] = diff_cfac2[3] = 0;
    diff_abfac[3] = diff_acfac[3] = diff_bcfac[3] = 0;

    /* derivatives w.r.t. alpha */
    diff_afac1[4] = ax.x * (beta.y * (-alpha.y * p.r3 + alpha.x * p.r2) - beta.x * p.r1);
    diff_bfac1[4] = ax.y * (beta.y * (-alpha.y * p.r6 + alpha.x * p.r5) - beta.x * p.r4);
    diff_cfac1[4] = ax.z * (beta.y * (-alpha.y * p.r9 + alpha.x * p.r8) - beta.x * p.r7);

    diff_afac2[4] = ax.x * (alpha.x * p.r3 + alpha.y * p.r2);
    diff_bfac2[4] = ax.y * (alpha.x * p.r6 + alpha.y * p.r5);
    diff_cfac2[4] = ax.z * (alpha.x * p.r9 + alpha.y * p.r8);

    diff_abfac[4] = ax.x * ax.y * (beta.y * p.r7 + beta.x * (-alpha.y * p.r9 + alpha.x * p.r8));
    diff_acfac[4] = ax.x * ax.z * (beta.y * p.r4 + beta.x * (-alpha.y * p.r6 + alpha.x * p.r5));
    diff_bcfac[4] = ax.y * ax.z * (beta.y * p.r1 + beta.x * (-alpha.y * p.r3 + alpha.x * p.r2));

    /* derivatives w.r.t. beta */
    diff_afac1[5] = ax.x * (beta.x * (alpha.x * p.r3 + alpha.y * p.r2) + beta.y * p.r1);
    diff_bfac1[5] = ax.y * (beta.x * (alpha.x * p.r6 + alpha.y * p.r5) + beta.y * p.r4);
    diff_cfac1[5] = ax.z * (beta.x * (alpha.x * p.r9 + alpha.y * p.r8) + beta.y * p.r7);

    diff_afac2[5] = diff_bfac2[5] = diff_cfac2[5] = 0;

    diff_abfac[5] = ax.x * ax.y * (beta.x * p.r7 - beta.y * (alpha.x * p.r9 + alpha.y * p.r8));
    diff_acfac[5] = ax.x * ax.z * (beta.x * p.r4 - beta.y * (alpha.x * p.r6 + alpha.y * p.r5));
    diff_bcfac[5] = ax.y * ax.z * (beta.x * p.r1 - beta.y * (alpha.x * p.r3 + alpha.y * p.r2));

    /* derivatives w.r.t. zeta, eta, xi */
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 3; ++i)
    {
        diff_afac1[i+6] = ax.x * (beta.y * (alpha.x * p.dr3[i] + alpha.y * p.dr2[i]) - beta.x * p.dr1[i]);
        diff_bfac1[i+6] = ax.y * (beta.y * (alpha.x * p.dr6[i] + alpha.y * p.dr5[i]) - beta.x * p.dr4[i]);
        diff_cfac1[i+6] = ax.z * (beta.y * (alpha.x * p.dr9[i] + alpha.y * p.dr8[i]) - beta.x * p.dr7[i]);

        diff_afac2[i+6] = ax.x * (alpha.y * p.dr3[i] - alpha.x * p.dr2[i]);
        diff_bfac2[i+6] = ax.y * (alpha.y * p.dr6[i] - alpha.x * p.dr5[i]);
        diff_cfac2[i+6] = ax.z * (alpha.y * p.dr9[i] - alpha.x * p.dr8[i]);

        diff_abfac[i+6] = ax.x * ax.y * (beta.y * p.dr7[i] + beta.x * (alpha.x * p.dr9[i] + alpha.y * p.dr8[i]));
        diff_acfac[i+6] = ax.x * ax.z * (beta.y * p.dr4[i] + beta.x * (alpha.x * p.dr6[i] + alpha.y * p.dr5[i]));
        diff_bcfac[i+6] = ax.y * ax.z * (beta.y * p.dr1[i] + beta.x * (alpha.x * p.dr3[i] + alpha.y * p.dr2[i]));
    }

    q.dx = sqrt(afac1 * afac1 + bfac1 * bfac1 + cfac1 * cfac1);
    q.dy1 = (afac1 * afac2 + bfac1 * bfac2 + cfac1 * cfac2) / q.dx;

    float dy2_tmp = sqrt(abfac * abfac + acfac * acfac + bcfac * bcfac);
    q.dy2 = dy2_tmp / q.dx;

    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < NP1; ++i)
    {
        q.diff_dx[i] = (afac1 * diff_afac1[i] +
            bfac1 * diff_bfac1[i] + cfac1 * diff_cfac1[i]) / q.dx;
        q.diff_dy1[i] =
            (afac1 * diff_afac2[i] + bfac1 * diff_bfac2[i] + cfac1 * diff_cfac2[i]) / q.dx +
            (diff_afac1[i] * afac2 + diff_bfac1[i] * bfac2 + diff_cfac1[i] * cfac2) / q.dx +
            -1 * q.dy1 * q.diff_dx[i] / q.dx;

        q.diff_dy2[i] = (abfac * diff_abfac[i] + acfac * diff_acfac[i] +
            bcfac * diff_bcfac[i]) / dy2_tmp / q.dx - q.dy2 * q.diff_dx[i] / q.dx;
    }

    q.x0 = ds * alpha.x * beta.y;
    q.y0 = ds * alpha.y;

    q.diff_x0[3] =       alpha.x * beta.y;
    q.diff_x0[4] = -ds * alpha.y * beta.y;
    q.diff_x0[5] =  ds * alpha.x * beta.x;

    q.diff_y0[3] =      alpha.y;
    q.diff_y0[4] = ds * alpha.x;

    q.diff_x0[0] = q.diff_x0[1] = q.diff_x0[2] =
        q.diff_x0[6] = q.diff_x0[7] = q.diff_x0[8] = 0;
    q.diff_y0[0] = q.diff_y0[1] = q.diff_y0[2] = q.diff_y0[5] =
        q.diff_y0[6] = q.diff_y0[7] = q.diff_y0[8] = 0;

    dual_integrand_t Ival, fval;

    Ival.val = 0;
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < NP1; ++j) Ival.deriv[j] = 0;

    for (int i = gid; i <= NMAX; i += gsz)
    {
        float theta_ang = i * DTHETA;

        float cos_theta, sin_theta;
        sin_theta = sincos(theta_ang, &cos_theta);
        float2 theta = (float2)(cos_theta, sin_theta);

        fval = compute_dual_integrand(theta, q);

        /* simpson's rule weight */
        int wght = ((i == 0) || (i == NMAX)) ? 1 : ((i & 0x1) ? 4 : 2);

        Ival.val += fval.val * wght;
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < NP1; ++j) Ival.deriv[j] += fval.deriv[j] * wght;
    }

    float integrand = ((1 - u1 - 2 * u2) * Ival.val.x +
        (u1 + 2 * u2) * Ival.val.y + u2 * Ival.val.z) * DTHETA / 3;
    float norm = 6. / (6 - 2 * u1 - u2);
    ret.val = integrand * norm;

    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < NP1; ++i)
    {
        float dintegrand = ((1 - u1 - 2 * u2) * Ival.deriv[i].x +
            (u1 + 2 * u2) * Ival.deriv[i].y + u2 * Ival.deriv[i].z) * DTHETA / 3;
        ret.deriv[i] = dintegrand * norm;
    }

    {
        float dintegrand = (-Ival.val.x + Ival.val.y) * DTHETA / 3;
        float dnorm = 2 * norm / (6 - 2 * u1 - u2);
        ret.deriv[NP1] = dintegrand * norm + integrand * dnorm;
    }

    {
        float dintegrand = (-2 * Ival.val.x + 2 * Ival.val.y + Ival.val.z) * DTHETA / 3;
        float dnorm = norm / (6 - 2 * u1 - u2);
        ret.deriv[NP1+1] = dintegrand * norm + integrand * dnorm;
    }

    return ret;
}

dual_flux_t ellipsoid_transit_flux_dual_unlocked_workgroup_body(int gid,
    int gsz, float2 alpha, float2 beta, dual_ang_params_t p,
    float3 ax, float ds, float u1, float u2)
{
    dual_params_t q;
    dual_flux_t ret;

    float diff_afac1[NP1], diff_bfac1[NP1], diff_cfac1[NP1];
    float diff_afac2[NP1], diff_bfac2[NP1], diff_cfac2[NP1];
    float diff_abfac[NP1], diff_acfac[NP1], diff_bcfac[NP1];

    /* derivatives w.r.t. a, b, c */
    diff_afac1[0] = p.r1;
    diff_bfac1[1] = p.r4;
    diff_cfac1[2] = p.r7;

    diff_afac1[1] = diff_afac1[2] = 0;
    diff_bfac1[0] = diff_bfac1[2] = 0;
    diff_cfac1[0] = diff_cfac1[1] = 0;

    diff_afac2[0] = p.r2;
    diff_bfac2[1] = p.r5;
    diff_cfac2[2] = p.r8;

    diff_afac2[1] = diff_afac2[2] = 0;
    diff_bfac2[0] = diff_bfac2[2] = 0;
    diff_cfac2[0] = diff_cfac2[1] = 0;

    diff_abfac[0] = ax.y * p.r9;
    diff_abfac[1] = ax.x * p.r9;
    diff_acfac[0] = ax.z * p.r6;
    diff_acfac[2] = ax.x * p.r6;
    diff_bcfac[1] = ax.z * p.r3;
    diff_bcfac[2] = ax.y * p.r3;

    diff_abfac[2] = diff_acfac[1] = diff_bcfac[0] = 0;

    /* values */
    float afac1 = ax.x * diff_afac1[0];
    float bfac1 = ax.y * diff_bfac1[1];
    float cfac1 = ax.z * diff_cfac1[2];

    float afac2 = ax.x * diff_afac2[0];
    float bfac2 = ax.y * diff_bfac2[1];
    float cfac2 = ax.z * diff_cfac2[2];

    float abfac = ax.x * diff_abfac[0];
    float acfac = ax.z * diff_acfac[2];
    float bcfac = ax.y * diff_bcfac[1];

    /* derivatives w.r.t. ds */
    diff_afac1[3] = diff_bfac1[3] = diff_cfac1[3] = 0;
    diff_afac2[3] = diff_bfac2[3] = diff_cfac2[3] = 0;
    diff_abfac[3] = diff_acfac[3] = diff_bcfac[3] = 0;

    /* derivatives w.r.t. alpha */
    diff_afac1[4] = diff_bfac1[4] = diff_cfac1[4] = 0;
    diff_afac2[4] = diff_bfac2[4] = diff_cfac2[4] = 0;
    diff_abfac[4] = diff_acfac[4] = diff_bcfac[4] = 0;

    /* derivatives w.r.t. beta */
    diff_afac1[5] = diff_bfac1[5] = diff_cfac1[5] = 0;
    diff_afac2[5] = diff_bfac2[5] = diff_cfac2[5] = 0;
    diff_abfac[5] = diff_acfac[5] = diff_bcfac[5] = 0;

    /* derivatives w.r.t. zeta, eta, xi */
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < 3; ++i)
    {
        diff_afac1[i+6] = ax.x * p.dr1[i];
        diff_bfac1[i+6] = ax.y * p.dr4[i];
        diff_cfac1[i+6] = ax.z * p.dr7[i];

        diff_afac2[i+6] = ax.x * p.dr2[i];
        diff_bfac2[i+6] = ax.y * p.dr5[i];
        diff_cfac2[i+6] = ax.z * p.dr8[i];

        diff_abfac[i+6] = ax.x * ax.y * p.dr9[i];
        diff_acfac[i+6] = ax.x * ax.z * p.dr6[i];
        diff_bcfac[i+6] = ax.y * ax.z * p.dr3[i];
    }

    q.dx = sqrt(afac1 * afac1 + bfac1 * bfac1 + cfac1 * cfac1);
    q.dy1 = (afac1 * afac2 + bfac1 * bfac2 + cfac1 * cfac2) / q.dx;

    float dy2_tmp = sqrt(abfac * abfac + acfac * acfac + bcfac * bcfac);
    q.dy2 = dy2_tmp / q.dx;

    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < NP1; ++i)
    {
        q.diff_dx[i] = (afac1 * diff_afac1[i] +
            bfac1 * diff_bfac1[i] + cfac1 * diff_cfac1[i]) / q.dx;

        q.diff_dy1[i] =
            (afac1 * diff_afac2[i] + bfac1 * diff_bfac2[i] + cfac1 * diff_cfac2[i]) / q.dx +
            (diff_afac1[i] * afac2 + diff_bfac1[i] * bfac2 + diff_cfac1[i] * cfac2) / q.dx +
            -1 * q.dy1 * q.diff_dx[i] / q.dx;

        q.diff_dy2[i] = (abfac * diff_abfac[i] + acfac * diff_acfac[i] +
            bcfac * diff_bcfac[i]) / dy2_tmp / q.dx - q.dy2 * q.diff_dx[i] / q.dx;
    }

    q.x0 = ds * alpha.x * beta.y;
    q.y0 = ds * alpha.y;

    q.diff_x0[3] =       alpha.x * beta.y;
    q.diff_x0[4] = -ds * alpha.y * beta.y;
    q.diff_x0[5] =  ds * alpha.x * beta.x;

    q.diff_y0[3] =      alpha.y;
    q.diff_y0[4] = ds * alpha.x;

    q.diff_x0[0] = q.diff_x0[1] = q.diff_x0[2] =
        q.diff_x0[6] = q.diff_x0[7] = q.diff_x0[8] = 0;
    q.diff_y0[0] = q.diff_y0[1] = q.diff_y0[2] = q.diff_y0[5] =
        q.diff_y0[6] = q.diff_y0[7] = q.diff_y0[8] = 0;

    dual_integrand_t Ival, fval;

    Ival.val = 0;
    __attribute__((opencl_unroll_hint))
    for (int j = 0; j < NP1; ++j) Ival.deriv[j] = 0;

    for (int i = gid; i <= NMAX; i += gsz)
    {
        float theta_ang = i * DTHETA;

        float cos_theta, sin_theta;
        sin_theta = sincos(theta_ang, &cos_theta);
        float2 theta = (float2)(cos_theta, sin_theta);

        fval = compute_dual_integrand(theta, q);

        /* simpson's rule weight */
        int wght = ((i == 0) || (i == NMAX)) ? 1 : ((i & 0x1) ? 4 : 2);

        Ival.val += fval.val * wght;
        __attribute__((opencl_unroll_hint))
        for (int j = 0; j < NP1; ++j) Ival.deriv[j] += fval.deriv[j] * wght;
    }

    float integrand = ((1 - u1 - 2 * u2) * Ival.val.x +
        (u1 + 2 * u2) * Ival.val.y + u2 * Ival.val.z) * DTHETA / 3;
    float norm = 6. / (6 - 2 * u1 - u2);
    ret.val = integrand * norm;

    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < NP1; ++i)
    {
        float dintegrand = ((1 - u1 - 2 * u2) * Ival.deriv[i].x +
            (u1 + 2 * u2) * Ival.deriv[i].y + u2 * Ival.deriv[i].z) * DTHETA / 3;
        ret.deriv[i] = dintegrand * norm;
    }

    {
        float dintegrand = (-Ival.val.x + Ival.val.y) * DTHETA / 3;
        float dnorm = 2 * norm / (6 - 2 * u1 - u2);
        ret.deriv[NP1] = dintegrand * norm + integrand * dnorm;
    }

    {
        float dintegrand = (-2 * Ival.val.x + 2 * Ival.val.y + Ival.val.z) * DTHETA / 3;
        float dnorm = norm / (6 - 2 * u1 - u2);
        ret.deriv[NP1+1] = dintegrand * norm + integrand * dnorm;
    }

    return ret;
}

#define LDA     (12)

kernel void ellipsoid_transit_flux_dual(global const float * restrict time,
    global const float * restrict params, ushort locked,
    global float * restrict flux, global float * restrict dflux)
{
    int sid = get_global_id(0);      /* sample index */
    int ssz = get_global_size(0);    /* number of samples */

    int gid = get_local_id(0);       /* summation group index */
    int gsz = get_local_size(0);     /* number of summation groups */

    sid /= gsz;
    ssz /= gsz;

    local float local_params[LDA];
    local float u1, u2, ds;
    local float2 alpha, beta, zeta, eta, xi;
    local float3 ax;
    local float alpha_ang, porb, q1, sqrt_q1, q2;

    /* pre-load the parameters for this work group */
    for (int off = gid; off < LDA; off += gsz)
        local_params[off] = params[off];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (gid == 0)
    {
        float a = local_params[0];   /* semiaxis along x */
        float b = local_params[1];   /* semiaxis along y */
        float c = local_params[2];   /* semiaxis along z */
        ax = (float3)(a, b, c);

        ds = local_params[3];           /* distance from ellipsoid to disk */
        porb = local_params[11];        /* orbital period */
        float t0 = local_params[4];     /* midtransit offset in time */

        float t = time[sid];
        float this_alpha = 2 * M_PI * (t / porb);
        float alpha0 = 2 * M_PI * (t0 / porb);

        {
            /* precompute trig for alpha */
            alpha_ang = this_alpha - alpha0;
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
        q1 = local_params[9];     /* limb darkening q1 */
        q2 = local_params[10];    /* limb darkening q2 */
        sqrt_q1 = sqrt(q1);
        u1 = 2 * sqrt_q1 * q2;
        u2 = sqrt_q1 * (1 - 2 * q2);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    dual_ang_params_t p = fill_dual_angular_parameters(zeta, eta, xi);
    dual_flux_t Ival = (locked) ?
        ellipsoid_transit_flux_dual_locked_workgroup_body(gid, gsz, alpha, beta, p, ax, ds, u1, u2) :
        ellipsoid_transit_flux_dual_unlocked_workgroup_body(gid, gsz, alpha, beta, p, ax, ds, u1, u2);

    dual_flux_t Ival_tot;
    Ival_tot.val = work_group_reduce_add(Ival.val);
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < NP1 + 2; ++i)
        Ival_tot.deriv[i] = work_group_reduce_add(Ival.deriv[i]);

    local float local_dflux[LDA];

    if (gid == 0)
    {
        if (alpha.x > 0)
        {
            local_dflux[0] = Ival_tot.deriv[0];     /* dIval / da */
            local_dflux[1] = Ival_tot.deriv[1];     /* dIval / db */
            local_dflux[2] = Ival_tot.deriv[2];     /* dIval / dc */
            local_dflux[3] = Ival_tot.deriv[3];     /* dIval / dds */

            /* dIval / dt0 from dIval / dalpha */
            local_dflux[4] = -2 * M_PI * Ival_tot.deriv[4] / porb;

            local_dflux[5] = Ival_tot.deriv[5];     /* dIval / dbeta */
            local_dflux[6] = Ival_tot.deriv[6];     /* dIval / dzeta */
            local_dflux[7] = Ival_tot.deriv[7];     /* dIval / deta */
            local_dflux[8] = Ival_tot.deriv[8];     /* dIval / dxi */

            /* dIval / dq1 and dIval / dq2 */
            local_dflux[9] = (sqrt_q1 == 0) ? 0 : ((q2 * Ival_tot.deriv[9] +
                0.5 * (1 - 2 * q2) * Ival_tot.deriv[10]) / sqrt_q1);
            local_dflux[10] = 2 * sqrt_q1 * (Ival_tot.deriv[9] - Ival_tot.deriv[10]);

            /* dIval / dporb from dIval / dalpha */
            local_dflux[11] = -alpha_ang * Ival_tot.deriv[4] / porb;

            flux[sid] = 1. - M_1_PI * Ival_tot.val;
        }
        else
        {
            flux[sid] = 1;
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < LDA; ++i) local_dflux[i] = 0;
        }
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    for (int off = gid; off < LDA; off += gsz)
        dflux[off*ssz+sid] = -M_1_PI * local_dflux[off];
}

kernel void ellipsoid_transit_flux_binned_dual(global const float * restrict time,
    global const float *params, ushort locked, float dt_bin, global float *flux,
    global float * restrict dflux)
{
    int sid = get_global_id(0);     /* sample index */
    int ssz = get_global_size(0);   /* number of samples */

    int gid1 = get_local_id(0);     /* workgroup dim 0, summation */
    int gsz1 = get_local_size(0);

    int gid2 = get_local_id(1);     /* workgroup dim 1, binning */
    int gsz2 = get_local_size(1);

    sid /= gsz1; ssz /= gsz1;

    local float local_params[LDA];
    local float u1, u2, ds, alpha_ang_mid, dalpha_bin;
    local float2 beta, zeta, eta, xi;
    local float3 ax;
    local float porb, q1, sqrt_q1, q2;

    /* pre-load the parameters for this work group */
    for (int off = gid1; (gid2 == 0) && (off < LDA); off += gsz1)
        local_params[off] = params[off];
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if ((gid1 == 0) && (gid2 == 0))
    {
        float a = local_params[0];   /* semiaxis along x */
        float b = local_params[1];   /* semiaxis along y */
        float c = local_params[2];   /* semiaxis along z */
        ax = (float3)(a, b, c);

        ds = local_params[3];           /* distance from ellipsoid to disk */
        porb = local_params[11];        /* orbital period */
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
        q1 = local_params[9];       /* limb darkening q1 */
        q2 = local_params[10];      /* limb darkening q2 */
        sqrt_q1 = sqrt(q1);
        u1 = 2 * sqrt_q1 * q2;
        u2 = sqrt_q1 * (1 - 2 * q2);
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    float h_alpha = dalpha_bin / (gsz2 - 1);
    float alpha_ang = alpha_ang_mid +
        (gid2 - (gsz2 - 1) / 2) * h_alpha;

    /* precompute trig for alpha */
    float sin_alpha, cos_alpha;
    sin_alpha = sincos(alpha_ang, &cos_alpha);
    float2 alpha = (float2)(cos_alpha, sin_alpha);

    dual_ang_params_t p = fill_dual_angular_parameters(zeta, eta, xi);
    dual_flux_t Ival = (locked) ?
        ellipsoid_transit_flux_dual_locked_workgroup_body(gid1, gsz1, alpha, beta, p, ax, ds, u1, u2) :
        ellipsoid_transit_flux_dual_unlocked_workgroup_body(gid1, gsz1, alpha, beta, p, ax, ds, u1, u2);

    float local_dflux[LDA];

    if (alpha.x > 0)
    {
        local_dflux[0] = Ival.deriv[0];     /* dIval / da */
        local_dflux[1] = Ival.deriv[1];     /* dIval / db */
        local_dflux[2] = Ival.deriv[2];     /* dIval / dc */
        local_dflux[3] = Ival.deriv[3];     /* dIval / dds */

        /* dIval / dt0 from dIval / dalpha */
        local_dflux[4] = -2 * M_PI * Ival.deriv[4] / porb;

        local_dflux[5] = Ival.deriv[5];     /* dIval / dbeta */
        local_dflux[6] = Ival.deriv[6];     /* dIval / dzeta */
        local_dflux[7] = Ival.deriv[7];     /* dIval / deta */
        local_dflux[8] = Ival.deriv[8];     /* dIval / dxi */

        /* dIval / dq1 and dIval / dq2 */
        local_dflux[9] = (sqrt_q1 == 0) ? 0 : ((q2 * Ival.deriv[9] +
            0.5 * (1 - 2 * q2) * Ival.deriv[10]) / sqrt_q1);
        local_dflux[10] = 2 * sqrt_q1 * (Ival.deriv[9] - Ival.deriv[10]);

        /* dIval / dporb from dIval / dalpha */
        local_dflux[11] = -alpha_ang * Ival.deriv[4] / porb;
    }
    else
    {
        Ival.val = 0;
        __attribute__((opencl_unroll_hint))
        for (int i = 0; i < LDA; ++i) local_dflux[i] = 0;
    }

    float weight = ((gid2 == 0) || (gid2 == gsz2 - 1)) ? 1 : ((gid2 & 0x1) ? 4 : 2);
    weight /= 3 * (gsz2 - 1);

    Ival.val *= weight;
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < NP1 + 2; ++i) local_dflux[i] *= weight;

    dual_flux_t Ival_tot;
    Ival_tot.val = work_group_reduce_add(Ival.val);
    __attribute__((opencl_unroll_hint))
    for (int i = 0; i < NP1 + 2; ++i)
        Ival_tot.deriv[i] = work_group_reduce_add(local_dflux[i]);

    if ((gid1 == 0) && (gid2 == 0)) flux[sid] = 1 - M_1_PI * Ival_tot.val;
    for (int off = gid1; (gid2 == 0) && (off < LDA); off += gsz1)
        dflux[off*ssz+sid] = -M_1_PI * Ival_tot.deriv[off];
}

#undef NP1
#undef LDA
#undef NMAX
#undef DTHETA

/* vim: set ft=opencl.doxygen: */
