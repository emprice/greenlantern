#define FTOL    (1e-6)
#define DTOL    (1e-4)
#define MAXREP  (16)

/**
 * @brief Quickly reduce the range of the mean anomaly to [-pi, pi]; this
 * step is necessary before the approximation step is valid
 * @param[in] M Mean anomaly
 * @return Range-reduced mean anomaly
 */
float meananom_range_reduce(float M)
{
    float M1;
    M1 = fmod(M, 2 * M_PI);
    if (fabs(M1) > M_PI)
        M1 -= copysign(2 * M_PI, M1);
    return M1;
}

/**
 * @brief Generate a first approximation of the eccentric anomaly based on
 * evaluating a series expansion
 * @param[in] M Mean anomaly
 * @param[in] e Orbit eccentricity
 * @return Eccentric anomaly approximation
 */
float eccanom_approx(float M, float e)
{
    float accum;

    float M1 = fabs(M) - M_PI_F;
    float ep1 = 1 + e;

    float e3  = e;
    float e5  = e * (-1 + 9 * e);
    float e7  = e * ( 1 + 9 * e * (-6 + 25 * e));
    float e9  = e * (-1 + 9 * e * (27 + e * (-459 + 1225 * e)));
    float e11 = e * ( 1 + 9 * e * (-112 + e * (5574 + 25 * e * (-2032 + 3969 * e))));

    float invep1 = 1. / ep1;
    float recip = invep1;
    float ratio = M1 * invep1;
    float ratio2 = ratio * ratio;

    accum  = ratio + M_PI; ratio *= ratio2;
    accum += e3  * ratio * recip / 6; ratio *= ratio2; recip *= invep1;
    accum += e5  * ratio * recip / 120; ratio *= ratio2; recip *= invep1;
    accum += e7  * ratio * recip / 5040; ratio *= ratio2; recip *= invep1;
    accum += e9  * ratio * recip / 362880; ratio *= ratio2; recip *= invep1;
    accum += e11 * ratio * recip / 39916800;

    return copysign(accum, M);
}

/**
 * @brief Evaluate the value and derivative of the defining equation for the
 * eccentric anomaly, M = E - e sin E
 * @param[in] M Mean anomaly
 * @param[in] e Orbit eccentricity
 * @param[in] E Proposal value for the eccentric anomaly
 * @param[out] df Derivative of the residual function evaluated at E
 * @return Value of the residual function evaluated at E
 */
float2 eccanom_value_and_derivative(float M, float e, float E)
{
    float sE, cE;
    float2 ret;

    sE = sincos(E, &cE);
    ret.y = 1 - e * cE;
    ret.x = (E - e * sE) - M;
    return ret;
}

/**
 * @brief Solve for the eccentric anomaly using Newton's method
 * @param[in] M Mean anomaly
 * @param[in] e Orbit eccentricity
 * @param[in] E Initial guess for eccentric anomaly
 * @return Computed value of eccentric anomaly
 */
float eccanom_newton_solve(float M, float e, float E)
{
    float dE;
    float2 fdf;
    int niter = 0;

    do
    {
        fdf = eccanom_value_and_derivative(M, e, E);
        dE = fdf.x / fdf.y;
        E -= dE;
    }
    while ((niter++ < MAXREP) && (fabs(fdf.x) > FTOL) && (fabs(dE) > DTOL));

    return E;
}

/**
 * @brief Convenience function for computing the eccentric anomaly from the
 * mean anomaly
 * @param[in] M Mean anomaly
 * @param[in] e Orbit eccentricity
 * @return Eccentric anomaly
 */
float eccanom_calc(float M, float e)
{
    float E;

    M = meananom_range_reduce(M);
    E = eccanom_approx(M, e);
    E = eccanom_newton_solve(M, e, E);

    return E;
}

float trueanom_calc(float M, float e)
{
    float E;
    E = eccanom_calc(M, e);
    return 2 * atan(sqrt((1 + e) / (1 - e)) * tan(0.5 * E));
}

float2 eccanom_and_trueanom_calc(float M, float e)
{
    float2 anomaly;
    anomaly.x = eccanom_calc(M, e);
    anomaly.y = 2 * atan(sqrt((1 + e) / (1 - e)) * tan(0.5 * anomaly.x));
    return anomaly;
}

float time_of_periastron_calc(float e, float om, float tc, float n)
{
    float eccanom = 2 * atan(sqrt((1 - e) / (1 + e)) * tan(0.5 * (om - M_PI_2)));
    float meananom = eccanom - e * sin(eccanom);
    return n * meananom + tc;
}

__kernel void eccentric_anomaly(__global float *M, float e, __global float *E)
{
    int gid = get_global_id(0);
    E[gid] = eccanom_calc(M[gid], e);
}

__kernel void true_anomaly(__global float *M, float e, __global float *F)
{
    int gid = get_global_id(0);
    F[gid] = trueanom_calc(M[gid], e);
}

/* vim: set ft=opencl.doxygen: */
