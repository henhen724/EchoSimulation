import numpy as np
import cupy as cp

ellipj = cp.ElementwiseKernel('float64 u, float64 m', 'float64 sn, float64 cn, float64 dn, float64 ph',
                              '''
        double ai, b, phi, t, twon, dnfac;
        double a[9], c[9];
        int j;

        /* Check for special cases */
        if (m < 0.0 || m > 1.0) {
            sn = NPY_NAN;
            cn = NPY_NAN;
            ph = NPY_NAN;
            dn = NPY_NAN;
            return;
        }
        if (m < 1.0e-9) {
            t = sin(u);
            b = cos(u);
            ai = 0.25 * m * (u - t * b);
            sn = t - ai * b;
            cn = b + ai * t;
            ph = u - ai;
            dn = 1.0 - 0.5 * m * t * t;
            return;
        }
        if (m >= 0.9999999999) {
            ai = 0.25 * (1.0 - m);
            b = cosh(u);
            t = tanh(u);
            phi = 1.0 / b;
            twon = b * sinh(u);
            sn = t + ai * (twon - u) / (b * b);
            ph = 2.0 * atan(exp(u)) - NPY_PI_2 + ai * (twon - u) / b;
            ai *= t * phi;
            cn = phi - ai * (twon - u);
            dn = phi + ai * (twon + u);
            return;
        }

        /* A. G. M. scale. See DLMF 22.20(ii) */
        a[0] = 1.0;
        b = sqrt(1.0 - m);
        c[0] = sqrt(m);
        twon = 1.0;
        j = 0;

        while (fabs(c[j] / a[j]) > MACHEP) {
            if (j > 7) {
                goto done;
            }
            ai = a[j];
            ++j;
            c[j] = (ai - b) / 2.0;
            t = sqrt(ai * b);
            a[j] = (ai + b) / 2.0;
            b = t;
            twon *= 2.0;
        }

    done:
        /* backward recurrence */
        phi = twon * a[j] * u;
        do {
            t = c[j] * sin(phi) / a[j];
            b = phi;
            phi = (asin(t) + phi) / 2.0;
        }
        while (--j);

        sn = sin(phi);
        t = cos(phi);
        cn = t;
        dnfac = cos(phi - b);
        /* See discussion after DLMF 22.20.5 */
        if (fabs(dnfac) < 0.1) {
            dn = sqrt(1 - m*(sn)*(sn));
        }
        else {
            dn = t / dnfac;
        }
        ph = phi;
        return;
    ''',
                              preamble='''
    __device__ static float nanf(void) {
        const union { int i; float f;} bint = {0x7fc00000UL};
        return bint.f;
    }

    #define NPY_PI_2      1.570796326794896619231321691639751442  /* pi/2 */
    #define NPY_NAN  ((double) nanf())
    #define MACHEP   1.11022302462515654042E-16
    ''',
                              name="jacobi_elliptic_trig")


def wrapped_ellipj_gpu(m, u):
    output_numpy = False
    if type(m) is np.ndarray:
        output_numpy = True
        m = cp.array(m)
    if type(u) is np.ndarray:
        output_numpy = True
        u = cp.array(u)

    sn, cn, dn, ph = ellipj(m, u)

    if output_numpy:
        return cp.asnumpy(sn), cp.asnumpy(cn), cp.asnumpy(dn), cp.asnumpy(ph)
    return sn, cn, dn, ph
