#include "greenlantern.h"

#define NO_IMPORT_POCKY
#include "pocky_api.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL greenlantern_ARRAY_API
#include <numpy/arrayobject.h>

PyObject *ellipsoid_transit_flux_dual(greenlantern_context_object *context,
    PyObject *args, PyObject *kwargs)
{
    char buf[BUFSIZ];
    char *keys[] = { "time", "params", "binsize", "locked", "flux", "dflux", "queue", NULL };

    PyObject *binsize = NULL, *queue_idx = NULL, *locked_flag = Py_False;
    pocky_bufpair_object *time, *params, *flux = NULL, *dflux = NULL;

    cl_int err;
    cl_command_queue queue;
    cl_kernel kernel = NULL;
    cl_event event;
    cl_ushort locked;

    long worksz[2], dfluxsz[2];
    size_t kernsz[2], locsz[2];

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|$O!O!O!O!O!", keys,
        pocky_api->bufpair_type, &time, pocky_api->bufpair_type, &params,
        &PyFloat_Type, &binsize, &PyBool_Type, &locked_flag,
        pocky_api->bufpair_type, &flux, pocky_api->bufpair_type, &dflux,
        &PyLong_Type, &queue_idx)) return NULL;

    if ((time->host == NULL) ||
        (!PyArray_CheckExact((PyArrayObject *) time->host)) ||
        (PyArray_NDIM((PyArrayObject *) time->host) != 1) ||
        (PyArray_TYPE((PyArrayObject *) time->host) != NPY_FLOAT32))
    {
        /* Invalid time array */
        PyErr_SetString(PyExc_ValueError,
            "Host time array should be a one-dimensional array of float32");
        return NULL;
    }

    if ((params->host == NULL) ||
        (!PyArray_CheckExact((PyArrayObject *) params->host)) ||
        (PyArray_NDIM((PyArrayObject *) params->host) != 2) ||
        (PyArray_DIM((PyArrayObject *) params->host, 0) != 1) ||
        (PyArray_TYPE((PyArrayObject *) params->host) != NPY_FLOAT32))
    {
        /* Invalid params array */
        PyErr_SetString(PyExc_ValueError,
            "Host parameters should be a two-dimensional array of float32");
        return NULL;
    }

    /* Construct the global dimensions for the kernel */
    worksz[0] = PyArray_DIM((PyArrayObject *) time->host, 0);
    worksz[1] = 1;

    /* Create a flux buffer if needed */
    if ((flux == NULL) &&
        (pocky_api->bufpair_empty_from_shape(context->pocky, 1, worksz, &flux)))
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not create a flux array on-the-fly");
        return NULL;
    }

    dfluxsz[0] = 12;
    dfluxsz[1] = worksz[0];

    /* Create a dflux buffer if needed */
    if ((dflux == NULL) &&
        (pocky_api->bufpair_empty_from_shape(context->pocky, 2, dfluxsz, &dflux)))
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not create a dflux array on-the-fly");
        return NULL;
    }

    if (worksz[0] == 0)
    {
        /* Early exit -- no work to do */
        return PyTuple_Pack(2, flux, dflux);
    }

    /* Extract the queue handle (first by default) and kernel */
    if (!queue_idx) queue = context->pocky->queues[0];
    else
    {
        long idx = PyLong_AsLong(queue_idx);
        queue = context->pocky->queues[idx];
    }

    /* Choose the appropriate kernel */
    if (!binsize) kernel = context->kernels.ellipsoid_transit_flux_dual;
    else kernel = context->kernels.ellipsoid_transit_flux_binned_dual;

    /* Copy data to the device */
    if (time->dirty == Py_True)
    {
        err = clEnqueueWriteBuffer(queue, time->device,
            CL_TRUE, 0, time->host_size * sizeof(cl_float),
            PyArray_DATA((PyArrayObject *) time->host), 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            snprintf(buf, BUFSIZ, greenlantern_ocl_fmt_internal,
                pocky_api->opencl_error_to_string(err), err);
            PyErr_SetString(greenlantern_ocl_error, buf);
            return NULL;
        }
    }

    if (params->dirty == Py_True)
    {
        err = clEnqueueWriteBuffer(queue, params->device,
            CL_TRUE, 0, params->host_size * sizeof(cl_float),
            PyArray_DATA((PyArrayObject *) params->host), 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            snprintf(buf, BUFSIZ, greenlantern_ocl_fmt_internal,
                pocky_api->opencl_error_to_string(err), err);
            PyErr_SetString(greenlantern_ocl_error, buf);
            return NULL;
        }
    }

    /* Ready to run kernel */
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(time->device));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &(params->device));

    locked = (locked_flag == Py_True) ? 1 : 0;
    clSetKernelArg(kernel, 2, sizeof(cl_ushort), &locked);

    if (!binsize)
    {
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &(flux->device));
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &(dflux->device));

        locsz[0] = 32;  /* number of threads working on summation */
        locsz[1] = 1;   /* number of points to bin together (unused) */
    }
    else
    {
        float binsize_val = (float) PyFloat_AsDouble(binsize);
        clSetKernelArg(kernel, 3, sizeof(cl_float), &binsize_val);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &(flux->device));
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &(dflux->device));

        locsz[0] = 16;  /* number of threads working on summation */
        locsz[1] = 17;  /* number of points to bin together */
    }

    kernsz[0] = worksz[0] * locsz[0];
    kernsz[1] = worksz[1] * locsz[1];

    err = clEnqueueNDRangeKernel(queue, kernel,
        2, NULL, kernsz, locsz, 0, NULL, &event);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, greenlantern_ocl_fmt_internal,
            pocky_api->opencl_error_to_string(err), err);
        PyErr_SetString(greenlantern_ocl_error, buf);
        return NULL;
    }

    /* Block until the kernel has run */
    err = clWaitForEvents(1, &event);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, greenlantern_ocl_fmt_internal,
            pocky_api->opencl_error_to_string(err), err);
        PyErr_SetString(greenlantern_ocl_error, buf);
        return NULL;
    }
    clReleaseEvent(event);

    /* Copy data back and block until we have all the results */
    err = clEnqueueReadBuffer(queue, flux->device,
        CL_TRUE, 0, flux->host_size * sizeof(cl_float),
        PyArray_DATA((PyArrayObject *) flux->host), 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, greenlantern_ocl_fmt_internal,
            pocky_api->opencl_error_to_string(err), err);
        PyErr_SetString(greenlantern_ocl_error, buf);
        return NULL;
    }

    err = clEnqueueReadBuffer(queue, dflux->device,
        CL_TRUE, 0, dflux->host_size * sizeof(cl_float),
        PyArray_DATA((PyArrayObject *) dflux->host), 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, greenlantern_ocl_fmt_internal,
            pocky_api->opencl_error_to_string(err), err);
        PyErr_SetString(greenlantern_ocl_error, buf);
        return NULL;
    }

    return PyTuple_Pack(2, flux, dflux);
}

/* vim: set ft=c.doxygen: */
