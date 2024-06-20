#include "greenlantern.h"

#define NO_IMPORT_POCKY
#include "pocky_api.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL greenlantern_ARRAY_API
#include <numpy/arrayobject.h>

PyObject *ellipsoid_transit_flux(greenlantern_context_object *context,
    PyObject *args, PyObject *kwargs)
{
    char buf[BUFSIZ];
    char *keys[] = { "alpha", "params", "binsize", "output", "queue", NULL };

    PyObject *queue_idx = NULL;
    PyObject *binsize = NULL;
    pocky_bufpair_object *input, *params, *output = NULL;

    cl_int err;
    cl_command_queue queue;
    cl_kernel kernel = NULL;
    cl_event event;

    long worksz[2];
    size_t kernsz[2], locsz[2];

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|$O!O!O!", keys,
        pocky_api->bufpair_type, &input, pocky_api->bufpair_type, &params,
        &PyFloat_Type, &binsize, pocky_api->bufpair_type, &output,
        &PyLong_Type, &queue_idx)) return NULL;

    if ((input->host == NULL) ||
        (!PyArray_CheckExact((PyArrayObject *) input->host)) ||
        (PyArray_NDIM((PyArrayObject *) input->host) != 1) ||
        (PyArray_TYPE((PyArrayObject *) input->host) != NPY_FLOAT32))
    {
        /* Invalid input array */
        PyErr_SetString(PyExc_ValueError,
            "Host input should be a one-dimensional array of float32");
        return NULL;
    }

    if ((params->host == NULL) ||
        (!PyArray_CheckExact((PyArrayObject *) params->host)) ||
        (PyArray_NDIM((PyArrayObject *) params->host) != 2) ||
        (PyArray_TYPE((PyArrayObject *) params->host) != NPY_FLOAT32))
    {
        /* Invalid input array */
        PyErr_SetString(PyExc_ValueError,
            "Host parameters should be a two-dimensional array of float32");
        return NULL;
    }

    /* Construct the global dimensions for the kernel */
    worksz[0] = PyArray_DIM((PyArrayObject *) params->host, 0);
    worksz[1] = PyArray_DIM((PyArrayObject *) input->host, 0);

    /* Create a buffer if needed */
    if ((output == NULL) &&
        (pocky_api->bufpair_empty_from_shape(context->pocky, 2, worksz, &output)))
    {
        PyErr_SetString(PyExc_RuntimeError, "Could not create an "
            "output array on-the-fly");
        return NULL;
    }

    if ((worksz[0] == 0) || (worksz[1] == 0))
    {
        /* Early exit -- no work to do */
        Py_INCREF(output);
        return (PyObject *) output;
    }

    /* Extract the queue handle (first by default) and kernel */
    if (!queue_idx) queue = context->pocky->queues[0];
    else
    {
        long idx = PyLong_AsLong(queue_idx);
        queue = context->pocky->queues[idx];
    }

    /* Choose the appropriate kernel */
    if (!binsize) kernel = context->kernels.ellipsoid_transit_flux_vector;
    else kernel = context->kernels.ellipsoid_transit_flux_binned_vector;

    /* Copy data to the device */
    if (input->dirty == Py_True)
    {
        err = clEnqueueWriteBuffer(queue, input->device,
            CL_TRUE, 0, input->host_size * sizeof(cl_float),
            PyArray_DATA((PyArrayObject *) input->host), 0, NULL, NULL);
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
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &(input->device));
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &(params->device));

    locsz[0] = 32;

    if (!binsize)
    {
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &(output->device));
        locsz[1] = 1;
    }
    else
    {
        float binsize_val = (float) PyFloat_AsDouble(binsize);
        clSetKernelArg(kernel, 2, sizeof(cl_float), &binsize_val);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &(output->device));
        locsz[1] = 17;   /* number of points to bin together */
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

    /* Copy data back and block until we have all the results */
    err = clEnqueueReadBuffer(queue, output->device,
        CL_TRUE, 0, output->host_size * sizeof(cl_float),
        PyArray_DATA((PyArrayObject *) output->host), 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        snprintf(buf, BUFSIZ, greenlantern_ocl_fmt_internal,
            pocky_api->opencl_error_to_string(err), err);
        PyErr_SetString(greenlantern_ocl_error, buf);
        return NULL;
    }

    Py_INCREF(output);
    return (PyObject *) output;
}

/* vim: set ft=c.doxygen: */
