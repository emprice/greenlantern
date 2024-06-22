#include "greenlantern.h"
#include "greenlantern_kernels.h"

#define NO_IMPORT_POCKY
#include "pocky_api.h"

PyTypeObject greenlantern_context_type;

PyObject *greenlantern_context_new(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
    greenlantern_context_object *self;

    if ((self = (greenlantern_context_object *) type->tp_alloc(type, 0)))
    {
        self->pocky = NULL;
        self->program = NULL;
    }

    return (PyObject *) self;
}

int greenlantern_context_init(greenlantern_context_object *context,
        PyObject *args, PyObject *kwargs)
{
    cl_int err;
    size_t idx;
    cl_kernel *kernels;
    cl_uint num_kernels;

    /* Expect a pocky Context object as the only argument */
    if (!PyArg_ParseTuple(args, "O!",
        pocky_api->context_type, &(context->pocky))) return -1;

    err = pocky_api->opencl_kernels_from_fragments(num_kernel_frags, kernel_frags,
        context->pocky->ctx, &(context->program), &num_kernels, &kernels);
    if (err != CL_SUCCESS) return -1;

    err = pocky_api->opencl_kernel_lookup_by_name(num_kernels, kernels,
        "ellipsoid_transit_flux_vector",
        &(context->kernels.ellipsoid_transit_flux_vector));
    if (err != CL_SUCCESS) return -1;
    clRetainKernel(context->kernels.ellipsoid_transit_flux_vector);

    err = pocky_api->opencl_kernel_lookup_by_name(num_kernels, kernels,
        "ellipsoid_transit_flux_binned_vector",
        &(context->kernels.ellipsoid_transit_flux_binned_vector));
    if (err != CL_SUCCESS) return -1;
    clRetainKernel(context->kernels.ellipsoid_transit_flux_binned_vector);

    /* Release any remaining kernel handles */
    for (idx = 0; idx < num_kernels; ++idx) clReleaseKernel(kernels[idx]);
    free(kernels);

    /* Make sure we retain a handle to pocky, even if it somehow
     * got garbage collected */
    Py_INCREF(context->pocky);

    return 0;       /* 0 indicates success for initproc */
}

void greenlantern_context_dealloc(greenlantern_context_object *self)
{
    /* Release all kernel handles */
    clReleaseKernel(self->kernels.ellipsoid_transit_flux_vector);
    clReleaseKernel(self->kernels.ellipsoid_transit_flux_binned_vector);

    /* Release other handles */
    clReleaseProgram(self->program);
    Py_TYPE(self)->tp_free((PyObject *) self);

    /* Release the handle to pocky */
    Py_DECREF(self->pocky);
}

PyMethodDef greenlantern_context_methods[] = {

    { "ellipsoid_transit_flux",
      (PyCFunction) ellipsoid_transit_flux,
      METH_VARARGS | METH_KEYWORDS,
      "ellipsoid_transit_flux(alpha: pocky.BufferPair, params: "
      "pocky.BufferPair, output: pocky.BufferPair) -> pocky.BufferPair\n"
      "Compute an ellipsoid transit lightcurve.\n\n"
      "Args:\n"
      "  alpha: Mean anomaly of the orbit\n"
      "  params: Planet and transit parameters\n"
      "  binsize: Mean anomaly bin size\n"
      "  output: Optional pre-allocated output buffer\n\n"
      "Returns:\n"
      "  A buffer of output values\n" },

    { NULL, NULL, 0, NULL }    /* sentinel */
};

/* vim: set ft=c.doxygen: */
