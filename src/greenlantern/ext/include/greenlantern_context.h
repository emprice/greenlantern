#ifndef GREENLANTERN_CONTEXT_H
#define GREENLANTERN_CONTEXT_H

/** Internal data for the @c greenlantern.ext.Context object */
typedef struct
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    PyObject_HEAD
#endif  /* DOXYGEN_SHOULD_SKIP_THIS */
    pocky_context_object *pocky;
    cl_program program;             /**< OpenCL program handle */

    struct
    {
        cl_kernel ellipsoid_transit_flux_single;
        cl_kernel ellipsoid_transit_flux_vector;
        cl_kernel ellipsoid_transit_flux_binned_vector;
    }
    kernels;    /**< Named kernel handles */
}
greenlantern_context_object;

/** Python type object for the @c greenlantern.ext.Context object type */
extern PyTypeObject greenlantern_context_type;

/** Methods table for the @c greenlantern.ext.Context object type */
extern PyMethodDef greenlantern_context_methods[];

/**
 * @brief Allocates and initializes an empty @c greenlantern.ext.Context object
 * @param[in] type Type of object to allocate
 * @param[in] args Python arguments to be parsed
 * @param[in] kwargs Python keyword arguments to be parsed
 * @return A new @c greenlantern.ext.Context object
 */
extern PyObject *greenlantern_context_new(PyTypeObject *type,
        PyObject *args, PyObject *kwargs);

/**
 * @brief Initializes a @c greenlantern.ext.Context object
 * @param[in] context Allocated object to initialize
 * @param[in] args Python arguments to be parsed
 * @param[in] kwargs Python keyword arguments to be parsed
 * @return On success, 0; otherwise, -1
 */
extern int greenlantern_context_init(greenlantern_context_object *context,
        PyObject *args, PyObject *kwargs);

/**
 * @brief Deallocates a Python @c greenlantern.ext.Context object
 * @param[in] self Object to be deallocated
 */
extern void greenlantern_context_dealloc(greenlantern_context_object *self);

#endif      /* GREENLANTERN_CONTEXT_H */

/* vim: set ft=c.doxygen: */
