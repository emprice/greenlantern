/**
 * @file greenlantern.h
 * @brief Common definitions for the @c greenlantern Python extension
 */

#ifndef GREENLANTERN_H
#define GREENLANTERN_H

/** Standardizes the definition of @c size_t for Python extensions */
#define PY_SSIZE_T_CLEAN

#include <Python.h>

/** Exception object for OpenCL-specific errors */
extern PyObject *greenlantern_ocl_error;

extern const char greenlantern_ocl_fmt_internal[];

#include "pocky.h"
#include "greenlantern_context.h"

extern PyObject *ellipsoid_transit_flux(greenlantern_context_object *context,
    PyObject *args, PyObject *kwargs);

#endif      /* GREENLANTERN_H */

/* vim: set ft=c.doxygen: */
