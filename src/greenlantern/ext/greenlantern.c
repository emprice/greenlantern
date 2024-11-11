#include "greenlantern.h"
#include "pocky_api.h"

#define PY_ARRAY_UNIQUE_SYMBOL greenlantern_ARRAY_API
#include <numpy/arrayobject.h>

PyObject *greenlantern_ocl_error;

/** Docstring for this Python extension */
static const char greenlantern_module_doc[] = "This is a stub.";

/** Docstring for the @c greenlantern.ext.Context type */
static const char greenlantern_context_type_doc[] = "This is a stub.";

const char greenlantern_ocl_fmt_internal[] =
    "Internal OpenCL error occurred with code %s (%d)";

/** Methods available at the module level */
static PyMethodDef greenlantern_methods[] = {
    { NULL, NULL, 0, NULL }     /* sentinel value */
};

/** Module definition */
struct PyModuleDef greenlantern_module = {
    PyModuleDef_HEAD_INIT,
    "greenlantern.ext",         /* module name */
    greenlantern_module_doc,    /* module documentation */
    -1,                         /* size of per-interpreter state of the module */
    greenlantern_methods        /* methods table */
};

/** Module entry point */
PyMODINIT_FUNC PyInit_ext(void)
{
    PyObject *mod;

    mod = PyModule_Create(&greenlantern_module);
    if (!mod) return NULL;

    /* Define exceptions */
    greenlantern_ocl_error = PyErr_NewException("greenlantern.ext.OpenCLError", NULL, NULL);

    /* Definition of the Context type */
    greenlantern_context_type = (PyTypeObject) {
        .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "greenlantern.ext.Context",
        .tp_doc = greenlantern_context_type_doc,
        .tp_basicsize = sizeof(greenlantern_context_object),
        .tp_itemsize = 0,
        .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .tp_new = greenlantern_context_new,
        .tp_init = (initproc) greenlantern_context_init,
        .tp_dealloc = (destructor) greenlantern_context_dealloc,
        .tp_methods = greenlantern_context_methods,
    };

    /* Attach the module components */
    if (PyModule_AddObject(mod, "OpenCLError", greenlantern_ocl_error) ||
        PyType_Ready(&greenlantern_context_type) ||
        PyModule_AddType(mod, &greenlantern_context_type))
    {
        Py_XDECREF(greenlantern_ocl_error);

        Py_DECREF(mod);
        return NULL;
    }

    import_array();
    import_pocky();

    return mod;
}

/* vim: set ft=c.doxygen: */
