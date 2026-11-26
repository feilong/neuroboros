/* Compiled kernel for neuroboros.surface.voronoi.
 *
 * The functions here must give exactly the results of the NumPy code that
 * they replace. Every product, sum and square root is therefore rounded on
 * its own, in the order in which it is written. The file must be compiled
 * without fused multiply-add and without fast-math (see setup.py);
 * multiply_add lets the caller check this when the module is loaded.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef __FAST_MATH__
#error "_voronoi_kernel.c must be compiled without fast-math."
#endif
/* GCC has no pragma for this and needs -ffp-contract=off. */
#if defined(_MSC_VER)
#pragma fp_contract(off)
#elif defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

PyDoc_STRVAR(multiply_add_doc,
"multiply_add(a, b, c)\n--\n\n"
"Return a * b + c, as this file computes it.\n\n"
"With separate rounding of the product and the sum, the result equals\n"
"Python's. A fused multiply-add gives another result for some values.");

static PyObject *
multiply_add(PyObject *self, PyObject *args)
{
    double a, b, c;

    if (!PyArg_ParseTuple(args, "ddd", &a, &b, &c)) {
        return NULL;
    }
    return PyFloat_FromDouble(a * b + c);
}

static PyMethodDef kernel_methods[] = {
    {"multiply_add", multiply_add, METH_VARARGS, multiply_add_doc},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef kernel_module = {
    PyModuleDef_HEAD_INIT,
    "_voronoi_kernel",
    "Compiled kernel for neuroboros.surface.voronoi.",
    -1,
    kernel_methods,
};

PyMODINIT_FUNC
PyInit__voronoi_kernel(void)
{
    return PyModule_Create(&kernel_module);
}
