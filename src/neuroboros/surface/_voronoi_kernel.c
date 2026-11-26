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

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

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

/* Fill ``view`` with the C-contiguous buffer of ``obj`` and check that it
 * holds floats (kind 'f') or signed integers (kind 'i') of the given size,
 * with the given number of dimensions. Return 0, or -1 with an exception.
 */
static int
get_array(PyObject *obj, Py_buffer *view, const char *name, char kind,
          Py_ssize_t itemsize, int ndim, int writable)
{
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    const char *format;

    if (writable) {
        flags |= PyBUF_WRITABLE;
    }
    if (PyObject_GetBuffer(obj, view, flags) < 0) {
        return -1;
    }
    format = view->format ? view->format : "B";
    if (*format == '@' || *format == '=') {
        format++;
    }
    if (view->ndim != ndim || view->itemsize != itemsize || format[1] != '\0'
            || !strchr(kind == 'f' ? "d" : "ilqn", format[0])) {
        PyErr_Format(PyExc_TypeError,
                     "%s must be a %d-D array of %d-byte %s.", name, ndim,
                     (int)itemsize, kind == 'f' ? "floats" : "integers");
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}

/* The arrays of label_samples, with their sizes. */
typedef struct {
    const double *ww;            /* (n_samples, 3) */
    const double *coords;        /* (n_vertices, 3) */
    const Py_ssize_t *faces;     /* (n_mesh_faces, 3) */
    const double *node_coords;   /* (n_nodes, 3) */
    const Py_ssize_t *face_nodes;/* (n_mesh_faces, n_boundary) */
    const Py_ssize_t *face_indices; /* (n_faces,) */
    const double *nnd;           /* (n_partitions, n_nodes) */
    const int32_t *nn;           /* (n_partitions, n_nodes) */
    const int32_t *single;       /* (n_partitions, n_mesh_faces) */
    int32_t *out;                /* (n_partitions, n_faces, n_samples) */
    Py_ssize_t n_samples, n_vertices, n_mesh_faces, n_nodes, n_boundary;
    Py_ssize_t n_faces, n_partitions;
} LabelArgs;

/* Label the samples of the faces. Return 0, 1 if out of memory, or 2 if an
 * index is out of range. Called without the interpreter lock.
 */
static int
label_samples_loop(const LabelArgs *a)
{
    const Py_ssize_t n_boundary = a->n_boundary, n_partitions = a->n_partitions;
    const Py_ssize_t n_samples = a->n_samples;
    Py_ssize_t f, j, p, s, k;
    /* Of the boundary nodes of one face: the coordinates and the distances
     * to one sample, then the distance and the label for each partition. */
    double *scratch = malloc((4 + n_partitions) * n_boundary * sizeof(double));
    int32_t *node_labels = malloc(n_partitions * n_boundary * sizeof(int32_t));
    double *bx = scratch, *by = bx + n_boundary, *bz = by + n_boundary;
    double *dist = bz + n_boundary, *node_dists = dist + n_boundary;

    if (scratch == NULL || node_labels == NULL) {
        free(scratch);
        free(node_labels);
        return 1;
    }
    for (f = 0; f < a->n_faces; f++) {
        const Py_ssize_t face = a->face_indices[f];
        const Py_ssize_t *corners, *nodes;
        const double *c0, *c1, *c2;
        int any_mixed = 0;

        if (face < 0 || face >= a->n_mesh_faces) {
            goto out_of_range;
        }
        for (p = 0; p < n_partitions; p++) {
            int32_t *out = a->out + (p * a->n_faces + f) * n_samples;
            const int32_t label = a->single[p * a->n_mesh_faces + face];

            if (label < 0) {
                any_mixed = 1;
                continue;
            }
            /* Every sample of the face has the same label. */
            for (s = 0; s < n_samples; s++) {
                out[s] = label;
            }
        }
        if (!any_mixed) {
            continue;
        }

        corners = a->faces + 3 * face;
        nodes = a->face_nodes + face * n_boundary;
        for (k = 0; k < 3; k++) {
            if (corners[k] < 0 || corners[k] >= a->n_vertices) {
                goto out_of_range;
            }
        }
        c0 = a->coords + 3 * corners[0];
        c1 = a->coords + 3 * corners[1];
        c2 = a->coords + 3 * corners[2];
        for (j = 0; j < n_boundary; j++) {
            const Py_ssize_t node = nodes[j];

            if (node < 0 || node >= a->n_nodes) {
                goto out_of_range;
            }
            bx[j] = a->node_coords[3 * node];
            by[j] = a->node_coords[3 * node + 1];
            bz[j] = a->node_coords[3 * node + 2];
            for (p = 0; p < n_partitions; p++) {
                node_dists[p * n_boundary + j] = a->nnd[p * a->n_nodes + node];
                node_labels[p * n_boundary + j] = a->nn[p * a->n_nodes + node];
            }
        }

        for (s = 0; s < n_samples; s++) {
            const double *w = a->ww + 3 * s;
            /* The sample, as _barycentric_points computes it. */
            const double x = w[0] * c0[0] + w[1] * c1[0] + w[2] * c2[0];
            const double y = w[0] * c0[1] + w[1] * c1[1] + w[2] * c2[1];
            const double z = w[0] * c0[2] + w[1] * c1[2] + w[2] * c2[2];

            /* Its distance to each boundary node, shared by the partitions. */
            for (j = 0; j < n_boundary; j++) {
                const double dx = x - bx[j], dy = y - by[j], dz = z - bz[j];

                dist[j] = sqrt(dx * dx + dy * dy + dz * dz);
            }
            for (p = 0; p < n_partitions; p++) {
                const double *pd = node_dists + p * n_boundary;
                double best_score;
                Py_ssize_t best = 0;

                if (a->single[p * a->n_mesh_faces + face] >= 0) {
                    continue;
                }
                /* The boundary node with the lowest score, and the first of
                 * them if several have it. */
                best_score = pd[0] + dist[0];
                for (j = 1; j < n_boundary; j++) {
                    const double score = pd[j] + dist[j];

                    if (score < best_score) {
                        best_score = score;
                        best = j;
                    }
                }
                a->out[(p * a->n_faces + f) * n_samples + s] =
                    node_labels[p * n_boundary + best];
            }
        }
    }
    free(scratch);
    free(node_labels);
    return 0;

out_of_range:
    free(scratch);
    free(node_labels);
    return 2;
}

PyDoc_STRVAR(label_samples_doc,
"label_samples(ww, coords, faces, node_coords, face_nodes, face_indices,\n"
"              nnd, nn, single, out)\n--\n\n"
"Label the samples of some faces for several partitions.\n\n"
"``out[p, i, s]`` becomes the label of sample ``s`` of face\n"
"``face_indices[i]`` in partition ``p``: ``single[p, face]`` if it is not\n"
"negative, and otherwise the label ``nn[p, node]`` of the boundary node of\n"
"the face with the lowest ``nnd[p, node]`` plus distance to the sample. The\n"
"first boundary node wins a tie. The distances from a sample to the\n"
"boundary nodes are computed once for all partitions.\n\n"
"The integer arrays have 32-bit elements, except ``faces``, ``face_nodes``\n"
"and ``face_indices``, which have the size of a pointer. The other arrays\n"
"are float64. All arrays are C-contiguous. The shapes are: ``ww``\n"
"(n_samples, 3); ``coords`` (nv, 3); ``faces`` (nf, 3); ``node_coords``\n"
"(n_nodes, 3); ``face_nodes`` (nf, n_boundary); ``face_indices`` (k,);\n"
"``nnd`` and ``nn`` (n_partitions, n_nodes); ``single`` (n_partitions, nf);\n"
"``out`` (n_partitions, k, n_samples).");

#define N_LABEL_ARRAYS 10

static PyObject *
label_samples(PyObject *self, PyObject *args)
{
    static const char *names[N_LABEL_ARRAYS] = {
        "ww", "coords", "faces", "node_coords", "face_nodes", "face_indices",
        "nnd", "nn", "single", "out"};
    static const char kinds[N_LABEL_ARRAYS + 1] = "ffifiifiii";
    static const int ndims[N_LABEL_ARRAYS] = {2, 2, 2, 2, 2, 1, 2, 2, 2, 3};
    /* The element sizes; 0 stands for the size of a pointer. */
    static const int sizes[N_LABEL_ARRAYS] = {8, 8, 0, 8, 0, 0, 8, 4, 4, 4};
    PyObject *objects[N_LABEL_ARRAYS];
    Py_buffer views[N_LABEL_ARRAYS];
    Py_ssize_t *shapes[N_LABEL_ARRAYS];
    LabelArgs a;
    int i, n_views = 0, status = 0;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOO", &objects[0], &objects[1],
                          &objects[2], &objects[3], &objects[4], &objects[5],
                          &objects[6], &objects[7], &objects[8], &objects[9])) {
        return NULL;
    }
    for (i = 0; i < N_LABEL_ARRAYS; i++) {
        const Py_ssize_t size = sizes[i] ? sizes[i] : (int)sizeof(Py_ssize_t);

        if (get_array(objects[i], &views[i], names[i], kinds[i], size,
                      ndims[i], i == N_LABEL_ARRAYS - 1) < 0) {
            goto fail;
        }
        shapes[i] = views[i].shape;
        n_views++;
    }

    a.ww = views[0].buf;
    a.coords = views[1].buf;
    a.faces = views[2].buf;
    a.node_coords = views[3].buf;
    a.face_nodes = views[4].buf;
    a.face_indices = views[5].buf;
    a.nnd = views[6].buf;
    a.nn = views[7].buf;
    a.single = views[8].buf;
    a.out = views[9].buf;
    a.n_samples = shapes[0][0];
    a.n_vertices = shapes[1][0];
    a.n_mesh_faces = shapes[2][0];
    a.n_nodes = shapes[3][0];
    a.n_boundary = shapes[4][1];
    a.n_faces = shapes[5][0];
    a.n_partitions = shapes[6][0];
    if (shapes[0][1] != 3 || shapes[1][1] != 3 || shapes[2][1] != 3
            || shapes[3][1] != 3 || shapes[4][0] != a.n_mesh_faces
            || a.n_boundary < 1 || shapes[6][1] != a.n_nodes
            || shapes[7][0] != a.n_partitions || shapes[7][1] != a.n_nodes
            || shapes[8][0] != a.n_partitions
            || shapes[8][1] != a.n_mesh_faces
            || shapes[9][0] != a.n_partitions || shapes[9][1] != a.n_faces
            || shapes[9][2] != a.n_samples) {
        PyErr_SetString(PyExc_ValueError,
                        "The shapes of the arrays do not match.");
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    status = label_samples_loop(&a);
    Py_END_ALLOW_THREADS
    if (status == 1) {
        PyErr_NoMemory();
        goto fail;
    }
    if (status == 2) {
        PyErr_SetString(PyExc_IndexError,
                        "A face, vertex or node index is out of range.");
        goto fail;
    }
    for (i = 0; i < n_views; i++) {
        PyBuffer_Release(&views[i]);
    }
    Py_RETURN_NONE;

fail:
    for (i = 0; i < n_views; i++) {
        PyBuffer_Release(&views[i]);
    }
    return NULL;
}

static PyMethodDef kernel_methods[] = {
    {"multiply_add", multiply_add, METH_VARARGS, multiply_add_doc},
    {"label_samples", label_samples, METH_VARARGS, label_samples_doc},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef kernel_module = {
    PyModuleDef_HEAD_INIT,
    "_voronoi_kernel",
    "Compiled kernel for neuroboros.surface.voronoi.",
    -1,
    kernel_methods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit__voronoi_kernel(void)
{
    return PyModule_Create(&kernel_module);
}
