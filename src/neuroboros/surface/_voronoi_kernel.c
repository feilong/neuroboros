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
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(_MSC_VER)
#pragma fp_contract(off)
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
 * holds floats (kind 'f'), signed integers (kind 'i') or booleans (kind 'b')
 * of the given size, with the given number of dimensions. Return 0, or -1
 * with an exception.
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
    if (view->ndim != ndim || view->itemsize != itemsize
            || format[0] == '\0' || format[1] != '\0'
            || !strchr(kind == 'f' ? "d" : kind == 'b' ? "?" : "ilqn",
                       format[0])) {
        PyErr_Format(PyExc_TypeError,
                     "%s must be a %d-D array of %d-byte %s.", name, ndim,
                     (int)itemsize,
                     kind == 'f' ? "floats" : kind == 'b' ? "booleans"
                                                         : "integers");
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
    const Py_ssize_t *inside_ptr;/* (n_partitions, n_mesh_faces + 1) */
    const double *inside_coords; /* (n_inside, 3) */
    const int32_t *inside_labels;/* (n_inside,) */
    int32_t *out;                /* (n_partitions, n_faces, n_samples) */
    Py_ssize_t n_samples, n_vertices, n_mesh_faces, n_nodes, n_boundary;
    Py_ssize_t n_faces, n_partitions, n_inside;
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
    /* At least one element: malloc(0) may return NULL. */
    int32_t *node_labels =
        malloc((n_partitions * n_boundary + 1) * sizeof(int32_t));
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
                const Py_ssize_t *ptr =
                    a->inside_ptr + p * (a->n_mesh_faces + 1) + face;

                if (ptr[0] < 0 || ptr[0] > ptr[1] || ptr[1] > a->n_inside) {
                    goto out_of_range;
                }
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
                const Py_ssize_t *ptr =
                    a->inside_ptr + p * (a->n_mesh_faces + 1) + face;
                double best_score;
                Py_ssize_t best = 0, q;
                int32_t label;

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
                label = node_labels[p * n_boundary + best];

                if (ptr[0] < ptr[1]) {
                    /* The nearest point inside the face, and the first of
                     * them if several are as near. It wins only if it is
                     * nearer than the score of the best boundary node. */
                    double nearest_dist = 0.0;
                    Py_ssize_t nearest = ptr[0];

                    for (q = ptr[0]; q < ptr[1]; q++) {
                        const double *point = a->inside_coords + 3 * q;
                        const double dx = x - point[0], dy = y - point[1];
                        const double dz = z - point[2];
                        const double d = sqrt(dx * dx + dy * dy + dz * dz);

                        if (q == ptr[0] || d < nearest_dist) {
                            nearest_dist = d;
                            nearest = q;
                        }
                    }
                    if (nearest_dist < best_score) {
                        label = a->inside_labels[nearest];
                    }
                }
                a->out[(p * a->n_faces + f) * n_samples + s] = label;
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
"              nnd, nn, single, inside_ptr, inside_coords, inside_labels,\n"
"              out)\n--\n\n"
"Label the samples of some faces for several partitions.\n\n"
"``out[p, i, s]`` becomes the label of sample ``s`` of face\n"
"``face_indices[i]`` in partition ``p``. It is ``single[p, face]`` if that\n"
"is not negative. Otherwise it is the label ``nn[p, node]`` of the boundary\n"
"node of the face with the lowest score, ``nnd[p, node]`` plus the distance\n"
"to the sample; the first boundary node wins a tie. The points of the\n"
"partition inside the face are ``inside_ptr[p, face]`` to\n"
"``inside_ptr[p, face + 1]`` of ``inside_coords`` and ``inside_labels``.\n"
"If the nearest of them, the first on a tie, is nearer to the sample than\n"
"that lowest score, the sample takes its label instead. The distances\n"
"from a sample to the boundary nodes are computed once for all partitions.\n\n"
"The integer arrays have 32-bit elements, except ``faces``, ``face_nodes``,\n"
"``face_indices`` and ``inside_ptr``, which have the size of a pointer. The\n"
"other arrays are float64. All arrays are C-contiguous. The shapes are:\n"
"``ww`` (n_samples, 3); ``coords`` (nv, 3); ``faces`` (nf, 3);\n"
"``node_coords`` (n_nodes, 3); ``face_nodes`` (nf, n_boundary);\n"
"``face_indices`` (k,); ``nnd`` and ``nn`` (n_partitions, n_nodes);\n"
"``single`` (n_partitions, nf); ``inside_ptr`` (n_partitions, nf + 1);\n"
"``inside_coords`` (n_inside, 3); ``inside_labels`` (n_inside,); ``out``\n"
"(n_partitions, k, n_samples).");

#define N_LABEL_ARRAYS 13

static PyObject *
label_samples(PyObject *self, PyObject *args)
{
    static const char *names[N_LABEL_ARRAYS] = {
        "ww", "coords", "faces", "node_coords", "face_nodes", "face_indices",
        "nnd", "nn", "single", "inside_ptr", "inside_coords", "inside_labels",
        "out"};
    static const char kinds[N_LABEL_ARRAYS + 1] = "ffifiifiiifii";
    static const int ndims[N_LABEL_ARRAYS] = {
        2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 3};
    /* The element sizes; 0 stands for the size of a pointer. */
    static const int sizes[N_LABEL_ARRAYS] = {
        8, 8, 0, 8, 0, 0, 8, 4, 4, 0, 8, 4, 4};
    PyObject *objects[N_LABEL_ARRAYS];
    Py_buffer views[N_LABEL_ARRAYS];
    Py_ssize_t *shapes[N_LABEL_ARRAYS];
    LabelArgs a;
    int i, n_views = 0, status = 0;

    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOO", &objects[0], &objects[1],
                          &objects[2], &objects[3], &objects[4], &objects[5],
                          &objects[6], &objects[7], &objects[8], &objects[9],
                          &objects[10], &objects[11], &objects[12])) {
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
    a.inside_ptr = views[9].buf;
    a.inside_coords = views[10].buf;
    a.inside_labels = views[11].buf;
    a.out = views[12].buf;
    a.n_samples = shapes[0][0];
    a.n_vertices = shapes[1][0];
    a.n_mesh_faces = shapes[2][0];
    a.n_nodes = shapes[3][0];
    a.n_boundary = shapes[4][1];
    a.n_faces = shapes[5][0];
    a.n_partitions = shapes[6][0];
    a.n_inside = shapes[10][0];
    if (shapes[0][1] != 3 || shapes[1][1] != 3 || shapes[2][1] != 3
            || shapes[3][1] != 3 || shapes[4][0] != a.n_mesh_faces
            || a.n_boundary < 1 || shapes[6][1] != a.n_nodes
            || shapes[7][0] != a.n_partitions || shapes[7][1] != a.n_nodes
            || shapes[8][0] != a.n_partitions
            || shapes[8][1] != a.n_mesh_faces
            || shapes[9][0] != a.n_partitions
            || shapes[9][1] != a.n_mesh_faces + 1 || shapes[10][1] != 3
            || shapes[11][0] != a.n_inside
            || shapes[12][0] != a.n_partitions || shapes[12][1] != a.n_faces
            || shapes[12][2] != a.n_samples) {
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
                        "A face, vertex, node or point index is out of range.");
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

static int
compare_keys(const void *a, const void *b)
{
    const int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;

    return (x > y) - (x < y);
}

/* The keys of the samples of one face, as distinct keys in increasing order
 * and the number of samples with each. ``keys`` is used as scratch space.
 * Return the number of distinct keys.
 */
static Py_ssize_t
count_face_keys(int64_t *keys, Py_ssize_t n_samples, int64_t *out_keys,
                int64_t *out_counts)
{
    /* Most faces have a few distinct keys, in long runs of samples. */
    enum { N_SMALL = 32 };
    Py_ssize_t n = 0, s, i, j;

    for (s = 0; s < n_samples; s++) {
        const int64_t key = keys[s];

        /* The key of the previous sample is the most likely one. */
        for (i = n - 1; i >= 0 && out_keys[i] != key; i--) {
        }
        if (i >= 0) {
            out_counts[i]++;
            if (i != n - 1) {
                /* Move it to the end, where the next sample looks first. */
                const int64_t count = out_counts[i];

                out_keys[i] = out_keys[n - 1];
                out_counts[i] = out_counts[n - 1];
                out_keys[n - 1] = key;
                out_counts[n - 1] = count;
            }
        }
        else if (n < N_SMALL) {
            out_keys[n] = key;
            out_counts[n] = 1;
            n++;
        }
        else {
            /* Too many distinct keys for this search: sort all of them. */
            qsort(keys, n_samples, sizeof(int64_t), compare_keys);
            n = 0;
            for (s = 0; s < n_samples; s++) {
                if (n > 0 && out_keys[n - 1] == keys[s]) {
                    out_counts[n - 1]++;
                }
                else {
                    out_keys[n] = keys[s];
                    out_counts[n] = 1;
                    n++;
                }
            }
            return n;
        }
    }
    /* Sort the distinct keys by insertion. */
    for (i = 1; i < n; i++) {
        const int64_t key = out_keys[i], count = out_counts[i];

        for (j = i; j > 0 && out_keys[j - 1] > key; j--) {
            out_keys[j] = out_keys[j - 1];
            out_counts[j] = out_counts[j - 1];
        }
        out_keys[j] = key;
        out_counts[j] = count;
    }
    return n;
}

PyDoc_STRVAR(count_labels_doc,
"count_labels(first, second, n_second, face_indices, selected,\n"
"             out_faces, out_keys, out_counts)\n--\n\n"
"Count the samples of each face by their labels in one or two partitions.\n\n"
"``first[i, s]`` and ``second[i, s]`` are the labels of sample ``s`` of\n"
"face ``face_indices[i]``. Its key is ``first[i, s] * n_second +\n"
"second[i, s]``, or ``first[i, s]`` if ``second`` is None. For each face\n"
"with ``selected[i]``, in order, and for each of its distinct keys, in\n"
"increasing order, one entry is written: the face index, the key and the\n"
"number of samples with that key. Return the number of entries.\n\n"
"``first`` and ``second`` are C-contiguous int32 arrays of shape (k,\n"
"n_samples), ``face_indices`` has integers of the size of a pointer, and\n"
"``selected`` has booleans. The outputs are 1-D arrays with room for\n"
"``n_samples`` entries for each selected face: ``out_faces`` like\n"
"``face_indices``, and ``out_keys`` and ``out_counts`` of int64.");

static PyObject *
count_labels(PyObject *self, PyObject *args)
{
    PyObject *objects[7];
    Py_buffer views[7];
    static const char *names[7] = {"first", "second", "face_indices",
                                   "selected", "out_faces", "out_keys",
                                   "out_counts"};
    static const char kinds[8] = "iiibiii";
    static const int ndims[7] = {2, 2, 1, 1, 1, 1, 1};
    static const int sizes[7] = {4, 4, 0, 1, 0, 8, 8};
    long long n_second;
    Py_ssize_t n_faces, n_samples, n_selected = 0, n_entries = 0, i, s;
    const int32_t *first, *second = NULL;
    const Py_ssize_t *face_indices;
    const char *selected;
    Py_ssize_t *out_faces;
    int64_t *out_keys, *out_counts, *keys = NULL;
    int has_second, n_views = 0, v, failed = 0;

    if (!PyArg_ParseTuple(args, "OOLOOOOO", &objects[0], &objects[1],
                          &n_second, &objects[2], &objects[3], &objects[4],
                          &objects[5], &objects[6])) {
        return NULL;
    }
    has_second = objects[1] != Py_None;
    for (v = 0; v < 7; v++) {
        const Py_ssize_t size = sizes[v] ? sizes[v] : (int)sizeof(Py_ssize_t);

        if (v == 1 && !has_second) {
            views[v].obj = NULL;
            n_views++;
            continue;
        }
        if (get_array(objects[v], &views[v], names[v], kinds[v], size,
                      ndims[v], v >= 4) < 0) {
            goto fail;
        }
        n_views++;
    }

    n_faces = views[0].shape[0];
    n_samples = views[0].shape[1];
    first = views[0].buf;
    if (has_second) {
        second = views[1].buf;
    }
    face_indices = views[2].buf;
    selected = views[3].buf;
    out_faces = views[4].buf;
    out_keys = views[5].buf;
    out_counts = views[6].buf;
    for (i = 0; i < (views[3].shape[0] == n_faces ? n_faces : 0); i++) {
        n_selected += selected[i] != 0;
    }
    if ((has_second && (views[1].shape[0] != n_faces
                        || views[1].shape[1] != n_samples))
            || views[2].shape[0] != n_faces || views[3].shape[0] != n_faces
            || views[4].shape[0] < n_selected * n_samples
            || views[5].shape[0] < n_selected * n_samples
            || views[6].shape[0] < n_selected * n_samples) {
        PyErr_SetString(PyExc_ValueError,
                        "The shapes of the arrays do not match.");
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    keys = malloc((n_samples > 0 ? n_samples : 1) * sizeof(int64_t));
    if (keys == NULL) {
        failed = 1;
    }
    for (i = 0; i < n_faces && !failed; i++) {
        const int32_t *a = first + i * n_samples;
        Py_ssize_t n, j;

        if (!selected[i]) {
            continue;
        }
        if (has_second) {
            const int32_t *b = second + i * n_samples;

            for (s = 0; s < n_samples; s++) {
                keys[s] = (int64_t)a[s] * n_second + b[s];
            }
        }
        else {
            for (s = 0; s < n_samples; s++) {
                keys[s] = a[s];
            }
        }
        n = count_face_keys(keys, n_samples, out_keys + n_entries,
                            out_counts + n_entries);
        for (j = 0; j < n; j++) {
            out_faces[n_entries + j] = face_indices[i];
        }
        n_entries += n;
    }
    free(keys);
    Py_END_ALLOW_THREADS
    if (failed) {
        PyErr_NoMemory();
        goto fail;
    }
    for (v = 0; v < n_views; v++) {
        if (views[v].obj != NULL) {
            PyBuffer_Release(&views[v]);
        }
    }
    return PyLong_FromSsize_t(n_entries);

fail:
    for (v = 0; v < n_views; v++) {
        if (views[v].obj != NULL) {
            PyBuffer_Release(&views[v]);
        }
    }
    return NULL;
}

static PyMethodDef kernel_methods[] = {
    {"multiply_add", multiply_add, METH_VARARGS, multiply_add_doc},
    {"label_samples", label_samples, METH_VARARGS, label_samples_doc},
    {"count_labels", count_labels, METH_VARARGS, count_labels_doc},
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
