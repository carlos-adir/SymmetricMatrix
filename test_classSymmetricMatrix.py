import numpy as np
import pytest
from classSymmetricMatrix import SymmetricMatrix
import transform


def test_Construction():
    M = [[1, 2, 4],
         [2, 3, 5],
         [4, 5, 6]]
    S = SymmetricMatrix(M)

    V = [1, 2, 3, 4, 5, 6]
    S = SymmetricMatrix(V)


def test_VerifyReadValues():
    M = [[1, 2, 4],
         [2, 3, 5],
         [4, 5, 6]]
    S = SymmetricMatrix(M)
    np.testing.assert_allclose(S.M, M)

    V = [1, 2, 3, 4, 5, 6]
    S = SymmetricMatrix(V)
    np.testing.assert_allclose(S.V, V)


def test_ConversionWithKnowValues_1():
    n = 3
    M = [[1, 2, 4],
         [2, 3, 5],
         [4, 5, 6]]
    V = [1, 2, 3, 4, 5, 6]
    L = [[1, 0, 0],
         [2, 1, 0],
         [4, 3, 1]]
    D = [[1, 0, 0],
         [0, -1, 0],
         [0, 0, -1]]
    d = [1, -1, -1]
    l = [2, 4, 3]
    M = np.array(M)
    V = np.array(V)
    D = np.array(D)
    L = np.array(L)
    l = np.array(l)
    d = np.array(d)

    # Using M as constructor
    S = SymmetricMatrix(M)

    np.testing.assert_allclose(n, S.n)
    np.testing.assert_allclose(M, S.M)
    np.testing.assert_allclose(V, S.V)
    np.testing.assert_allclose(L, S.L)
    np.testing.assert_allclose(D, S.D)
    np.testing.assert_allclose(l, S.l)
    np.testing.assert_allclose(d, S.d)
    LDL = L @ D @ L.T
    SLDL = S.L @ S.D @ (S.L).T
    np.testing.assert_allclose(LDL, M)
    np.testing.assert_allclose(SLDL, S.M)

    # Using V as constructor
    S = SymmetricMatrix(V)
    np.testing.assert_allclose(n, S.n)
    np.testing.assert_allclose(M, S.M)
    np.testing.assert_allclose(V, S.V)
    np.testing.assert_allclose(L, S.L)
    np.testing.assert_allclose(D, S.D)
    np.testing.assert_allclose(l, S.l)
    np.testing.assert_allclose(d, S.d)
    LDL = L @ D @ L.T
    SLDL = S.L @ S.D @ (S.L).T
    np.testing.assert_allclose(LDL, M)
    np.testing.assert_allclose(SLDL, S.M)


def test_ConversionWithKnowValues_2():
    n = 5
    M = [[2, -1, 0, 0, 0],
         [-1, 3, -1, 0, 0],
         [0, -1, 3, -1, 0],
         [0, 0, -1, 3, -1],
         [0, 0, 0, -1, 2]]
    V = [2, -1, 3, 0, -1, 3, 0, 0, -1, 3, 0, 0, 0, -1, 2]
    L = [[1, 0, 0, 0, 0],
         [-1 / 2, 1, 0, 0, 0],
         [0, -2 / 5, 1, 0, 0],
         [0, 0, -5 / 13, 1, 0],
         [0, 0, 0, -13 / 34, 1]]
    D = [[2, 0, 0, 0, 0],
         [0, 5 / 2, 0, 0, 0],
         [0, 0, 13 / 5, 0, 0],
         [0, 0, 0, 34 / 13, 0],
         [0, 0, 0, 0, 55 / 34]]
    d = [2, 5 / 2, 13 / 5, 34 / 13, 55 / 34]
    l = [-1 / 2, 0, -2 / 5, 0, 0, -5 / 13, 0, 0, 0, -13 / 34]
    M = np.array(M)
    V = np.array(V)
    D = np.array(D)
    L = np.array(L)
    l = np.array(l)
    d = np.array(d)

    # # Using M as constructor
    S = SymmetricMatrix(M)
    np.testing.assert_allclose(n, S.n)
    np.testing.assert_allclose(M, S.M)
    np.testing.assert_allclose(V, S.V)
    np.testing.assert_allclose(L, S.L)
    np.testing.assert_allclose(D, S.D)
    np.testing.assert_allclose(l, S.l)
    np.testing.assert_allclose(d, S.d)
    LDL = L @ D @ L.T
    SLDL = S.L @ S.D @ (S.L).T
    np.testing.assert_allclose(LDL, M)
    np.testing.assert_allclose(SLDL, S.M)

    # Using V as constructor
    S = SymmetricMatrix(V)
    np.testing.assert_allclose(n, S.n)
    np.testing.assert_allclose(M, S.M)
    np.testing.assert_allclose(V, S.V)
    np.testing.assert_allclose(L, S.L)
    np.testing.assert_allclose(D, S.D)
    np.testing.assert_allclose(l, S.l)
    np.testing.assert_allclose(d, S.d)
    LDL = L @ D @ L.T
    SLDL = S.L @ S.D @ (S.L).T
    np.testing.assert_allclose(LDL, M)
    np.testing.assert_allclose(SLDL, S.M)


def test_ConversionWithRandomValues():
    Ntests = 10
    for i in range(Ntests):
        for n in range(2, 10):
            A = np.random.rand(n, n)
            # We transform a random array into a symmetric array
            # by summing the transpose
            M = A + np.transpose(A)

            Sym = SymmetricMatrix(M)

            V = transform.M2V(M)
            L, d = transform.M2Ld(M)
            D = np.diag(d)
            l = transform.L2l(L)

            np.testing.assert_allclose(M, Sym.M)
            np.testing.assert_allclose(V, Sym.V)
            np.testing.assert_allclose(l, Sym.l)
            np.testing.assert_allclose(d, Sym.d)
            np.testing.assert_allclose(L, Sym.L)
            np.testing.assert_allclose(D, Sym.D)


test_Construction()
test_VerifyReadValues()
test_ConversionWithKnowValues_1()
test_ConversionWithKnowValues_2()
# test_ConversionWithRandomValues()
