import numpy as np
import pytest
from classSymmetricMatrix import SymmetricMatrix
import UsefulFunctions as UF
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
    np.testing.assert_array_almost_equal(S.M, M)

    V = [1, 2, 3, 4, 5, 6]
    S = SymmetricMatrix(V)
    np.testing.assert_array_almost_equal(S.V, V)


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

    np.testing.assert_array_almost_equal(n, S.n)
    np.testing.assert_array_almost_equal(M, S.M)
    np.testing.assert_array_almost_equal(V, S.V)
    np.testing.assert_array_almost_equal(L, S.L)
    np.testing.assert_array_almost_equal(D, S.D)
    np.testing.assert_array_almost_equal(l, S.l)
    np.testing.assert_array_almost_equal(d, S.d)
    LDL = L @ D @ L.T
    SLDL = S.L @ S.D @ (S.L).T
    np.testing.assert_array_almost_equal(LDL, M)
    np.testing.assert_array_almost_equal(SLDL, S.M)

    # Using V as constructor
    S = SymmetricMatrix(V)
    np.testing.assert_array_almost_equal(n, S.n)
    np.testing.assert_array_almost_equal(M, S.M)
    np.testing.assert_array_almost_equal(V, S.V)
    np.testing.assert_array_almost_equal(L, S.L)
    np.testing.assert_array_almost_equal(D, S.D)
    np.testing.assert_array_almost_equal(l, S.l)
    np.testing.assert_array_almost_equal(d, S.d)
    LDL = L @ D @ L.T
    SLDL = S.L @ S.D @ (S.L).T
    np.testing.assert_array_almost_equal(LDL, M)
    np.testing.assert_array_almost_equal(SLDL, S.M)


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
    np.testing.assert_array_almost_equal(n, S.n)
    np.testing.assert_array_almost_equal(M, S.M)
    np.testing.assert_array_almost_equal(V, S.V)
    np.testing.assert_array_almost_equal(L, S.L)
    np.testing.assert_array_almost_equal(D, S.D)
    np.testing.assert_array_almost_equal(l, S.l)
    np.testing.assert_array_almost_equal(d, S.d)
    LDL = L @ D @ L.T
    SLDL = S.L @ S.D @ (S.L).T
    np.testing.assert_array_almost_equal(LDL, M)
    np.testing.assert_array_almost_equal(SLDL, S.M)

    # Using V as constructor
    S = SymmetricMatrix(V)
    np.testing.assert_array_almost_equal(n, S.n)
    np.testing.assert_array_almost_equal(M, S.M)
    np.testing.assert_array_almost_equal(V, S.V)
    np.testing.assert_array_almost_equal(L, S.L)
    np.testing.assert_array_almost_equal(D, S.D)
    np.testing.assert_array_almost_equal(l, S.l)
    np.testing.assert_array_almost_equal(d, S.d)
    LDL = L @ D @ L.T
    SLDL = S.L @ S.D @ (S.L).T
    np.testing.assert_array_almost_equal(LDL, M)
    np.testing.assert_array_almost_equal(SLDL, S.M)


def test_ConversionWithRandomValues():
    Ntests = 10
    for i in range(Ntests):
        for n in range(2, 10):
            M = UF.getRandomSymmetricMatrix(n)

            Sym = SymmetricMatrix(M)

            V = transform.M2V(M)
            L, d = transform.M2Ld(M)
            D = np.diag(d)
            l = transform.L2l(L)

            np.testing.assert_array_almost_equal(M, Sym.M)
            np.testing.assert_array_almost_equal(V, Sym.V)
            np.testing.assert_array_almost_equal(l, Sym.l)
            np.testing.assert_array_almost_equal(d, Sym.d)
            np.testing.assert_array_almost_equal(L, Sym.L)
            np.testing.assert_array_almost_equal(D, Sym.D)


def test_InverseOfSymmetricMatrix():
    Ntests = 1000
    nmax = 10
    for i in range(Ntests):
        for n in range(2, nmax + 1):
            M = UF.getRandomSymmetricMatrix(n)
            Sym = SymmetricMatrix(M)
            Minv = Sym.inv()
            np.testing.assert_array_almost_equal(Minv @ M, np.eye(n))
            np.testing.assert_array_almost_equal(Minv @ M, np.eye(n))
