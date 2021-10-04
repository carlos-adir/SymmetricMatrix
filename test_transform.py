import numpy as np
import pytest
import transform


def test_transformationMintoV():
    M = np.zeros((1, 1))
    Vgood = np.zeros(1)
    Vtest = transform.M2V(M)
    np.testing.assert_allclose(Vgood, Vtest)

    M = np.zeros((2, 2))
    Vgood = np.zeros(3)
    Vtest = transform.M2V(M)
    np.testing.assert_allclose(Vgood, Vtest)

    M = np.zeros((3, 3))
    Vgood = np.zeros(6)
    Vtest = transform.M2V(M)
    np.testing.assert_allclose(Vgood, Vtest)

    M = np.eye(2)
    Vgood = np.array([1, 0, 1])
    Vtest = transform.M2V(M)
    np.testing.assert_allclose(Vgood, Vtest)

    M = np.eye(3)
    Vgood = np.array([1, 0, 1, 0, 0, 1])
    Vtest = transform.M2V(M)
    np.testing.assert_allclose(Vgood, Vtest)

    M = np.array([[1, 2, 4],
                  [2, 3, 5],
                  [4, 5, 6]])
    Vgood = np.array([1, 2, 3, 4, 5, 6])
    Vtest = transform.M2V(M)
    np.testing.assert_allclose(Vgood, Vtest)

    M = np.array([[1, 2, 4, 7],
                  [2, 3, 5, 8],
                  [4, 5, 6, 9],
                  [7, 8, 9, 10]])
    Vgood = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Vtest = transform.M2V(M)
    np.testing.assert_allclose(Vgood, Vtest)


def test_transformationVintoM():
    V = np.zeros(1)
    Mgood = np.zeros((1, 1))
    Mtest = transform.V2M(V)
    np.testing.assert_allclose(Mgood, Mtest)

    V = np.zeros(3)
    Mgood = np.zeros((2, 2))
    Mtest = transform.V2M(V)
    np.testing.assert_allclose(Mgood, Mtest)

    V = np.zeros(6)
    Mgood = np.zeros((3, 3))
    Mtest = transform.V2M(V)
    np.testing.assert_allclose(Mgood, Mtest)

    V = np.array([1, 0, 1])
    Mgood = np.eye(2)
    Mtest = transform.V2M(V)
    np.testing.assert_allclose(Mgood, Mtest)

    V = np.array([1, 0, 1, 0, 0, 1])
    Mgood = np.eye(3)
    Mtest = transform.V2M(V)
    np.testing.assert_allclose(Mgood, Mtest)

    V = np.array([1, 2, 3, 4, 5, 6])
    Mgood = np.array([[1, 2, 4],
                      [2, 3, 5],
                      [4, 5, 6]])
    Mtest = transform.V2M(V)
    np.testing.assert_allclose(Mgood, Mtest)

    V = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    Mgood = np.array([[1, 2, 4, 7],
                      [2, 3, 5, 8],
                      [4, 5, 6, 9],
                      [7, 8, 9, 10]])
    Mtest = transform.V2M(V)
    np.testing.assert_allclose(Mgood, Mtest)


def test_creationWithRandomMatrix():
    for n in range(2, 10):
        A = np.random.rand(n, n)
        Mgood = A + np.transpose(A)
        Vtest = transform.M2V(Mgood)
        Mtest = transform.V2M(Vtest)
        np.testing.assert_allclose(Mgood, Mtest)

    for n in range(2, 10):
        size = int(n * (n + 1) / 2)
        Vgood = np.random.rand(size)
        Mtest = transform.V2M(Vgood)
        Vtest = transform.M2V(Mtest)
        np.testing.assert_allclose(Vgood, Vtest)


def test_transformationLDL():
    M = np.eye(3)
    dgood = np.ones(3)
    Lgood = np.eye(3)
    Ltest, dtest = transform.M2Ld(M)
    np.testing.assert_allclose(Lgood, Ltest)
    np.testing.assert_allclose(dgood, dtest)


def test_transformationLDLwithRandom():
    Ntests = 100
    for i in range(Ntests):
        for n in range(2, 10):
            Lgood = np.random.rand(n, n)
            for k in range(n):
                Lgood[k, k] = 1
                for j in range(k + 1, n):
                    Lgood[k, j] = 0
            dgood = np.random.rand(n)
            M = Lgood @ np.diag(dgood) @ Lgood.T
            Ltest, dtest = transform.M2Ld(M)
            np.testing.assert_allclose(Lgood, Ltest)
            np.testing.assert_allclose(dgood, dtest)


def test_transformation_V2ld_withKnowValues():
    tolerance = 5e-2

    V = [0.16, 1.64, 1.06, 1.22, 0.79, 1.53]
    lgood = [10.25, 7.625, 0.744]
    dgood = [0.16, -15.75, 0.941]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)

    V = [1.15, 0.93, 0.7, 1.67, 1.07, 1.93]
    lgood = [0.809, 1.452, 5.386]
    dgood = [1.15, -0.052, 1.016]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)

    V = [0.07, 1.78, 0.05, 0.66, 1.74, 0.59]
    lgood = [25.429, 9.429, 0.333]
    dgood = [0.07, -45.213, -0.628]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)

    V = [0.59, 0.74, 1.85, 1.04, 1.72, 0.33]
    lgood = [1.254, 1.763, 0.451]
    dgood = [0.59, 0.922, -1.691]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)

    V = [0.55, 1.01, 0.28, 0.87, 0.34, 1.8]
    lgood = [1.836, 1.582, 0.799]
    dgood = [0.55, -1.575, 1.428]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)

    V = [1.25, 1.49, 1.7, 0.78, 0.26, 0.37]
    lgood = [1.192, 0.624, 8.803]
    dgood = [1.25, -0.076, 5.779]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)

    V = [0.9, 1.15, 0.97, 1.43, 0.28, 1.39]
    lgood = [1.278, 1.589, 3.098]
    dgood = [0.9, -0.499, 3.911]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)

    V = [0.98, 0.91, 0.3, 1.37, 0.8, 0.9]
    lgood = [0.929, 1.398, 0.866]
    dgood = [0.98, -0.545, -0.606]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)

    V = [1.06, 0.92, 1.25, 0.9, 1.52, 0.06]
    lgood = [0.868, 0.849, 1.636]
    dgood = [1.06, 0.452, -1.913]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)

    V = [0.36, 0.93, 0.36, 0.82, 1.21, 1.47]
    lgood = [2.583, 2.278, 0.445]
    dgood = [0.36, -2.043, 0.006]
    ltest, dtest = transform.V2ld(V)
    np.testing.assert_allclose(lgood, ltest, rtol=tolerance)
    np.testing.assert_allclose(dgood, dtest, rtol=tolerance)


def test_transformation_V2ld_withRandomValues():
    Ntests = 100
    for i in range(Ntests):
        for n in range(2, 10):
            M = np.random.rand(n, n)
            M += np.transpose(M)
            V = transform.M2V(M)

            L, dgood = transform.M2Ld(M)
            lgood = transform.L2l(L)

            ltest, dtest = transform.V2ld(V)
            np.testing.assert_allclose(lgood, ltest)
            np.testing.assert_allclose(dgood, dtest)
