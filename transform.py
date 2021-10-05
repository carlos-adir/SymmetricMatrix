import numpy as np
import UsefulFunctions as UF


def M2V(M):
    UF.VerifyValidM(M)
    M = np.array(M)
    n = M.shape[0]
    V = np.zeros((n * (n + 1)) // 2)
    for i in range(n):
        for j in range(i + 1):
            index = j + (i * (i + 1)) // 2
            V[index] = M[i, j]
    return V


def V2M(V):
    UF.VerifyValidV(V)
    n = int(np.floor(np.sqrt(2 * len(V))))
    tempM = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            tempM[i, j] = V[j + (i * (i + 1)) // 2]
    tempM += np.transpose(tempM)
    for i in range(n):
        index = (i * (i + 3)) // 2
        tempM[i, i] = V[index]
    return tempM


def M2Ld(M):
    UF.VerifyValidM(M)

    n = M.shape[0]
    d = np.zeros(n)
    L = np.eye(n)

    d[0] = M[0, 0]
    for i in range(1, n):
        for j in range(i):
            soma = np.sum(d[:j] * L[j, :j] * L[i, :j])
            L[i, j] = (M[i, j] - soma) / d[j]

        soma = np.sum(d[:i] * L[i, :i]**2)
        d[i] = M[i, i] - soma

    return L, d


def M2LD(M):
    L, d = M2Ld(M)
    D = np.diag(d)
    return L, D


def L2l(L):
    UF.VerifyValidL(L)
    n = L.shape[0]
    lf = np.zeros((n * (n - 1)) // 2)
    for i in range(n - 1):
        for j in range(i + 1):
            lf[j + (i * (i + 1)) // 2] = L[i + 1, j]
    return lf


def M2ld(M):
    UF.VerifyValidM(M)
    L, d = M2Ld(M)
    lf = L2l(L)
    return lf, d


def l2L(lf):
    UF.VerifyValidV(lf)
    n = int(np.floor(np.sqrt(2 * len(lf)))) + 1
    L = np.eye(n)
    for i in range(n - 1):
        indexi = i * (i + 1) // 2
        for j in range(i + 1):
            L[i + 1, j] = lf[j + indexi]
    return L


def V2ld(V):
    UF.VerifyValidV(V)
    M = V2M(V)
    L, d = M2Ld(M)
    lf = L2l(L)

    n = int(np.floor(np.sqrt(2 * len(V))))
    d = np.zeros(n)
    lf = np.zeros((n * (n - 1) // 2))

    for i in range(n):
        index = i * (i + 3) // 2
        d[i] = V[index]
    for i in range(1, n):
        indexi = i * (i - 1) // 2
        index1 = i * (i + 1) // 2
        vecti = lf[indexi:indexi + i]
        for j in range(i):
            indexj = (j - 1) * j // 2
            vectj = lf[indexj:indexj + j]
            soma = np.sum(d[:j] * vectj * vecti[:j])
            lf[indexi + j] = (V[index1 + j] - soma) / d[j]
        d[i] -= np.sum(d[:i] * vecti**2)
    return lf, d


def L2Linv(L):
    UF.VerifyValidL(L)
    n = L.shape[0]
    A = -np.copy(L)

    for i in range(n):
        for j in range(i - 1, -1, -1):
            # soma = 0
            # for k in range(j + 1, i):
            #     soma += A[i, k] * L[k, j]
            A[i, j] -= np.sum(A[i, j + 1:i] * L[j + 1:i, j])

    for i in range(n):
        A[i, i] = 1
        for j in range(i + 1, n):
            A[i, j] = 0
    return A


def l2linv(lf):
    UF.VerifyValidV(lf)
    n = int(np.floor(np.sqrt(2 * len(lf)))) + 1
    lfinv = -np.copy(lf)
    for i in range(n):
        indexi = i * (i - 1) // 2
        for j in range(i - 1, -1, -1):
            soma = 0
            for k in range(j + 1, i):
                indexk = k * (k - 1) // 2
                soma += lfinv[indexi + k] * lf[j + indexk]
            lfinv[indexi + j] -= soma
    return lfinv
