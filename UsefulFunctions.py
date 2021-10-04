import numpy as np


def isNumpyArray(M):
    if not isinstance(M, np.ndarray):
        try:
            M = np.array(M)
        except Exception:
            return False
    return True


def isMatrix(M):
    if not isNumpyArray(M):
        return False
    M = np.array(M)
    if len(M.shape) != 2:
        return False
        raise ValueError("The given M is not 2D numpy array")
    return True


def isSquareMatrix(M):
    if not isMatrix(M):
        return False
    M = np.array(M)
    if M.shape[0] != M.shape[1]:
        return False
    else:
        return True


def isSymmetricMatrix(M):
    if not isSquareMatrix(M):
        return False
    M = np.array(M)
    MT = np.transpose(M)
    diff = M - MT
    if np.all(np.abs(diff) < 1e-7):
        return True
    else:
        return False


def isValidM(M):
    try:
        VerifyValidM(M)
        return True
    except Exception:
        return False


def isValidV(V):
    try:
        VerifyValidV(V)
        return True
    except Exception:
        return False


def VerifyValidM(M):
    if not isNumpyArray(M):
        raise TypeError("Could not convert the parameter M into numpy array")
    if not isMatrix(M):
        raise TypeError("The given parameter M is not a Matrix")
    if not isSquareMatrix(M):
        raise ValueError("The given matrix M is not a square matrix")
    if not isSymmetricMatrix(M):
        raise ValueError("The given matrix M is not symmetric")


def VerifyValidV(V):
    if not isNumpyArray(V):
        raise TypeError("Could not convert the parameter V into numpy array")
    V = np.array(V)
    if not len(V.shape) == 1:
        errormsg = "The received parameter must be 1D-array.\n"
        errormsg += "Received shape %s" % str(V.shape)
        raise ValueError(errormsg)
    n = (1 + np.sqrt(1 + 8 * len(V))) / 2
    if np.abs(int(n) - n) > 1e-5:
        nmin = int(np.floor(n))
        nmax = int(np.ceil(n))
        errormsg = "The size of V is not valid!\nValid values are: %d < %.3f < %d\n %d < len(V)=%d < %d" % (
            nmin, n, nmax, nmin * (nmin + 1) // 2, len(V), nmax * (nmax + 1) // 2)
        raise ValueError(errormsg)


def VerifyValidL(L):
    if not isNumpyArray(L):
        raise TypeError("Could not convert the parameter L into numpy array")
    if not isMatrix(L):
        raise TypeError("The given parameter L is not a matrix")
    if not isSquareMatrix(L):
        raise ValueError("The given matrix L is not a square matrix")
    if not np.all(np.diagonal(L) == 1):
        raise ValueError("All the values in diagonal of L must be 1")
    n = L.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if not L[i, j] == 0:
                errormsg = "L is a lower matrice. You must have L[i, j] = 0 forall i < j"
                raise ValueError(errormsg)


def getNfromV(V):
    return int(np.floor(np.sqrt(2 * len(V))))
