import UsefulFunctions as UF
import transform
import numpy as np


class SymmetricMatrix:

    def __throwEntryError(self, M):
        try:
            M = np.array(M)
        except Exception:
            errormsg = "We could not convert the parameter M into an array"
            raise TypeError(errormsg)

        if len(M.shape) == 1:
            UF.VerifyValidV(M)
        elif len(M.shape) == 2:
            UF.VerifyValidM(M)
        else:
            errormsg = "The received parameter M is not (1D) or (2D) array"
            raise ValueError(errormsg)

    def __init__(self, M):
        if UF.isValidM(M):
            M = np.array(M)
            self.__n = int(M.shape[0])
            self.__V = transform.M2V(M)
        elif UF.isValidV(M):
            self.__V = np.copy(M)
            self.__n = UF.getNfromV(self.__V)
        else:
            self.__throwEntryError(M)

        self.__l = None
        self.__d = None

    def compute_ld(self):
        l, d = transform.V2ld(self.__V)
        self.__l = l
        self.__d = d

    def __str__(self):
        return str(self.M)

    @property
    def d(self):
        if self.__d is None:
            self.compute_ld()
        return np.copy(self.__d)

    @property
    def l(self):
        if self.__l is None:
            self.compute_ld()
        return np.copy(self.__l)

    @property
    def V(self):
        return np.copy(self.__V)

    @property
    def M(self):
        return transform.V2M(self.__V)

    @property
    def D(self):
        if self.__d is None:
            self.compute_ld()
        return np.diag(self.__d)

    @property
    def L(self):
        if self.__l is None:
            self.compute_ld()
        return transform.l2L(self.__l)

    @property
    def n(self):
        return self.__n

    def getLinv(self):
        linv = transform.l2linv(self.l)
        return transform.l2L(linv)

    def inv(self):
        Linv = self.getLinv()
        if np.any(self.d == 0):
            errormsg = "Impossible Invert Singular Matrix: det(M) = 0"
            raise ValueError(errormsg)
        dinv = 1 / self.d
        Dinv = np.diag(dinv)
        return np.transpose(Linv) @ Dinv @ Linv

    def __add__(self, S):
        if not isinstance(S, SymmetricMatrix):
            raise Exception("Can sum only symmetric matrix")

    def __sub__(self, S):
        newV = self.__V - S.V
        return newV

    def __neg__(self):
        return SymmetricMatrix(V=-self.V)
