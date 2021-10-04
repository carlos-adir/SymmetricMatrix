import numpy as np
import pytest
from UsefulFunctions import *


def test_InvalidM():
    M = [[1, 1],
         [0, 1]]
    with pytest.raises(ValueError):
        VerifyValidM(M)


def test_ValidM():
    M = [[2, 1],
         [1, 2]]
    VerifyValidM(M)


def test_InvalidV():
    lenght = 5
    V = np.random.rand(lenght)
    with pytest.raises(ValueError):
        VerifyValidV(V)


def test_ValidV():
    for lenght in (1, 3, 6, 10, 15, 21, 28, 36):
        V = np.random.rand(lenght)
        VerifyValidV(V)
