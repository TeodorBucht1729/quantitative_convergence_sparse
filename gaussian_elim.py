import copy
import numpy as np
from fractions import Fraction


def solveLinear(A: np.ndarray, b: np.ndarray):
    A = copy.deepcopy(A)
    b = copy.deepcopy(b)
    n = A.shape[0]
    m = A.shape[1]
    rank = 0
    col = list(range(m))
	# vi col(m); iota(all(col), 0);
    percentage_done = 0
    for i in range(m):
        # print("processing column", i)
        new_percentage = int(i/m * 100)
        if new_percentage > percentage_done:
            # print(f"{new_percentage}% done")
            percentage_done = new_percentage
        # print(i, A, b)
        v = Fraction(0, 1)
        bv = Fraction(0, 1)
        for r in range(i, n):
            for c in range(i, m):
                v = A[r][c]
                # if (v != Fraction(0, 1)):
                if ((bv == Fraction(0, 1) and v != Fraction(0, 1)) or (v != Fraction(0, 1) and abs(v) < abs(bv))):
                    br = r
                    bc = c
                    bv = v
                    # break
            if bv != Fraction(0, 1):
                break
        # print("bv:", bv, type(bv))
        if (bv == Fraction(0, 1)):
            for j in range(i, n):
                if b[j] != Fraction(0, 1):
                  return None, -1
            break
        A[[i, br]] = A[[br, i]]
        b[[i, br]] = b[[br, i]]
        col[i], col[bc] = col[bc], col[i]

        for j in range(n):
            A[j][i], A[j][bc] = A[j][bc], A[j][i]

        bv = Fraction(1, 1) / A[i][i]
        for j in range(i+1, n):
            fac = A[j][i] * bv
            # the rest of this loop will do nothing in this case
            if fac == Fraction(0, 1):
                continue
            b[j] -= fac * b[i]
            for k in range(i+1, m):
                A[j][k] -= fac*A[i][k]
        rank += 1
    # print(A, b)
    x = np.zeros(m, dtype=object)
    for i in range(m):
        x[i] = Fraction(0, 1)
    for i in range(rank)[::-1]:
        b[i] /= A[i][i]
        x[col[i]] = b[i]
        for j in range(0, i):
            b[j] -= A[j][i] * b[i]

    return x, rank

if __name__ == "__main__":
    A = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 3.0, 1.0, 1.0], [4.0, -1.0, 1.0, -1.0]])
    b = np.array([5.0, 9.0, -2.0])
    print(solveLinear(A, b))