import numpy as np
import scipy.linalg as la
from numpy.linalg import solve
from scipy import linalg


def solver():
    """A:coefficient"""
    print("enter A:")
    print("")
    a = matrix_maker()
    print("")

    """"B:constant"""
    print("enter B:")
    print("")
    b = matrix_maker()

    augmented = np.hstack([a, b])
    print("augmented matrix:")
    print(augmented)

    M1 = add_row(augmented, 2, 1, 0)
    print("M1:", M1)

    M2 = add_row(M1, -3, 3, 0)
    print("M2:", M2)

    temp = add_row(M2, -1, 4, 0)
    print("Temp:", temp)

    M3 = scale_row(temp, 1 / 5, 1)
    print("M3:", M3)
    #
    M4 = add_row(M3, 1, 2, 1)
    print("M4:", M4)
    #
    M5 = add_row(M4, 13, 3, 1)
    print("M5:", M5)
    #
    M6 = add_row(M5, 1, 4, 1)
    print("M6:", M6)
    #
    M7 = scale_row(M6, 5 / 21, 2)
    print("M7:", M7)
    #
    M8 = add_row(M7, -58 / 5, 3, 2)
    print("M8:", M8)
    #
    M9 = add_row(M8, 44 / 5, 4, 2)
    print("M9:", M9)

    M10 = scale_row(M9, 7 / 187, 3)
    print("M10:", M10)

    M11 = add_row(M10, 12 / 7, 4, 3)
    print("M11:", M11)  #

    M12 = scale_row(M11, 187 / 1196, 4)
    print("M12:", M12)

    M13 = add_row(M12, -12 / 187, 3, 4)
    print("M13:", M13)  #

    M14 = add_row(M13, -5 / 7, 2, 4)
    print("M14:", M14)

    M15 = add_row(M14, -2, 1, 4)
    print("M15:", M15)

    M16 = add_row(M15, -3, 0, 4)
    print("M16:", M16)

    M17 = add_row(M16, 9 / 7, 2, 3)
    print("M17:", M17)

    M18 = add_row(M17, 2 / 5, 1, 3)
    print("M18:", M18)

    M19 = add_row(M18, 4, 0, 3)
    print("M19:", M19)

    M20 = add_row(M19, -6 / 5, 1, 2)
    print("M20:", M20)

    M21 = add_row(M20, -2, 0, 2)
    print("M21:", M21)

    M22 = add_row(M21, -3, 0, 1)
    print("M22:", M22)

    #
    x = M22[:, 5].reshape(5, 1)
    print("X:", x)

    # if (invertible()):
    print("Solution of linear equations:", solve(a, b))
    return solve(a, b)


def matrix_maker():
    # A basic code for matrix input from user

    R = int(input("Enter the number of rows:"))
    C = int(input("Enter the number of columns:"))

    # Initialize matrix
    matrix = []
    print("Enter the entries rowwise:")

    # For user input
    for i in range(R):  # A for loop for row entries
        a = []
        for j in range(C):  # A for loop for column entries
            a.append(int(input()))
        matrix.append(a)

        # For printing the matrix
    for i in range(R):
        for j in range(C):
            print(matrix[i][j], end=" ")
        print()
    return np.array(matrix)


def add_row(A, k, i, j):
    "Add k times row j to row i in matrix A."
    n = A.shape[0]
    E = np.eye(n)
    if i == j:
        E[i, i] = k + 1
    else:
        E[i, j] = k
    return E @ A


def scale_row(A, k, i):
    "Multiply row i by k in matrix A."
    n = A.shape[0]
    E = np.eye(n)
    E[i, i] = k
    return E @ A


def switch_rows(A, i, j):
    "Switch rows i and j in matrix A."
    n = A.shape[0]
    E = np.eye(n)
    E[i, i] = 0
    E[j, j] = 0
    E[i, j] = 1ر
    E[j, i] = 1
    return E @ A


def quick_solver():
    A = matrix_maker()
    b = matrix_maker()

    x = la.solve(A, b)
    print("res:")
    print(x)


def invertible(a):
    # return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]
    try:
        print(linalg.inv(a))
    except:
        print("not consistent")


solver()
