import numpy as np
from numpy.linalg import matrix_rank
from scipy.linalg import lu
from scipy.linalg import null_space


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


#


A = matrix_maker()
ns = null_space(A)
print("null space:", ns)
dim_ns = ns.shape
print("dimension of null space:", dim_ns)

print("rank A:")
print(matrix_rank(A))
U=lu(A)
print(U)
