import math
import copy
from Matrix import QuadMatrix, TripleDiagMatrix, Matrix

'''
Метод простой итерации:
Ax = b
x = Bx + g

'''
def Solution_of_SLE(A: QuadMatrix, b: list, epsilon: float):
    '''
    A -> M - N (Метод Якоби)
    M = D, D - Diagonal matrix
    '''

    def stop_condition(X: list, epsilon: float):
        norm = sum(elem ** 2 for elem in X)
        return math.sqrt(norm) < epsilon       

    def prepare_SLE(A: QuadMatrix, b: list):
        size = len(A)
        
        result_matrix = QuadMatrix(size, 0)
        result_coef = Matrix(1, size, 0)

        for row in range(size):
            for col in range(size):
                if row == col:
                    result_matrix[row, col] = 0
                else:
                    result_matrix[row, col] = - A[row, col] / A[row, row]

            result_coef[row] = b[row] / A[row, row]

        return result_matrix, result_coef

    B, g = prepare_SLE(A, b)
    x = copy.copy(g)
    flag = False
    iterations_cnt = 0

    while not flag:
        prev_x = copy.copy(x)
        x = B @ prev_x + g
        iterations_cnt += 1
        flag = stop_condition(prev_x - x, epsilon)
    return x, iterations_cnt
        


def main():
    size = int(input("Введите размер матрицы: "))
    print('Введите значения матрицы:\n')
    data = [[] for _ in range(size)]
    for row in range(size):
        data[row] = list(map(int, input().split()))
    A = TripleDiagMatrix(size, data)
    
    b_coef = list(map(int, input("Введите коэффициенты: ").split()))
   # eps = float(input('Введите эпсилон: '))

    x = Solution_of_SLE(A, b_coef, 0.1)

   # print('Solution:', x)


if __name__ == '__main__':
    main()
