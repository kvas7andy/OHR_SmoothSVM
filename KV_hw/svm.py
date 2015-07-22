"""
Модуль с реализованными функциями для вычислений и экспериментов.
Эксперементы проведены в IPython Notebook файле Kvasov_Coursework.ipynb
Исследование Квасова А. для курсовой работы
"Байесовский подход к обучению распознаванию образов с учетом критерия
гладкости решающего правила на основе метода опорных векторов"
"""


import numpy as np
import time
from cvxopt import matrix, spmatrix, sparse, spdiag, solvers
from scipy import linalg

from config_svm import Eps


def tridiagonal(a, b, c):
    """
    Создать трехдиагональную матрицу NxN,
    размеры которой определяются по векторам a, b, c:
        a - вектор на главной диагонали, Nx1 (или размерности 1);
        b - вектор над главной диагональю, (N-1)x1 (или размерности 1);
        c - вектор под главной диагональю, (N-1)x1 (или размерности 1);
    """
    return np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)


def smooth_qp_dual_solver(X_train, y_train,
    C=0.1, alpha=0, tol=1e-6, max_iter=10**6, verbose=False):
    """Функции для решения задачи SVM:
        svm_qp_dual_solver(X, y, C=0.1, alpga=0,
            tol=1e-6, max_iter=10**6,verbose=False)
    Описание параметров:
        • X — переменная типа numpy.array, матрица размера N × D,
            признаковые описания объектов из обучающей выборки,
        • y — переменная типа numpy.array, матрица размера N × 1,
            правильные ответы на обучающей выборке,
        • C — параметр регуляризации,
        • alpha — параметр гладкости,
        • tol — требуемая точность,
        • max_iter — максимальное число итераций,
        • verbose — в случае True, требуется выводить отладочную информацию
            на экран (номер итерации, значение целевой функции),
    
    Возвращает словарь с полями:
        • ’a’ — numpy.array, вектор параметров модели,
            векторная часть, матрица размера K × 1,
        • 'b' — скаляр, сдвиг в векторе параметров модели
        • 'lambdas' — numpy.array, матрица размера N×1, значения
        двойственных переменных,
        • ’status’ — {0, 1, 2}, 0 — если метод вышел по критерию останова,
                                1 — если оптимизация невыполнима,
        • ’time’ — время работы метода.
        • 'dual' — максимальное значение двойственной функции 
    """
    tm = time.time()
    # prepare qp matrixes:
    N = X.shape[0]
    D = X.shape[1]
    """P is a square dense or sparse real matrix, representing a positive semidefinite symmetric matrix in 'L' storage,
    i.e., only the lower triangular part of P is referenced. q is a real single-column dense matrix.
    The arguments h and b are real single-column dense matrices. G and A are real dense or sparse matrices. """
    B_diags = np.ones((3, D))
    B_diags[1]= 2*np.ones((D,))
    B_diags[1, 0] = 1
    B_diags[1,-1] = 1
    B_diags[0] = -1
    B_diags[2] = -1
    B_diags = alpha*B_diags
    B_diags[1] += 1

    B_inv = linalg.solve_banded((1, 1), B_diags, np.eye(D)) #scipy.linalg solver for banded matrix 
    
    P = matrix(X * y) * matrix(B_inv) * matrix(X * y).T
    q = matrix(-1.0, (N, 1))

    I = spmatrix(1, range(N), range(N))
    G = sparse([[-I, I]])
    h = matrix(0.0, (N + N, 1))
    h[N:] = C
    A = matrix(y).T
    b = matrix(0.0)

    #solve convex problem
    solvers.options['maxiters'] = max_iter
    solvers.options['reltol'] = tol
    solvers.options['show_progress'] = verbose
    sol = solvers.qp(P, q, G, h, A, b)
    lambdas = np.array(sol['x'])
    lambdas[lambdas < Eps] = 0.0
    #write results
    ret_dic = {
    'a': np.sum(lambdas.T*y.T*np.dot(B_inv, X.T), axis=1).reshape(-1, 1),
    'b': -((np.sum(np.dot(a.T, (lambdas*X).T)) + C*np.sum(y[np.abs(lambdas - C) < Eps])) /
            np.sum(lambdas[np.abs(lambdas - C) >= Eps])),###???
    'lambdas': lambdas,
    'dual': -sol['primal objective'],
    'status': (2 * (sol['iterations'] == max_iter) + 
                            1 * (sol['iterations'] != max_iter and
                                     sol['status'] == 'unknown')),
    'time': time.time() - tm}
    return ret_dic


def smooth_qp_primal_real_solver(X, y, fines=np.array([0, 0, 0]), gamma=1,
    alpha=0, C=0.1, tol=1e-6, max_iter=10**6, verbose=False):
    """Функции для решения задачи SVM на реальных данных:
        svm_qp_primal_real_solver(X, y, fines=np.array([0, 0, 0]),
            alpha=0, C=0.1, tol=1e-6, max_iter=10**6, verbose=False)
    Описание параметров:
        • (!!!) X — переменная типа numpy.ma (masked array), uncomressed!,
            матрица размера N × 2 x max_D, признаковые описания объектов обучающей выборки, 
        • y — переменная типа numpy.array, матрица размера N × 1,
            правильные ответы на обучающей выборке,
        • fines — переменная типа numpy.array, матрица размера 3,
                        штраф за совпадение ai, bj,
                              за растяжение по i (bj фиксир),
                              за сжатие по i (ai фиксир);
        • gamma — скорость роста штрафа при возрастании различия r(wi, wj)
        • C — параметр регуляризации,
        • alpha — параметр гладкости,
        • tol — требуемая точность,
        • max_iter — максимальное число итераций,
        • verbose — в случае True, требуется выводить отладочную информацию
            на экран (номер итерации, значение целевой функции),
    
    Возвращает словарь с полями:
        • ’a’ — numpy.array, вектор параметров модели,
            веторна часть матрица размера K × 1,
        • 'b' — скаляр, сдвиг в векторе параметров модели
        • 'lambdas' — numpy.array, матрица размера N×1, значения
        двойственных переменных,
        • ’status’ — {0, 1, 2}, 0 — если метод вышел по критерию останова,
                                1 — если оптимизация невыполнима,
        • ’time’ — время работы метода.
        • 'dual' — максимальное значение двойственной функции 
    """
    import sigproc
    tm_all = time.time()
    B = sigproc.m_distance_features(X, fines=fines) #matrix of distances (DTW)
    if verbose:
        print("Time elapsed [DTW]: ", time.time() - tm_all)
        tm = time.time()
    X_n = -B*y.T
    # prepare qp matrixes:
    N = X_n.shape[0]
    D = N
    """P is a square dense or sparse real matrix, representing a positive semidefinite symmetric matrix in 'L' storage,
    i.e., only the lower triangular part of P is referenced. q is a real single-column dense matrix.
    The arguments h and b are real single-column dense matrices. G and A are real dense or sparse matrices. """
    B[B!=0] = B[B!=0]**(-gamma)
    B = np.eye(D) + alpha*(-B + np.eye(D)*np.sum(B, axis=1))# D==N
    P = spdiag([matrix(B), spmatrix([], [], [], (N, N))])
    q = matrix(0.0, (D + N, 1))# D==N
    q[D:] = C

    I = spmatrix(1, range(N), range(N))
    O = spmatrix([], [], [], (N, D))
    G = sparse([[-matrix(y*X_n), O, -I], [-I, -I, O]])
    h = matrix(0., (N+N+N, 1))
    h[:N] = -1

    solvers.options['maxiters'] = max_iter
    solvers.options['reltol'] = tol
    solvers.options['show_progress'] = verbose
    if verbose:
        print("Time elapsed [Cone QP preparation]: ", time.time() - tm)
        tm = time.time()
    sol = solvers.qp(P, q, G, h)

    ret_dic = {
    'a': np.array(sol['x'][:D]).reshape(-1, 1),
    #w /= np.sum(w * w, axis=0) ** (1 / 2)  # normalize?
    'b': 0,
    'ksi': np.array(sol['x'][D:]).reshape(-1, 1),
    'primal': sol['primal objective'],
    'status': (2 * (sol['iterations'] == max_iter) + 
                            1 * (sol['iterations'] != max_iter and
                                     sol['status'] == 'unknown')),
    'time': time.time() - tm_all}
    return ret_dic

def predict(X_test, data_type, problem_type, a, b=0,
    X_train=None, y_train=None, fines=np.array([0, 0, 0]), multi_solver=False):
    dt = data_type
    pt = problem_type
    if dt == 'real':
        if pt == 'primal':
            import sigproc
            R = -y_train.T*sigproc.m_distance_features(X_test, fines, X_train)
            y_pred = np.sign(np.sum(R*a.T, axis=1)).reshape(-1, 1)
        else: pass
    else: pass

    return y_pred


def multi_solver(X, y, data_type, problem_type, **kwargs):
    """
    Возвращет:
        a_list - numpy.array,  из значений N весов (a - вектор, размера количества объектов)
            и нулевы значений
    """
    dt = data_type
    pt = problem_type
    if dt == 'real':
        if pt == 'primal':
            a_list = []
            for y_cl in np.unique(y):
                ?????????res = smooth_qp_primal_real_solver(X, y, **kwargs)
                ????????a_list += [[res['a'], res['b']]
        else: pass
    else: pass
    return np.array(a_list)

def multi_predict(X_test, data_type, problem_type, a_list,**kwargs):
    dt = data_type
    pt = problem_type
    if dt == 'real':
        if pt == 'primal':
            import sigproc
            R = -y_train.T*sigproc.m_distance_features(X_test, fines, X_train)
            y_pred = np.argmax(R.dot(a_list[:, :-1]))
        else: pass
    else: pass

