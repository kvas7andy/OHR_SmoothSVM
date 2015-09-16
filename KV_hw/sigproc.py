import numpy as np
import svm
from ipyparallel import interactive

@interactive
def dtw(a, b, do_path=False, fines=np.array([0, 0, 0]),
        metric='euclidean', d=None, ver=1):
    """
    Проводим трасформацию пары сигналов.
    Описание параметров:
        a, b - numpy.array, пара сигналов, вектор столбец Mx2 и Nx2;
        do_path - boolean, найти и вывести путь;
        fines - numpy.array, штраф за совпадение ai, bj,
                              за растяжение по i (bj фиксир),
                              за сжатие по i (ai фиксир);
        metric - метрика сходства (при 0 значении) между векторами измерений;

    Возвращается кортеж значений:
        res - double, вычисленное значение несходства; 
        Если do_path==True:
            path - numpy.array, таблица парных соответствий,
                                        Kx2, max(m, n) <= K <= m+n;
        D - numpy.array,
            матрица наименьших расстояний до сигналов a,b с концами в (i, j), MxN;
    """

    M, N = a.shape[0], b.shape[0]; D = np.zeros((M+1, N+1))
    D[0,:] = np.inf; D[:,0] = np.inf; D[0, 0] = 0

    #distance matrix inside D[1:, 1:]
    if metric=='euclidean':
        D[1:, 1:] = (np.sum(a**2, axis=1)[:, np.newaxis] +
                np.sum(b**2, axis=1)[np.newaxis, :] - 2*a.dot(b.T))**(1/2)

    #make D[i,j] - optimal distance between a1,...,ai and b1,...,bj time series
    if d is None:
        d = M+N
    sum_fines = 0

    for i in range(1, M+1):
        for j in range(1, N+1):
            if np.abs(i-j) > d: D[i, j] = np.inf; continue
            if ver != 2:
                tmp_arr = np.array([D[i-1, j-1], D[i-1, j], D[i, j-1]]) + fines
                dmin = np.min(tmp_arr)
                D[i, j] += dmin
                if ver == 3:
                    ind = np.argmin(tmp_arr)
                    sum_fines += fines[ind]
            else:
                dmin = np.min(np.array([D[i-1, j-1], D[i-1, j], D[i, j-1]]) +
                              fines*np.array([D[i, j], D[i, j], D[i, j]]))
                D[i, j] = dmin
            

    if do_path:
    #Traceback from end 
        path = [[M, N]]; i, j = M, N
        while i > 0 and j > 0:
            min_ind = np.argmin(np.array([D[i-1, j-1], D[i-1, j], D[i, j-1]]) +
                                fines)### n_optimal
            i = i - (min_ind != 2)
            j = j - (min_ind != 1)
            path = [[i, j]] + path
        if i > 0 or j > 0:
            path = [[1, 1]] + path
        path = np.array(path)

    if ver != 2:
        if ver != 3 or sum_fines == 0:
            res = D[M, N]
        else:
            res = D[M, N] / sum_fines - 1;
    else:
        res = D[M, N]/(M+N)
    D = D[1:, 1:]
    return (res, path, D) if do_path else (res, D)

@interactive
def m_distance_features(X1, fines, X2=None, d=None, ver=1):
    if X2 is None: #DTW distance of X1 objects => N(N-1)/2
        X = X1
        R = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
        for i in range(X.shape[0]-1):
            for j in range(i+1, X.shape[0]):
                a = X[i].compressed().reshape(2, -1).T
                b = X[j].compressed().reshape(2, -1).T
                R[i][j] = dtw(a, b, do_path=False, fines=fines, d=d, ver=ver)[0]
                R[j][i] = R[i][j]
    else: #DTW distance of X1 to X2 objects => N*M
        R = np.zeros((X1.shape[0], X2.shape[0]), dtype=np.float64)
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                a = X1[i].compressed().reshape(2, -1).T
                b = X2[j].compressed().reshape(2, -1).T
                R[i][j] = dtw(a, b, do_path=False, fines=fines, d=d, ver=ver)[0]
    return R


def hold_out_modelhw(dataX, datay, K, N_max,
    alpha, N_train,n_boarder='N_train', exper_iters=100):
    N_train_cl = N_train // 2 # D = 100 ?? Сколько берем объектов одного класса для эксперимента
    N_test_cl = N_train_cl#N_cl - N_train_cl

    if n_boarder == 'N_max':
        N_boarder = N_max
    else:
        N_boarder=N_train
    
    res_sum = []
    #par_sum = []
    for i in range(K.size):
        res = []
        #param = []
        for j in range(i+1, K.size):
            results = np.empty((exper_iters,), dtype='float64')
            for exper_iter in range(exper_iters):
                num1 = i #Номер первого класса
                num2 = j #Номер второго класса
                #Сделаем обучающую выборку
                ind1 = np.random.choice(N_boarder, size=N_train_cl, replace=False)
                ind2 = np.random.choice(N_boarder, size=N_train_cl, replace=False)
                ind = (num1*N_max + ind1).tolist() + (num2*N_max + ind2).tolist()
                X_train = dataX[ind, :]
                y_train = np.ones((2*N_train_cl, 1))
                y_train[:N_train_cl] = -1
                #Сделаем контрольную выборку
                ind3 = np.array(list(set(np.arange(N_boarder)) - set(ind1)))
                ind4 = np.array(list(set(np.arange(N_boarder)) - set(ind2)))
                ind3 = np.random.choice(ind3, size=N_test_cl, replace=False)
                ind4 = np.random.choice(ind4, size=N_test_cl, replace=False)
                ind = (num1*N_max + ind3).tolist() + (num2*N_max + ind4).tolist()
                X_test = dataX[ind, :]
                y_test = np.ones((2*N_test_cl, 1))
                y_test[:N_test_cl] = -1
                #Проводим обучение
                sol = svm.smooth_qp_dual_solver(X_train, y_train, alpha=alpha, verbose=False)
                #Предсказания классификатора на контрольной выборке
                y_pred = np.sign(np.dot(X_test, sol['a']) + sol['b']) #??немного схитрил умножив наоборот не y = sgn(res['a'].T * X.T  + ).T
                #Доля правильно распознанных объектов
                accuracy = np.sum(y_pred != y_test) / (2*N_test_cl) #N_test_cl - количество контрольных объектов в одном классе
                results[exper_iter] = accuracy
            #param += [[(num1, num2), (ind1, ind2), sol]]
            res += [results]
        #if not param:
        #    param = [np.nan]
        if res:
            res_sum += [res]
        #par_sum += [param]
    #res = np.zeros((K.size, K.size), dtype='float64')
    #for i in range(K.size - 1):
    #    res[i, i+1:] = np.mean(res_sum[i], axis=-1)
    return res_sum#, par_sum

#a way of getting general pathed symbol
if __name__ == '__main__':
    a = np.arange(20).reshape(5, 4)
    l = np.array([[1, 0], [0, 1], [1, 1]])
    print(a[l.T[0], l.T[1]])