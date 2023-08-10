import numpy as np

def initial_solution(A):

# 一次元最適化
def oneD_optimize(A, range, var, base):
    if range[1] - range[0] == 0:
        return var + range[1] * base, df(A, var, range[1] * base)
    elif range[1] - range[0] == 1:
        if  df(A, var, base) < 0:
            return var + range[1] * base, df(A, var, range[1] * base)
        else:
            return var + range[0] * base, df(A, var, range[0] * base)
    else:
        central = np.floor(np.mean(range)).astype(np.int32)
        if df(A, var + central * base, -base) < 0:
            return oneD_optimize(A, np.array([range[0], central]), var, base)
        else:
            if df(A, var + central * base, base) < 0:
                return oneD_optimize(A, np.array([central + 1, range[1]]), var, base)
            else:
                return var + central * base, df(A, var, central * base)

# 函数の差分
def df(A, var, dv):
    df: np.float32 = 0
    for i in range(A.shape[0]):
        df += np.dot(A[i]**2, dv * (var + dv))


