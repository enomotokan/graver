import numpy as np
import time

# 整基底を見つけるアルゴリズムの中で使うユークリッド互除法アルゴリズム
def euqlid(A, i, j, k):
    n = 0
    if A[i, j] == 0:
        if A[i, k] != 0:
            A[:, j], A[:, k] = A[:, k].copy(), A[:, j].copy()
    if A[i, k] == 0:
        pass

    elif A[i, j] >= A[i, k]:
        while True:
            if n == 0:
                A[:, j] -= np.floor(A[i, j].astype(np.float32) / A[i, k].astype(np.float32)).astype(np.int32) * A[:, k]
                if A[i, j] == 0:
                    A[:, j], A[:, k] = A[:, k].copy(), A[:, j].copy()
                    break
            else:
                A[:, k] -= np.floor(A[i, k].astype(np.float32) / A[i, j].astype(np.float32)).astype(np.int32) * A[:, j]
                if A[i, k] == 0:
                    break
            n += 1
            n %= 2
    else:
        while True:
            if n == 0:
                A[:, k] -= np.floor(A[i, k].astype(np.float32) / A[i, j].astype(np.float32)).astype(np.int32) * A[:, j]
                if A[i, k] == 0:
                    break
            else:
                A[:, j] -= np.floor(A[i, j].astype(np.float32) / A[i, k].astype(np.float32)).astype(np.int32) * A[:, k]
                if A[i, j] == 0:
                    A[:, j], A[:, k] = A[:, k].copy(), A[:, j].copy()
                    break
            n += 1
            n %= 2
    return A

# 入力を検証する
def verify_input(A):
    if type(A) != np.ndarray:
        print("Error: Input is not Numpyarray")
        return False
    else:
        if A.dtype != int:
            print("Error: Dtype of input must be int")
            return False
        else:
            if A.ndim != 2:
                print("Error: Dimension of input matrix must be two")
                return False
            else:
                if A.shape[1] <= A.shape[0]:
                    print("Error: Width of input matrix must be larger than length")
                    return False
                else:
                    return True

# Aの核の格子点の整基底を計算する
def basis_intker(A):
    if verify_input(A):
        N = A.shape[0]
        A = np.append(A, np.eye(A.shape[1]).astype(np.int32), axis=0)
        n_deleted: int = 0
        for i in range(N):
            if np.all(A[i - n_deleted][i - n_deleted:] == 0):
                A = np.delete(A , i - n_deleted)
                n_deleted += 1
                continue

            equal_1 = np.where(A[i - n_deleted, i - n_deleted:] == 1)[0]
            if len(equal_1) == 0:
                for j in range(i - n_deleted + 1, A.shape[1]):
                    if A[i, j] < -1:
                        A[:, j] *= -1
                    A = euqlid(A, i - n_deleted, i - n_deleted, j)
            else:
                A[:, i - n_deleted], A[:, i - n_deleted + equal_1[0]] = A[:, i - n_deleted + equal_1[0]].copy(), A[:, i - n_deleted].copy()
                for j in range(i - n_deleted + 1, A.shape[1]):
                    A[:, j] -= A[i, j] * A[:, i - n_deleted]
        
        return A[i - n_deleted + 1:, i - n_deleted + 1:].T

# 与えられた整基底の生成する加群のグレーバー基底を計算する
def graver(basis):
    if verify_input(basis):
        M = basis.copy().astype(np.float32)
        linear_independent = np.array([], dtype=np.int32)
        for i in range(M.shape[0]):
            equal_0 = np.abs(M[i]) <= 2**-15
            if np.all(equal_0):
                M[i] = np.zeros(M.shape[1])
                continue
            else:
                k = np.arange(M.shape[1])[np.logical_not(equal_0)][0]
                M[i] /= M[i, k]
                for j in np.delete(np.arange(M.shape[0]), i):
                    M[j] -= M[i] * M[j, k]
                linear_independent =np.append(linear_independent, k)
                if len(linear_independent) == M.shape[0]:
                    break

        basis = basis.astype(np.int32)
        projected_basis = basis[:, [i in linear_independent for i in range(basis.shape[1])]]

        # 射影された基底に対してPottierアルゴリズムを適用する
        G = np.array([projected_basis[np.int32(n / 2)] if n % 2 == 0 else - projected_basis[np.int32((n - 1) / 2)] for n in range(2 * projected_basis.shape[0])])
        # G = projected_basis.copy()
        C = np.empty((0, projected_basis.shape[1]), dtype=np.int32)
        for i in range(projected_basis.shape[0]):
            for j in range(i + 1, projected_basis.shape[0]):
                C = np.append(C, [projected_basis[i] + projected_basis[j]], axis=0)
                C = np.append(C, [- projected_basis[i] - projected_basis[j]], axis=0)
                C = np.append(C, [projected_basis[i] - projected_basis[j]], axis=0)
                C = np.append(C, [- projected_basis[i] + projected_basis[j]], axis=0)

        while len(C) != 0:
            s = C[0]
            C = np.delete(C, 0, axis=0)
            for g in G:
                if np.all(s == 0):
                    s = 0
                    break
                elif np.all(np.sign(g) * np.sign(s) >= 0):
                    if np.all(np.abs(g) <= np.abs(s)):
                        s -= np.min(np.round(s.astype(np.float32)[g != 0] / g.astype(np.float32)[g != 0])).astype(np.int32) * g
            if not np.all(s == 0):
                for g in G:
                    C = np.append(C, [g + s], axis=0)
                G = np.append(G, [s], axis=0)
        
        # 極小元でない元を除く
        G_ = np.empty((0, G.shape[1]), dtype=np.int32)
        minimum = True
        not_minimum_list = np.array([], dtype=np.int32)
        while len(G) != 0:
            minimum = True
            for j in range(1, len(G)):
                if np.all(np.sign(G[0]) * np.sign(G[j]) >= 0):
                    if np.all(np.abs(G[0]) >= np.abs(G[j])) and not np.all(np.abs(G[0]) == np.abs(G[j])):
                        minimum = False
                    elif np.all(np.abs(G[0]) <= np.abs(G[j])):
                        not_minimum_list = np.append(not_minimum_list, j)
            if minimum:
                G_ = np.append(G_, [G[0]], axis=0)
            G = np.delete(G, not_minimum_list, axis=0)
            not_minimum_list = np.array([], dtype=np.int32)
            G = np.delete(G, 0, axis=0)
        
        # 射影・持ち上げアルゴリズム
        projected_basis_inv = np.linalg.inv(projected_basis.astype(np.float32))
        notin_linearindependent = [i not in linear_independent for i in range(basis.shape[1])]
        F_notin_linearindependent = np.round(np.dot(np.dot(G_, projected_basis_inv), basis[:, notin_linearindependent])).astype(np.int32)
        
        F = np.empty((G_.shape[0], basis.shape[1]), dtype=np.int32)
        n_added_in_linearindependent = 0
        n_added_notin_linearindependent = 0
        for i in range(basis.shape[1]):
            if i in linear_independent:
                F[:, i] = G_[:, n_added_in_linearindependent]
                n_added_in_linearindependent += 1
            else:
                F[:, i] = F_notin_linearindependent[:, n_added_notin_linearindependent]
                n_added_notin_linearindependent += 1

        G = F.copy()
        C_ = np.empty((0, G.shape[1]), dtype=np.int32)

        for i in range(G.shape[0]):
            for j in range(i + 1, G.shape[0]):
                if not np.all(G[i] == -G[j]):
                    C_ = np.append(C_, [G[i] + G[j]], axis=0)
        
        project =[i in linear_independent for i in range(basis.shape[1])]

        C = [np.empty((0, basis.shape[1]), dtype=np.int32)]
        for c in C_:
            norm_c = np.sum(np.abs(c)).astype(np.int32)
            if len(C) - 1 < norm_c:
                for i in range(norm_c - len(C) + 1):
                    C.append(np.empty((0, basis.shape[1]), dtype=np.int32))
            C[norm_c] = np.append(C[norm_c], [c], axis=0)
        norm: int = 0
        while True:
            s = 0
            for norm in range(1, len(C)):
                if len(C[norm]) != 0:
                    s = C[norm][0]
                    C[norm] = np.delete(C[norm], 0, axis=0)
                    break
            if np.all(s == 0):
                break
            sign_s = np.sign(s)
            # if not np.any([np.all(np.sign(v) * sign_s >= 0) and np.all(v <= s) for v in G]): 
            if np.all([np.any(np.sign(v) * sign_s < 0) or np.any(np.abs(v) > np.abs(s)) for v in G]):
                for g in G:
                    if np.all(np.sign(s[project]) * np.sign(g[project]) >= 0):
                        norm = np.sum(np.abs(s + g))
                        if len(C) - 1 < norm:
                            for i in range(norm - len(C) + 1):
                                C.append(np.empty((0, basis.shape[1]), dtype=np.int32))
                        C[norm] = np.append(C[norm], [s + g], axis=0)
                G = np.append(G, [s], axis=0)
        return G

def graver_original(projected_basis):
    G = np.array([projected_basis[np.int32(n / 2)] if n % 2 == 0 else - projected_basis[np.int32((n - 1) / 2)] for n in range(2 * projected_basis.shape[0])])
    # G = projected_basis.copy()
    C = np.empty((0, projected_basis.shape[1]), dtype=np.int32)
    for i in range(projected_basis.shape[0]):
        for j in range(i + 1, projected_basis.shape[0]):
            C = np.append(C, [projected_basis[i] + projected_basis[j]], axis=0)
            C = np.append(C, [- projected_basis[i] - projected_basis[j]], axis=0)
            C = np.append(C, [projected_basis[i] - projected_basis[j]], axis=0)
            C = np.append(C, [- projected_basis[i] + projected_basis[j]], axis=0)

    while len(C) != 0:
        s = C[0]
        C = np.delete(C, 0, axis=0)
        for g in G:
            if np.all(s == 0):
                s: int = 0
                break
            elif np.all(np.sign(g) * np.sign(s) >= 0):
                if np.all(np.abs(g) <= np.abs(s)):
                    s -= np.min(np.round(s.astype(np.float32)[g != 0] / g.astype(np.float32)[g != 0])).astype(np.int32) * g
        if not np.all(s == 0):
            for g in G:
                C = np.append(C, [g + s], axis=0)
            G = np.append(G, [s], axis=0)
                
    # 極小元でない元を除く
    G_ = np.empty((0, G.shape[1]), dtype=np.int32)
    minimum = True
    not_minimum_list = np.array([], dtype=np.int32)
    while len(G) != 0:
        minimum = True
        for j in range(1, len(G)):
            if np.all(np.sign(G[0]) * np.sign(G[j]) >= 0):
                if np.all(np.abs(G[0]) >= np.abs(G[j])) and not np.all(np.abs(G[0]) == np.abs(G[j])):
                    minimum = False
                elif np.all(np.abs(G[0]) <= np.abs(G[j])):
                    not_minimum_list = np.append(not_minimum_list, j)
        if minimum:
            G_ = np.append(G_, [G[0]], axis=0)
        G = np.delete(G, not_minimum_list, axis=0)
        not_minimum_list = np.array([], dtype=np.int32)
        G = np.delete(G, 0, axis=0)
    return G_


if __name__ == "__main__":
    A = np.array([[2, -1, 0, -3, 2, -2], [1, 5, -4, 0, 0, 0]], dtype=np.int32)
    basis = basis_intker(A)
    t = time.time()
    proj_lift = graver(basis)
    print(proj_lift)
    print(len(proj_lift))
    t1 = time.time()
    print(t1 - t)
    original = graver_original(basis)
    print(original)
    print(len(original))
    print(time.time() - t1)
    