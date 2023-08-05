import numpy as np 
import types
from graver import *

# 左辺のとりうる最小値を求める
def max_lefthand(bound, constant, coefficient, variable_n):
    max_lefthand = -constant
    for variable_name in coefficient.keys():
        if coefficient[variable_name] >= 0:
            max_lefthand += bound[variable_n[variable_name]][1] * coefficient[variable_name]
        else:
            max_lefthand += bound[variable_n[variable_name]][0] * coefficient[variable_name]
    return max_lefthand

# 左辺のとりうる最大値を求める
def min_lefthand(bound, constant, coefficient, variable_n):
    max_lefthand = -constant
    for variable_name in coefficient.keys():
        if coefficient[variable_name] >= 0:
            max_lefthand += bound[variable_n[variable_name]][0] * coefficient[variable_name]
        else:
            max_lefthand += bound[variable_n[variable_name]][1] * coefficient[variable_name]
    return max_lefthand

# 整数線型制約分離可能凸整数最適化
# Separable Convex Integer Optimization with Linear Integer Conditions
class SCIOLIC():
    def __init__(self):

        # 実行可能解を最大何回探索するか
        self.N_search_feasible = 3

        # 近似度
        self.alpha = 0.9

        # ログ出力するかどうか
        self.verbose = True

        # エラーがあるかどうか
        self.errored = False

        # 変数名対番号の辞書
        self.variable_n = {}

        # 変数名対分離可能凸函数の辞書
        self.convexfunc = {}

        # 番号対変数名の辞書
        self.n_variable = {}

        # 変数の数
        self.N_variable = 0

        # 変数の上下限
        self.bound = []

        # 係数のリスト
        self.coefficient = []

        # 条件式の不等号のリスト
        self.condition = []

        # 標準形のA,b
        self.A = 0
        self.b = []

    # 変数の追加
    def add_variable(self, var_name: str, lower_bound: int, upper_bound: int):

        if type(var_name) != str:
            print("Error: input var_name must be string")
            self.errored = True
        else:
            if type(lower_bound) != int:
                print("Error: input lower_bound must be integer")
                self.errored = True
            else:
                if type(upper_bound) != int:
                    print("Error: input upper_bound must be integer")
                    self.errored = True
                else:
                    if not var_name in self.variable_n.keys():
                        self.variable_n[var_name] = self.N_variable
                        self.n_variable[self.N_variable] = var_name
                        self.bound.append([lower_bound, upper_bound])
                        self.N_variable += 1
                    else:
                        self.bound[self.variable_n[var_name]]=[lower_bound, upper_bound]

    # 分離可能凸函数の追加
    def add_convexfunc(self, variable_name: str, f: types.FunctionType or int or float):
        if not (type(f) == types.FunctionType or int or float):
            print("Error: input f must be function or a real number which is the coefficient of one variable linear function")
        else:
            if not variable_name in self.variable_n.keys():
                print("Error: input variable_name does not exist")
            else:
                self.convexfunc[variable_name] = f
                    
    # 条件式の不等号と定数を追加
    def add_condition(self, inequality_sign: str, constant: int):
        if inequality_sign not in ["=", "<=", ">=", "<", ">"]:
            print("Error: input inequality sign must be selected from \"=\", \"<=\", \">=\", \"<\", \">\"")
            self.errored = True
        else:
            if type(constant) != int:
                print("Error: input constant must be integer")
                self.errored = True
            else:
                if inequality_sign in ["=", "<=", ">="]:
                    self.condition.append(inequality_sign)
                    self.coefficient.append({})
                    self.b.append(constant)
                elif inequality_sign == "<":
                    self.condition.append("<=")
                    self.coefficient.append({})
                    self.b.append(constant - 1)
                else:
                    self.condition.append(">=")
                    self.coefficient.append({})
                    self.b.append(constant + 1)
                    

    # 追加した条件式に対し係数を定義
    def define_coefficient(self, condition_num: int, variable_name: str, coefficient: int):
        if not condition_num <= len(self.condition) - 1:
            print("Error: specified condition number does not exist")
            self.errored = True
        else:
            if variable_name not in self.variable_n.keys():
                print("Error: Specified variable name does not exist")
                self.errored = True
            else:
                self.coefficient[condition_num][variable_name] = coefficient

    # 解を探索
    def search(self):
        if self.verbose:
            print("Welcome to SCIOLIC, a solver for separatable convex integer optimization with linear integer condition")
            print("Created by Enomoto Kan, Oita prefecture, Japan")
            print("Contact: enomotokan@gmail.com\n")
        if self.errored:
            print("Error: definition of the problem has error.")
        else:
            # ログ出力を行う
            if self.verbose:
                
                print("Starting to reserch a feasible solution...")
                

            # 条件式の中で全ての係数が0であるものがないかどうか調べる
            for condition_num in range(len(self.condition)):
                if self.condition[condition_num] == {}:
                    print("Error: All coefficient of each variable of condition " + str(condition_num) + " is 0")
                    self.errored = True
            
            # if len(self.condition) == 0:


            # 入力した情報を標準形に変換
            self.A = np.zeros((len(self.condition), self.N_variable), dtype=np.int32)
            self.b = np.array(self.b)
            self.max_array = np.array([])

            for n in range(len(self.condition)):

                if self.condition[n] == ">=":
                    max_lefthand_n = max_lefthand(self.bound, self.b[n], self.coefficient[n], self.variable_n)
                    if max_lefthand_n < 0:
                        print("Error: condition " + str(n) +" cannot be satisfied")
                        self.errored = True
                    self.max_array = np.append(self.max_array, max_lefthand_n)
                    self.bound.append([0, max_lefthand_n])
                    for variable_name in self.coefficient[n].keys():
                        self.A[n][self.variable_n[variable_name]] = self.coefficient[n][variable_name]
                    self.A = np.append(self.A, np.array([np.zeros(len(self.condition))]).astype(np.int32).T, axis=1)
                    self.A[n][-1] = -1

                elif self.condition[n] == "<=":
                    min_lefthand_n = min_lefthand(self.bound, self.b[n], self.coefficient[n], self.variable_n)
                    if min_lefthand_n > 0:
                        print("Error: condition " + str(n) +" cannot be satisfied")
                        self.errored = True
                    self.max_array = np.append(self.max_array, - min_lefthand_n)
                    self.bound.append([0, -min_lefthand_n])
                    for variable_name in self.coefficient[n].keys():
                        self.A[n][self.variable_n[variable_name]] = self.coefficient[n][variable_name]
                    self.A = np.append(self.A, np.array([np.zeros(len(self.condition))]).astype(np.int32).T, axis=1)
                    self.A[n][-1] = 1

                elif self.condition[n] == "=":
                    max_lefthand_n = max_lefthand(self.bound, self.b[n], self.coefficient[n], self.variable_n)
                    min_lefthand_n = min_lefthand(self.bound, self.b[n], self.coefficient[n], self.variable_n)
                    if not max_lefthand_n >= 0 >= min_lefthand_n:
                        print("Error: condition " + str(n) +" cannot be satisfied")
                        self.errored = True
                    self.max_array = np.append(self.max_array, np.max([max_lefthand_n, - min_lefthand_n]))
                    for variable_name in self.coefficient[n].keys():
                        self.A[n][self.variable_n[variable_name]] = self.coefficient[n][variable_name]
            self.bound = np.array(self.bound)

            # 条件が充たされているか
            satisfied = True

            for i in range(self.N_search_feasible):
                # 実行可能解を探索
                var = search_feasible_solution(self.A, self.b, self.max_array, self.bound, self.N_variable)

                for n in range(len(self.condition)):
                    lefthand = 0
                    for variable_name, coef in self.coefficient[n].items():
                        lefthand += var[self.variable_n[variable_name]] * coef
                    if self.condition[n] == "=":
                        if lefthand != self.b[n]:
                            satisfied = False
                    elif self.condition[n] == ">=":
                        var = np.append(var, lefthand - self.b[n])
                        if lefthand < self.b[n]:
                            satisfied = False
                    else:
                        var = np.append(var, - lefthand + self.b[n])
                        if lefthand > self.b[n]:
                            satisfied = False
                            
                if satisfied:
                    break
            if not satisfied:
                self.errored = True
                print("Error: condition is not satisfied. Feasible solution may not exist or may be hard to find")
            elif self.verbose:
                print("One feasible solution was discovered")
            
            if self.errored:
                return None
                

            if self.verbose:
                print("Configuring Graver basis...")
            graver_A = graver(basis_intker(self.A))
            if self.verbose:
                print("Configuration was completed. Optimizing the problem...")

            # graver最良増加法
            equal_func = np.array([type(f) == types.FunctionType for f in list(self.convexfunc.values())])

            coef = np.array([])
            convfunc = []
            for f in list(self.convexfunc.values()):
                if type(f) == (int or float):
                    coef = np.append(coef, f)
                else:
                    convfunc.append(f)
            graver_A_ = np.empty((0, graver_A.shape[1]), dtype=np.int32)
            for base in graver_A:
                notin_graver_A_ = True
                for base_ in graver_A_:
                    if np.all(base == -base_):
                        notin_graver_A_ = False
                        pass
                    else:
                        continue
                if notin_graver_A_:
                    graver_A_ = np.append(graver_A_, [base], axis=0)
            graver_A = graver_A_
            # 最適解かどうか
            bool_optimal = False

            for i in range(np.ceil(np.log(1 - self.alpha) / np.log(1 - 1 / (2 * (self.A.shape[1] - 1)))).astype(np.int32)):
                var_list = np.empty((0, len(var)), dtype=np.int32)
                df_list = np.array([])
                for base in graver_A:
                    plus_range = np.iinfo(np.int32).max
                    minus_range = np.iinfo(np.int32).min
                    for n in range(graver_A.shape[1]):
                        if np.sign(base[n]) == 1:
                            plus_range = np.min(np.append(np.floor((self.bound[n][1] - var[n]) / base[n]).astype(np.int32), plus_range))
                            minus_range = np.max(np.append(np.ceil((self.bound[n][0] - var[n]) / base[n]).astype(np.int32), minus_range))
                        elif np.sign(base[n]) == -1:
                            plus_range = np.min(np.append(np.floor((self.bound[n][0] - var[n]) / base[n]).astype(np.int32), plus_range))
                            minus_range = np.max(np.append(np.ceil((self.bound[n][1] - var[n]) / base[n]).astype(np.int32), minus_range))
                    var_, df_ = oneD_optimize(coef, convfunc, [minus_range, plus_range], var, equal_func, base, self.N_variable)
                    var_list = np.append(var_list, [var_], axis=0)
                    df_list = np.append(df_list, df_)
                df_argmin = np.argmin(df_list)
                if np.abs(df_list[df_argmin]) <= 2**-15:
                    bool_optimal = True
                    break
                var = var_list[df_argmin]
            
            if self.verbose:
                print("Optimization succeeded")
            
            result = {}
            for n, variable in self.n_variable.items():
                result[variable] = var[n]
            return result, bool_optimal
                    
# 一次元最適化
def oneD_optimize(coef, convfunc, range, var, equal_func, base, N_variable):
    if range[1] - range[0] == 0:
        return var + range[1] * base, df(coef, convfunc, var[:N_variable], equal_func, range[1] * base[:N_variable])
    elif range[1] - range[0] == 1:
        if  df(coef, convfunc, var[:N_variable], equal_func, base[:N_variable]) < 0:
            return var + range[1] * base, df(coef, convfunc, var[:N_variable], equal_func, range[1] * base[:N_variable])
        else:
            return var + range[0] * base, df(coef, convfunc, var[:N_variable], equal_func, range[0] * base[:N_variable])
    else:
        central = np.floor(np.mean(range)).astype(np.int32)
        if df(coef, convfunc, var[:N_variable] + central * base[:N_variable], equal_func, -base[:N_variable]) < 0:
            return oneD_optimize(coef, convfunc, np.array([range[0], central]), var, equal_func, base, N_variable)
        else:
            if df(coef, convfunc, var[:N_variable] + central * base[:N_variable], equal_func, base[:N_variable]) < 0:
                return oneD_optimize(coef, convfunc, np.array([central + 1, range[1]]), var, equal_func, base, N_variable)
            else:
                return var + central * base, df(coef, convfunc, var[:N_variable], equal_func, central * base[:N_variable])

# dv動かしたときのfの差分
def df(coef, convfunc, var, equal_func, dvar):
    df = np.dot(coef, dvar[np.logical_not(equal_func)])
    if not np.all(np.logical_not(equal_func)):
        for f, v, dv in convfunc, var[equal_func], dvar[equal_func]:
            df += f(v + dv) - f(v)
    return df

                
            
# 自然数を0,1の結合で表すための結合ベクトルを生成する函数
def n_alpha(n):
    alpha=np.array([])

    while True:
        lognint = np.int64(np.log2(n + 1))
        alpha = np.append(alpha, np.array([2**(lognint - i - 1) for i in range(lognint)]))
        n -= 2**lognint - 1
        if n == 0:
            break
    return alpha

# 連続変数焼き鈍し法の確率変数を生成
def update(betadE):
    if np.abs(betadE) <= 2**-15:
        return np.random.rand()
    elif betadE >=0:
        return - np.log((np.exp(- betadE) - 1) * np.random.rand() + 1) / betadE
    else:
        return 1 - np.log((1-np.exp(betadE) * np.random.rand() + np.exp(betadE))) / betadE

# 実行可能解を探索する
def search_feasible_solution(A, b, max_array, bound, N_variable):
    coefficient_representation = []
    len_representation = np.array([], dtype=np.int32)
    for n in range(bound.shape[0]):
        coefficient_representation.append(n_alpha(bound[n][1] - bound[n][0]))
        len_representation = np.append(len_representation, len(coefficient_representation[-1]))
    # 焼き鈍し法の変数の数
    N_var_anneal = np.sum(len_representation, dtype=np.int32)
    
    # 焼き鈍し法の函数の係数
    b_ = b.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            b_[i] -= A[i, j] * bound[j][0]
    A_ = np.zeros((A.shape[0], N_var_anneal), dtype=np.float32)
    n = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(len_representation[j]):
                A_[i, n] = A[i, j] * coefficient_representation[j][k]
                n += 1
        n = 0
    A_ = A_.astype(np.float32)
    b_ = b_.astype(np.float32)
    for i in range(A_.shape[0]):
        A_[i] /= np.array([max_array * A.shape[0]], dtype=np.float32)[i]
        b_[i] /= np.array([max_array * A.shape[0]], dtype=np.float32)[i]

    anneal_A = np.zeros((N_var_anneal, N_var_anneal))
    for i in range(N_var_anneal):
        anneal_A[i, i] += A_[n, i] * (A_[n, i] - 2 * b_[n]) / 2
        for j in range(i + 1, N_var_anneal):
            for n in range(A.shape[0]):
                anneal_A[i, j] += 2 * A_[n, i] * A_[n, j]
    
    anneal_A += anneal_A.T
                    
    # 焼き鈍し法の変数を初期化
    beta = 0.01
    gamma = 1000000**(1 / (30 * N_variable))
    x = np.random.rand(N_var_anneal)
    for n in range(30 * N_var_anneal):
        for k in range(N_var_anneal):
            x[k] = update(beta * (anneal_A[k][k] + np.dot(np.delete(anneal_A[k], k), np.delete(x, k))))
        beta *= gamma
    x = np.round(x)

    var = np.array([], dtype=np.int32)
    m = 0
    n = 0
    for coef in coefficient_representation:
        var= np.append(var, np.dot(x[n:n + len(coef)], coef).astype(np.int32) + bound[m][0])
        n += len(coef)
        m += 1
    
    return var[:N_variable]


if __name__ == "__main__":
    Solver = SCIOLIC()

    # 実行可能解を最大何回探索するか
    Solver.N_search_feasible = 10
    # 近似度
    Solver.alpha = 0.9
    # ログ出力するかどうか
    Solver.verbose = True

    Solver.add_variable("x", 3, 5)
    Solver.add_variable("y", 1, 4)
    Solver.add_variable("z", -2, 3)

    Solver.add_condition(">", 2)
    Solver.define_coefficient(0, "x", 2)
    Solver.define_coefficient(0, "y", -1)
    Solver.define_coefficient(0, "z", -2)

    Solver.add_convexfunc("x", 3)
    Solver.add_convexfunc("y", 4)
    Solver.add_convexfunc("z", 2)

    print(Solver.search())

    # print(Solver.A)
    # print(Solver.b)
    # print(Solver.bound)


