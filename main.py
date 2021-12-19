import json
import time
import random
import numpy as np
import logging
from typing import List
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from humanfriendly import format_timespan
from pymoo.core.evaluator import Evaluator
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.factory import get_crossover, get_mutation
from utils import mat_file_to_np_array, calculate_decidability_with_numpy, calculate_equal_error_rate

start_time = time.time()
path = '/home/matheus_enacom/PycharmProjects/TCC/feature-selection-pso-mono/mono-matheus/features/NICEIris256.mat'
mat_file = mat_file_to_np_array(path)
testImg = mat_file[2]
testLabel = mat_file[3]
testLabelT = np.transpose(testLabel)
# trainImg = mat_file[0]
# trainLabel = mat_file[1]
# trainLabelT = np.transpose(trainLabel)
algorithm = None
algorithm_choice = 'NSGA'
result = None
pop_size = 1
n_gen = 1
teste = "test_all_features_2"

logging.warning(' ############ Inicializando otimizaćão - Matheus Lemes #####################')


def __generate_list_with_random_int(qtde: int) -> List[int]:
    random_list = []
    for _ in range(qtde):
        n = random.randint(0, 255)
        random_list.append(n)
    return random_list


def create_first_population(pop_size: int):
    x_matrix = np.full((pop_size, 256), True, dtype=bool)
    idx = 1
    for i in range(1, pop_size):
        random_index = __generate_list_with_random_int(qtde=idx)
        x_matrix[i, :][random_index] = False
        idx += 1
        if i % 255 == 0:
            idx = 1
    return x_matrix


def minimize_features(x):
    sizes = list()
    for X in x:
        y = np.sum(X)
        sizes.append(y)
    sizes = np.asarray(sizes, dtype=int)
    sizes = sizes.reshape(-1, 1)
    return sizes


def maximize_decidability(x):
    X = 1 * x
    out_y = list()
    for x in X:
        y = calculate_decidability_with_numpy(testImg * x, testLabelT)
        out_y.append(y)
    out_y = np.asarray(out_y, dtype=np.float64)
    out_y = out_y.reshape(-1, 1)
    return out_y


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=256,
                         n_obj=2,
                         n_constr=0,
                         xl=0,
                         xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = minimize_features(x)
        f2 = maximize_decidability(x)

        out['F'] = np.column_stack([f1, f2])


my_problem = MyProblem()
logging.warning('==> Gerando primeira populacão <==')
x = create_first_population(pop_size)
pop = Population.new('X', x)
Evaluator().eval(my_problem, pop)

if algorithm_choice == 'NSGA':
    algorithm = NSGA2(
        pop_size=pop_size,
        n_offsprings=pop_size,
        sampling=pop,
        crossover=get_crossover("bin_exp"),
        mutation=get_mutation("real_pm"),
        eliminate_duplicates=True)

elif algorithm_choice == 'RNSGA':
    ref_points = np.array([[100.0, -2.0]])
    algorithm = RNSGA2(
        ref_points=ref_points,
        pop_size=pop_size,
        sampling=pop,
        epsilon=0.01,
        normalization='front',
        extreme_points_as_reference_points=False,
        weights=np.array([0.5, 0.5]))

logging.warning('==> Iniciando Otimizacão <== ')
if algorithm is not None:
    result = minimize(my_problem,
                      algorithm,
                      ('n_gen', n_gen),
                      verbose=True,
                      save_history=True)

end_otm = time.time() - start_time
t_otm = format_timespan(end_otm)

_, F = result.opt.get('X', 'F')

hist = result.history
logging.warning('==> Calculando Hiper Volume <== ')
# Hiper Volume
n_evals = []
hist_F = []
hist_cv = []
hist_cv_avg = []

for algo in hist:
    n_evals.append(algo.evaluator.n_eval)
    opt = algo.opt
    hist_cv.append(opt.get("CV").min())
    hist_cv_avg.append(algo.pop.get("CV").mean())
    feas = np.where(opt.get('feasible'))[0]
    hist_F.append(opt.get('F')[feas])

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)

metric = Hypervolume(ref_point=np.array([256, 10.0]),
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)
hv = [metric.do(_F) for _F in hist_F]

logging.warning('==> Calculando Equal Error Rate <== ')
# Equal Error Rate
eer = []
if result is not None:
    for xis in result.X.astype(int):
        error = calculate_equal_error_rate(testImg * xis, testLabel)
        eer.append(error)

a1 = result.X.astype(int)
a2 = result.F
a3 = 1 * x
f_pop = a3.tolist()
x = a1.tolist()
f = a2.tolist()

end_time = time.time() - start_time
t = format_timespan(end_time)
logging.warning('==> Finalizando Execucão <== ')
print("Total execution time: ", format_timespan(end_time))
resultado = {'test': teste, 'Algorithm': algorithm_choice, 'Number_of_Generations': n_gen, 'Population_Size': pop_size,
             'Optimization_time': t_otm, 'Execution_time': t, 'First_population': f_pop, 'Features': x,
             'Objective_Function': f, 'Equal Error Rate': eer, 'HyperVolum': hv}

with open(f'teste_{teste}_{algorithm_choice}.json', 'w') as fp:
    json.dump(resultado, fp)

logging.warning('==> Salvando Histórico <== ')
historico = {}
for index, line in enumerate(hist_F):
    historico[f'generation {index+1}'] = line.tolist()

with open(f'historico_{teste}_{algorithm_choice}.json', 'w') as fp:
    json.dump(historico, fp)

print("Function value: %s" % result.F)
