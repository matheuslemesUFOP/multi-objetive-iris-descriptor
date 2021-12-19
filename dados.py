import json
from utils import plot_pareto_front, plot_hypervolum

f = open('historico_test_oficial_3_NSGA.json')
hist_data_nsga = json.load(f)
plot_pareto_front(hist_data=hist_data_nsga)

hv = open('teste_test_oficial_3_NSGA.json')
hv_data = json.load(hv)
hv_hist_data = hv_data['HyperVolum']
plot_hypervolum(hv_data=hv_hist_data)

f2 = open('historico_test_oficial_3_RNSGA.json')
hist_data_nsga2 = json.load(f2)
plot_pareto_front(hist_data=hist_data_nsga2)

hv2 = open('teste_test_oficial_3_RNSGA.json')
hv_data2 = json.load(hv2)
hv_hist_data2 = hv_data2['HyperVolum']
plot_hypervolum(hv_data=hv_hist_data2)

