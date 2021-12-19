import scipy.io as sio
import numpy as np
from sklearn.metrics import euclidean_distances
import operator
import plotly.graph_objects as px
import pandas as pd


def mat_file_to_np_array(path: str):
    mat_file = sio.loadmat(path)
    key_mat_file = sorted(mat_file.keys())
    mat_array = mat_file[key_mat_file[-1]][0][0]

    return mat_array


def calculate_decidability_with_numpy(_x_embeddings, _y):
    # Calculate the genuine and impostor distributions
    _x_embeddings = _x_embeddings[:5000, :]
    _y = _y[:5000]
    genuine, impostor = [], []
    distances = euclidean_distances(_x_embeddings, _x_embeddings)
    for i, yi in enumerate(_y):
        for j, yj in enumerate(_y):
            if yi == yj:
                genuine.append(distances[i, j])
            else:
                impostor.append(distances[i, j])

    return np.abs(np.mean(genuine) - np.mean(impostor)) / np.sqrt(0.5 * (np.std(genuine) ** 2 + np.std(impostor) ** 2))


def calculate_equal_error_rate(feature, label):
    label = label.reshape(-1, 1)
    _x_embeddings = feature[:10000, :]
    _y = label[:10000]
    genuine, impostor = [], []
    distances = euclidean_distances(_x_embeddings, _x_embeddings)
    for i, yi in enumerate(_y):
        for j, yj in enumerate(_y):
            if yi == yj:
                genuine.append(distances[i, j])
            else:
                impostor.append(distances[i, j])

    dmin = np.amin(genuine)
    dmax = np.amax(impostor)

    # Calculate False Match Rate and False NonMatch Rate for different thresholds
    FMR = np.zeros(5000)
    FNMR = np.zeros(5000)
    t = np.linspace(dmin, dmax, 5000)

    for t_val in range(5000):
        fm = np.sum(impostor <= t[t_val])
        FMR[t_val] = fm / len(impostor)

    for t_val in range(5000):
        fnm = np.sum(genuine > t[t_val])
        FNMR[t_val] = fnm / len(genuine)

    abs_diffs = np.abs(FMR - FNMR)
    min_index = np.argmin(abs_diffs)
    eer = (FMR[min_index] + FNMR[min_index]) / 2

    return eer


def plot_pareto_front(hist_data):
    fig = px.Figure()
    fig.update_layout(
        title={'text': 'Fronteira Pareto',
               'y': 0.95,
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top'
               },
        xaxis_title='Quantidade de features',
        yaxis_title='Decidibilidade',
        legend_title='Gerações',
        font=dict(
            size=18
        )
    )
    for gen in list(hist_data.keys()):
        number_of_gen = int(gen.split()[-1])
        if number_of_gen % 10 == 0 or number_of_gen == 1:
            pareto_front = sorted(hist_data[gen], key=operator.itemgetter(0))
            df = pd.DataFrame(pareto_front, columns=['x', 'y'])
            df['y'] = df['y'] * -1
            trace = px.Scatter(
                x=df['x'],
                y=df['y'],
                name=gen,
                showlegend=True,
                mode='lines+markers',
                marker_size=10
            )
            fig.add_trace(trace)

    fig.show()


def plot_hypervolum(hv_data):
    fig = px.Figure()
    fig.update_layout(
        yaxis_title='Hipervolume',
        xaxis_title='Gerações',
        font=dict(
            size=18
        )
    )

    trace = px.Scatter(
        y=hv_data,
        mode='lines+markers',
        marker_size=10,
        textposition="top center"
    )
    fig.add_trace(trace)
    fig.add_annotation(
        x=len(hv_data),
        y=hv_data[-1],
        showarrow=False,
        text=str(round(hv_data[-1], 2)),
        yshift=10
    )
    fig.update_layout(showlegend=False)
    fig.show()
