"""Figures françaises partagées avec le notebook."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap


def shap_figure(result, kind='beeswarm', feature=None):
    values = np.asarray(result['values'])
    data = np.asarray(result['data'])
    explanation = shap.Explanation(values=values, base_values=np.asarray(result['base_values']),
        data=data, feature_names=result['features'])
    plt.close('all')
    if kind == 'waterfall':
        shap.plots.waterfall(explanation[0], max_display=12, show=False)
    elif kind == 'dependence':
        i = result['features'].index(feature)
        shap.plots.scatter(explanation[:, i], show=False)
        plt.xlabel(feature + ' (code pour les catégories)')
        plt.ylabel('Contribution SHAP à la probabilité positive')
    else:
        shap.plots.beeswarm(explanation, max_display=15, show=False, plot_size=(9, 6))
        plt.xlabel('Contribution à la probabilité positive')
    fig = plt.gcf()
    fig.suptitle({'waterfall': 'Explication du client : probabilité positive',
                  'dependence': 'Dépendance SHAP', 'beeswarm': 'Influence globale SHAP'}[kind], fontsize=12)
    fig.tight_layout()
    return fig
