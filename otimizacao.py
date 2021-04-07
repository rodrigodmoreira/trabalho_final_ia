import pandas as pd
import pickle

from base_am.resultado import Fold
from base_am.avaliacao import Experimento

from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from particular_am.metodo_particular import MetodoClassificacao
from particular_am.preprocessamento_atributos_particular import preprocessar_dataframe
from particular_am.avaliacao_particular import OtimizacaoObjetivoMLP

from datasets.helper import load_and_preprocess_dataset

# ============ LEITURA E FORMATACAO DO DATASET
dataframe, _, col_classe = load_and_preprocess_dataset()


# ============ CONSTRUÇÃO DE FOLDS
arr_folds = Fold.gerar_k_folds(dataframe, val_k=5, col_classe=col_classe, num_repeticoes=2, num_folds_validacao=4, num_repeticoes_validacao=2)


# ============ CONSTRUÇÃO DOS MÉTODOS
scikit_mlp_method = MLPClassifier(random_state=2, solver='lbfgs', activation='logistic', hidden_layer_sizes=(10,), learning_rate_init=.1, max_iter=200)
ml_mlp_method = MetodoClassificacao(scikit_mlp_method, col_classe)


# ============ OTIMIZAÇÃO DE PARÂMETROS
experimento_mlp = Experimento([arr_folds[0]], ml_method=ml_mlp_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoMLP,
                    num_trials=20)

macro_f1_mlp = experimento_mlp.macro_f1_avg

print()
print(f'Melhor Macro F1 mlp: {macro_f1_mlp}')
best_params_mlp = experimento_mlp.studies_per_fold[0].best_params
print(f'Melhores parâmetros: {best_params_mlp}')
