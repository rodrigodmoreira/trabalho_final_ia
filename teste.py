import pandas as pd
import pickle

from base_am.resultado import Fold
from base_am.avaliacao import Experimento

from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from particular_am.metodo_particular import MetodoClassificacao
from particular_am.preprocessamento_atributos_particular import preprocessar_dataframe
from particular_am.avaliacao_particular import OtimizacaoObjetivoSVM

from datasets.helper import load_and_preprocess_dataset

# ============ LEITURA E FORMATACAO DO DATASET
df_treino, df_teste, col_classe = load_and_preprocess_dataset()


# ============ AVALIAÇÃO DO TESTE
best_n_first_layer = 10
scikit_mlp_method = MLPClassifier(random_state=2, solver='lbfgs', activation='logistic', hidden_layer_sizes=(best_n_first_layer,), learning_rate_init=.1, max_iter=200)
ml_mlp_method = MetodoClassificacao(scikit_mlp_method, col_classe)
result = ml_mlp_method.eval(df_treino=df_treino, df_data_to_predict=df_teste, col_classe=col_classe)

print('\n=============== TESTE ================')
print(f'\nMacro F1: {result.macro_f1}')
print('\nMatriz de confusão:')
print(result.mat_confusao)
print('\nRelatório de classificação:')
print(classification_report(result.y, result.predict_y))
