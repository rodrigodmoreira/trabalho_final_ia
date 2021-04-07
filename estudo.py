import pandas as pd
import pickle

from base_am.resultado import Fold
from base_am.avaliacao import Experimento

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from particular_am.metodo_particular import MetodoClassificacao
from particular_am.avaliacao_particular import OtimizacaoObjetivoRandomForest, OtimizacaoObjetivoSVM, OtimizacaoObjetivoMLP

from datasets.helper import load_and_preprocess_dataset

# ============ LEITURA E FORMATACAO DO DATASET
dataframe, _, col_classe = load_and_preprocess_dataset()


# ============ CONSTRUÇÃO DE FOLDS
arr_folds = Fold.gerar_k_folds(dataframe, val_k=2, col_classe=col_classe, num_repeticoes=1, num_folds_validacao=2, num_repeticoes_validacao=1)


# ============ CONSTRUÇÃO DOS MÉTODOS
scikit_rndforest_method = RandomForestClassifier(random_state=2, class_weight='balanced', n_estimators=64)
ml_rndforest_method = MetodoClassificacao(scikit_rndforest_method, col_classe)

scikit_svm_method = LinearSVC(random_state=2, class_weight='balanced')
ml_svm_method = MetodoClassificacao(scikit_svm_method, col_classe)

scikit_gaussianNB = GaussianNB()
ml_gaussianNB_method = MetodoClassificacao(scikit_gaussianNB, col_classe)

scikit_mlp = MLPClassifier(random_state=2, solver='lbfgs', activation='logistic', hidden_layer_sizes=(10,), learning_rate_init=.1, max_iter=200)
ml_mlp_method = MetodoClassificacao(scikit_mlp, col_classe)


# ============ OTIMIZAÇÃO DE PARÂMETROS
experimento_rndforest = Experimento([arr_folds[0]], ml_method=ml_rndforest_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoRandomForest,
                    num_trials=5)

experimento_svm = Experimento([arr_folds[0]], ml_method=ml_svm_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoSVM,
                    num_trials=5)

experimento_gaussianNB = Experimento([arr_folds[0]], ml_method=ml_gaussianNB_method, num_trials=1)

experimento_mlp = Experimento([arr_folds[0]], ml_method=ml_mlp_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoMLP,
                    num_trials=5)


macro_f1_rndforest = experimento_rndforest.macro_f1_avg
macro_f1_svm = experimento_svm.macro_f1_avg
macro_f1_gaussianNB = experimento_gaussianNB.macro_f1_avg
macro_f1_mlp = experimento_mlp.macro_f1_avg

print()
print(f'Melhor Macro F1 RandomForest: {macro_f1_rndforest}')
print(f'Melhor Macro F1 SVM: {macro_f1_svm}')
print(f'Melhor Macro F1 GaussianNaiveBayes: {macro_f1_gaussianNB}')
print(f'Melhor Macro F1 Multi-Layer Perceptron: {macro_f1_mlp}')
