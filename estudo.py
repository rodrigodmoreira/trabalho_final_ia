import pandas as pd
import pickle

from base_am.resultado import Fold
from base_am.avaliacao import Experimento

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from particular_am.metodo_particular import MetodoClassificacao
from particular_am.preprocessamento_atributos_particular import preprocessar_dataframe
from particular_am.avaliacao_particular import OtimizacaoObjetivoRandomForest, OtimizacaoObjetivoSVM


# ============ LEITURA E FORMATACAO DO DATASET
# twitter_dataset = pickle.load(open('datasets/SS-Twitter/raw.pickle', 'rb'))
# youtube_dataset = pickle.load(open('datasets/SS-Youtube/raw.pickle', 'rb'))
# olympic_dataset = pickle.load(open('datasets/Olympic/raw.pickle', 'rb'))
kaggle_dataset = pickle.load(open('datasets/kaggle-insults/raw.pickle', 'rb'))
df_kaggle = {
    'text': kaggle_dataset['texts'],
    'label': [entry['label'] for entry in kaggle_dataset['info']]
}
df_kaggle = pd.DataFrame(df_kaggle)
col_classe = 'label'

# preprocessamento do dataset a ser utilizado
df_kaggle = preprocessar_dataframe(df_kaggle)


# ============ CONSTRUÇÃO DE FOLDS
df_treino = df_kaggle.sample(frac=.7, random_state=2)
df_teste = df_kaggle.drop(df_treino.index)
arr_folds = Fold.gerar_k_folds(df_kaggle, val_k=2, col_classe=col_classe, num_repeticoes=1, num_folds_validacao=2, num_repeticoes_validacao=1)


# ============ CONSTRUÇÃO DOS MÉTODOS
scikit_rndforest_method = RandomForestClassifier(random_state=2, class_weight='balanced', n_estimators=64)
ml_rndforest_method = MetodoClassificacao(scikit_rndforest_method, col_classe)

scikit_svm_method = LinearSVC(random_state=2, class_weight='balanced')
ml_svm_method = MetodoClassificacao(scikit_svm_method, col_classe)

scikit_gaussianNB = GaussianNB()
ml_gaussianNB_method = MetodoClassificacao(scikit_gaussianNB, col_classe)


# ============ OTIMIZAÇÃO DE PARÂMETROS
experimento_rndforest = Experimento([arr_folds[0]], ml_method=ml_rndforest_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoRandomForest,
                    num_trials=5)

experimento_svm = Experimento([arr_folds[0]], ml_method=ml_svm_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoSVM,
                    num_trials=5)

experimento_gaussianNB = Experimento([arr_folds[0]], ml_method=ml_gaussianNB_method, num_trials=1)

macro_f1_rndforest = experimento_rndforest.macro_f1_avg
macro_f1_svm = experimento_svm.macro_f1_avg
macro_f1_gaussianNB = experimento_gaussianNB.macro_f1_avg

print()
print(f'Melhor Macro F1 RandomForest: {macro_f1_rndforest}')
print(f'Melhor Macro F1 SVM: {macro_f1_svm}')
print(f'Melhor Macro F1 GaussianNaiveBayes: {macro_f1_gaussianNB}')
