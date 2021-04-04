import pandas as pd
import pickle

from base_am.resultado import Fold
from base_am.avaliacao import Experimento

from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from particular_am.metodo_particular import MetodoClassificacao
from particular_am.preprocessamento_atributos_particular import preprocessar_dataframe
from particular_am.avaliacao_particular import OtimizacaoObjetivoSVM


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


# ============ AVALIAÇÃO DO TESTE
best_exp_cost = 1.251066014107722
scikit_svm_method = LinearSVC(random_state=2, C=2**best_exp_cost)
ml_svm_method = MetodoClassificacao(scikit_svm_method, col_classe)
result = ml_svm_method.eval(df_treino=df_treino, df_data_to_predict=df_teste, col_classe=col_classe)

print('\n=============== TESTE ================')
print(f'\nMacro F1: {result.macro_f1}')
print('\nMatriz de confusão:')
print(result.mat_confusao)
print('\nRelatório de classificação:')
print(classification_report(result.y, result.predict_y))
