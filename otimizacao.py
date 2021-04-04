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
arr_folds = Fold.gerar_k_folds(df_kaggle, val_k=5, col_classe=col_classe, num_repeticoes=2, num_folds_validacao=4, num_repeticoes_validacao=2)


# ============ CONSTRUÇÃO DOS MÉTODOS
scikit_svm_method = LinearSVC(random_state=2, class_weight='balanced')
ml_svm_method = MetodoClassificacao(scikit_svm_method, col_classe)


# ============ OTIMIZAÇÃO DE PARÂMETROS
experimento_svm = Experimento([arr_folds[0]], ml_method=ml_svm_method,
                    ClasseObjetivoOtimizacao=OtimizacaoObjetivoSVM,
                    num_trials=50)

macro_f1_svm = experimento_svm.macro_f1_avg

print()
print(f'Melhor Macro F1 SVM: {macro_f1_svm}')
best_params_svm = experimento_svm.studies_per_fold[0].best_params
print(f'Melhores parâmetros: {best_params_svm}')
