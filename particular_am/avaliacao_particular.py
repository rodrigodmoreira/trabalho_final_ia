from base_am.avaliacao import OtimizacaoObjetivo
from base_am.metodo import MetodoAprendizadoDeMaquina
from base_am.resultado import Fold, Resultado
from particular_am.metodo_particular import MetodoClassificacao
import optuna
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

class OtimizacaoObjetivoRandomForest(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, col_classe:str):
        super().__init__(fold, col_classe)

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        # https://www.researchgate.net/publication/230766603_How_Many_Trees_in_a_Random_Forest
        n_estimators = trial.suggest_int('n_estimators', 64, 128)

        scikit_method = RandomForestClassifier(random_state=2, class_weight='balanced', n_estimators=n_estimators)

        return MetodoClassificacao(scikit_method, self.col_classe)

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1

class OtimizacaoObjetivoSVM(OtimizacaoObjetivo):
    def __init__(self, fold:Fold, col_classe:str):
        super().__init__(fold, col_classe)

    def obtem_metodo(self,trial: optuna.Trial)->MetodoAprendizadoDeMaquina:
        #Um custo adequado para custo pode variar muito, por ex, para uma tarefa 
        #o valor de custo pode ser 10, para outra, 32000. 
        #Assim, normalmente, para conseguir valores mais distintos,
        #usamos c=2^exp_cost
        exp_cost = trial.suggest_uniform('exp_cost', 0, 3) 

        scikit_method = LinearSVC(C=2**exp_cost, random_state=2, class_weight='balanced')

        return MetodoClassificacao(scikit_method, self.col_classe)

    def resultado_metrica_otimizacao(self,resultado: Resultado) -> float:
        return resultado.macro_f1
