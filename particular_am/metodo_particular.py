from base_am.metodo import MetodoAprendizadoDeMaquina
import pandas as pd
from particular_am.preprocessamento_atributos_particular import gerar_atributos_texto
from base_am.resultado import Resultado
from typing import Union, List
from sklearn.base import ClassifierMixin, RegressorMixin

class MetodoClassificacao(MetodoAprendizadoDeMaquina):
    def __init__(self, ml_method:Union[ClassifierMixin,RegressorMixin], col_classe=''):
        self.ml_method = ml_method
        self.col_classe = col_classe

    def obtem_x(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str):
        x_treino = df_treino.drop(col_classe, axis = 1)
        x_to_predict = df_data_to_predict.drop(col_classe, axis = 1)
        return x_treino, x_to_predict

    def obtem_y(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str):
        y_treino = df_treino[col_classe]
        y_to_predict = df_data_to_predict[col_classe]
        return y_treino,y_to_predict

    def eval(self, df_treino:pd.DataFrame, df_data_to_predict:pd.DataFrame, col_classe:str, seed:int=1):
        #separação da classe 
        x_treino, x_to_predict = self.obtem_x(df_treino, df_data_to_predict, col_classe)
        y_treino, y_to_predict = self.obtem_y(df_treino, df_data_to_predict, col_classe)
        # print(f"x_treino: {y_treino} y_to_predict:{y_to_predict}")
        
        #geração dos atributos
        x_treino_bow, x_to_predict_bow = gerar_atributos_texto(x_treino, x_to_predict)
        
        #geração do modelo e prediçaõ do primeiro nivel
        self.ml_method.fit(x_treino_bow, y_treino)
        arr_predict_prim_nivel = self.ml_method.predict(x_to_predict_bow)
        
        return Resultado(y_to_predict, arr_predict_prim_nivel)
