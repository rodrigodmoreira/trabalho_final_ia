import pandas as pd
from base_am.preprocessamento_atributos import BagOfWords, BagOfItems
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def gerar_atributos_texto(df_treino:pd.DataFrame, df_data_to_predict: pd.DataFrame, max_df:float = .9) -> pd.DataFrame:
    bow_amostra = BagOfWordsLyrics(max_df)
    df_bow_treino = bow_amostra.cria_bow(df_treino,"text")
    df_bow_data_to_predict = bow_amostra.aplica_bow(df_data_to_predict,"text")

    return df_bow_treino,df_bow_data_to_predict

lem = WordNetLemmatizer()
def lem_tokenizer(lyrics, to_lower = True, only_alphanum = False):
    tokens = lyrics
    
    # tokenizar apenas se não tokenizado
    if isinstance(lyrics, str):
        tokens = word_tokenize(lyrics)
    
    # processar tokens: lower case > limitar a alfanumericos > lemmatize
    processed_tokens = []
    for i, tk in enumerate(tokens):
        if to_lower:
            tk = tk.lower()

        if only_alphanum:
            tk = re.sub(r'\W+', '', tk)

        if tk == '':
            continue

        processed_tokens.append(lem.lemmatize(tk))
        # processed_tokens.append(tk)
    pos_tags = nltk.pos_tag(tokens)

    return processed_tokens, pos_tags

def preprocessar_dataframe(df: pd.DataFrame, skip_empty = True):
    df_preprocessado = []

    for row in df.values:
        text = row[0]

        # pular instancia caso nao tenha texto
        if not isinstance(text, str):
          if skip_empty:
            continue
          else:
            text = ''

        tokens, _ = lem_tokenizer(text)

        # reconstruir texto
        processed_text = ''
        for tk in tokens:
            processed_text += tk + ' '
        
        # reconstruir dataframe
        df_preprocessado.append([str(processed_text)] + list(row[1:]))
    
    return pd.DataFrame(df_preprocessado, columns=list(df.columns))

stop_list, _ = lem_tokenizer(["i","he","she","it","a","the","almost","do","does"])
# stop_list, _ = lem_tokenizer(stopwords.words('english'))
class BagOfWordsLyrics(BagOfWords):
    def __init__(self, max_df:float = .9):
        #O TfidfVectorizer que é resposavel por gerar a representação BOW
        #você pode mudar a parametrização do mesmo (inclusive, na fase de avaliação)
        #norm: normalização para que todos os valores fiquem entre 0 e 1
        #max_df: remove palavras que ocorrem em mais que 90% dos documentos
        #stop_words: lista das stopwords a serem removidas
        self.vectorizer = TfidfVectorizer(norm="l2",max_df=max_df, stop_words=stop_list)
        pass


