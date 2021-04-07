import pandas as pd
import pickle

from particular_am.preprocessamento_atributos_particular import preprocessar_dataframe

def load_and_preprocess_dataset() -> pd.DataFrame:
    twitter_dataset = pickle.load(open('datasets/SS-Twitter/raw.pickle', 'rb'))
    youtube_dataset = pickle.load(open('datasets/SS-Youtube/raw.pickle', 'rb'))
    kaggle_dataset = pickle.load(open('datasets/kaggle-insults/raw.pickle', 'rb'))


    # extract only positives from twitter
    df_twitter = { 'text': [], 'label': [] }
    for i, text in enumerate(twitter_dataset['texts']):
        if twitter_dataset['info'][i]['label'] == 1:
            df_twitter['text'].append(text)
            df_twitter['label'].append(0)
    df_twitter = pd.DataFrame(df_twitter)

    # extract only positives from youtube
    df_youtube = { 'text': [], 'label': [] }
    for i, text in enumerate(youtube_dataset['texts']):
        if youtube_dataset['info'][i]['label'] == 1:
            df_youtube['text'].append(text)
            df_youtube['label'].append(0)
    df_youtube = pd.DataFrame(df_youtube)

    # extract all neutral comments and insults from kaggle dataset
    df_kaggle = {
        'text': kaggle_dataset['texts'],
        'label': [entry['label'] for entry in kaggle_dataset['info']]
    }
    df_kaggle = pd.DataFrame(df_kaggle)


    col_classe = 'label'

    # preprocessamento do dataset a ser utilizado
    df_final = preprocessar_dataframe(pd.concat([df_twitter, df_youtube, df_kaggle]))
    
    # divisão consistente de 70% em treino e 30% de teste
    df_treino = df_final.sample(frac=.7, random_state=2)
    df_teste = df_final.drop(df_treino.index)
    return df_treino, df_teste, col_classe
        