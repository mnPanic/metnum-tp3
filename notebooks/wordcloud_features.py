import unidecode
import pandas as pd
import matplotlib.pyplot as plt

from typing import List
from wordcloud import WordCloud, STOPWORDS

def add_frequency_features(df: pd.DataFrame) -> (WordCloud, WordCloud):
    """
    Agrega las features avg_freq_title y avg_freq_desc.
    Devuelve los dos wordclouds que fueron generados para poder plotearlos
    con plot_wc().
    """
    titles_freq, titles_wc = get_wc(df, 'titulo')
    desc_freq, desc_wc = get_wc(df, 'descripcion')

    # las normalizamos con min-max de 0 a 1
    df['avg_freq_title'] = avg_freq_rows(df['titulo'], titles_wc, titles_freq)
    min_max_norm(df, 'avg_freq_title')

    df['avg_freq_desc'] = avg_freq_rows(df['descripcion'], desc_wc, desc_freq)
    min_max_norm(df, 'avg_freq_desc')

    return desc_wc, titles_wc

def remove_accents(accented_string):
    # https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    return unidecode.unidecode(accented_string)

def get_wc(df: pd.DataFrame, field: str) -> (dict, WordCloud):
    """Devuelve un dict con las frecuencias y el word cloud"""
    words = ""
    for row in df[field].values:
        words += remove_accents(str(row)) + "\n"
    
    # uso stopwords custom a parte de las de wordcloud
    # sacamos las preposiciones porque no aportan mucho.
    stopwords = set([
        # preposiciones
        "el", "para", "en", "de", "la", "del",
        "nan", "los", "las", "se", "con", "al",
        "es", "lo",
        # html escapado
        "nbsp", "li", "br", "aacute", "hr", "col",
    ]).union(STOPWORDS)
    
    wc = WordCloud(
        width = 800, height = 800, 
        stopwords = stopwords,
        background_color ='white', 
        collocations=False, # evita collocations (ex. en casa, en venta)
    )
    
    
    freq = wc.process_text(words)
    # por si queremos printear las frecuencias en orden
    #print({k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)})

    wc.fit_words(freq)

    return freq, wc

def plot_wc(wc: WordCloud):
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wc)
    plt.axis("off")

def min_max_norm(df, col):
    # https://en.wikipedia.org/wiki/Feature_scaling
    max_ = df[col].max()
    min_ = df[col].min()
    df[col]=(df[col]-min_)/(max_-min_)

def avg_freq(row, wc, frequencies) -> float:
    sum_freqs = 0      # suma de las frecuencias

    # Proceso el texto para obtener las palabras
    # omitiendo los stopwords y eso que componen
    # al row del dataframe.
    # Nos quedamos con solo las palabras (las keys)
    words = wc.process_text(row).keys()
    
    # Vemos la frecuencia de cada una buscandola en
    # el dict de frecuencias.
    considered = 0
    for word in words:
        if word not in frequencies:
            # ni la consideramos
            continue

        sum_freqs += frequencies[word]
        considered += 1
    
    if considered == 0:
        return 0
    
    # retornamos el promedio
    return sum_freqs / considered

def avg_freq_rows(rows, wc, frequencies) -> List[float]:
    # para poder usarlo para generar una nueva columna
    res = []
    for row in rows:
        res.append(avg_freq(str(row), wc, frequencies))

    return res