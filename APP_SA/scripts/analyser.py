dicneg = ["faillite","crise","chômage","récession","déflation","dette","baisse","pertes","déclin","instabilité","risque","inflation","endette","effondre","insolvab","inquiet","incert","rupture","pénurie","retrait","obstacle","déficit","ralenti","trouble","diffic","crack","échec","suspicion","dérive","régression","fiasco","dislocation","repli","chute","crainte","dysfonctionnement","contrecoup","panne","lamentable","emprunt","rouge",'guerre','recul']
dicpos = ["croissance","augment", "force","vert","efficac", "apport", "asc", 'succes','hausse','remporte',"prospér","innov","créativ","opportun","investi","dévelop","évolu","avanc","succès","rentab","product","progres","expansion","réussi","vitalité","stab","riche","progr","optim","abondan","amélior","prospère","opulence","florissant","essor","engagement","croître","positif","gagne","stable","fort","promet","enrichi","dynamique","satisf","favorable","fluide","réussi","fiable","profit","favoris","perform","triomph","viable","constructif","fluent","opérationnel","bénéf","lucratif","durable","fructueux","prolifique","épanoui","fécond","croissant","compétitif","florissante","résilient","effervescent","vivant","vibrant","vertueux","précieux","proactif","radieux","opérant","favorabl","floraison","rentab","avancé","soutenu","flamboyant","puissan","construit","substantiel","stimulant","accompli","excellent","abondant","avant-gardiste","prestigieux","amélior","durabl","évoluant","énergique","flamboyant","fructif","prospective","puissance","évolu","saisissant","confortable","inspirant","enthousias","radieux","attra","harmoni","rayonn","glorieux","éclatant","inspirateur","efficacement","lumi","inspirant","gagn","vigoureux"]




from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options as FirefoxOptions
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
def convert_to_lowercase(txt):
    """
    Convertit un texte en minuscules.
    
    Args:
    - txt (str): Le texte à convertir en minuscules.
    
    Returns:
    - str: Le texte converti en minuscules.
    """
    return txt.lower()



def remove_stop_words(txt):
    """
    Supprime les mots vides (stop words) d'un texte en français.
    
    Args:
    - txt (str): Le texte à nettoyer.
    
    Returns:
    - str: Le texte sans les mots vides.
    """
    # Chargement des mots vides en français
    stop_words = set(stopwords.words('french'))
    
    # Tokenisation du texte en mots
    words = txt.split()
    
    # Filtrage des mots vides
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # Reconstruction du texte sans les mots vides
    clean_text = ' '.join(filtered_words)
    
    return clean_text


def lemmatization(texte):
    """
    Effectue la lemmatisation des mots en français.
    
    Args:
    - texte (str): Le texte à lemmatiser.
    
    Returns:
    - str: Le texte avec uniquement les radicaux des mots.
    """
    # Initialisation du lemmatiseur
    lemmatizer = WordNetLemmatizer()
    
    # Tokenisation du texte en mots
    mots = word_tokenize(texte, language='french')

    # Lemmatisation de chaque mot et reconstruction du texte
    texte_lemmatise = ' '.join([lemmatizer.lemmatize(mot) for mot in mots])

    return texte_lemmatise

def preprocess(txt):
    LC = convert_to_lowercase(txt)
    SW = remove_stop_words(LC)
    LEM = lemmatization(SW)
    return LEM

def extract_sentiment(text):
    # Initialiser l'analyseur de sentiments VADER
    analyzer = SentimentIntensityAnalyzer()
    
    # Analyser le texte pour obtenir les scores de sentiment
    sentiment_scores = analyzer.polarity_scores(text)
    
    # Extraire les scores de sentiment
    positive_score = sentiment_scores['pos']
    negative_score = sentiment_scores['neg']
    neutral_score = sentiment_scores['neu']
    
    return positive_score, negative_score, neutral_score

def spliter(texte):
    mots = texte.split()
    return mots


def compteur(liste):
    c=0
    for i in liste :
        c+=1
    return c

def ana_mot(rac, mot):
    if rac in mot :
        return 1
    return 0


def S_A(texte):
    liste = spliter(texte)
    c = compteur(liste)/4
    cp = 0
    cn = 0
    for mot in liste:
        for motneg in dicneg :
            cn += ana_mot(motneg, mot)
        for motpos in dicpos :
            cp += ana_mot(motpos, mot)
    tp = cp/c
    tn = cn/c
    tN = 1-(tp+tn)
    return [tp, tn, tN]

def sentiment_analyser(titles):
    pos = 0
    neut = 0
    neg = 0
    c = 0

    for txt in titles:
        # Obtenir les scores de sentiment pour chaque titre
        texte = preprocess(txt)
        positive_score, negative_score, neutral_score = S_A(texte)

        # Ajouter les scores au total
        pos += positive_score
        neg += negative_score
        neut += neutral_score
        c += 1
    # Calculer les moyennes des scores de sentiment
    if c > 0:
        pos = pos * 100 / c
        neg = neg * 100 / c
        neut = neut * 100 / c
    return [pos, neg, neut]
    
def get_news_titles(search_query):
    # Chemin du driver de votre navigateur (Mozilla Firefox dans cet exemple)
    driver_path = r'C:\Users\chouf\Desktop\drivers\geckodriver.exe'  # Assurez-vous de télécharger geckodriver et de spécifier le bon chemin

    # Créer des options pour le navigateur Firefox
    firefox_options = FirefoxOptions()

    driver = webdriver.Firefox(options=firefox_options)

    # Ouvrir Google News
    driver.get("https://news.google.com/")
    time.sleep(2)  # Attendre un peu pour que la page se charge

    # Rechercher le sujet spécifié
    search_box = driver.find_element("css selector", 'input[type="text"]')
    search_box.send_keys(search_query)
    search_box.send_keys(Keys.RETURN)
    time.sleep(20)  # Attendre un peu pour que les résultats de recherche se chargent
    news_titles = []
    # Récupérer les titres des articles
    for i in range(10):
        news_titles += driver.find_elements("xpath", f'//*[@id="yDmH0d"]/c-wiz[2]/div/main/div[2]/c-wiz/c-wiz[{i}]/c-wiz/article/div[1]/div[2]/div/a')
    titles = [title.text for title in news_titles]

    # Fermer le navigateur
    driver.quit()

    return titles

def analyser(stock):
    titles = get_news_titles(stock)
    return [sentiment_analyser(titles),titles]

"""
def main():
    subj = input("Donnez le nom de l'action ou monnaie à évaluer: ")
    deba = input("Donnez la date de début de la fenêtre d'analyse (au format YYYY-MM-DD): ")
    

    # Appeler la fonction pour récupérer les titres des articles
    titles = get_news_titles(subj)
    
    # Afficher les titres des articles
    print("Titres des articles sur", subj, ":", titles)
    sentiment_analyser(titles)

if __name__ == "__main__":
    main()
"""
#print(analyser('attijari wafabank'))