import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Fonction pour calculer la moyenne mobile simple (MMS)
def moyenne_mobile_simple(data, window_size):
    """
    Calcule la moyenne mobile simple (MMS) des données avec une fenêtre spécifiée.
    
    Args:
    - data: Liste des prix de l'action
    - window_size: Taille de la fenêtre pour la MMS
    
    Returns:
    - Liste des valeurs de la MMS
    """
    mms_values = []
    for i in range(len(data) - window_size + 1):
        window = data[i : i + window_size]
        mms = sum(window) / window_size
        mms_values.append(mms)
    return mms_values

def MMS(symbole_action,start_date,end_date):
    data = yf.download(symbole_action, start=start_date, end=end_date)
    prix_action = data['Close'].values

# Paramètres de la MMS
    fenetre_mms = int(input('donnez la taille de la fenetre (en jours):'))

# Calcul de la MMS
    mms = moyenne_mobile_simple(prix_action, fenetre_mms)

# Affichage des résultats
    print(f"Prix de l'action {symbole_action} :", prix_action)
    print(f"Moyenne mobile simple (fenêtre = {fenetre_mms}) :", mms)

# Tracé des prix de l'action et de la MMS
    plt.plot(prix_action, label='Prix de l\'action', color='blue')
    plt.plot(range(fenetre_mms - 1, len(prix_action)), mms, label=f'MMS ({fenetre_mms} jours)', color='red')
    plt.xlabel('Jours')
    plt.ylabel('Prix')
    plt.title(f'Moyenne Mobile Simple (MMS) pour {symbole_action}')
    plt.legend()
    plt.grid(True)
    plt.show()



# Fonction pour récupérer les données historiques à partir de Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Fonction pour préparer les données pour la régression logistique
def prepare_data(stock_data):
    stock_data['Price_Up'] = np.where(stock_data['Close'].shift(-1) > stock_data['Close'], 1, 0)
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values[:-1]
    y = stock_data['Price_Up'].values[:-1]
    return X, y

# Fonction principale
def RegLog(symbol,start_date,end_date):

    # Récupération des données historiques
    stock_data = get_stock_data(symbol, start_date, end_date)

    # Préparation des données pour la régression logistique
    X, y = prepare_data(stock_data)

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Construction et entraînement du modèle de régression logistique
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Prédictions
    predictions = model.predict(X_test)

    # Évaluation du modèle
    accuracy = metrics.accuracy_score(y_test, predictions)
    print("Précision du modèle : {:.2f}%".format(accuracy * 100))

    # Affichage du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Prédictions')
    plt.plot(y_test, label='Valeurs réelles')
    plt.title("Prédictions vs Valeurs Réelles")
    plt.xlabel("Jours")
    plt.ylabel("Prix (Up:1, Down:0)")
    plt.legend()
    plt.show()





# Fonction pour calculer la moyenne mobile exponentielle (MME)
def moyenne_mobile_exponentielle(data, window_size):
    """
    Calcule la moyenne mobile exponentielle (MME) des données avec une fenêtre spécifiée.
    
    Args:
    - data: Liste des prix de l'action
    - window_size: Taille de la fenêtre pour la MME
    
    Returns:
    - Liste des valeurs de la MME
    """
    alpha = 2 / (window_size + 1)
    mme_values = [data[0]]
    for i in range(1, len(data)):
        mme = alpha * data[i] + (1 - alpha) * mme_values[-1]
        mme_values.append(mme)
    return mme_values

def MME(symbole_action,start_date,end_date):
# Téléchargement des données d'action depuis Yahoo Finance
    data = yf.download(symbole_action, start=start_date, end=end_date)

# Sélection des prix de clôture de l'action
    prix_action = data['Close'].values
    fenetre_mme = int(input('donnez la taille de la fenetre (en jours):'))

# Calcul de la MME
    mme = moyenne_mobile_exponentielle(prix_action, fenetre_mme)

# Affichage des résultats
    print(f"Prix de l'action {symbole_action} :", prix_action)
    print(f"Moyenne mobile exponentielle (fenêtre = {fenetre_mme}) :", mme)

# Tracé des prix de l'action et de la MME
    plt.plot(prix_action, label='Prix de l\'action', color='blue')
    plt.plot(mme, label=f'MME ({fenetre_mme} jours)', color='red')
    plt.xlabel('Jours')
    plt.ylabel('Prix')
    plt.title(f'Moyenne Mobile Exponentielle (MME) pour {symbole_action}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Fonction pour récupérer les données depuis Yahoo Finance
def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Fonction pour calculer la moyenne mobile exponentielle (EMA)
def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

# Fonction pour prédire les prix futurs en utilisant l'EMA
def predict_prices(data, ema_short, ema_long):
    signals = pd.DataFrame(index=data.index)
    signals['Price'] = data['Close']
    signals['EMA_Short'] = ema_short
    signals['EMA_Long'] = ema_long
    signals['Signal'] = 0.0
    signals['Signal'][ema_short.index[0]:] = \
        np.where(signals['EMA_Short'][ema_short.index[0]:] 
                 > signals['EMA_Long'][ema_short.index[0]:], 1.0, 0.0)
    signals['Position'] = signals['Signal'].diff()
    return signals
def EMA(symbol,start_date,end_date):

# Récupération des données
    stock_data = get_stock_data(symbol, start_date, end_date)

# Calcul de l'EMA court terme et long terme
    ema_short = calculate_ema(stock_data, 12)
    ema_long = calculate_ema(stock_data, 26)

# Prédiction des prix futurs
    predicted_prices = predict_prices(stock_data, ema_short, ema_long)

# Affichage des graphiques
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'], label='Prix de clôture')
    plt.plot(ema_short, label='EMA Court Terme', color='orange')
    plt.plot(ema_long, label='EMA Long Terme', color='purple')
    plt.title('Prévision des Prix avec Moyenne Mobile Exponentielle (EMA)')
    plt.xlabel('Date')
    plt.ylabel('Prix de clôture')
    plt.legend()
    plt.show()




# Fonction pour préparer les données pour la régression linéaire
def prepare_data(stock_data):
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Date'] = stock_data['Date'].map(pd.Timestamp.to_julian_date)
    X = stock_data[['Date']].values
    y = stock_data['Close'].values
    return X, y

# Fonction pour entraîner le modèle de régression linéaire
def train_linear_regression_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Fonction pour afficher les prédictions
def plot_predictions_Lreg(X_test, y_test, model):
    plt.scatter(X_test, y_test, color='gray')
    plt.plot(X_test, model.predict(X_test), color='red', linewidth=2)
    plt.xlabel('Date (Julian)')
    plt.ylabel('Closing Price')
    plt.title('Linear Regression Predictions')
    plt.show()

# Fonction principale
def TSFLReg(symbol,start_date,end_date):
    # Récupération des données historiques
    stock_data = get_stock_data(symbol, start_date, end_date)

    # Préparation des données pour la régression linéaire
    X, y = prepare_data(stock_data)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle de régression linéaire
    model = train_linear_regression_model(X_train, y_train)

    # Affichage des prédictions
    plot_predictions_Lreg(X_test, y_test, model)




# Fonction pour préparer les données pour le modèle de réseau de neurones
def prepare_data(stock_data, look_back=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))
    X, y = [], []
    for i in range(len(stock_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    return np.array(X), np.array(y)

# Fonction pour créer et entraîner le modèle de réseau de neurones
def build_lstm_model(X_train, y_train, look_back):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    return model

# Fonction pour afficher les prédictions
def plot_predictions_lstm(predictions, y_test):
    plt.plot(predictions, color='red', label='Predicted Prices')
    plt.plot(y_test, color='blue', label='Actual Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Fonction principale
def LSTMM(symbol, start_date, end_date):
    
    # Récupération des données historiques
    stock_data = get_stock_data(symbol, start_date, end_date)

    # Préparation des données pour le modèle LSTM
    look_back = 60
    X, y = prepare_data(stock_data, look_back)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Construction et entraînement du modèle LSTM
    model = build_lstm_model(X_train, y_train, look_back)

    # Prédictions
    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1)

    # Affichage des prédictions
    plot_predictions_lstm(predictions, y_test)


def main():
    stock = input("saisir le code de l'action ou monnaie a evaluer: ")
    start_date = input("saisir la date de debut YYYY-MM-DD: ")
    end_date = input('saisir la date de fin YYYY-MM-DD: ')
    meth = input("saisir la methode choisie :")
    if meth == 'LSTM':
        LSTMM(stock,start_date,end_date)
    elif meth == 'TSFLReg':
        TSFLReg(stock,start_date,end_date)
    elif meth == 'EMA':
        EMA(stock, start_date, end_date)
    elif meth == 'MME':
        MME(stock,start_date,end_date)
    elif meth == 'MMS':
        MMS(stock,start_date, end_date)
    elif meth == 'RegLog':
        RegLog(stock,start_date,end_date)


if __name__ == "__main__":
    main()
