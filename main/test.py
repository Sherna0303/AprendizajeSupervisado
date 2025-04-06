import joblib
import pandas as pd

modelo_tiempo = joblib.load("main\models\modelo_tiempo.pkl")
modelo_precio = joblib.load("main\models\modelo_precio.pkl")

le_origen = joblib.load("main\models\le_origen.pkl")
le_destino = joblib.load("main\models\le_destino.pkl")
le_transporte = joblib.load("main\models\le_transporte.pkl")

# Datos de entrada
origen_input = 'A'
destino_input = 'G'
distancia_input = 15.88
transporte_input = 'Metro'
hora_pico_input = 1

if (origen_input in le_origen.classes_ and
    destino_input in le_destino.classes_ and
    transporte_input in le_transporte.classes_):

    origen_cod = le_origen.transform([origen_input])[0]
    destino_cod = le_destino.transform([destino_input])[0]
    transporte_cod = le_transporte.transform([transporte_input])[0]

    entrada = pd.DataFrame([{
        'origen': origen_cod,
        'destino': destino_cod,
        'distancia_km': distancia_input,
        'transporte': transporte_cod,
        'hora_pico': hora_pico_input
    }])

    tiempo_estimado = modelo_tiempo.predict(entrada)[0]
    precio_estimado = modelo_precio.predict(entrada)[0]

    print(f"Predicción para ruta {origen_input} → {destino_input}")
    print(f"Distancia: {distancia_input} km | Transporte: {transporte_input} | Hora pico: {hora_pico_input}")
    print(f"Tiempo estimado: {tiempo_estimado:.1f} min")
    print(f"Precio estimado: {precio_estimado:.0f} COP")
else:
    print("Entrada inválida: verifica origen, destino o transporte.")
