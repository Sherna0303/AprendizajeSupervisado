import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("dataset_transporte_masivo.csv")

le_origen = LabelEncoder()
le_destino = LabelEncoder()
le_transporte = LabelEncoder()

df['origen'] = le_origen.fit_transform(df['origen'])
df['destino'] = le_destino.fit_transform(df['destino'])
df['transporte'] = le_transporte.fit_transform(df['transporte'])

X = df[['origen', 'destino', 'distancia_km', 'transporte', 'hora_pico']]
y_tiempo = df['tiempo_total_min']
y_precio = df['precio_total_cop']

X_train, X_test, y_train_tiempo, y_test_tiempo = train_test_split(X, y_tiempo, test_size=0.3, random_state=42)
_, _, y_train_precio, y_test_precio = train_test_split(X, y_precio, test_size=0.3, random_state=42)

modelo_tiempo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_precio = RandomForestRegressor(n_estimators=100, random_state=42)

modelo_tiempo.fit(X_train, y_train_tiempo)
modelo_precio.fit(X_train, y_train_precio)

pred_tiempo = modelo_tiempo.predict(X_test)
pred_precio = modelo_precio.predict(X_test)

mae_tiempo = mean_absolute_error(y_test_tiempo, pred_tiempo)
mae_precio = mean_absolute_error(y_test_precio, pred_precio)

print(f"MAE tiempo estimado: {mae_tiempo:.2f} minutos")
print(f"MAE precio estimado: {mae_precio:.2f} COP")