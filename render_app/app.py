from flask import Flask, render_template, request
import pandas as pd
import joblib
import json
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_pipeline.joblib"
INFO_PATH = BASE_DIR / "modelo_info.json"

# Cargar modelo
modelo = joblib.load(MODEL_PATH)

# Cargar información del modelo
with open(INFO_PATH, "r", encoding="utf-8") as f:
    modelo_info = json.load(f)

WEATHER_OPTIONS = [
    "Clear",
    "Clouds",
    "Mist",
    "Rain",
    "Snow",
    "Drizzle",
    "Haze",
    "Thunderstorm",
    "Fog",
    "Smoke",
    "Squall"
]


def calcular_is_weekend(day_of_week):
    """
    Calcula si el día corresponde a fin de semana.
    En pandas, day_of_week usa:
    0 = lunes, 1 = martes, ..., 5 = sábado, 6 = domingo.
    """
    return 1 if day_of_week in [5, 6] else 0


def validar_entrada(hour, day_of_week, temp, clouds_all, weather_main):
    errores = []

    if hour < 0 or hour > 23:
        errores.append("La hora debe estar entre 0 y 23.")

    if day_of_week < 0 or day_of_week > 6:
        errores.append("El día de la semana debe estar entre 0 y 6.")

    if temp < 230 or temp > 330:
        errores.append("La temperatura debe estar en un rango razonable entre 230 y 330 Kelvin.")

    if clouds_all < 0 or clouds_all > 100:
        errores.append("La nubosidad debe estar entre 0 y 100 por ciento.")

    if weather_main not in WEATHER_OPTIONS:
        errores.append("La condición climática seleccionada no es válida.")

    return errores


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    errores = []

    valores = {
        "hour": "8",
        "day_of_week": "0",
        "is_weekend": "0",
        "temp": "285.15",
        "clouds_all": "40",
        "weather_main": "Clouds"
    }

    if request.method == "POST":
        valores = {
            "hour": request.form.get("hour", ""),
            "day_of_week": request.form.get("day_of_week", ""),
            "temp": request.form.get("temp", ""),
            "clouds_all": request.form.get("clouds_all", ""),
            "weather_main": request.form.get("weather_main", "")
        }

        try:
            hour = int(valores["hour"])
            day_of_week = int(valores["day_of_week"])
            temp = float(valores["temp"])
            clouds_all = float(valores["clouds_all"])
            weather_main = valores["weather_main"]

            # Calcular automáticamente si es fin de semana
            is_weekend = calcular_is_weekend(day_of_week)

            # Guardar el valor calculado para mantenerlo disponible en la plantilla
            valores["is_weekend"] = str(is_weekend)

            errores = validar_entrada(
                hour,
                day_of_week,
                temp,
                clouds_all,
                weather_main
            )

            if not errores:
                entrada = pd.DataFrame([{
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "temp": temp,
                    "clouds_all": clouds_all,
                    "weather_main": weather_main
                }])

                prediccion = modelo.predict(entrada)[0]

                # Evitar mostrar predicciones negativas si llegaran a ocurrir
                prediccion = max(0, prediccion)

                resultado = round(prediccion, 0)

        except ValueError:
            errores.append("Verifica que los campos numéricos tengan valores válidos.")

        except Exception as e:
            errores.append(f"Ocurrió un error al generar la predicción: {str(e)}")

    return render_template(
        "index.html",
        resultado=resultado,
        errores=errores,
        valores=valores,
        weather_options=WEATHER_OPTIONS,
        modelo_info=modelo_info
    )


if __name__ == "__main__":
    app.run(debug=True)
