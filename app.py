# Copyright (c) 2023 Yasmany Fernández Fernández, Jefferson Iván Narváez Quendi, Sira M. Allende Alonso, Ridelio Miranda Pérez
# Licencia MIT - Ver archivo LICENSE para detalles
from flask import Flask, request, jsonify, render_template
import json
import os
import sys
import webbrowser
from threading import Timer
from optimizacion import ejecutar_optimizacion  # Importa la función de optimización

app = Flask(__name__)

# Ruta para guardar los datos
@app.route('/guardar_datos', methods=['POST'])
def guardar_datos():
    try:
        data = request.get_json()
        
        # Guardar los datos en un archivo JSON
        with open('datos_optimizacion.json', 'w') as f:
            json.dump(data, f, indent=4)
            
        return jsonify({
            "status": "success",
            "message": "Datos guardados correctamente"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Ruta para cargar los datos guardados
@app.route('/cargar_datos', methods=['GET'])
def cargar_datos():
    try:
        if os.path.exists('datos_optimizacion.json'):
            with open('datos_optimizacion.json', 'r') as f:
                data = json.load(f)
            return jsonify({
                "status": "success",
                "data": data
            })
        return jsonify({
            "status": "success",
            "data": None
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
# Nueva ruta para ejecutar la optimización
@app.route('/ejecutar_optimizacion', methods=['GET'])
def ejecutar_optimizacion_route():
    try:
        resultados = ejecutar_optimizacion()
        if resultados:
            return jsonify({
                "status": "success",
                "resultados": resultados
            })
        return jsonify({
            "status": "error",
            "message": "No se pudieron obtener resultados"
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Ruta para cargar los resultados de optimización
@app.route('/cargar_resultados', methods=['GET'])
def cargar_resultados():
    try:
        if os.path.exists('resultados_optimizacion.json'):
            with open('resultados_optimizacion.json', 'r') as f:
                data = json.load(f)
            return jsonify({
                "status": "success",
                "data": data
            })
        return jsonify({
            "status": "success",
            "data": None
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 
    
@app.route('/obtener_datos_grafo', methods=['GET'])
def obtener_datos_grafo():
    try:
        if os.path.exists('datos_optimizacion.json'):
            with open('datos_optimizacion.json', 'r') as f:
                datos = json.load(f)
            
            # Estructura para el grafo estático
            grafo_estatico = {
                "nodos": {
                    "salida": datos['etiquetasA'],
                    "transito": datos['etiquetasR'],
                    "llegada": datos['etiquetasF']
                },
                "conexiones": datos['distancias']
            }
            
            # Estructura para el grafo dinámico (se completará con los resultados)
            grafo_dinamico = {
                "nodos": {
                    "salida": datos['etiquetasA'],
                    "transito": datos['etiquetasR'],
                    "llegada": datos['etiquetasF']
                },
                "conexiones": datos['distancias'],
                "rutas": []
            }
            
            # Si hay resultados, añadimos las rutas al grafo dinámico
            if os.path.exists('resultados_optimizacion.json'):
                with open('resultados_optimizacion.json', 'r') as f:
                    resultados = json.load(f)
                    if 'reporte_rutas' in resultados:
                        grafo_dinamico['rutas'] = resultados['reporte_rutas']
            
            return jsonify({
                "status": "success",
                "grafo_estatico": grafo_estatico,
                "grafo_dinamico": grafo_dinamico
            })
        return jsonify({
            "status": "success",
            "data": None
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/')
def index():
    return render_template('index.html')

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()  # Abre el navegador después de 1 segundo
    app.run(debug=False)
