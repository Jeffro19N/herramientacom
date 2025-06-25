# Copyright (c) 2023 Yasmany Fernández Fernández y Jefferson Narváez
# Licencia MIT - Ver archivo LICENSE para detalles
import json
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpInteger, LpStatus
import numpy as np

class ModeloOptimizacion:
    def __init__(self, datos):
        self.datos = datos
        self.problema = LpProblem("Modelo_Emergencias", LpMinimize)
        
        # Inicializar estructuras de datos
        self.X = {}  # Variables de flujo
        self.Y = {}  # Variables de activación
        
    def crear_variables(self):
        # Crear variables Y (binarias) para nodos R y F
        for rf in self.datos['etiquetasR'] + self.datos['etiquetasF']:
            self.Y[rf] = LpVariable(f"Y_{rf}", cat=LpBinary)
        
        # Crear variables X (enteras) para flujos
        for id_fam in self.datos['idFamilias']:
            h = int(self.datos['idf'][id_fam]['h'])
            for i in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']:
                for j in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']:
                    if self.datos['distancias'].get(i, {}).get(j, 0) > 0:  # Solo si hay conexión
                        self.X[(id_fam, h, i, j)] = LpVariable(
                            f"X_{id_fam}_{h}_{i}_{j}", 
                            lowBound=0, 
                            cat=LpInteger
                        )
    
    def definir_funcion_objetivo(self):
        # Primer término: sum[(id,h,i,j), c*d(i,j)*ord(h)*X(id,h,i,j)]
        termino1 = lpSum(
            self.datos['costoPorKm'] * 
            self.datos['distancias'][i][j] * 
            h * 
            self.X[(id_fam, h, i, j)]
            for id_fam in self.datos['idFamilias']
            for h in [int(self.datos['idf'][id_fam]['h'])]
            for i in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
            for j in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
            if (id_fam, h, i, j) in self.X
        )
        
        # Segundo término: sum[rf, Ac(rf)*Y(rf)]
        termino2 = lpSum(
            self.datos['ac'].get(rf, 0) * self.Y[rf]
            for rf in self.datos['etiquetasR'] + self.datos['etiquetasF']
        )
        
        self.problema += termino1 + termino2
    
    def agregar_restricciones(self):
        # Restricción Robust_Salidas_C1
        for id_fam in self.datos['idFamilias']:
            h = int(self.datos['idf'][id_fam]['h'])
            ns = self.datos['idf'][id_fam]['ns']
            
            suma_salida = lpSum(
                self.X.get((id_fam, h, ns, j), 0)
                for j in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                if (id_fam, h, ns, j) in self.X
            )
            
            suma_entrada = lpSum(
                self.X.get((id_fam, h, j, ns), 0)
                for j in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                if (id_fam, h, j, ns) in self.X
            )
            
            valor_idf = self.datos['idf'][id_fam]['valor']
            self.problema += (suma_salida - suma_entrada == valor_idf), f"Robust_Salidas_C1_{id_fam}_{h}_{ns}"
        
        # Restricción capac_llegada_origen
        for ns in self.datos['etiquetasA']:
            suma = lpSum(
                h * self.X.get((id_fam, h, ns, j), 0)
                for id_fam in self.datos['idFamilias']
                for h in [int(self.datos['idf'][id_fam]['h'])]
                for j in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                if (id_fam, h, ns, j) in self.X
            )
            alpha = self.datos.get('alpha', 0)
            self.problema += (suma <= alpha), f"capac_llegada_origen_{ns}"
        
        # Restricción Robust_Flujo_Transito
        for id_fam in self.datos['idFamilias']:
            h = int(self.datos['idf'][id_fam]['h'])
            for nt in self.datos['etiquetasR']:
                suma_entrada = lpSum(
                    self.X.get((id_fam, h, i, nt), 0)
                    for i in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                    if (id_fam, h, i, nt) in self.X
                )
                
                suma_salida = lpSum(
                    self.X.get((id_fam, h, nt, i), 0) * 
                    (1 if self.datos['distancias'][nt].get(i, 0) > 0 else 0)
                    for i in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                    if (id_fam, h, nt, i) in self.X
                )
                
                self.problema += (suma_entrada == suma_salida), f"Robust_Flujo_Transito_{id_fam}_{h}_{nt}"
        
        # Restricción cap_llegada_punto_transito
        for nt in self.datos['etiquetasR']:
            suma = lpSum(
                h * self.X.get((id_fam, h, nt, j), 0)
                for id_fam in self.datos['idFamilias']
                for h in [int(self.datos['idf'][id_fam]['h'])]
                for j in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                if (id_fam, h, nt, j) in self.X
            )
            beta = self.datos.get('beta', 0)
            self.problema += (suma <= beta * self.Y[nt]), f"cap_llegada_punto_transito_{nt}"
        
        # Restricción Robust_Llegada_C2
        for nll in self.datos['etiquetasF']:
            suma_entrada = lpSum(
                h * self.X.get((id_fam, h, i, nll), 0)
                for id_fam in self.datos['idFamilias']
                for h in [int(self.datos['idf'][id_fam]['h'])]
                for i in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                if (id_fam, h, i, nll) in self.X
            )
            
            suma_salida = lpSum(
                h * self.X.get((id_fam, h, nll, i), 0)
                for id_fam in self.datos['idFamilias']
                for h in [int(self.datos['idf'][id_fam]['h'])]
                for i in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                if (id_fam, h, nll, i) in self.X
            )
            
            pi_nll = self.datos['pi'].get(nll, 0)
            self.problema += (suma_entrada - suma_salida <= pi_nll * self.Y[nll]), f"Robust_Llegada_C2_{nll}"
        
        # Restricción cap_llegada_centro_seguro
        for nll in self.datos['etiquetasF']:
            suma = lpSum(
                h * self.X.get((id_fam, h, i, nll), 0)
                for id_fam in self.datos['idFamilias']
                for h in [int(self.datos['idf'][id_fam]['h'])]
                for i in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                if (id_fam, h, i, nll) in self.X
            )
            gamma = self.datos.get('gamma', 0)
            self.problema += (suma <= gamma), f"cap_llegada_centro_seguro_{nll}"
        
        # Restricción Equilibrio1
        for id_fam in self.datos['idFamilias']:
            h = int(self.datos['idf'][id_fam]['h'])
            
            suma1 = lpSum(
                self.X.get((id_fam, h, ns, rf), 0)
                for ns in self.datos['etiquetasA']
                for rf in self.datos['etiquetasR'] + self.datos['etiquetasF']
                if (id_fam, h, ns, rf) in self.X
            ) - lpSum(
                self.X.get((id_fam, h, rf, ns), 0)
                for ns in self.datos['etiquetasA']
                for rf in self.datos['etiquetasR'] + self.datos['etiquetasF']
                if (id_fam, h, rf, ns) in self.X
            )
            
            suma2 = lpSum(
                self.X.get((id_fam, h, ar, nll), 0)
                for ar in self.datos['etiquetasA'] + self.datos['etiquetasR']
                for nll in self.datos['etiquetasF']
                if (id_fam, h, ar, nll) in self.X
            ) - lpSum(
                self.X.get((id_fam, h, nll, ar), 0)
                for ar in self.datos['etiquetasA'] + self.datos['etiquetasR']
                for nll in self.datos['etiquetasF']
                if (id_fam, h, nll, ar) in self.X
            )
            
            self.problema += (suma1 == suma2), f"Equilibrio1_{id_fam}_{h}"
        
        # Restricción Equilibrio2
        for id_fam in self.datos['idFamilias']:
            h = int(self.datos['idf'][id_fam]['h'])
            for nll in self.datos['etiquetasF']:
                suma_salida = lpSum(
                    self.X.get((id_fam, h, nll, j), 0)
                    for j in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                    if (id_fam, h, nll, j) in self.X
                )
                
                suma_entrada = lpSum(
                    self.X.get((id_fam, h, j, nll), 0)
                    for j in self.datos['etiquetasA'] + self.datos['etiquetasR'] + self.datos['etiquetasF']
                    if (id_fam, h, j, nll) in self.X
                )
                
                self.problema += (suma_salida <= suma_entrada), f"Equilibrio2_{id_fam}_{h}_{nll}"
    
    @staticmethod
    def generar_reporte_rutas(datos, resultados):
        reporte = []
        
        # Construir grafo de conexiones
        grafo = {}
        for i in datos['etiquetasA'] + datos['etiquetasR'] + datos['etiquetasF']:
            grafo[i] = [j for j in datos['etiquetasA'] + datos['etiquetasR'] + datos['etiquetasF'] 
                    if datos['distancias'].get(i, {}).get(j, 0) > 0]
        
        # Función para reconstruir TODAS las rutas utilizadas por una familia
        def encontrar_rutas_utilizadas(origen, flujos):
            # Creamos un grafo de flujos reales
            grafo_flujos = {}
            for flujo in flujos:
                if flujo['cantidad'] > 0:
                    if flujo['desde'] not in grafo_flujos:
                        grafo_flujos[flujo['desde']] = []
                    grafo_flujos[flujo['desde']].append(flujo['hacia'])
            
            # Función auxiliar DFS para encontrar rutas completas
            def dfs(nodo_actual, camino_actual, rutas_completas):
                camino_actual = camino_actual + [nodo_actual]
                
                # Si llegamos a un destino final (etiquetasF)
                if nodo_actual in datos['etiquetasF']:
                    rutas_completas.append(camino_actual)
                    return
                
                # Explorar vecinos en el grafo de flujos reales
                for vecino in grafo_flujos.get(nodo_actual, []):
                    if vecino not in camino_actual:  # Evitar ciclos
                        dfs(vecino, camino_actual, rutas_completas)
            
            rutas_completas = []
            dfs(origen, [], rutas_completas)
            return rutas_completas
        
        # Procesar cada familia
        for id_fam, familia_data in resultados['resumen']['flujos_por_familia'].items():
            h = familia_data['tamaño_familia']
            origen = familia_data['origen']
            
            # Encontrar TODAS las rutas utilizadas por esta familia
            rutas_utilizadas = encontrar_rutas_utilizadas(origen, familia_data['flujos'])
            
            # Procesar cada ruta utilizada
            for ruta in rutas_utilizadas:
                destino = ruta[-1]
                ruta_str = "->".join(ruta)
                
                # Calcular distancia total exacta
                distancia = sum(datos['distancias'][ruta[i]][ruta[i+1]] for i in range(len(ruta)-1))
                
                # Calcular el flujo mínimo en la ruta (cuello de botella)
                flujo_minimo = min(
                    next(
                        (flujo['cantidad'] for flujo in familia_data['flujos'] 
                        if flujo['desde'] == ruta[i] and flujo['hacia'] == ruta[i+1]
                    )
                    for i in range(len(ruta)-1)
                )
                )
                personas_en_ruta = flujo_minimo * h
                num_nodos_ruta = len(ruta)-1
                num_nodos_ruta = num_nodos_ruta + 1
                
                # Agregar al reporte
                reporte.append({
                    "id_familia": int(id_fam),
                    "tamaño_familia": h,
                    "familias_en_ruta": flujo_minimo,
                    "personas_en_ruta": personas_en_ruta,
                    "ruta_str": ruta_str,
                    "num_nodos_ruta": num_nodos_ruta,
                    "distancia": distancia
                    
                })
        
        return reporte

    def resolver(self):
        self.crear_variables()
        self.definir_funcion_objetivo()
        self.agregar_restricciones()

        # Resolver el problema
        self.problema.solve()
        
        # Preparar resultados
        resultados = {
            "status": LpStatus[self.problema.status],
            "valor_objetivo": self.problema.objective.value(),
            "variables_X": {},
            "variables_Y": {},
            "resumen": {
                "nodos_activados": [],
                "flujos_por_familia": {}
            }
        }
        
        # Recoger valores de Y
        for rf, var in self.Y.items():
            resultados["variables_Y"][rf] = var.value()
            if var.value() == 1:
                resultados["resumen"]["nodos_activados"].append(rf)
        
        # Recoger valores de X
        for (id_fam, h, i, j), var in self.X.items():
            if var.value() > 0:
                if id_fam not in resultados["variables_X"]:
                    resultados["variables_X"][id_fam] = []
                resultados["variables_X"][id_fam].append({
                    "desde": i,
                    "hacia": j,
                    "cantidad": var.value(),
                    "personas": h * var.value()
                })
                
                # Resumen por familia
                if id_fam not in resultados["resumen"]["flujos_por_familia"]:
                    resultados["resumen"]["flujos_por_familia"][id_fam] = {
                        "tamaño_familia": h,
                        "origen": self.datos['idf'][id_fam]['ns'],
                        "flujos": []
                    }
                resultados["resumen"]["flujos_por_familia"][id_fam]["flujos"].append({
                    "desde": i,
                    "hacia": j,
                    "cantidad": var.value(),
                    "personas": h * var.value()
                })

                    # Generar reporte de rutas (solo si la solución es óptima)
            if resultados["status"] == "Optimal":
                resultados['reporte_rutas'] = ModeloOptimizacion.generar_reporte_rutas(self.datos, resultados)
            else:
                resultados['reporte_rutas'] = []
        return resultados

def cargar_datos_desde_archivo():
    try:
        with open('datos_optimizacion.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def guardar_resultados(resultados):
    with open('resultados_optimizacion.json', 'w') as f:
        json.dump(resultados, f, indent=4)

def ejecutar_optimizacion():
    datos = cargar_datos_desde_archivo()
    if not datos:
        print("No se encontraron datos de entrada. Por favor, ingrese los datos primero.")
        return None
    
    modelo = ModeloOptimizacion(datos)
    resultados = modelo.resolver()
    guardar_resultados(resultados)
    return resultados