import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


data = {
    "Jugador": [
        "J1", "J2", "J3", "J4", "J5", "J6", "J7", "J8", "J9", "J10",
        "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18", "J19", "J20",
        "J21", "J22", "J23", "J24", "J25", "J26", "J27", "J28", "J29", "J30"
    ],
    "Posiciones": [
        "GK", "GK",
        "LB", "LB",
        "RB", "RB",
        "CB", "CB", "CB", "CB", "CB", "CB",
        "CDM", "CDM",
        "CM", "CM", "CM", "CM",
        "CAM", "CAM",
        "LW", "LW",
        "RW", "RW",
        "ST", "ST", "ST", "ST", "ST", "ST"
    ],
    "Rendimiento": [
        8.7, 8.2, 7.5, 7.0, 7.4, 6.9, 8.1, 7.9, 8.3, 7.8,
        7.6, 8.0, 7.7, 7.2, 8.0, 7.6, 7.8, 7.3, 8.5, 8.2,
        8.1, 7.4, 8.0, 7.5, 9.2, 8.8, 8.5, 7.9, 7.6, 8.3
    ],
    "Goles": [
        0, 0, 1, 0, 1, 0, 2, 1, 3, 1,
        1, 2, 1, 0, 3, 2, 2, 1, 5, 6,
        7, 4, 6, 5, 12, 10, 9, 6, 5, 8
    ],
    "Asistencias": [
        0, 0, 4, 3, 3, 2, 1, 1, 1, 0,
        1, 2, 4, 3, 6, 5, 4, 3, 10, 9,
        8, 5, 7, 6, 3, 4, 3, 2, 2, 4
    ],
    "Pases": [
        75, 73, 85, 82, 84, 80, 90, 88, 91, 87,
        86, 89, 88, 85, 90, 89, 87, 85, 88, 87,
        84, 82, 83, 81, 78, 79, 80, 77, 76, 79
    ],
    "Defensa": [
        65, 60, 78, 75, 76, 72, 90, 88, 91, 87,
        86, 89, 85, 82, 70, 68, 65, 63, 50, 48,
        45, 40, 42, 38, 30, 28, 25, 22, 20, 27
    ],
    "Condicion": [
        "Buena", "Buena", "Buena", "Fatiga", "Buena", "Lesion leve",
        "Buena", "Buena", "Buena", "Buena", "Fatiga", "Buena",
        "Buena", "Fatiga", "Buena", "Buena", "Buena", "Fatiga",
        "Buena", "Buena", "Buena", "Fatiga", "Buena", "Buena",
        "Buena", "Buena", "Fatiga", "Buena", "Lesion leve", "Buena"
    ]
}

df = pd.DataFrame(data)
df.to_excel("df_prueba.xlsx")

# =====================================================
# CONDICIÓN FÍSICA
# =====================================================

condicion_factor = {
    "Buena": 1.0,
    "Fatiga": 0.75,
    "Lesion leve": 0.5,
    "Lesionado": 0.0
}

df["Factor_condicion"] = df["Condicion"].map(condicion_factor)

# =====================================================
# VARIABLES DEL PERCEPTRÓN
# =====================================================

features = [
    "Rendimiento",
    "Goles",
    "Asistencias",
    "Pases",
    "Defensa",
    "Factor_condicion"
]

# =====================================================
# FORMACIONES POR ESTILO
# =====================================================

formaciones = {
    "ofensivo": {
        "formacion": "4-3-3",
        "posiciones": ["GK", "LB", "CB", "CB", "RB", "CM", "CAM", "CM", "LW", "ST", "RW"]
    },
    "defensivo": {
        "formacion": "5-4-1",
        "posiciones": ["GK", "LB", "CB", "CB", "CB", "RB", "CDM", "CM", "CM", "CAM", "ST"]
    },
    "equilibrado": {
        "formacion": "4-4-2",
        "posiciones": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    }
}

# =====================================================
# POSICIONES EQUIVALENTES
# =====================================================

equivalencias = {
    "GK": ["GK"],
    "LB": ["LB"],
    "RB": ["RB"],
    "CB": ["CB"],
    "CDM": ["CDM", "CM"],
    "CM": ["CM", "CAM", "CDM"],
    "CAM": ["CAM", "CM"],
    "LM": ["LW", "CM"],
    "RM": ["RW", "CM"],
    "LW": ["LW", "ST"],
    "RW": ["RW", "ST"],
    "ST": ["ST", "LW", "RW"]
}

# =====================================================
# ETIQUETAS PARA ENTRENAR EL PERCEPTRÓN
# =====================================================

def crear_etiqueta(row, estilo):
    if row["Factor_condicion"] == 0:
        return 0

    if estilo == "ofensivo":
        return int(
            row["Goles"] >= 5 or
            row["Asistencias"] >= 6 or
            row["Rendimiento"] >= 8.3
        )

    elif estilo == "defensivo":
        return int(
            row["Defensa"] >= 80 or
            row["Pases"] >= 88 or
            row["Rendimiento"] >= 8.0
        )

    elif estilo == "equilibrado":
        return int(
            row["Rendimiento"] >= 7.8 and
            row["Factor_condicion"] >= 0.75
        )

    else:
        raise ValueError("Estilo no válido")

# =====================================================
# ENTRENAMIENTO DEL PERCEPTRÓN
# =====================================================

modelos = {}

for estilo in ["ofensivo", "defensivo", "equilibrado"]:

    df[f"Etiqueta_{estilo}"] = df.apply(
        lambda row: crear_etiqueta(row, estilo),
        axis=1
    )

    X = df[features]
    y = df[f"Etiqueta_{estilo}"]

    # Separar datos para entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("perceptron", Perceptron(
            max_iter=1000,
            eta0=0.1,
            random_state=42
        ))
    ])

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicciones
    y_pred = modelo.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n====================================")
    print(f"MODELO {estilo.upper()}")
    print("====================================")

    print(f"Accuracy: {accuracy:.2f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    modelos[estilo] = modelo

# =====================================================
# CLUSTERING PARA QUÍMICA ENTRE JUGADORES
# =====================================================

features_quimica = [
    "Rendimiento",
    "Goles",
    "Asistencias",
    "Pases",
    "Defensa"
]

scaler_quimica = StandardScaler()
X_quimica = scaler_quimica.fit_transform(df[features_quimica])

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster_quimica"] = kmeans.fit_predict(X_quimica)
print("CLUSTERS DE QUÍMICA")

print(df[["Jugador", "Cluster_quimica"]])

print("\nPROMEDIOS POR CLUSTER")

print(
    df.groupby("Cluster_quimica")[
        ["Rendimiento", "Goles", "Asistencias", "Pases", "Defensa"]
    ].mean()
)

# =====================================================
# FUNCIÓN DE SCORE INDIVIDUAL
# =====================================================

def calcular_score(row, estilo):
    if estilo == "ofensivo":
        score = (
            0.35 * row["Rendimiento"] +
            0.30 * row["Goles"] +
            0.20 * row["Asistencias"] +
            0.15 * row["Pases"] / 10
        )

    elif estilo == "defensivo":
        score = (
            0.30 * row["Rendimiento"] +
            0.40 * row["Defensa"] / 10 +
            0.20 * row["Pases"] / 10 +
            0.10 * row["Asistencias"]
        )

    elif estilo == "equilibrado":
        score = (
            0.30 * row["Rendimiento"] +
            0.20 * row["Goles"] +
            0.20 * row["Asistencias"] +
            0.15 * row["Pases"] / 10 +
            0.15 * row["Defensa"] / 10
        )

    else:
        raise ValueError("Estilo no válido")

    return score * row["Factor_condicion"]

# =====================================================
# FUNCIÓN DE QUÍMICA
# =====================================================

def calcular_quimica(candidato, alineacion_actual, df_temp):
    if len(alineacion_actual) == 0:
        return 1.0

    jugadores_actuales = [
        jugador["Jugador"]
        for jugador in alineacion_actual
        if jugador["Jugador"] != "Sin candidato"
    ]

    if len(jugadores_actuales) == 0:
        return 1.0

    cluster_candidato = candidato["Cluster_quimica"]

    clusters_actuales = df_temp[
        df_temp["Jugador"].isin(jugadores_actuales)
    ]["Cluster_quimica"]

    coincidencias = (clusters_actuales == cluster_candidato).sum()

    quimica = coincidencias / len(jugadores_actuales)

    return quimica

# =====================================================
# RECOMENDAR ESTILO SEGÚN HISTORIAL
# =====================================================

def recomendar_estilo_por_historial():
    print("Ingresa los resultados de los últimos 5 partidos contra este equipo.")
    print("Ejemplo: goles a favor = 2, goles en contra = 1\n")

    gf_total = 0
    gc_total = 0
    victorias = 0
    empates = 0
    derrotas = 0

    for i in range(1, 6):
        print(f"Partido {i}")

        gf = int(input("Goles a favor: "))
        gc = int(input("Goles en contra: "))

        gf_total += gf
        gc_total += gc

        if gf > gc:
            victorias += 1
        elif gf == gc:
            empates += 1
        else:
            derrotas += 1

        print()

    diferencia_goles = gf_total - gc_total

    print("Resumen del historial:")
    print(f"Victorias: {victorias}")
    print(f"Empates: {empates}")
    print(f"Derrotas: {derrotas}")
    print(f"Goles a favor: {gf_total}")
    print(f"Goles en contra: {gc_total}")
    print(f"Diferencia de goles: {diferencia_goles}")

    if victorias >= 3 and diferencia_goles > 0:
        estilo_recomendado = "ofensivo"
    elif derrotas >= 3 or diferencia_goles < -2:
        estilo_recomendado = "defensivo"
    else:
        estilo_recomendado = "equilibrado"

    print(f"\nEstilo recomendado por el sistema: {estilo_recomendado}")

    respuesta = input("¿Deseas aceptar este estilo? (si/no): ").lower()

    if respuesta == "si":
        return estilo_recomendado

    print("\nElige otro estilo:")
    print("1. ofensivo")
    print("2. defensivo")
    print("3. equilibrado")

    opcion = input("Selecciona una opción: ")

    if opcion == "1":
        return "ofensivo"
    elif opcion == "2":
        return "defensivo"
    elif opcion == "3":
        return "equilibrado"
    else:
        print("Opción inválida. Se usará equilibrado por defecto.")
        return "equilibrado"

# =====================================================
# ALINEACIÓN FINAL
# =====================================================

def generar_alineacion(df, estilo):
    estilo = estilo.lower()

    if estilo not in formaciones:
        raise ValueError("Elige: ofensivo, defensivo o equilibrado")

    df_temp = df.copy()

    formacion = formaciones[estilo]["formacion"]
    posiciones_necesarias = formaciones[estilo]["posiciones"]

    df_temp["Recomendado_perceptron"] = modelos[estilo].predict(df_temp[features])

    df_temp["Score_individual"] = df_temp.apply(
        lambda row: calcular_score(row, estilo),
        axis=1
    )

    alineacion = []
    jugadores_usados = set()

    for posicion in posiciones_necesarias:
        posiciones_validas = equivalencias[posicion]

        candidatos = df_temp[
            (~df_temp["Jugador"].isin(jugadores_usados)) &
            (df_temp["Factor_condicion"] > 0) &
            (df_temp["Recomendado_perceptron"] == 1) &
            (df_temp["Posiciones"].apply(
                lambda x: any(pos in x.split(",") for pos in posiciones_validas)
            ))
        ].copy()

        # Respaldo si el perceptrón no recomienda candidatos para esa posición
        if candidatos.empty:
            candidatos = df_temp[
                (~df_temp["Jugador"].isin(jugadores_usados)) &
                (df_temp["Factor_condicion"] > 0) &
                (df_temp["Posiciones"].apply(
                    lambda x: any(pos in x.split(",") for pos in posiciones_validas)
                ))
            ].copy()

        if candidatos.empty:
            alineacion.append({
                "Posición": posicion,
                "Jugador": "Sin candidato",
                "Recomendado": "-",
                "Cluster química": "-",
                "Química": 0,
                "Score individual": 0,
                "Score final": 0
            })
            continue

        candidatos["Quimica"] = candidatos.apply(
            lambda row: calcular_quimica(row, alineacion, df_temp),
            axis=1
        )

        candidatos["Score_final"] = (
            (0.80 * candidatos["Score_individual"] +
            0.20 * candidatos["Quimica"] )* 10
        )

        mejor = candidatos.sort_values(
            by="Score_final",
            ascending=False
        ).iloc[0]

        alineacion.append({
            "Posición": posicion,
            "Jugador": mejor["Jugador"],
            "Recomendado IA": int(mejor["Recomendado_perceptron"]),
            "Cluster química": int(mejor["Cluster_quimica"]),
            "Química": round(mejor["Quimica"], 2),
            "Score individual": round(mejor["Score_individual"], 2),
            "Score final": round(mejor["Score_final"], 2)
        })

        jugadores_usados.add(mejor["Jugador"])

    return formacion, pd.DataFrame(alineacion)

# =====================================================
# 13. EJECUCIÓN DEL AGENTE
# =====================================================

estilo = recomendar_estilo_por_historial()

formacion, alineacion = generar_alineacion(df, estilo)

print("\n==============================")
print("RESULTADO FINAL DEL AGENTE")
print("==============================")
print(f"Estilo seleccionado: {estilo}")
print(f"Formación recomendada: {formacion}")
print("\nAlineación propuesta:")
print(alineacion)

alineacion.to_excel("Datos_Alineacion.xlsx")

print("\nScore promedio de la alineación:")
print(round(alineacion["Score final"].mean(), 2))
