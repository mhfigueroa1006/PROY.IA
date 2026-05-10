# Agente Inteligente para Selección de Alineaciones de Fútbol

## Descripción

Este proyecto implementa un agente inteligente capaz de generar alineaciones de fútbol automáticamente utilizando técnicas de Inteligencia Artificial.

El sistema analiza:
- rendimiento de jugadores
- goles
- asistencias
- pases
- defensa
- condición física
- historial reciente

Posteriormente selecciona:
- estilo táctico
- formación
- alineación óptima

## Técnicas de IA utilizadas

- Aprendizaje supervisado:
  - Perceptrón

- Aprendizaje no supervisado:
  - K-Means

- Sistema multicriterio de scoring

## Estructura del proyecto

```text
ProyectoIA/
├── src/
├── data/
├── results/
├── experiments/
```

## Instalación

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejemplo de ejecución

Al ejecutar el sistema:

```bash
python src/agente_futbol.py
```

el agente analiza el contexto reciente del partido, evalúa el rendimiento y estado físico de los jugadores y genera automáticamente:

- una recomendación táctica
- una alineación óptima
- scores individuales
- análisis de química entre jugadores

Los resultados se exportan automáticamente en archivos Excel dentro de la carpeta `results/`.

## Dependencias

- pandas
- scikit-learn
- openpyxl
- numpy

## Resultados

El sistema:
- recomienda estilos tácticos
- genera alineaciones
- evalúa jugadores
- analiza química entre jugadores
El sistema exporta automáticamente los resultados y alineaciones generadas en archivos Excel dentro de la carpeta `results/`.
## Autores

Proyecto desarrollado para la materia de Inteligencia Artificial.
