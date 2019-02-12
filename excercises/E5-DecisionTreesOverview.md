# E5 - Decision Trees Overview

Los arboles de decisión son un tipo de algoritmo de machine learning que busca abordar todos los posibles resultados dependiendo de las posibles combinaciones existentes en un problema objetivo, este tipo de algoritmo funciona cuando existe un número finito de combinaciones entre decisiones en donde las combinaciones indican la profundidad del árbol y los incentivos para la selección de decisiones cambia según las probabilidad, costo de selección o beneficios que tengan las posibles opciones de respuesta. 
 Los arboles de decisión buscan encontrar una serie de decisiones óptimas para alcanzar un objetivo, para esto existen los siguientes tipos de árboles de decisión:
Bagging: construye arboles de decisión con multiples iteraciones de entrenamiento alterando aleatoriamente la selección muestral con el objetivo de votar por el mejor árbol de predicción por consenso. 

Random Forest: árbol de clasificación que utiliza un número X de arboles en función de encontrar la mejor tasa de clasificación 

Boosted trees: es un tipo de árboles de clasificación para problemas relacionados con regresiones

Rotation Trees: Tipo de árbol de decisión que utiliza Análisis de Componentes Principales (PCA en inglés) para entrar cada árbol con un set de datos totalmente aleatorio. 

Las ventajas de utilizar random forest son:

1.	Fácil comprensión de cómo un algoritmo encuentra patrones de decisión. 
2.	Este algoritmo hace automáticamente selección de variables y encuentra el set característico para predecir la variable dependiente. 
3.	Es muy útil cuando la relación entre mis variables no es lineal 

Usualmente el uso que se da a los arboles de decisión es precisamente conocer el set probabilístico en cada nodo de decisión que me determina la proyección de mi variable dependiente en función de cada característica participar de mis observaciones. 

