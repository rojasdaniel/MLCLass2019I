Primero definamos Gradient Boosting y  XGB Classifier:

Gradient Boosting: * Busca optimizar multiples algoritmos de clasificación debiles y generar un solo algoritmo que recoja lo "bueno" de cada algoritmo separado en uno solo. Es decir que, en una primera interación se entrena y detectan aquellos arboles donde su aprendizaje fue menor para en la siguiente iteración poder mejorar los resultados obtenidos en comparación con la primera iteración, así sucesivamente. 

XGB Classifier: * Tiene el mismo principio que Gradient Boosting, solo que cuenta de diferencial los siguientes atributos:

* Penalización inteligente de los árboles
* Un encogimiento proporcional de los nudos foliares
* Newton Boosting
* Parámetro de aleatorización adicional

Ya con implementar newton boosting garantiza que se encuentre una aproximación más cercana al mínimo gradiente descendiente de manera rápida. Adicionalmente, con tener penalización inteligente de los árboles, estamos hablando de garantizar darle mejores pesos a aquellos algoritmos de clasificación que tienen un mejor performarce de manera automática y así potenciar la metodología de Gradient Boosting. 
Ambos algoritmos de boosting son muy populares en el mundo del data science (se evidencia más uso sobre los algoritmos basados en XGB que en Gradient Boosting), XGB Boosting cuenta con ventajas como la construcción de los arboles en función de controlar el over-fiting por ejemplo y computacionalmente es más exigente que Gradient Boosting. 

En términos computacionales, utilizar XGB requiere utilizar todo el poder de la CPU, y puede tomar bastante tiempo lograr entrenar grandes modelos de arboles de decisión o inclusive puede ocurrir que el árbol sea lo suficientemente grande y no quepa en la memoria RAM imposibilitando su ejecución y entrenamiento. 


