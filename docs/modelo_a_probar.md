parte a:
baseline: 
- isolation forest
- pca

modelos : 
- ae 
- vae
- DAGMM
- deep insolation forest
- internal Contrastive Learning

- usar un regresor logisto o un clasificador que lo que haga es que en cada interacion prediga, los que predicio mal lo guarda, si ve que siempre predice mal las mismas esas son anomalas


experimentos: 
- ver como afeta el sacar ndvi o no 
- busqueda de hyperparamentros (botelneck, cantidad de capas, en vae variar el β-)
- rebalanceo(deberia dar peor por que dejan de ser anomalas)
- comparacion de modelos
- regularizaciones
- comparacion con cosas mas modernas
- ver como reaciona cuando se le saca data de una fuente. 
- ver si logra predecir la sequia del 2023
- probar clustering (dbescan, gmm, kmenas, usar DAGMM directamente( mezcla ae con gmm))  ¿existen tipos distintos de anomalías?
- ¿Qué tan temprano detecta anomalías? esta analizarla por que por como tenemos los datos tendriamos que ver
- analises de sensebilidad, perturbarlo 

- probar agrupar en x regiones tipo chunks para ver como es el performance



parte b:
baseline: 
- mean 
- rl con l1 y l2


modelos: 
- xgboost
- nn
- ver como funciona forecasting

experimtos: 
- con el output de la parte A, sin eso o unicamente la parte a con input(seguramente a un ensemble de arboles le de peor la parte del output)
- vae para generar muestras anomalas o no anomalas(robada de santi la idea) ademas de rebalanceo
- profundidad de nn, activaciones, loss, densidad de conexiones
- recurente nn, puede servir para ver que tan temprano predice el rinde
-  regularizaciones l2, dropout, profundidades de arboels y demas cosas
