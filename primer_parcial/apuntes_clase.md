## Aprendizaje Automático

### 7/8/26

- Esteban y Facundo
- Campus Virtual // Comunicación por correo electrónico
- Clases virtuales a demanda (en principio)
- Parciales -> Viernes 18/9 y Viernes 6/11
    - Promoción: parciales $\geq$ 7 y sumar 16 entre ambos
    - Final: parciales $\geq$ 4 y sumar 12 entre ambos
    - Recuperatorios: pisan notal de parcial
        - 13/11 recuperatorio del primer parcial
        - 20/11 recuperatorio del segundo parcial
- Evaluación continua:
    - Completar cuestionarios, consignas de contenidos, guías, papers. Se avisa con 2 semanas de anticipación.
    - Habrá mínimo 3 cuestaionarios
    - Son aprobados o desaprobados
    - Hay que aprobar $\frac{N-1}{N}$. Se puede recuperar 1 de los $N$
- Es más una materia de matemática que de programación

---
#### Ciclo de vida de un proyecto de ML

Formulación del problema -> Diseño del experimento -> Conseguir Datos -> Exploración de Datos -> Entrenamiento del modelo (generalmente predictivo) -> Análisis del modelo -> Presentar o desplegarlo

- A veces tenemos que los datos antes de formular la pregunta
- No es lineal, osea las etapas no son sólo estas y se pueden conectar de otras formas
- Por sobre todas las cosas, es un ciclo iterativo

> Esta materia trata el entrenamiento del modelo

A diferencia del software tradicional, no hacemos desarrollo sino entrenamiento de modelos. Las métricas van a guiar el proceso de desarrollo. El entrenamiento, a través de un optimizador, va a mejorar la métrica.

Todos los modelos tienen:
- Algoritmo (modelo),
- Optimizador,
- Métrica (función de costo).

---
#### Notación

Fuente: *The elements of Statistical Learning*

Todos los vectores son columnas. Matrices en negrita, vectores no.

- $N$: filas
- $p$: columnas
- $X$: vector variable de entrada
- $Y$: vector de salida cuantitativa
- $G$: vector de salida cualitativo (quizas lo vemos como $Y$ en un abuso de notación)

#### ¿Cuál es la relación entre aprendizaje automático y estadística?
Chequear lo que dice la slide

- Correlación: dirección privilegiada
- En 2D es fácil de ver
- *Describir estadisticamente* lo que sucede con las variables
- Caracterizar una distribución que la pueda generar
- Podemos hablar de densidades y direcciones

Aprendizaje supervisado: buscamos distribución conjunta.

Aprendizaje no supervisado: busca $p(x)$

---
#### Aprendizaje Supervisado

$$T=\{(x, y)\}^N$$

- Con $x$ de tamaño $p$,
- $(x, y)$ las observaciones,
- y $T$ el dataset

Asumimos una relación funcional entre la entrada y la salida

$$y=f(x)+\epsilon$$

- $f$ es la información sistemática,
- $\epsilon$ son las cosas no modeladas

¿Cómo obtenemos $f$?

El algoritmo de aprendizaje se encarga de ver la diferencia entre lo real y lo predicho. La ventaja de la prediccón es que puede ser mucho más barata que ir a buscar un dato real. Nos interesa la inferencia, entender las relaciones entre $X$ e $Y$.

> Atribución > Inferencia > Predicción

Reglas de la probabilidad: suma; producto; teorema de Bayes. (Casos discreto y continuo)

En las tablas del ejemplo 1, cuando paso de una tabla que tiene info de $X$ a la siguiente aplicando Bayes, pierdo justamente esa info, de la distribución de $X$, pero se convierte en una tabla mucho más útil.

¿Cuál es más fácil de trabajar? Depende, para la distribución poblacional, la de arriba (la conjunta $P(X, Y)$), para predecir, la de $P(Y|X)$.

Muchas veces nos vamos a confundir la distribución teórica con la empírica.

Conocer una distribución no es poder predecir. Falta una regla de decisión, elegir una etiqueta candidata, fundamentar esa decisión es la teoría de decisión.

---
### 14/8/26

Aprendizaje supervisado -> identificación de $P(X)$ con $X$ con $p$ coordenadas

$$Y=F(X)+\Epsilon$$

- $Y$: lo que quiero predecir,
- $X$: lo que uso para predecir,
- $F$: la parte determinista,
- $\Epsilon$: la parte probabilística

¿Por qué buscar $F$?:
- Si encontrar $F$ es menos costoso que $Y$, es conveniente
- Inferencia: Entender relaciones entre $X$ e $Y$, poder intervenir en $X$ para ver cómo reacciona $Y$.

Si tengo la probabilidad conjunta, tengo una descripción completa del problema

$$P(X, Y) \to P(Y|X)$$

El $\to$ es a través de Bayes. Perdemos información de $X$, pero sigue siendo útil. Pero falta algo para predecir, nos falta algo para asignar una etiqueta. ¿Asignar la más probable? Tiene sentido en algunos ámbitos.

[hasta acá, repaso de la clase pasada]

---
#### Teoría de decisión - Clasificación
- Toma de decisión óptima (bajo cierto punto de vista, minimizando algún aspecto)
- Teoría de decisión a la Bishop
- Minimizar la probabilidad de error, maximizar el acierto, poner la frontera en la intersección.

[ver sección de cuentas de los apuntes de cátedra]

Ejemplo:

Supongamos que tengo un $X \in [0,2]$ y que $P(X)=\frac{X}{2}$. Además $P(C_2|X)=\frac{X}{2}$ y $P(C_1|X)=1-\frac{X}{2}$

$$P(E)=\int_{R_1} P(C_2|X) P(x) dx + \int_{R_2} P(C_1|X) P(x) dx $$

Como quiero minimizar el error, es un problema de optimización. Para simplificar, digamos que hasta cierto $\alpha$ asigno $C_1$ y a la derecha, $C_2$. Desarrollando $P(E)$, se obtiene $\alpha=1$.

Es equivalente a decir: "asigná la clase más probable de cada lado".

En el otro punto de vista más general, usamos una función de pérdida, es una matriz que a cada valor es un costo asignado.

En el problema

$$P(G|X)=\dfrac{P(X,G)}{P(X)}$$

$P(X, G)$ suele ser muy difícil por la alta dimensionalidad. ¿La solución? modelar, introducimos sesgo para achicar dimensionalidad. ¿Qué modelamos? Inferencia: obtener la conjunta o la condicional; Decisión: predecimos (usando Bayes o no)

Aproximaciones: modelos generativos (naive Bayes, ADL); normalizados (AD, regresión logística); función discriminante (perceptrón, SVM)

#### Teoría de decisión - Regresión

Cambia la forma de medir el error.

Tener en cuenta para los ejercicios (los tres puntos hablan de la maldición de la dimensionalidad):
- ¿Cuándo es un mínimo de datos para estimar un promedio?
- Para cada variable, cuántos valores pueden tomar (o intervalos)
- ¿Cuántos escenarios quedan?

---
### 21/8/26

#### Regresión

Formulación general

$$f(x)=\beta_0+\sum_{j=1}^p x_j \beta_j$$
$$f(x)=x^T \beta$$

¿Qué modelamos con $f$? La esperanza de la condicional $E(Y|X)$

¿Cómo es un nivel más arriba de esto último, de $P(Y|X)$? Asumiendo que $P$ es una distribución normal. Esto sucede *antes* de tomar la esperanza.

¿Qué son los $X$? Lo que uno quiera. Es cómodo pensar que son variables continuas, pero pueden ser categorías, interacciones entre variables. La relación con los parámetros son los que indican que estamos usando una regresión lineal.

**Entrenamiento**

- Tenemos $n$ instancias y vemos lo que mejor ajusta. ¿Cómo medimos qué es mejor? Una función de costo, en este caso $RSS$.
- El método es $OLS$ (cuadrados mínimos)
- Lo único que pueo modificar para ajustar $RSS$ es $\beta$
- $\frac{\partial RSS}{\partial \beta}$ da como resultado un gradiente
- $\frac{\partial RSS^2}{\partial^2 \beta}$ da como resultado un hessiano

Resulta en $$\hat{\beta_{OLS}} = (X^TX)^{-1}X^T y$$

**Geometría de los resultados**
- $x_1$ y $x_2$ son variables
- simbolizan el subespacio generado por esas variables
- por construcción de $X^T(y-X\beta)=0$, los valores predichos son la predicción ortogonal que minimiza la distancia del espacio generado al espacio de las observaciones.
- la dimensión de los vectores columna $X$ que generan un subespacio tiene dimensión $p$
- $\lambda_i=0 \Rightarrow det()=0$
- $X^TX$ son las correlaciones, si tengo mal diseñado esot, la matriz es LD. Con multicolinealidad esto queda lejos del caso ideal y de hecho la matriz no es inversible

---
### 25/8/26

**Matriz de covarianzas**

- Grado de variación conjunta. Nos permite medir correlación
- La covarianza de una variable consigo misma es la varianza de esa variable
- Preferimos trabajar de manera matricial
- La forma está determinada por la cantidad de variables
- $\Sigma=\Sigma^T$ osea que es simétrica, también debe ser al menos semidifinida positiva o definida positiva
- Si no "estandarizamos" las unidades de la matriz de covarianza, no hay forma de comparar. Se usa la matriz de correlaciones
- Independencia estadística y covarianza: ¿Cómo se relacionan?: si son independientes, la covarianza es 0, pero no es al reves, porque puede ser que haya una dependencia no lineal.
- Matriz mal condicionada: autovalores muy grandes o muy chicos. Hay un factor que se calcula dividiendo el más grande con el más chico.

---
Entrenamiento - cuadrados mínimos.

Sesgo, varianza, distribución de coeficientes.

---
### 4/9/26

Planteamos problema de regresión lineal, optimizamos y obtenemos

$$\hat{\beta} = (X^TX)^{-1}X^T y$$

- Es la solución de cuadrados mínimos
- Con máxima verosimilitud llegamos a lo mismo
- Los modelos necesitan una métrica para saber cuáles son los parámetros que mejor optimizan

---

Si hay multicolinealidad: **SVD**

$$X=UDV^T$$

Se le suele decir el teorema fundamental del álgebra lineal

Además me puedo inventar una cuasi inversa al menos por izquierda

$$X^{-1}=VD^{-1}U^T$$

Esta es la pseudo inversa de Moore-Penrose, que nos deja la forma más eficiente de resolver cuadrados mínimos

$$\hat{\beta}=VD^{-1}U^Ty$$

A considerar, que si el rango de $(X^TX)$ es menor a $p$ entonces no existe su inversa, es decir que hay algún autovalor $\lambda$ igual a cero. De manera muy bruta, lo tiramos. Al ser chico, tiene poca varianza, entonces no estamos perdiendo información. No termino teniendo mayor problema con los predictores tampoco.

---
### 11/9/26

#### Ridge

Usando SVD llegamos a que $$\hat{\beta}=VD^{-1}U^Ty$$

- Problemas que no podemos resolver: si tiene autovalores chicos (mal condicionada, multicolinealidad, etc)
- Pero en este escenario, es fácil sacar los valores problemáticos, acomodando el sistema
- Recordemos que OLS y MLE con diferentes métodos de optimización analíticos, son el mismo estimador
- Descenso por gradiente es otra forma, pero es numérico, no analítico como los otros dos

- ¿Si no hay rango completo?
    - SVD: pero es una solución muy brusca
    - Ridge: aborda el mismo problema, de manera más suave. En general, son métodos de regularización
        - Dos formas de entrar: problemas de multicolinealidad; o Gauss-Markov

Ahora optimizamos pero con restricciones

$$\hat{\beta}_{RIDGE}=\argmin\{||y-X\beta||^2+\lambda||\beta||^2\}$$

- Es la forma más usual de escribirlo
- $+\lambda||\beta||^2$ es el término controlador

Forma más lagrangiana de escribirlo:

$$\hat{\beta}_{RIDGE}=\{||y-X\beta||^2\}_{MÍNIMO} \,\,\,\, \text{sujeto a } \beta^T\beta \overline{v}=0$$

- Prestarle atención a normalización y centralización

$$\hat{\beta}_{RIDGE}=(X^TX+\lambda I)^{-1}X^Ty$$

> Ojo: no penaliza el intercepto!!

- Ridge tiene solución analítica (Lasso no)
- Tienen que estar estandarizadas las variables

$$x_i \sim \mathcal{N}(\mu,\sigma^2)\,\,\,\,\overline{x}=\dfrac{\sum x_i}{N}\,\,\,\, \hat{\sigma}^2=\dfrac{\sum (x_i-\overline{x})^2}{N}\,\,\,\, \epsilon(x)=\dfrac{\sigma^2}{\sqrt{n}}$$

> ¿Por qué funciona Ridge? Gracias a SVD. Nos va a ayudar a calcularlo y también a entender por qué funciona.