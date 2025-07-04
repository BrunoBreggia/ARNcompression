## Representación latente de datos secuenciales

Secuencias largas de datos con alfabetos reducidos, como por ejemplo el ARN ($\mathcal{A}=\{A,U,G,C\}$),
tienen elevadas probabilidades de que se repitan patrones comunes a lo largo de la secuencia.
En la práctica, se sabe que hay regines conservadas que son comunes en los seres vivos en general o 
entre algunas familias o especies. Por lo tanto podemos asumir que hay reglas generales sobre la 
combinación de estos caracteres. En tal caso, debería ser posible que un modelo de machine learning
sea capaz de aprender estas reglas y probablemente poder comprimir las secuencias en otras de longitud
mucho menor usando el mismo alfabeto del texto original.

Intentamos desarrollar un enconder-decoder con capas recurrentes con la finalidad de generar a la 
salida del encoder una representación latente, idealmente de menor longitud que la secuencia de input
y con el mismo alfabeto. La finalidad del decoder es asegurarnos que sea viable volver a la secuencia
original a partir de la representación latente. El encoder aprenderá a codificar, y el decoder a 
decodificar. Debemos priorizar que la salida sea identica a la entrada, y que la representación 
intermedia sea de la menor longitud posible respecto de la entrada.


