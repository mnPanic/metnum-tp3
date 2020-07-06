# TODO

- Implementacion
  - Implementación de `linear_regression.cpp`
  - Tests en python

- Experimentos
  - Meta
    - Copiar cross validation
    - Implementar metricas
      - Metricas RMSE y RMSLE
      - https://www.investopedia.com/terms/r/r-squared.asp#:~:text=R%2Dsquared%20(R2),variables%20in%20a%20regression%20model

  - Modelos para explicar caracteristicas
    - Precio
      - **Pensar que features elegir**
      - centros comerciales cercanos
      - habitaciones
      - hacer word cloud de los titulos / descripciones para ver
        si hay palabras que aparecen mas que el resto.
        Correlacionar las palabras que mas aparecen con el precio

    - **Buscar la otra, y pensar que features**
      - Predecir Antiguedad
        - metros cubiertos: casas antiguas son mas grandes
        - lat, lng: casas antiguas estan
      - Predecir cant baños
      - Predecir m2
        - habitaciones, baños (obvio)

    - Proyecciones
      - Hacer forwards y backwards substitution
      - Graficar mejora en score over steps
      - Graficar score over labels (donde las labels son las features tomadas)

  - Segmentacion
    - **Proponer ~2**
      - provincia
      - tipo de propiedad
    - Ciudad

  - Feature engineering
    - **Armar ~2**
      - "centrico", cercania de cosas
        - centros comerciales cercanos
        - escuelas cercanas
      - country, va a ser cheto
        - gym
        - escuelas cercanas
        - piscina

- Informe
  - Desarrollo
  - Plantear experimentos
