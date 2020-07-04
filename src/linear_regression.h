#pragma once

#include "types.h"

// Cuadrados Minimos Lineales
// Interfaz basada en sklearn.linear_model.LinearRegression
//
//     https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
//
// Uso usual
//
//      clf = LinearRegression()
//      clf.fit(X_train, y_train)
//      y_pred = clf.predict(X_test)
//
class LinearRegression {
public:
    LinearRegression();

    /* Nuestro caso particular

    Dadas f1, ..., fn funciones aplicadas sobre las features,
    x_i, y_i, w_i features

        A = f1(x_1, y_1, w_1),  f2, ..., fn
            f1(x_2, ...),       f2, ..., fn
            .
            .
            .
            f1(x_n, y_n, w_n),  f2, ..., fn

        b = y_1     # precio|feature 1
            y_2     # precio|feature 2
            .
            .
            .
            y_n     # precio|feature n
    
    Podria ser lo siguiente, donde las funciones son proyecciones

        A = x_1     y_1     w_1
            x_2     y_2     w_2
            .
            .
            .
            x_n     y_n     w_n

    */

    // Fitea el modelo lineal
    // Resuelve CML con ecuaciones normales, siendo
    //
    //  - X la matriz A de CML
    //  - y el vector b
    //
    // Encuentra el x que minimiza
    //
    //    ||Ax - b||_2
    //
    // Mediante ecuaciones normales, se obtiene a partir de
    //
    //    AtAx = Atb
    //
    void fit(Matrix X, Matrix y);

    // Predice usando el modelo lineal
    // Recibe X=A', y devuelve b=b', la prediccion, haciendo A'x
    // (donde x es la solucion de CML almacenada en el predict)
    Matrix predict(Matrix X);

    // DEBUGGING
    Matrix get_x();

private:

    // Solucion de CML. Se setea durante el predict.
    // Para nuestro caso de uso, contiene los coeficientes
    // de la combinación lineal de la familia de fns.
    Matrix x;

};
