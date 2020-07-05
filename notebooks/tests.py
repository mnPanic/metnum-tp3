import metnum
import unittest
import numpy as np
import sklearn.linear_model

class TestLinearRegression(unittest.TestCase):
    def assert_allclose(self, got, want):
        self.assertTrue(np.allclose(want, got), f"expected {want} but got {got}")
    
    def test_trivial(self):
        """
        Testea que se comporta de forma correcta para una tabla de
        datos trivial.
        """

        # Lo entrenamos con algo trivial, la identidad y un canonico
        A = np.eye(3)
        b = np.array([
            [1],
            [0],
            [0]
        ])

        clf = metnum.LinearRegression()
        # AtA = I
        # AtA x = At b sii x = b
        clf.fit(A, b)
        
        # Hacemos predict de la identidad para obtener el vector debajo
        y_pred = clf.predict(np.eye(3))
        self.assert_allclose(y_pred, b)

        # Ahora predict de otra matriz un poco mas complicada
        y_pred = clf.predict(np.diag([3, 3, 3]))
        self.assert_allclose(y_pred, np.array([3, 0, 0]).reshape(3, 1))


    def test_compare_numpy(self):
        """Compara con numpy"""
        A = np.array([
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ])
        # Ecuaciones normales es malo con un numero de condicion alto
        # print(np.linalg.cond(A))

        b = np.array([1, 2, 3]).reshape(3, 1)

        # fit
        clf = metnum.LinearRegression()
        clf.fit(A, b)

        # predict
        A_monio = np.array([
            [0, 1, 1],
            [1, 1, 4],
            [0, 0, 0],
        ])

        y_pred = clf.predict(A_monio)

        # con np.linalg
        x = np.linalg.solve((A.T@A), A.T@b)
        y = A_monio@x
    
        self.assert_allclose(y_pred, y)

if __name__ == "__main__":
    unittest.main()