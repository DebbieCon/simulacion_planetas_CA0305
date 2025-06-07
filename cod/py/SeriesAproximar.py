import numpy as np
import pandas as pd

class SeriesAproximar:
    def __init__(self, m, k=0.07436680):
        '''
        Inicializa el solver del problema de dos cuerpos.

        Parámetros
        ----------
        m : float  
            masa del cuerpo en términos de la masa del cuerpo central

        k : float  
            constante gravitacional gaussiana (por defecto para sistema geocéntrico)
        '''
        self.m = m
        self.k = k
        self.mu = 1 + m

        # Coeficientes f_j y g_j precalculados
        self.f_coeffs = [self.f0, self.f1, self.f2, self.f3, self.f4, 
                         self.f5, self.f6, self.f7, self.f8, self.f9, self.f10]

        self.g_coeffs = [self.g0, self.g1, self.g2, self.g3, self.g4, 
                         self.g5, self.g6, self.g7, self.g8, self.g9, self.g10]


    def __str__(self):
        '''
        Retorna una representación en cadena del objeto SeriesAproximar.

        Retorna
        -------
        str
            Representación en cadena del objeto.
        '''
        return f'Objeto SeriesAproximar: m={self.m}, k={self.k}, mu={self.mu}'
    
    # Definición de los coeficientes f_j
    def f0(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f0

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        int 
            el coeficiente es 1
        '''
        return 1

    def f1(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f1

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        int 
            el coeficiente es 0
        '''
        return 0

    def f2(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f2

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente f2
        '''
        return -self.epsilon(r0)/2

    def f3(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f3

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente f3
        '''
        return self.epsilon(r0)*lambda0/2

    def f4(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f4

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente f4
        '''
        eps = self.epsilon(r0)
        return -eps*(2*eps + 15*lambda0**2 - 3*psi0)/24

    def f5(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f5

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente f5
        '''
        eps = self.epsilon(r0)
        return eps*lambda0*(2*eps + 7*lambda0**2 - 3*psi0)/8

    def f6(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f6

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente f6
        '''
        eps = self.epsilon(r0)
        term1 = 22*eps**2 + 6*eps*(70*lambda0**2 - 11*psi0)
        term2 = 45*(21*lambda0**4 - 14*lambda0**2*psi0 + psi0**2)
        return -eps*(term1 + term2)/720

    def f7(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f7

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente f7
        '''
        eps = self.epsilon(r0)
        term1 = 12*eps**2 + 4*eps*(22*lambda0**2 - 9*psi0)
        term2 = 5*(33*lambda0**4 - 30*lambda0**2*psi0 + 5*psi0**2)
        return eps*lambda0*(term1 + term2)/80

    def f8(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f8

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente f8
        '''
        eps = self.epsilon(r0)
        term1 = 584*eps**3 + 36*eps**2*(560*lambda0**2 - 73*psi0)
        term2 = 54*eps*(1925*lambda0**4 - 1120*lambda0**2*psi0 + 67*psi0**2)
        term3 = 315*(429*lambda0**6 - 495*lambda0**4*psi0 + 135*lambda0**2*psi0**2 - 5*psi0**3)
        return -eps*(term1 + term2 + term3)/40320

    def f9(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f9

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente f9
        '''
        eps = self.epsilon(r0)
        term1 = 2368*eps**3 + 444*eps**2*(77*lambda0**2 - 24*psi0)
        term2 = 18*eps*(7007*lambda0**4 - 5698*lambda0**2*psi0 + 827*psi0**2)
        term3 = 189*(715*lambda0**6 - 1001*lambda0**4*psi0 + 385*lambda0**2*psi0**2 - 35*psi0**3)
        return eps*lambda0*(term1 + term2 + term3)/24192

    def f10(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente f10

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente f10
        '''
        eps = self.epsilon(r0)
        term1 = 28384*eps**4 + 48*eps**3*(31735*lambda0**2 - 3548*psi0)
        term2 = 54*eps**2*(245245*lambda0**4 - 126940*lambda0**2*psi0 + 6559*psi0**2)
        term3 = 90*eps*(420420*lambda0**6 - 441441*lambda0**4*psi0 + 107514*lambda0**2*psi0**2 - 3461*psi0**3)
        term4 = 14175*(2431*lambda0**8 - 4004*lambda0**6*psi0 + 2002*lambda0**4*psi0**2 - 308*lambda0**2*psi0**3 + 7*psi0**4)
        return -eps*(term1 + term2 + term3 + term4)/3628800

    # Definición de los coeficientes g_j
    def g0(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g0

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        int 
            el coeficiente es 1
        '''
        return 1

    def g1(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g1

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        int 
            el coeficiente es 0
        '''
        return 0

    def g2(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g2

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        int 
            el coeficiente es 0
        '''
        return 0

    def g3(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g3

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente g3
        '''
        return -self.epsilon(r0)/6

    def g4(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g4

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente g4
        '''
        return self.epsilon(r0)*lambda0/4

    def g5(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g5

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente g5
        '''
        eps = self.epsilon(r0)
        return -eps*(8*eps + 45*lambda0**2 - 9*psi0)/120

    def g6(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g6

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente g6
        '''
        eps = self.epsilon(r0)
        return eps*lambda0*(5*eps + 14*lambda0**2 - 6*psi0)/24

    def g7(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g7

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente g7
        '''
        eps = self.epsilon(r0)
        term1 = 172*eps**2 + 36*eps*(70*lambda0**2 - 11*psi0)
        term2 = 225*(21*lambda0**4 - 14*lambda0**2*psi0 + psi0**2)
        return -eps*(term1 + term2)/5040

    def g8(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g8

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente g8
        '''
        eps = self.epsilon(r0)
        term1 = 52*eps**2 + 14*eps*(25*lambda0**2 - 9*psi0)
        term2 = 15*(33*lambda0**4 - 30*lambda0**2*psi0 + 5*psi0**2)
        return eps*lambda0*(term1 + term2)/320

    def g9(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g9

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente g9
        '''
        eps = self.epsilon(r0)
        term1 = 7136*eps**3 + 108*eps**2*(1785*lambda0**2 - 23*psi0)
        term2 = 432*eps*(1925*lambda0**4 - 1120*lambda0**2*psi0 + 67*psi0**2)
        term3 = 2205*(429*lambda0**6 - 495*lambda0**4*psi0 + 135*lambda0**2*psi0**2 - 5*psi0**3)
        return -eps*(term1 + term2 + term3)/362880

    def g10(self, r0, lambda0, psi0):
        '''
        Calcula el coeficiente g10

        Parametros
        ---------
        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        Retorna 
        -------
        float 
            el valor del coeficiente g10
        '''
        eps = self.epsilon(r0)
        term1 = 15220*eps**3 + 12*eps**2*(14938*lambda0**2 - 4647*psi0)
        term2 = 81*eps*(7007*lambda0**4 - 5698*lambda0**2*psi0 + 827*psi0**2)
        term3 = 756*(715*lambda0**6 - 1001*lambda0**4*psi0 + 385*lambda0**2*psi0**2 - 35*psi0**3)
        return eps*lambda0*(term1 + term2 + term3)/120960

    def epsilon(self, r0):
        ''' 
        Calcula ε₀ = μ/r₀³

        Parametros 
        ----------
        r0: float  
            distancia inicial

        Retorna
        -------
        float 
            el valor de ε₀ 
        '''
        return self.mu / (r0**3)

    def compute_initial_parameters(self, r0_vec, v0_vec):
        '''
        Calcula los parámetros iniciales necesarios para las series.

        Parámetros
        ----------
        r0_vec: np.ndarray 
            vector de posición inicial [x0, y0, z0]

        v0_vec: np.ndarray 
            vector de velocidad inicial [vx0, vy0, vz0]

        Retorna
        -------
        r0: float  
            distancia inicial

        lambda0: float
            parámetro lambda inicial

        psi0: float
            parámetro psi inicial
        '''
        r0 = np.linalg.norm(r0_vec)
        lambda0 = np.dot(r0_vec, v0_vec) / (r0**2)
        psi0 = np.dot(v0_vec, v0_vec) / (r0**2)
        return r0, lambda0, psi0

    def compute_FG_series(self, tau, r0, lambda0, psi0, n_terms=10):
        '''
        Calcula las series F y G y sus derivadas hasta n términos.

        Parámetros
        ----------
        tau: float 
            tiempo modificado τ = k(t - t0)

        r0: float  
            distancia inicial

        lambda0: float  
            parámetro lambda inicial

        psi0: float  
            parámetro psi inicial

        n_terms: int  
            número de términos a usar en la serie (máx 10)

        Retorna
        -------
        F, G: float 
            valores de las series F(τ) y G(τ)

        F_prime, G_prime: float  
            derivadas de las series
        '''
        F = 0.0
        G = 0.0
        F_prime = 0.0
        G_prime = 0.0

        # Limitar el número de términos a 10 como máximo
        n_terms = min(n_terms, 10)

        for j in range(n_terms + 1):
            fj = self.f_coeffs[j](r0, lambda0, psi0)
            gj = self.g_coeffs[j](r0, lambda0, psi0)

            F += fj * (tau**j)
            G += gj * (tau**j)

            if j > 0:
                F_prime += j * fj * (tau**(j-1))
                G_prime += j * gj * (tau**(j-1))

        return F, G, F_prime, G_prime

    def solve(self, t, t0, r0_vec, v0_vec, n_terms=10):
        '''
        Resuelve el problema de dos cuerpos para un tiempo t dado.

        Parámetros
        ----------
        t: float 
            tiempo final

        t0: float 
            tiempo inicial

        r0_vec: np.ndarray
            vector de posición inicial [x0, y0, z0]

        v0_vec: np.ndarray
            vector de velocidad inicial [vx0, vy0, vz0]

        n_terms: np.ndarray
            número de términos a usar en las series (máx 10)

        Retorna
        --------
        r_vec: np.ndarray
            vector de posición en tiempo t [x, y, z]

        v_vec: np.ndarray
            vector de velocidad en tiempo t [vx, vy, vz]

        r: float 
            distancia en tiempo t

        v: float 
            velocidad en tiempo t
        '''
        # Calcular tiempo modificado
        tau = self.k * (t - t0)

        # Calcular parámetros iniciales
        r0, lambda0, psi0 = self.compute_initial_parameters(r0_vec, v0_vec)

        # Calcular series F y G
        F, G, F_prime, G_prime = self.compute_FG_series(tau, r0, lambda0, psi0, n_terms)

        # Calcular posición y velocidad en tiempo t
        r_vec = F * r0_vec + G * v0_vec
        v_vec = F_prime * r0_vec + G_prime * v0_vec

        # Calcular distancia y velocidad escalares
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        return r_vec, v_vec, r, v

    def resultados(self, tf, t0, r0_vec, v0_vec, periodo, n_terms=10):
        '''
        Calcula y almacena los resultados de la posicion y la velocidad de un cuerpo 
        en un rango de tiempo determinado y los guarda en un archivo csv 

        Parametros 
        ----------
        tf : float 
            tiempo final

        t0 : float
            tiempo inicial

        r0_vec: np.ndarray
            vector de posición inicial [x0, y0, z0]

        v0_vec: np.ndarray
            vector de velocidad inicial [vx0, vy0, vz0]

        periodo: int 
            intervalo de tiempo

        n_terms: int 
            numero de terminos, fijo en 10 

        Retorna
        -------
        df : DataFrame
            contiene los resultados de la posicion y velocidad del cuerpo 
        '''
        datos = []
        for t in range(t0, tf, periodo):
            r_vec, v_vec, r, v = self.solve(tf, t0, r0_vec, v0_vec)
            datos.append({
                't (días)': t,
                'r_x': r_vec[0],
                'r_y': r_vec[1],
                'r_z': r_vec[2],
                'v_x': v_vec[0],
                'v_y': v_vec[1],
                'v_z': v_vec[2]
            })
        df = pd.DataFrame(datos)
        df.to_csv('data\\resultados.csv', index=False)
        return df

