from Cuerpo import Cuerpo
from Simulacion_N_Cuerpos import Simulacion_N_Cuerpos as snc
from Simulacion2D import Simulacion2D as s2d
from Simulacion3D import Simulacion3D as s3d
from SeriesAproximar import SeriesAproximar
import numpy as np
import math



# SIMULACION EN 2D

UA = 1.496e11

## Sistema Sol-Tierra-Marte-Intruso

sol = Cuerpo([0,0], [0,0], 2.989e30)
tierra = Cuerpo([UA,0], [0,24.1e3], 6.39e23)
marte = Cuerpo([1.524*UA,0], [0,24.1e3], 5.39e23)
intruso = Cuerpo([0.8*UA,0-3*UA],[-35e3,40e3], 2e14)

sim = s2d([sol,tierra,marte,intruso], G = 6.67430e-11, h=15000)
sim.simular(pasos = 5000)
sim.animar()
#sim.guardar_animacion('simulacion_2d_1.mp4')

#Sistema estrella, dos cuerpos y cometa

estrella = Cuerpo([0,0], [0,-2000],2e30)
planeta1 = Cuerpo([-400e9,160e9], [4000,0], 2e14)
planeta2 = Cuerpo([150e9,0], [0,30000], 6e24)
cometa = Cuerpo([206e9,0], [0,24000], 6e23)


sim = s2d([estrella,planeta1,planeta2,cometa], G = 6.67430e-11, h=15000)
sim.simular(pasos = 5000)
sim.animar()
#sim.guardar_animacion('simulacion_2d_2.mp4')


# SIMULACIÓN EN 3D


G = 6.67430e-11  # Constante gravitacional
UA = 1.496e11     # 1 Unidad Astronómica (metros)
h = 15000        # Paso de tiempo (para mayor detalle)


cuerpos_5 = [
    # Estrella central masiva
    Cuerpo(
        pos=[0, 0, 0],
        vel=[0, 0, 0],
        masa=3e30
    ),
    # Estrella compañera (órbita elíptica)
    Cuerpo(
        pos=[0.3 * UA, 0, 0],
        vel=[0, 25e3, 0],
        masa=1.5e30
    ),
    # Planeta en órbita estable
    Cuerpo(
        pos=[1.0 * UA, 0, 0],
        vel=[0, 30e3, 0],
        masa=6e24
    ),
    # Planeta con inclinación orbital (eje z)
    Cuerpo(
        pos=[0.5 * UA, 0.5 * UA, 0.3 * UA],
        vel=[-20e3, 15e3, 5e3],
        masa=3e24
    ),
    # Intruso caótico (cometa/asteroide masivo)
    Cuerpo(
        pos=[-0.7 * UA, -0.7 * UA, 0.2 * UA],
        vel=[40e3, -30e3, -10e3],
        masa=1e22
    )
]

sim = s3d(cuerpos_5, h, G)
sim.simular(pasos=15000)  # Más pasos para trayectorias complejas
sim.animar()

# APROXIMACIÓN DE LA ÓRBITA

m_tierra = 0.000003004  #Masa de la tierra en términos de la masa del Sol
k_heliocentric = 0.01720209895

t0 = 0
t_f = 20

r0 = np.array([-0.9803066574733390, -0.1935141489038212, 0.00001498630022526094])
v0 = np.array([0.003057689999970376, -0.016948417777335137, 0.0000005102168370891322])

solver = SeriesAproximar(m = m_tierra, k = k_heliocentric)

df = solver.resultados(t_f,t0,r0,v0,1,n_terms=10)

