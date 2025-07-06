import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Datos(Dataset):
    def __init__(self, url, fecha='fecha', planetas=None):
        self._datos = pd.read_csv(url, index_col=0)

        if planetas is None:
            planetas = ['mercurio','venus','tierra','marte','jupiter','saturno','urano','neptuno']

        self._planetas = planetas
        self._num_planetas = len(planetas)
        self._num_datos = len(self._datos)

        self._t = None
        self._id_obj = None
        self._r = None
        self._v = None

        self._scaler_t = MinMaxScaler()
        self._scaler_pos = StandardScaler()
        self._scaler_r = StandardScaler()
        self._scaler_v = StandardScaler()

        self._t_s = None
        self._pos_s = None
        self._r_s = None
        self._v_s = None

        self._pos_mtx = None
        self._vel_mtx = None

        self._datos[fecha] = pd.to_datetime(self._datos[fecha])
        self._datos['t'] = (self._datos[fecha] - self._datos[fecha].min()).dt.days.astype(np.float32)
        self._datos.drop(columns=[fecha], inplace=True)
        self._t = self._datos['t'].values.reshape(-1,1).astype(np.float32)

    # --- Properties con setters ---
    @property
    def datos(self): return self._datos
    @datos.setter
    def datos(self, valor): self._datos = valor

    @property
    def planetas(self): return self._planetas
    @planetas.setter
    def planetas(self, valor): self._planetas = valor

    @property
    def num_planetas(self): return self._num_planetas
    @num_planetas.setter
    def num_planetas(self, valor): self._num_planetas = valor

    @property
    def num_datos(self): return self._num_datos
    @num_datos.setter
    def num_datos(self, valor): self._num_datos = valor

    @property
    def t(self): return self._t
    @t.setter
    def t(self, valor): self._t = valor

    @property
    def id_obj(self): return self._id_obj
    @id_obj.setter
    def id_obj(self, valor): self._id_obj = valor

    @property
    def r(self): return self._r
    @r.setter
    def r(self, valor): self._r = valor

    @property
    def v(self): return self._v
    @v.setter
    def v(self, valor): self._v = valor

    @property
    def scaler_t(self): return self._scaler_t
    @scaler_t.setter
    def scaler_t(self, valor): self._scaler_t = valor

    @property
    def scaler_pos(self): return self._scaler_pos
    @scaler_pos.setter
    def scaler_pos(self, valor): self._scaler_pos = valor

    @property
    def scaler_r(self): return self._scaler_r
    @scaler_r.setter
    def scaler_r(self, valor): self._scaler_r = valor

    @property
    def scaler_v(self): return self._scaler_v
    @scaler_v.setter
    def scaler_v(self, valor): self._scaler_v = valor

    @property
    def t_s(self): return self._t_s
    @t_s.setter
    def t_s(self, valor): self._t_s = valor

    @property
    def pos_s(self): return self._pos_s
    @pos_s.setter
    def pos_s(self, valor): self._pos_s = valor

    @property
    def r_s(self): return self._r_s
    @r_s.setter
    def r_s(self, valor): self._r_s = valor

    @property
    def v_s(self): return self._v_s
    @v_s.setter
    def v_s(self, valor): self._v_s = valor

    @property
    def pos_mtx(self): return self._pos_mtx
    @pos_mtx.setter
    def pos_mtx(self, valor): self._pos_mtx = valor

    @property
    def vel_mtx(self): return self._vel_mtx
    @vel_mtx.setter
    def vel_mtx(self, valor): self._vel_mtx = valor

    # --- Métodos de procesamiento ---
    def posiciones_velocidades(self, obj='tierra'):
        pos = np.zeros((self.num_datos, self.num_planetas, 3), dtype=np.float32)
        vel = np.zeros((self.num_datos, self.num_planetas, 3), dtype=np.float32)

        for i, p in enumerate(self.planetas):
            pos[:, i, 0] = self.datos[f'{p}_x']
            pos[:, i, 1] = self.datos[f'{p}_y']
            pos[:, i, 2] = self.datos[f'{p}_z']
            vel[:, i, 0] = self.datos[f'{p}_vx']
            vel[:, i, 1] = self.datos[f'{p}_vy']
            vel[:, i, 2] = self.datos[f'{p}_vz']

        self.id_obj = self.planetas.index(obj)
        self.r = pos[:, self.id_obj, :]
        self.v = vel[:, self.id_obj, :]
        self.pos_mtx = pos
        self.vel_mtx = vel

    def escalar_y_transformar(self):
        self.scaler_t.fit(self.t)
        self.scaler_pos.fit(self.pos_mtx.reshape(self.num_datos, -1))
        self.scaler_r.fit(self.r)
        self.scaler_v.fit(self.v)

        self.t_s = torch.tensor(self.scaler_t.transform(self.t), dtype=torch.float32)
        self.pos_s = torch.tensor(
            self.scaler_pos.transform(self.pos_mtx.reshape(self.num_datos, -1)).reshape(self.num_datos, self.num_planetas, 3),
            dtype=torch.float32
        )
        self.r_s = torch.tensor(self.scaler_r.transform(self.r), dtype=torch.float32)
        self.v_s = torch.tensor(self.scaler_v.transform(self.v), dtype=torch.float32)

    def inv_escalar_pos(self, r_escalado):
        r_np = r_escalado.detach().cpu().numpy()
        return self.scaler_r.inverse_transform(r_np)

    def masas_planetas_dic(self):
        masas_dict = {
            "sol": 1.9885e30, "mercurio": 3.301e23, "venus": 4.867e24,
            "tierra": 5.972e24, "marte": 6.417e23, "jupiter": 1.898e27,
            "saturno": 5.683e26, "urano": 8.681e25, "neptuno": 1.024e26
        }
        return [masas_dict[p.lower()] for p in self.planetas]

    def datos_id(self, idx):
        return (self.t_s[idx], self.pos_s[idx]), (self.r_s[idx], self.v_s[idx])
    
    def __len__(self):
        return self.num_datos

    def __getitem__(self, idx):
        return (self.t_s[idx], self.pos_s[idx]), (self.r_s[idx], self.v_s[idx])
