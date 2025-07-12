import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
from Datos import Datos

class ModeloPINN(Datos):
    def __init__(self, ruta, planetas, fecha = 'fecha', device='cpu'):
        '''
        Instancia de un objeto ModeloPINN que hereda de Datos.

        Parámetros
        ----------
            ruta : str
                Ruta al archivo CSV con los datos de los planetas.
            planetas : list
                Lista de nombres de los planetas a considerar.
            fecha : str, opcional
                Nombre de la columna que contiene las fechas (default es 'fecha').
            device : str, opcional
                Dispositivo para PyTorch ('cpu' o 'cuda', default es 'cpu').

        Retorna
        -------
        '''
        super().__init__(ruta, fecha, planetas=planetas)

        self._disp = torch.device(device)
        self._modelo = None
        self._optim = None
        self._scheduler = None
        self._train_loader = None
        self._val_loader = None
        self._G = 1.036e-12
        self._masas = torch.tensor(self.masas_planetas_dic(), dtype=torch.float32, device=self._disp).view(1, -1, 1)

    @property
    def modelo(self): 
        '''
        Retorna el modelo de red neuronal utilizado para la simulación.

        Parámetros
        ----------

        Retorna
        -------
            nn.Module
                Modelo de red neuronal.
        '''
        return self._modelo

    @modelo.setter
    def modelo(self, red): 
        ''' 
        Asigna un nuevo modelo de red neuronal.

        Parámetros
        ----------
            red : nn.Module
                Nuevo modelo de red neuronal a asignar.
        Retorna
        -------
        '''
        self._modelo = red

    @property
    def optim(self): 
        '''
        Retorna el optimizador utilizado para entrenar el modelo.   

        Parámetros
        ----------
        Retorna
        -------
            optim.Optimizer
                Optimizador de PyTorch.
        '''
        return self._optim
    
    @optim.setter
    def optim(self, optimizador): 
        '''
        Asigna el optimizador utilizado para entrenar el modelo.

        Parámetros
        ----------
            optimizador : torch.optim.Optimizer
                Instancia del optimizador de PyTorch.

        Retorna
        -------
        '''
        self._optim = optimizador

    @property
    def scheduler(self): 
        '''
        Retorna el scheduler de tasa de aprendizaje.

        Parámetros
        ----------

        Retorna
        -------
            _scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
                Scheduler que ajusta la tasa de aprendizaje según la pérdida.
        '''
        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, sched): 
        '''
        Asigna un nuevo scheduler de tasa de aprendizaje.

        Parámetros
        ----------
            sched : torch.optim.lr_scheduler.ReduceLROnPlateau
                Scheduler que ajusta la tasa de aprendizaje según la pérdida.
        Retorna
        -------
        '''
        self._scheduler = sched

    @property
    def disp(self): 
        '''
        Retorna el dispositivo en el que se ejecuta el modelo.

        Parámetros
        ----------
        Retorna
        -------
            torch.device
                Dispositivo PyTorch (CPU o GPU).
        '''
        return self._disp

    @property
    def train_loader(self): 
        '''
        Retorna el DataLoader para el conjunto de entrenamiento.
        
        Parámetros
        ----------

        Retorna
        -------
            DataLoader
                DataLoader que itera sobre el conjunto de entrenamiento.
        '''
        return self._train_loader
    
    @train_loader.setter
    def train_loader(self, loader):
        '''
        Asigna el DataLoader para el conjunto de entrenamiento.

        Parámetros
        ----------
            loader : torch.utils.data.DataLoader
                Cargador de datos para entrenamiento.

        Retorna
        -------
        '''
        self._train_loader = loader

    @property
    def val_loader(self):
        '''
        Retorna el DataLoader utilizado para el conjunto de validación.

        Parámetros
        ----------

        Retorna
        -------
            val_loader : torch.utils.data.DataLoader
                Cargador de datos para validación.
        '''
        return self._val_loader

    @val_loader.setter
    def val_loader(self, loader):
        '''
        Asigna el DataLoader para el conjunto de validación.

        Parámetros
        ----------
            loader : torch.utils.data.DataLoader
                Cargador de datos para validación.

        Retorna
        -------
        '''
        self._val_loader = loader

    @property
    def G(self):
        '''
        Retorna la constante gravitacional utilizada en los cálculos físicos.

        Parámetros
        ----------

        Retorna
        -------
            G : float
                Valor de la constante gravitacional.
        '''
        return self._G

    @property
    def masas(self):
        '''
        Retorna las masas de los planetas utilizadas en la simulación.

        Parámetros
        ----------

        Retorna
        -------
            masas : torch.Tensor
                Tensor con las masas de los planetas.
        '''
        return self._masas


    def construir_red(self, neuronas=None):
        '''
        Construye la arquitectura de la red neuronal secuencial y la asigna al modelo.

        Parámetros
        ----------
            neuronas : list[int], opcional
                Lista que define el número de neuronas por capa. Si no se especifica,
                se usa una arquitectura por defecto basada en el número de planetas.

        Retorna
        -------
        '''
        if neuronas is None:
            neuronas = [1 + self.num_planetas * 3, 64, 64, 3]

        capas = []
        for i in range(len(neuronas) - 2):
            capas += [nn.Linear(neuronas[i], neuronas[i+1]), nn.SiLU()]
        capas.append(nn.Linear(neuronas[-2], neuronas[-1]))
        capas.append(nn.Tanh())
        self.modelo = nn.Sequential(*capas).to(self.disp)

    def optimizar_red(self, paciencia, min_lr):
        '''
        Configura el optimizador y el scheduler para el entrenamiento del modelo.

        Parámetros
        ----------
            paciencia : int
                Número de épocas sin mejora antes de reducir la tasa de aprendizaje.
            min_lr : float
                Tasa de aprendizaje mínima permitida por el scheduler.

        Retorna
        -------
        '''
        self.optim = optim.Adam(self.modelo.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, factor=0.5, paciencia=paciencia, min_lr=min_lr
        )

    def particionar_datos(self, tam_test=0.2, batch_size=32):
        '''
        Divide los datos en conjuntos de entrenamiento y validación, y configura los DataLoaders.

        Parámetros
        ----------
            tam_test : float
                Proporción de datos usada para validación.
            batch_size : int
                Tamaño de lote para los DataLoaders.

        Retorna
        -------
        '''
        n_test = int(self.num_datos * tam_test)
        n_train = self.num_datos - n_test
        train_ds, val_ds = random_split(self, [n_train, n_test])

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    def calcular_factor_escalado_aceleracion(self):
        '''
        Calcula el factor de escalado para la aceleración física en unidades normalizadas.

        Parámetros
        ----------

        Retorna
        -------
            factor_acel : float
                Factor de escalado para la aceleración en unidades normalizadas (UA/día²).
        '''
        sigma_pos = self.scaler_pos.scale_[0]
        sigma_t = self.scaler_t.scale_[0]

        factor_acel = (1 / sigma_pos) * (1 / sigma_t)**2
        return factor_acel

    def residuos_fisicos(self, t, pos_planetas):
        '''
        Calcula los residuos físicos entre la aceleración predicha por la red y la aceleración gravitacional real.

        Parámetros
        ----------
            t : torch.Tensor
                Tiempos en formato tensorial, de forma [batch] o [batch, 1].
            pos_planetas : torch.Tensor
                Posiciones de los planetas en cada instante, de forma [batch, num_planetas, 3].

        Retorna
        -------
            residuos : torch.Tensor
                Diferencia entre aceleración predicha (escalada) y aceleración física.
        '''
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.requires_grad_(True)

        inp = torch.cat([t, pos_planetas.view(t.size(0), -1)], dim=1)
        r_pred = self.modelo(inp)

        v_pred = torch.autograd.grad(r_pred, t, torch.ones_like(r_pred), create_graph=True)[0]
        a_pred = torch.autograd.grad(v_pred, t, torch.ones_like(v_pred), create_graph=True)[0]

        diffs = pos_planetas - r_pred.unsqueeze(1)
        dist3 = torch.norm(diffs, dim=2, keepdim=True) ** 3 + 1e-9
        a_fis = (self.G * self.masas * diffs / dist3).sum(dim=1)

        factor = self.calcular_factor_escalado_aceleracion()
        return a_pred * factor - a_fis

    def entrenar_pinn(self, epochs=2000, peso_fis=1e3, early=200):
        '''
        Entrena el modelo PINN incorporando restricciones físicas y aplica early stopping.

        Parámetros
        ----------
            epochs : int
                Número máximo de épocas de entrenamiento.
            peso_fis : float
                Peso del término de pérdida física.
            early : int
                Número de épocas sin mejora para detener el entrenamiento anticipadamente.

        Retorna
        -------
            None
        '''
        best_val, paciencia = np.inf, 0
        history = {
            'train_loss': [], 'val_loss': [],
            'train_r': [], 'train_v': [], 'train_f': [],
            'val_r': [], 'val_v': [], 'val_f': []
        }

        for ep in range(1, epochs + 1):
            self.modelo.train()
            running_train = 0

            for (t_b, pos_b), (r_b, v_b) in self.train_loader:
                t_b = t_b.to(self.disp).requires_grad_(True)
                pos_b = pos_b.to(self.disp)
                r_b = r_b.to(self.disp)
                v_b = v_b.to(self.disp)

                self.optim.zero_grad()
                inp = torch.cat([t_b, pos_b.view(t_b.size(0), -1)], dim=1)
                r_pred = self.modelo(inp)
                loss_r = nn.MSELoss()(r_pred, r_b)

                v_pred = [torch.autograd.grad(
                    outputs=r_pred[:, i], inputs=t_b,
                    grad_outputs=torch.ones_like(r_pred[:, i]),
                    create_graph=True, retain_graph=True
                )[0] for i in range(r_pred.size(1))]
                v_pred = torch.cat(v_pred, dim=1)
                loss_v = nn.MSELoss()(v_pred, v_b)

                res = self.residuos_fisicos(t_b.squeeze(-1), pos_b)
                loss_fis = (res**2).mean()

                loss = loss_r + loss_v + peso_fis * loss_fis
                history['train_r'].append(loss_r.item())
                history['train_v'].append(loss_v.item())
                history['train_f'].append(loss_fis.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), max_norm=1.0)
                self.optim.step()
                running_train += loss.item() * t_b.size(0)

            train_loss = running_train / len(self.train_loader.dataset)

            self.modelo.eval()
            running_val = 0

            for (t_b, pos_b), (r_b, v_b) in self.val_loader:
                t_b = t_b.to(self.disp).requires_grad_(True)
                pos_b = pos_b.to(self.disp)
                r_b = r_b.to(self.disp)
                v_b = v_b.to(self.disp)

                inp = torch.cat([t_b, pos_b.view(t_b.size(0), -1)], dim=1)
                r_pred = self.modelo(inp)
                loss_r = nn.MSELoss()(r_pred, r_b)

                v_pred = [torch.autograd.grad(
                    outputs=r_pred[:, i], inputs=t_b,
                    grad_outputs=torch.ones_like(r_pred[:, i]),
                    create_graph=True, retain_graph=True
                )[0] for i in range(r_pred.size(1))]
                v_pred = torch.cat(v_pred, dim=1)
                loss_v = nn.MSELoss()(v_pred, v_b)

                res = self.residuos_fisicos(t_b.squeeze(-1), pos_b)
                loss_fis = (res**2).mean()
                history['val_r'].append(loss_r.item())
                history['val_v'].append(loss_v.item())
                history['val_f'].append(loss_fis.item())

                loss = loss_r + loss_v + peso_fis * loss_fis
                running_val += loss.item() * t_b.size(0)

            val_loss = running_val / len(self.val_loader.dataset)
            self.scheduler.step(val_loss)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val:
                best_val, paciencia = val_loss, 0
                best_state = self.modelo.state_dict()
            else:
                paciencia += 1
                if paciencia >= early:
                    print(f"Early stopping en epoch {ep}")
                    break

            print(f"Ep {ep:4d} -> Train {train_loss:.3e}, Val {val_loss:.3e}")

        self.modelo.load_state_dict(best_state)

    def predecir(self, t_np, pos_todos_np):
        '''
        Realiza predicciones con el modelo entrenado a partir de tiempos y posiciones normalizadas.

        Parámetros
        ----------
            t_np : np.ndarray
                Tiempos en formato NumPy.
            pos_todos_np : np.ndarray
                Posiciones de todos los planetas en cada instante.

        Retorna
        -------
            r_pred : np.ndarray
                Posición predicha del objeto de interés en coordenadas físicas.
        '''
        t_s = self.scaler_t.transform(t_np.reshape(-1, 1))
        pos_s = self.scaler_pos.transform(pos_todos_np.reshape(len(pos_todos_np), -1)).reshape(len(pos_todos_np), self.num_planetas, 3)

        t_t = torch.from_numpy(t_s).float().to(self.disp)
        pos_t = torch.from_numpy(pos_s).float().to(self.disp)

        self.modelo.eval()
        with torch.no_grad():
            inp = torch.cat([t_t, pos_t.view(t_t.size(0), -1)], dim=1)
            r_s = self.modelo(inp).cpu()

        return self.inv_escalar_pos(r_s)

    def simular(self, inicio, dias_sim, paso_dias=1):
        '''
        Simula la trayectoria del objeto de interés a lo largo de varios días.

        Parámetros
        ----------
            inicio : str
                Fecha de inicio de la simulación (formato compatible con pandas).
            dias_sim : int
                Número total de días a simular.
            paso_dias : int
                Intervalo de tiempo entre predicciones.

        Retorna
        -------
            traj : np.ndarray
                Trayectoria simulada del objeto de interés en coordenadas físicas.
        '''
        t0 = (pd.to_datetime(inicio) - pd.to_datetime(self.scaler_t.data_min_[0])).days
        idx0 = int(np.argmin(np.abs(self.scaler_t.inverse_transform(self.t_s) - t0)))

        t_cur = self.scaler_t.inverse_transform(self.t_s[idx0:idx0 + 1].cpu().numpy())
        pos_todos0 = self.pos_s[idx0:idx0 + 1].unsqueeze(0).cpu().numpy()

        traj = []
        for _ in range(int(dias_sim // paso_dias)):
            r_pred = self.predecir(t_cur, pos_todos0)[0]
            traj.append(r_pred.copy())
            t_cur += paso_dias
            pos_todos0[0, self.id_obj, :] = self.scaler_r.transform(r_pred.reshape(1, 3))

        return np.array(traj)

    def entrenar_cv(self, k_folds=5, epochs=2000, peso_fis=1e3, early=200, batch_size=32):
        '''
        Entrena el modelo usando validación cruzada k-fold para evaluar su robustez.

        Parámetros
        ----------
            k_folds : int
                Número de particiones para la validación cruzada.
            epochs : int
                Número máximo de épocas por fold.
            peso_fis : float
                Peso del término de pérdida física.
            early : int
                Número de épocas sin mejora para aplicar early stopping.
            batch_size : int
                Tamaño de lote para los DataLoaders.

        Retorna
        -------
            historiales : list[dict]
                Lista de historiales de entrenamiento por cada fold.
        '''
        fold_size = len(self) // k_folds
        indices = np.arange(len(self))
        np.random.shuffle(indices)

        historiales = []

        for k in range(k_folds):
            val_idx = indices[k * fold_size : (k + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            train_ds = Subset(self, train_idx)
            val_ds = Subset(self, val_idx)

            self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            self.construir_red()
            self.optimizar_red(paciencia=early // 4, min_lr=1e-5)

            print(f"\n Fold {k+1}/{k_folds}")
            hist = self.entrenar_pinn(epochs=epochs, peso_fis=peso_fis, early=early)
            historiales.append(hist)

        return historiales

    def calcular_peso_fisico(self, muestra_t=100):
        '''
        Calcula un peso físico inverso proporcional a la aceleración promedio en una muestra.

        Parámetros
        ----------
            muestra_t : int
                Número de muestras aleatorias para estimar el peso físico.

        Retorna
        -------
            peso_fis : float
                Peso físico sugerido para balancear la pérdida durante el entrenamiento.
        '''
        idxs = np.random.choice(len(self), muestra_t, replace=False)
        t_sample = self.t_s[idxs].float().to(self.disp)
        pos_sample = self.pos_s[idxs].float().to(self.disp)

        if self.modelo is None:
            self.construir_red()

        try:
            res = self.residuos_fisicos(t_sample, pos_sample)
            norma_media = res.norm(dim=1).mean().item()
        except RuntimeError as e:
            print(f"t_sample.shape: {t_sample.shape}, pos_sample.shape: {pos_sample.shape}")
            raise e

        peso_fis = 1.0 / (norma_media + 1e-8)
        return peso_fis

    def simular_corregido(self, inicio, dias_sim, paso_dias=1):
        '''
        Simula la trayectoria del objeto de interés usando posiciones sin interpolación previa.

        Parámetros
        ----------
            inicio : str
                Fecha de inicio de la simulación.
            dias_sim : int
                Número total de días a simular.
            paso_dias : int
                Intervalo de tiempo entre predicciones.

        Retorna
        -------
            traj : np.ndarray
                Trayectoria simulada del objeto de interés en coordenadas físicas.
        '''
        t0 = (pd.to_datetime(inicio) - pd.to_datetime(self.scaler_t.data_min_[0])).days
        idx0 = int(np.argmin(np.abs(self.scaler_t.inverse_transform(self.t_s) - t0)))

        t_cur = self.scaler_t.inverse_transform(self.t_s[idx0:idx0 + 1].cpu().numpy())
        pos_todos0 = self.pos_s[idx0:idx0 + 1].cpu().numpy()

        traj = []
        for _ in range(int(dias_sim // paso_dias)):
            r_pred = self.predecir(t_cur, pos_todos0)[0]
            traj.append(r_pred.copy())
            t_cur += paso_dias
            pos_todos0[0, self.id_obj, :] = self.scaler_r.transform(r_pred.reshape(1, 3))

        return np.array(traj)