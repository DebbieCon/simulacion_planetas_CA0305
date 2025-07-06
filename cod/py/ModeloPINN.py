import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from Datos import Datos
from sklearn.model_selection import KFold
from copy import deepcopy
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModeloPINN(Datos):
    def __init__(self, ruta, planetas, fecha = 'fecha', device='cpu'):
        super().__init__(ruta, fecha, planetas=planetas)

        self._disp = torch.device(device)
        self._modelo = None
        self._optim = None
        self._scheduler = None
        self._train_loader = None
        self._val_loader = None
        self._G = 1.036e-12
        self._masas = torch.tensor(self.masas_planetas_dic(), dtype=torch.float32, device=self._disp).view(1, -1, 1)

    # --- Getters y setters ---
    @property
    def modelo(self): return self._modelo
    @modelo.setter
    def modelo(self, red): self._modelo = red

    @property
    def optim(self): return self._optim
    @optim.setter
    def optim(self, optimizador): self._optim = optimizador

    @property
    def scheduler(self): return self._scheduler
    @scheduler.setter
    def scheduler(self, sched): self._scheduler = sched

    @property
    def disp(self): return self._disp

    @property
    def train_loader(self): return self._train_loader
    @train_loader.setter
    def train_loader(self, loader): self._train_loader = loader

    @property
    def val_loader(self): return self._val_loader
    @val_loader.setter
    def val_loader(self, loader): self._val_loader = loader

    @property
    def G(self): return self._G

    @property
    def masas(self): return self._masas

    # --- Métodos funcionales ---
    def construir_red(self, neuronas=None):
        if neuronas is None:
            neuronas = [1 + self.num_planetas * 3, 64, 64, 3]

        capas = []
        for i in range(len(neuronas) - 2):
            capas += [nn.Linear(neuronas[i], neuronas[i+1]), nn.SiLU()]
        capas.append(nn.Linear(neuronas[-2], neuronas[-1]))
        self.modelo = nn.Sequential(*capas).to(self.disp)

    def optimizar_red(self, paciencia, min_lr):
        self.optim = optim.Adam(self.modelo.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, factor=0.5, patience=paciencia, min_lr=min_lr
        )

    def particionar_datos(self, tam_test=0.2, batch_size=32):
        n_test = int(self.num_datos * tam_test)
        n_train = self.num_datos - n_test
        train_ds, val_ds = random_split(self, [n_train, n_test])

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    def residuos_fisicos(self, t, pos_planetas):
        t = t.unsqueeze(-1).requires_grad_(True)
        inp = torch.cat([t, pos_planetas.view(t.size(0), -1)], dim=1)
        r_pred = self.modelo(inp)
        v_pred = torch.autograd.grad(r_pred, t, torch.ones_like(r_pred), create_graph=True)[0]
        a_pred = torch.autograd.grad(v_pred, t, torch.ones_like(v_pred), create_graph=True)[0]

        diffs = pos_planetas - r_pred.unsqueeze(1)
        dist3 = torch.norm(diffs, dim=2, keepdim=True) ** 3 + 1e-9
        a_fis = (self.G * self.masas * diffs / dist3).sum(dim=1)

        return a_pred - a_fis

    def entrenar_pinn(self, epochs=2000, peso_fis=1e3, early=200):
        best_val, paciencia = np.inf, 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_r': [],
            'train_v': [],
            'train_f': [],
            'val_r': [],
            'val_v': [],
            'val_f': []
        }

        for ep in range(1, epochs + 1):
            self.modelo.train()
            running_train = 0

            for (t_b, pos_b), (r_b, v_b) in self.train_loader:
                t_b = t_b.to(self.disp)
                pos_b = pos_b.to(self.disp)
                r_b = r_b.to(self.disp)
                v_b = v_b.to(self.disp)


                t_b = t_b.requires_grad_(True)

                self.optim.zero_grad()

                inp = torch.cat([t_b, pos_b.view(t_b.size(0), -1)], dim=1)
                r_pred = self.modelo(inp)

                loss_r = nn.MSELoss()(r_pred, r_b)

                v_pred = []

                for i in range(r_pred.size(1)):  # para cada coordenada x, y, z
                    grad_i = torch.autograd.grad(
                        outputs=r_pred[:, i],
                        inputs=t_b,
                        grad_outputs=torch.ones_like(r_pred[:, i]),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    v_pred.append(grad_i)

                v_pred = torch.cat(v_pred, dim=1)  # ahora v_pred tiene shape [batch, 3]
                loss_v = nn.MSELoss()(v_pred, v_b)

                res = self.residuos_fisicos(t_b.squeeze(-1), pos_b)
                loss_fis = (res**2).mean()

                loss = loss_r + loss_v + peso_fis * loss_fis

                # Dentro del loop de entrenamiento
                history['train_r'].append(loss_r.item())
                history['train_v'].append(loss_v.item())
                history['train_f'].append(loss_fis.item())

                loss.backward()
                self.optim.step()
                running_train += loss.item() * t_b.size(0)


            train_loss = running_train / len(self.train_loader.dataset)

            self.modelo.eval()
            running_val = 0

            for (t_b, pos_b), (r_b, v_b) in self.val_loader:
                t_b = t_b.to(self.disp)
                pos_b = pos_b.to(self.disp)
                r_b = r_b.to(self.disp)
                v_b = v_b.to(self.disp)

                t_b = t_b.requires_grad_(True)

                inp = torch.cat([t_b, pos_b.view(t_b.size(0), -1)], dim=1)
                r_pred = self.modelo(inp)

                loss_r = nn.MSELoss()(r_pred, r_b)
                
                v_pred = []

                for i in range(r_pred.size(1)):  # para cada coordenada x, y, z
                    grad_i = torch.autograd.grad(
                        outputs=r_pred[:, i],
                        inputs=t_b,
                        grad_outputs=torch.ones_like(r_pred[:, i]),
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    v_pred.append(grad_i)

                v_pred = torch.cat(v_pred, dim=1)  # ahora v_pred tiene shape [batch, 3]

                loss_v = nn.MSELoss()(v_pred, v_b)

                res = self.residuos_fisicos(t_b.squeeze(-1), pos_b)
                loss_fis = (res**2).mean()
                # Dentro del bloque de evaluación (justo antes de acumular running_val)
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

        with torch.no_grad():
            all_true, all_pred = [], []

            for (t_b, pos_b), (r_b, _) in self.val_loader:
                t_b = t_b.to(self.disp)
                pos_b = pos_b.to(self.disp)
                r_b = r_b.to(self.disp)

                inp = torch.cat([t_b, pos_b.view(t_b.size(0), -1)], dim=1)
                pred = self.modelo(inp)

                all_true.append(r_b)
                all_pred.append(pred)

            r_true = torch.cat(all_true, dim=0)
            r_pred = torch.cat(all_pred, dim=0)
            history['metrics'] = self.calcular_metricas(r_true, r_pred)
            for nombre, valor in history['metrics'].items():
                print(f"{nombre}: {valor:.5f}")

        return history
    
    def predecir(self, t_np, pos_todos_np):

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
    
    def validacion_cruzada(self, k=5, epochs=1000, batch_size=32, early=100):

        historiales = []
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        idxs = np.arange(self.num_datos)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(idxs)):
            print(f"\nFold {fold+1}/{k}")
            train_set = torch.utils.data.Subset(self, train_idx)
            val_set = torch.utils.data.Subset(self, val_idx)

            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

            self.construir_red()
            self.optimizar_red(paciencia=20, min_lr=1e-5)
            hist = self.entrenar_pinn(epochs=epochs, peso_fis=0.0, early=early)

            historiales.append(deepcopy(hist))

        return historiales
    
    def refinar_con_fisica(self, epochs=1000, peso_fis=1e3, early=200):
        self.optimizar_red(paciencia=30, min_lr=1e-6)
        hist = self.entrenar_pinn(epochs=epochs, peso_fis=peso_fis, early=early)
        return hist
    
    def simular_objeto(self, fecha_inicio, r0_xyz, id_obj, pasos=200, paso_dias=1):
        t0 = (pd.to_datetime(fecha_inicio) - pd.to_datetime(self.scaler_t.data_min_[0])).days
        t0_norm = self.scaler_t.transform([[t0]])
        r0_norm = self.scaler_r.transform(r0_xyz.reshape(1, 3))

        t_actual = torch.tensor(t0_norm, dtype=torch.float32, device=self.disp)
        pos_cur = self.pos_s[0:1].clone().to(self.disp)  # dummy base
        pos_cur[:, id_obj, :] = torch.tensor(r0_norm, dtype=torch.float32, device=self.disp)

        trayectoria = []
        for _ in range(pasos):
            with torch.no_grad():
                entrada = torch.cat([t_actual, pos_cur.view(1, -1)], dim=1)
                r_pred = self.modelo(entrada)
                r_np = self.inv_escalar_pos(r_pred).squeeze()
                trayectoria.append(r_np)

            r_pred_norm = self.scaler_r.transform(r_np.reshape(1, 3))
            pos_cur[:, id_obj, :] = torch.tensor(r_pred_norm, dtype=torch.float32, device=self.disp)
            t_actual += paso_dias

        return np.array(trayectoria)
    
    def calcular_metricas(self,y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        r2 = r2_score(y_true, y_pred)

        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

