import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from scipy.stats import pareto
from .traficoparetopython_3 import simular_trafico

import numpy as np
import random

class RedesOpticasEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    
    def __init__(self, render_mode=None, seed=0, num_ont=16, TxRate=10e9, B_guaranteed=[], n_ciclos=200):

        print("Selecciona opciones del entorno:")
        print("1: SLA fijo (400 Mbps)")
        print("2: SLA aleatorio")
        self.SLA_seleccionado= input("Introduce el numero de la opción (1 o 2): ")
        self.num_ont = num_ont
        self.temp_ciclo = 0.002
        self.B_available = TxRate * self.temp_ciclo
        self.B_guaranteed = B_guaranteed.copy()
        self.B_max = np.array(B_guaranteed) * self.temp_ciclo

        self.B_guaranteed_change = np.array([random.choice([i / 10 for i in range(2000, 8000)]) for _ in range(self.num_ont)])

        self.max_queue = self.B_max*1.333333
        self.b_excess = 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_ont,), dtype=np.float64)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_ont,), dtype=np.float64)

        self.step_durations = []

        self.B_alloc = np.zeros(self.num_ont)
        self.trafico_entrada = np.zeros(self.num_ont)
        self.B_demand = np.zeros(self.num_ont)
        self.prev_demand = np.zeros(self.num_ont)

        self.rng = np.random.default_rng(seed)
        
        self.instantes=0
        self.n_ciclos=n_ciclos-1
        self.state = None
        self.onts = None
        self.instantes_cambio = np.array([random.choice([i / 10 for i in range(500, 4000)]) for _ in range(self.num_ont)])

    def _get_obs(self):
        return self.B_demand / self.max_queue

    def _get_info(self):
        return {
            "B_demand": self.B_demand.copy(),
            "B_alloc": self.B_alloc.copy(),
            "trafico_entrada": self.trafico_entrada.copy(),
            "prev_demand": self.prev_demand.copy(),
            "B_max": self.B_max.copy(),
            "B_available": self.B_available,
            "max_cola": self.max_queue.copy(),
        }

    def _calculate_reward(self):
        delta_cola = self.prev_demand - self.B_demand
        delta_cola_norm = np.mean(delta_cola / self.max_queue)
        
        uso_bw = np.sum(self.B_alloc) / np.sum(self.B_max)

        max_excess = np.sum(self.B_max) - self.B_available
        lim_olt = self.b_excess / max_excess
        self.b_excess = 0

        # Combine rewards with weights
        reward = (
            0.5 * delta_cola_norm +       # Aumentar prioridad de reducción de colas
            1.3 * uso_bw +                # Reducir peso de eficiencia general
            0.5 * lim_olt
        )

        return float(reward)  # Ensure reward is a scalar
    

    def step(self, action):
        start_time = time.time()

        #self.B_guaranteed = [600e6]* self.num_ont
        #self.B_max = np.array(self.B_guaranteed) * self.temp_ciclo

        for i in range(self.num_ont):
            if self.instantes >= self.instantes_cambio[i] and self.SLA_seleccionado == "2":
                self.B_guaranteed[i] = self.B_guaranteed_change[i] * 1e6
            elif self.instantes >= self.instantes_cambio[i] and self.SLA_seleccionado == "1":
                self.B_guaranteed[i] = 400e6
            else:
                self.B_guaranteed[i] = 600e6
    
        self.B_max = np.array(self.B_guaranteed) * self.temp_ciclo
                                        
        #if self.instantes >=  instantes_cambio[self.num_ont]:
            #self.B_guaranteed = [400e6] * self.num_ont
            #self.B_max = np.array(self.B_guaranteed) * self.temp_ciclo

        self.prev_demand = np.copy(self.B_demand)
        
        self.trafico_entrada_por_ciclo, self.onts = simular_trafico(self.onts)
        self.trafico_entrada = self.trafico_entrada_por_ciclo
        
        self.B_alloc = np.clip(action, 0, 1) * self.B_max

        # Comprobamos si el ancho de banda asignado excede el disponible
        total = np.sum(self.B_alloc)
        if total > self.B_available:
            self.b_excess = total - self.B_available
            self.B_alloc *= (self.B_available / total)

        # Procesamos el trafico de entrada y el asignado
        for i in range(self.num_ont):
            self.B_demand[i] += self.trafico_entrada[i]
            
            if self.B_alloc[i] > self.B_demand[i]:
                self.B_alloc[i] = self.B_demand[i]

            self.B_demand[i] -= self.B_alloc[i]

            self.B_demand[i] = np.clip(self.B_demand[i], 0, self.max_queue[i])
     
        reward = self._calculate_reward()

        if self.instantes==self.n_ciclos:
            done=True
        else:
            done=False

        self.instantes+=1

        end_time = time.time()
        step_duration = end_time - start_time
        self.step_durations.append(step_duration)

        info = self._get_info()

        return self._get_obs(), reward, done, False, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.n_ciclos = self.n_ciclos
        self.instantes = 0 

        self.onts = [None]
        self.trafico_entrada_por_ciclo, self.onts = simular_trafico()

        self.B_demand = np.zeros(self.num_ont)
        self.trafico_entrada = np.zeros(self.num_ont)
        self.B_alloc = np.zeros(self.num_ont)

        self.prev_demand = np.zeros(self.num_ont)
        self.b_excess = 0

        observation = self._get_obs()
        self.rng = np.random.default_rng(seed)
        info = self._get_info()
      

        return observation, info

from gymnasium.envs.registration import register

register(
    id="RedesOpticasEnv-v0",
    entry_point="custom_env.redes_opticas_env:RedesOpticasEnv",
)

