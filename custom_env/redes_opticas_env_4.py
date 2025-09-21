import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from scipy.stats import pareto
from .traficoparetopython_4 import simular_trafico
import itertools

class RedesOpticasEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, seed=0, num_ont=16, TxRate=10e9, B_guaranteed=[], n_ciclos=200):

        self.epsilon = 1e-9

        self.num_ont = num_ont
        self.num_guaranteed_ont = 6  
        self.temp_ciclo = 0.002  # 2 ms
        self.B_available = TxRate * self.temp_ciclo  

        self.B_max = np.zeros(self.num_ont)
        for i in range(self.num_ont):
            if i < self.num_guaranteed_ont:  
                self.B_max[i] = 600e6 * self.temp_ciclo
            else: 
                self.B_max[i] = 800e6 * self.temp_ciclo

        self.B_guaranteed_max = 800e6
        self.B_max_ONT2 = self.B_guaranteed_max * self.temp_ciclo
        self.B_min_ONT2 = self.B_guaranteed_max * self.temp_ciclo * 0.6

        self.max_queue = [1600000] * self.num_ont
        self.b_excess = 0

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_ont,), dtype=np.float64)
        #self.observation_space = spaces.MultiDiscrete([181] * self.num_ont)  # 181 niveles: 0 a 180
        self.action_space = spaces.MultiDiscrete([181] * self.num_ont)

        self.step_durations = []
        self.B_alloc = np.zeros(self.num_ont)
        self.trafico_entrada = np.zeros(self.num_ont)
        self.B_demand = np.zeros(self.num_ont)
        self.prev_demand = np.zeros(self.num_ont)

        self.rng = np.random.default_rng(seed)
        self.instantes = 0
        self.n_ciclos = n_ciclos - 1
        self.state = None
        self.onts = None
     
        self.lim_olt= 0.0
        

    def _get_obs(self):
        return self.B_demand / self.max_queue
    
    #def _get_obs(self):
        #clipped_demand = np.minimum(self.B_demand, self.B_max)
        #obs_discrete = (clipped_demand / self.B_max * 900) // 5
        #obs_discrete = np.clip(obs_discrete.astype(int), 0, 180)
        #return obs_discrete

    def _get_info(self):

        flex_allocs = []
        for i in range(self.num_guaranteed_ont, self.num_ont):
            if self.B_demand[i] > self.B_min_ONT2:
                flex_allocs.append(self.B_alloc[i])

        if len(flex_allocs) > 1 and np.sum(flex_allocs) > 0:
            flex_allocs = np.array(flex_allocs)
            fairness_index = (np.sum(flex_allocs) ** 2) / (len(flex_allocs) * np.sum(flex_allocs ** 2))
        else:
            fairness_index = 1.0
        return {
            "B_demand": self.B_demand.copy(),
            "B_alloc": self.B_alloc.copy(),
            "trafico_entrada": self.trafico_entrada.copy(),
            "prev_demand": self.prev_demand.copy(),
            "B_max": self.B_max.copy(),
            "B_available": self.B_available,
            "max_cola": self.max_queue.copy(),
            "jfi_flex": fairness_index
        }
    
    def _calculate_reward(self):

        epsilon = 1e-9  
        total = np.sum(self.B_alloc)

        if total > self.B_available + self.epsilon:
            self.b_excess = total - self.B_available
        else:
            self.b_excess = epsilon

        # 1. Penalización por exceso de OLT
        if self.b_excess > epsilon:     
            self.lim_olt = 0.0  
        else:
            self.lim_olt= 1.0

        # 2. ONTs garantizadas
        guaranteed_ont_reward = 0

        for i in range(self.num_guaranteed_ont):

            if self.B_demand[i] > self.epsilon:
                clipped_demand = min(self.B_demand[i], self.B_max[i])

                if self.B_alloc[i] > clipped_demand * (0.99 - self.epsilon):
                    guaranteed_ont_reward += min(self.B_alloc[i], clipped_demand) / (clipped_demand + self.epsilon)

        # 3. ONTs flexibles
        flex_ont_reward = 0
        flex_ont_min_penalty = 0

        for i in range(self.num_guaranteed_ont, self.num_ont):
            
            clipped_demand = min(self.B_demand[i], self.B_max[i])

            if self.B_demand[i] > epsilon:
                flex_ont_reward += min(self.B_alloc[i], clipped_demand) / (clipped_demand + self.epsilon)

            if self.B_demand[i] > self.B_min_ONT2 + epsilon and self.B_alloc[i] < self.B_min_ONT2 - self.epsilon:
                flex_ont_min_penalty += 1.0
            if self.B_demand[i] < self.B_min_ONT2 - epsilon and self.B_alloc[i] < self.B_demand[i] * 0.002 - self.epsilon:
                flex_ont_min_penalty += 1.0

        # 4. Índice JFI con selección iterativa (water-filling)

        def water_filling_onts():

            demands = [min(d, m) for d, m in zip(self.B_demand, self.B_max)]
            flex_indices = [i for i in range(self.num_guaranteed_ont, self.num_ont) if self.B_demand[i] > self.B_min_ONT2]
            remaining = set(flex_indices)
            available = np.sum(self.B_alloc[self.num_guaranteed_ont:])
            while remaining:
                if not remaining:
                    break
                quota = available / len(remaining)
                low_demand = [i for i in remaining if demands[i] < quota]
                if not low_demand:
                    break
                for i in low_demand:
                    available -= demands[i]
                    remaining.remove(i)
                

            return list(remaining)

        competing = water_filling_onts()
        flex_allocs = [self.B_alloc[i] for i in competing]

        if len(flex_allocs) > 1 and np.sum(flex_allocs) > self.epsilon:

            flex_allocs = np.array(flex_allocs)
            fairness_index = (np.sum(flex_allocs)**2) / (len(flex_allocs) * np.sum(flex_allocs**2) + self.epsilon)
        else:
            fairness_index = 1.0

        guaranteed_ont_reward = guaranteed_ont_reward/self.num_guaranteed_ont    
        flex_ont_min_penalty = flex_ont_min_penalty/(self.num_ont - self.num_guaranteed_ont)     
        flex_ont_reward = flex_ont_reward/(self.num_ont - self.num_guaranteed_ont)      

        # 5. Recompensa final
        reward = (
              2.1  * self.lim_olt
            + 2.0  * guaranteed_ont_reward     
            - 1.4  * flex_ont_min_penalty      
            + 0.25 * flex_ont_reward           
            + 2.0  * fairness_index             
        )

        return float(reward)

    def step(self, action):
        start_time = time.time()

        self.prev_demand = np.copy(self.B_demand)

        self.trafico_entrada_por_ciclo, self.onts = simular_trafico(self.onts)
        self.trafico_entrada = self.trafico_entrada_por_ciclo

        action = np.array(action)
        # Cada acción representa un múltiplo de 100 (0, 100, ..., 900)
        action_real = action * 5 / 900  # Normaliza a [0, 1]
        self.B_alloc = np.clip(action_real, self.epsilon, 1) * self.B_max


        for i in range(self.num_ont):
          
            self.B_demand[i] += self.trafico_entrada[i]

            self.B_alloc[i] = min(self.B_alloc[i], self.B_demand[i])
            self.B_demand[i] -= self.B_alloc[i]
            self.B_demand[i] = np.clip(self.B_demand[i], self.epsilon, self.max_queue[i])

        reward = self._calculate_reward()
        done = self.instantes == self.n_ciclos
        self.instantes += 1

        self.step_durations.append(time.time() - start_time)

        return self._get_obs(), reward, done, False, self._get_info()    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.n_ciclos = self.n_ciclos
        self.instantes = 0

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
