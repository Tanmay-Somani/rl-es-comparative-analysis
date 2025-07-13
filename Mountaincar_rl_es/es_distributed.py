# es_distributed.py
import numpy as np
import gymnasium as gym
import multiprocessing as mp
import time
import psutil
from tqdm import tqdm

# This helper class remains the same as in your original code
class SGD:
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v

# This worker function must be at the top level for multiprocessing
def get_reward(config, net_shapes, params, seed_and_id):
    env = gym.make(config['game'])
    seed, k_id = seed_and_id
    
    np.random.seed(seed)
    sign = -1. if k_id % 2 == 0 else 1.
    params += sign * config['sigma'] * np.random.randn(params.size)
    
    p, start = [], 0
    for shape in net_shapes:
        n_w, n_b = shape[0] * shape[1], shape[1]
        p.append(params[start: start + n_w].reshape(shape))
        p.append(params[start + n_w: start + n_w + n_b].reshape((1, shape[1])))
        start += n_w + n_b

    s, _ = env.reset()
    ep_r = 0.
    for _ in range(config['ep_max_step']):
        x = s[np.newaxis, :]
        x = np.tanh(x.dot(p[0]) + p[1])
        x = np.tanh(x.dot(p[2]) + p[3])
        x = x.dot(p[4]) + p[5]
        a = np.argmax(x, axis=1)[0]
        s, r, terminated, truncated, _ = env.step(a)
        ep_r += r
        if terminated or truncated:
            break
    env.close()
    return ep_r

class DistributedESAgent:
    def __init__(self, config, n_individuals, learning_rate):
        self.config = config
        self.n_individuals = n_individuals
        self.net_shapes, self.net_params = self._build_net()
        self.optimizer = SGD(self.net_params, learning_rate)
        
        base = n_individuals
        rank = np.arange(1, base + 1)
        util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
        self.utility = util_ / util_.sum() - 1 / base

    def _build_net(self):
        def linear(n_in, n_out):
            w = np.random.randn(n_in * n_out).astype(np.float32) * 0.1
            b = np.random.randn(n_out).astype(np.float32) * 0.1
            return (n_in, n_out), np.concatenate((w, b))
        s0, p0 = linear(self.config['n_feature'], 30)
        s1, p1 = linear(30, 20)
        s2, p2 = linear(20, self.config['n_action'])
        return [s0, s1, s2], np.concatenate((p0, p1, p2))

    def train(self, n_generations, log_data):
        start_time = time.time()
        process = psutil.Process()
        
        n_cores = max(1, mp.cpu_count() - 1)
        with mp.Pool(processes=n_cores) as pool:
            for _ in tqdm(range(n_generations), desc="ES Distributed"):
                noise_seed = np.random.randint(0, 2**32 - 1, size=self.n_individuals // 2, dtype=np.uint32).repeat(2)
                
                jobs = [pool.apply_async(get_reward, (self.config, self.net_shapes, self.net_params.copy(), [noise_seed[k_id], k_id])) for k_id in range(self.n_individuals)]
                rewards = np.array([j.get() for j in jobs])
                
                kids_rank = np.argsort(rewards)[::-1]
                cumulative_update = np.zeros_like(self.net_params)
                for ui, k_id in enumerate(kids_rank):
                    np.random.seed(noise_seed[k_id])
                    sign = -1. if k_id % 2 == 0 else 1.
                    cumulative_update += self.utility[ui] * sign * np.random.randn(self.net_params.size)
                
                gradients = self.optimizer.get_gradients(cumulative_update / (self.n_individuals * self.config['sigma']))
                self.net_params += gradients

                log_data['es_distributed']['cpu_usage'].append(process.cpu_percent())
                log_data['es_distributed']['memory_usage'].append(process.memory_info().rss / (1024 * 1024))
                log_data['es_distributed']['training_time'].append(time.time() - start_time)
                log_data['es_distributed']['convergence_speed'].append(rewards.mean())
        return log_data

def run_es_distributed(log_data):
    """Entry point to set up and run the Distributed ES benchmark."""
    print("\nRunning Distributed Evolutionary Strategies...")
    GAME_CONFIG = {
        "game": "MountainCar-v0", "n_feature": 2, "n_action": 3, 
        "ep_max_step": 200, "sigma": 0.05,
    }
    agent = DistributedESAgent(GAME_CONFIG, n_individuals=20, learning_rate=0.05)
    log_data = agent.train(n_generations=100, log_data=log_data)
    return log_data