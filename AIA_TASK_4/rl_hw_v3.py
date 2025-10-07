import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, List, Optional
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces

class PoleEnvironmentGymnasium(gym.Env):
    """
    Custom Gymnasium Environment pro Pole prostředí
    Agent se musí naučit dosáhnout cílové pozice a vyhnout se překážkám
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, size: int = 8, 
                 obstacles: Optional[List[Tuple[int, int]]] = None, 
                 goal: Optional[Tuple[int, int]] = None,
                 render_mode: Optional[str] = None):
        super(PoleEnvironmentGymnasium, self).__init__()
        
        self.size = size
        self.obstacles = obstacles or [(2, 2), (3, 3), (4, 4), (5, 2), (6, 5)]
        self.goal = goal or (size-1, size-1)
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        self.render_mode = render_mode
        
        # Akce: 0=nahoru, 1=doprava, 2=dolů, 3=doleva
        self.action_space = spaces.Discrete(4)
        
        # Observation space: pozice (x, y)
        self.observation_space = spaces.Box(
            low=0, 
            high=size-1, 
            shape=(2,), 
            dtype=np.int32
        )
        
        # Rewards
        self.goal_reward = 100
        self.obstacle_penalty = -50
        self.move_penalty = -1
        self.out_of_bounds_penalty = -10
        
        self.actions_map = {0: 'nahoru ↑', 1: 'doprava →', 2: 'dolů ↓', 3: 'doleva ←'}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Resetuje prostředí podle Gymnasium API"""
        super().reset(seed=seed)
        self.current_pos = self.start_pos
        observation = np.array(self.current_pos, dtype=np.int32)
        info = {}
        return observation, info
    
    def step(self, action: int):
        """
        Provede akci v prostředí podle Gymnasium API
        Returns: (observation, reward, terminated, truncated, info)
        """
        x, y = self.current_pos
        
        # Aplikuj akci
        if action == 0:  # nahoru
            new_pos = (x, max(0, y-1))
        elif action == 1:  # doprava
            new_pos = (min(self.size-1, x+1), y)
        elif action == 2:  # dolů
            new_pos = (x, min(self.size-1, y+1))
        elif action == 3:  # doleva
            new_pos = (max(0, x-1), y)
        else:
            raise ValueError(f"Neplatná akce: {action}")
        
        # Výpočet reward a terminated flag
        terminated = False
        truncated = False
        
        if new_pos == self.current_pos and action in [0, 1, 2, 3]:
            # Agent narazil na hranici
            reward = self.out_of_bounds_penalty
        elif new_pos in self.obstacles:
            # Agent narazil na překážku
            reward = self.obstacle_penalty
            terminated = True
        elif new_pos == self.goal:
            # Agent dosáhl cíle
            reward = self.goal_reward
            terminated = True
        else:
            # Normální pohyb
            reward = self.move_penalty
        
        self.current_pos = new_pos
        observation = np.array(self.current_pos, dtype=np.int32)
        info = {'action_name': self.actions_map[action]}
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Vykreslí prostředí"""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self):
        """Textové vykreslení"""
        grid = {}
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) == self.current_pos:
                    grid[(x, y)] = '🤖'
                elif (x, y) == self.goal:
                    grid[(x, y)] = '🎯'
                elif (x, y) in self.obstacles:
                    grid[(x, y)] = '🚫'
                else:
                    grid[(x, y)] = '⬜'
        
        print("\n" + "="*50)
        print("  ", end="")
        for x in range(self.size):
            print(f"{x:2}", end="")
        print()
        
        for y in range(self.size):
            print(f"{y} ", end="")
            for x in range(self.size):
                print(f"{grid[(x, y)]} ", end="")
            print()
        print(f"Pozice: {self.current_pos}")
    
    def _render_rgb_array(self):
        """Vrátí RGB array pro vizualizaci"""
        grid = np.zeros((self.size, self.size))
        
        # Překážky - červená
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = 2
        
        # Cíl - modrá
        grid[self.goal[1], self.goal[0]] = -2
        
        # Agent - bílá
        grid[self.current_pos[1], self.current_pos[0]] = 0.5
        
        return grid
    
    def get_state_index(self, pos: Optional[Tuple[int, int]] = None) -> int:
        """Převede pozici na index stavu pro Q-table"""
        if pos is None:
            pos = self.current_pos
        return pos[1] * self.size + pos[0]
    
    def visualize_policy(self, q_table: np.ndarray):
        """Vizualizuje naučenou politiku"""
        grid = self._render_rgb_array()
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(grid, cmap='coolwarm', vmin=-2, vmax=2, alpha=0.7)
        
        # Přidej šipky pro nejlepší akce
        arrow_props = {'head_width': 0.2, 'head_length': 0.2, 'fc': 'black', 'ec': 'black', 'alpha': 0.8, 'linewidth': 1.5}
        
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in self.obstacles and (x, y) != self.goal and (x, y) != self.start_pos:
                    state_idx = self.get_state_index((x, y))
                    best_action = np.argmax(q_table[state_idx])
                    
                    dx, dy = 0, 0
                    if best_action == 0:  # nahoru
                        dy = -0.35
                    elif best_action == 1:  # doprava
                        dx = 0.35
                    elif best_action == 2:  # dolů
                        dy = 0.35
                    elif best_action == 3:  # doleva
                        dx = -0.35
                    
                    ax.arrow(x, y, dx, dy, **arrow_props)
        
        # Zvýrazni START - zelený čtverec
        start_x, start_y = self.start_pos
        ax.add_patch(plt.Rectangle((start_x-0.4, start_y-0.4), 0.8, 0.8, 
                                   fill=False, edgecolor='lime', linewidth=4))
        ax.text(start_x, start_y-0.6, '★ START', ha='center', va='top', 
               fontsize=12, fontweight='bold', color='lime',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))
        
        # Zvýrazni CÍL - žlutý čtverec
        goal_x, goal_y = self.goal
        ax.add_patch(plt.Rectangle((goal_x-0.4, goal_y-0.4), 0.8, 0.8, 
                                   fill=False, edgecolor='yellow', linewidth=4))
        ax.text(goal_x, goal_y+1.5, '★ CÍL', ha='center', va='bottom', 
               fontsize=12, fontweight='bold', color='yellow',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='darkblue', alpha=0.7))
        
        ax.set_title('Naučená Politika - Q-Learning Agent', fontsize=18, fontweight='bold', pad=20)
        
        # Legenda s barevnými značkami
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                      markersize=15, label='🚫 Překážky'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                      markersize=15, label='🎯 Cíl'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lime', 
                      markersize=15, label='★ Start'),
            plt.Line2D([0], [0], marker='>', color='black', markersize=15, 
                      label='Nejlepší akce', linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
        
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_xlabel('X souřadnice', fontsize=12)
        ax.set_ylabel('Y souřadnice', fontsize=12)
        plt.tight_layout()
        plt.show()

class QLearningAgent:
    """Q-Learning agent kompatibilní s Gymnasium"""
    
    def __init__(self, state_space: int, action_space: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, 
                 epsilon_min: float = 0.01):
        
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Inicializuj Q-table
        self.q_table = np.zeros((state_space, action_space))
    
    def choose_action(self, state: int) -> int:
        """Epsilon-greedy strategie"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state: int, action: int, reward: float, next_state: int, terminated: bool):
        """Q-Learning update"""
        current_q = self.q_table[state, action]
        
        if terminated:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-Learning update
        self.q_table[state, action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent_gymnasium(env: PoleEnvironmentGymnasium, agent: QLearningAgent, 
                         episodes: int = 1000) -> List[float]:
    """Trénuje agenta v Gymnasium prostředí"""
    rewards_per_episode = []
    
    print(f"\n🎓 Začíná trénink na {episodes} epizod...")
    
    for episode in range(episodes):
        observation, info = env.reset()
        state = env.get_state_index(tuple(observation))
        total_reward = 0
        steps = 0
        max_steps = 200
        
        while steps < max_steps:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = env.get_state_index(tuple(observation))
            
            agent.learn(state, action, reward, next_state, terminated)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        rewards_per_episode.append(total_reward)
        
        if episode % 200 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:]) if len(rewards_per_episode) >= 100 else np.mean(rewards_per_episode)
            print(f"Episode {episode:4d} | Reward: {total_reward:6.1f} | Avg(100): {avg_reward:6.1f} | ε: {agent.epsilon:.3f} | Steps: {steps}")
    
    return rewards_per_episode

def test_agent_gymnasium(env: PoleEnvironmentGymnasium, agent: QLearningAgent, 
                        episodes: int = 5, render: bool = True):
    """Testuje agenta v Gymnasium prostředí"""
    print(f"\n🧪 Testování agenta na {episodes} epizodách...")
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Vypni exploration
    
    success_count = 0
    total_rewards = []
    
    for episode in range(episodes):
        observation, info = env.reset()
        state = env.get_state_index(tuple(observation))
        total_reward = 0
        steps = 0
        max_steps = 100
        
        print(f"\n--- Test Episode {episode + 1} ---")
        
        while steps < max_steps:
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            next_state = env.get_state_index(tuple(observation))
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if render and steps % 5 == 0:
                env.render()
            
            if terminated or truncated:
                if tuple(observation) == env.goal:
                    success_count += 1
                    print(f"✅ SUCCESS! Dosažen cíl za {steps} kroků")
                else:
                    print(f"❌ FAIL! Naražení do překážky")
                break
        else:
            print(f"⏰ TIMEOUT po {max_steps} krocích")
        
        total_rewards.append(total_reward)
        print(f"Reward: {total_reward:.1f}")
    
    agent.epsilon = original_epsilon
    
    success_rate = success_count / episodes
    avg_reward = np.mean(total_rewards)
    
    print(f"\n📊 VÝSLEDKY TESTOVÁNÍ:")
    print(f"Success Rate: {success_rate:.1%} ({success_count}/{episodes})")
    print(f"Průměrný Reward: {avg_reward:.1f}")
    
    return avg_reward

def plot_training_results(rewards: List[float]):
    """Vykreslí výsledky tréninku"""
    plt.figure(figsize=(14, 5))
    
    # Rewards per episode
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6)
    plt.title('Rewards per Episode', fontsize=14)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    
    # Moving average
    plt.subplot(1, 2, 2)
    window_size = 50
    moving_avg = []
    for i in range(len(rewards)):
        start = max(0, i - window_size + 1)
        moving_avg.append(np.mean(rewards[start:i+1]))
    
    plt.plot(moving_avg, color='green', linewidth=2)
    plt.title(f'Moving Average (window={window_size})', fontsize=14)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def visualize_q_values(agent: QLearningAgent, env: PoleEnvironmentGymnasium):
    """Vizualizuje Q-values pro každou akci"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    action_names = ['Nahoru ↑', 'Doprava →', 'Dolů ↓', 'Doleva ←']
    
    for idx, ax in enumerate(axes.flat):
        q_values = agent.q_table[:, idx].reshape(env.size, env.size)
        
        sns.heatmap(q_values, annot=True, fmt='.1f', cmap='viridis', 
                   square=True, cbar_kws={'label': 'Q-value'}, ax=ax)
        ax.set_title(f'Q-values pro akci: {action_names[idx]}', fontsize=12)
        ax.set_xlabel('X pozice')
        ax.set_ylabel('Y pozice')
    
    plt.tight_layout()
    plt.show()

# HLAVNÍ PROGRAM
if __name__ == "__main__":
    print("🚀 POLE ENVIRONMENT S GYMNASIUM A Q-LEARNING")
    print("="*60)
    
    # Vytvoř Gymnasium prostředí s polem 10x10
    env = PoleEnvironmentGymnasium(
        size=10,
        obstacles=[
            # Horní levá sekce
            (2, 2), (3, 2), (2, 3),
            # Vertikální zeď
            (5, 1), (5, 2), (5, 3), (5, 4),
            # Horizontální bariéra
            (1, 5), (2, 5), (3, 5),
            # Bloky rozmístěné
            (7, 3), (8, 3),
            (3, 7), (4, 7),
            (7, 6), (8, 6), (7, 7),
            # Dolní překážky
            (1, 8), (2, 8),
            (5, 9), (6, 9)
        ],
        goal=(9, 9),
        render_mode='human'
    )
    
    print(f"🌍 Prostředí: {env.size}x{env.size} grid")
    print(f"📦 Observation space: {env.observation_space}")
    print(f"🎮 Action space: {env.action_space}")
    
    # Vytvoř agenta
    state_space = env.size * env.size
    agent = QLearningAgent(
        state_space=state_space,
        action_space=env.action_space.n,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    print(f"🤖 Q-Learning Agent s {state_space} stavy a {env.action_space.n} akcemi\n")
    
    # Zobraz počáteční prostředí
    print("📋 Počáteční prostředí:")
    env.reset()
    env.render()
    
    # Trénink - 2000 epizod pro 10x10 prostředí
    episodes = 2000
    rewards = train_agent_gymnasium(env, agent, episodes=episodes)
    
    # Zobraz výsledky tréninku
    print(f"\n📊 Výsledky tréninku po {episodes} epizodách:")
    print(f"Nejlepší reward: {max(rewards):.1f}")
    print(f"Průměrný reward (posledních 100): {np.mean(rewards[-100:]):.1f}")
    
    plot_training_results(rewards)
    
    # Zobraz naučenou politiku
    print("\n🗺️ Naučená politika:")
    env.visualize_policy(agent.q_table)
    
    # Vizualizuj Q-values
    print("\n🔥 Vizualizace Q-values:")
    visualize_q_values(agent, env)
    
    # Testuj agenta
    test_agent_gymnasium(env, agent, episodes=3, render=False)
    
    # Zavři prostředí
    env.close()
    
    print(f"\n✅ Program dokončen!")
    print(f"📈 Final epsilon: {agent.epsilon:.3f}")