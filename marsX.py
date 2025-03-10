import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import agentpy as ap

# --- Parámetros del modelo ---
SPAWN_RATE_EXPLORER = 10
GRID_SIZE = 20
PEATON_NUMBER = 10
DOME_NUMBER = 5
N_STEPS = 50
DAY_NIGHT_TIME = 10
NUMBER_OF_DAY_NIGHT_CYCLES = DAY_NIGHT_TIME / N_STEPS 

# --- 1. Define la tabla de color-a-número ---
color_mapping = {
    (255, 0, 0): 1,     # Building (Red: FF0000)
    (255, 180, 0): 2,   # Street (Orange: FFB400)
    (0, 255, 38): 3,    # Sidewalk (Green: 00FF26)
    (255, 0, 157): 4,   # Crosswalk (Pink: FF009D)
    (0, 49, 255): 5,    # Traffic Light (Blue: 0031FF)
    (0, 255, 253): 6,   # Building/Dome Spawn Area (Cyan: 00FFFD)
    (0, 0, 0): 7        # Agent Spawn Points (Black: 000000)
}

# --- 2. Carga la imagen base de la ciudad ---
image_path = "citygrid.png"  # Ajusta la ruta si es necesario
image = Image.open(image_path).convert("RGB")
image_array = np.array(image)

# --- 3. Crea la grilla base ---
base_grid = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=int)
for i in range(image_array.shape[0]):
    for j in range(image_array.shape[1]):
        pixel = tuple(image_array[i, j])
        base_grid[i, j] = color_mapping.get(pixel, 0)

# --- 4. Expande la grilla base ---
def expand_city_grid(base, repetitions_x=2, repetitions_y=2):
    return np.tile(base, (repetitions_y, repetitions_x))

city_grid = expand_city_grid(base_grid, repetitions_x=GRID_SIZE, repetitions_y=GRID_SIZE)
city_size = city_grid.shape


# ============= AGENTES =============

class MovingAgent(ap.Agent):
    def setup(self):
        valid_positions = np.argwhere(city_grid == 7)
        if len(valid_positions) == 0:
            raise ValueError("No valid spawn points (7) found!")
        self.x, self.y = map(int, valid_positions[np.random.randint(len(valid_positions))])
        self.oxigenLevel = 20
        self.dome_id = None

    def step(self):
        step_size = 6
        directions = [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]
        np.random.shuffle(directions)

        for dx, dy in directions:
            new_x, new_y = self.x + dx, self.y + dy
            if (0 <= new_x < city_size[0] and 0 <= new_y < city_size[1] and
                all(city_grid[self.x + i * np.sign(dx), self.y + i * np.sign(dy)] in (2, 4, 7)
                    for i in range(1, step_size + 1))):
                self.x, self.y = new_x, new_y
                return

    def action(self):
        if self.oxigenLevel > 0:
            self.oxigenLevel -= 1  # Reduce oxygen by 1 per step

class Dome(ap.Agent):
    """
    - Spawnea en celdas 6 (Cyan).
    - Cada step crea un MovingAgent.
    - Desaparece si todos sus explorers tienen oxígeno 0.
    """
    def setup(self):
        self.spawned_explorers = []
        self.spawn_rate = SPAWN_RATE_EXPLORER
        valid_positions = np.argwhere(city_grid == 6)
        if len(valid_positions) == 0:
            raise ValueError("No valid dome spawn points (6) found!")
        px = valid_positions[np.random.randint(len(valid_positions))]
        self.x, self.y = map(int, px)

    def action(self):
        new_explorers = []
        for _ in range(self.spawn_rate):
            explorer = MovingAgent(self.model)
            explorer.dome_id = self.id
            self.spawned_explorers.append(explorer)
            new_explorers.append(explorer)
            self.model.agents.append(explorer)
            self.model.all_agents.append(explorer)

            # Spawnear al explorer en celdas 7 (manualmente)
            valid_positions = np.argwhere(city_grid == 7)
            px = valid_positions[np.random.randint(len(valid_positions))]
            explorer.x, explorer.y = map(int, px)

        # Si todos sus explorers tienen oxígeno <= 0 => remove dome
        if self.spawned_explorers and all(ex.oxigenLevel <= 0 for ex in self.spawned_explorers):
            self.model.remove_agent(self)
            print(f"Dome {self.id} ha desaparecido.")

class OxigenPoint(ap.Agent):
    """
    Puntos de oxígeno: SOLO en celdas 2 (calle).
    """
    def setup(self):
        pass  # La posición se asigna en el modelo

class Semaforo(ap.Agent):
    """
    Uno por cada celda 5 en la grilla expandida.
    Alterna GREEN / RED cada 10 pasos.
    """
    def setup(self):
        self.state = 'GREEN'
        self.timer = 0
        self.change_interval = 10

    def step(self):
        self.timer += 1
        if self.timer % self.change_interval == 0:
            self.state = 'RED' if self.state == 'GREEN' else 'GREEN'

    def action(self):
        pass

class Peaton(ap.Agent):
    """
    Spawnea en aceras (3).
    - Pisa (3) libremente.
    - Pisa (4) o (5) solo si al menos un semáforo está en GREEN.
    - Muere si colisiona con un MovingAgent.
    """
    def setup(self):
        valid_positions = np.argwhere(city_grid == 3)
        if len(valid_positions) == 0:
            raise ValueError("No valid sidewalk (3) to spawn a Peaton!")
        px = valid_positions[np.random.randint(len(valid_positions))]
        self.x, self.y = map(int, px)

    def step(self):
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        np.random.shuffle(directions)
        # Si hay al menos un semáforo en verde
        any_green = any(s.state == 'GREEN' for s in self.model.semaforos)

        for dx, dy in directions:
            nx = self.x + dx
            ny = self.y + dy
            if 0 <= nx < city_size[0] and 0 <= ny < city_size[1]:
                tile = city_grid[nx, ny]
                # Piso acera (3) sin restricción
                if tile == 3:
                    self.x, self.y = nx, ny
                    break
                # Piso cruce (4) o semáforo (5) si hay semáforo en verde
                elif tile in (4, 5):
                    if any_green:
                        self.x, self.y = nx, ny
                    break

    def action(self):
        # Colisión con MovingAgent
        for agent in self.model.agents:
            if agent.x == self.x and agent.y == self.y:
                self.model.remove_agent(self)
                break


# ============= Agente "Sol" que alterna día/noche =============
class Sol(ap.Agent):
    """
    Alterna entre 'día' y 'noche' cada cierto número de steps.
    """
    def setup(self):
        self.estado = "día"
        self.cambio_intervalo = 10
        self.contador = 0

    def step(self):
        self.contador += 1
        if self.contador % self.cambio_intervalo == 0:
            self.estado = "noche" if self.estado == "día" else "día"
            print(f"*** SOL => Ahora es {self.estado.upper()} ***")


# ============= MODELO DE LA CIUDAD =============
class CityModel(ap.Model):
    def setup(self):
        self.grid = city_grid

        # 1) Domes
        self.domeAgents = ap.AgentList(self, DOME_NUMBER, Dome)

        # 2) MovingAgents (creados por Dome.action())
        self.agents = ap.AgentList(self)

        # 3) Creamos semáforos (uno por cada celda 5)
        positions_5 = np.argwhere(city_grid == 5)
        self.semaforos = ap.AgentList(self, len(positions_5), Semaforo)
        for s, pos in zip(self.semaforos, positions_5):
            s.x, s.y = pos

        # 4) OxigenPoints SOLO en celdas 2
        positions_2 = np.argwhere(city_grid == 2)
        n_oxy = 5
        self.oxygen_points = ap.AgentList(self, n_oxy, OxigenPoint)
        for ox in self.oxygen_points:
            rnd_pos = positions_2[np.random.randint(len(positions_2))]
            ox.x, ox.y = rnd_pos

        # 5) Peatones
        self.peatones = ap.AgentList(self, PEATON_NUMBER, Peaton)

        # 6) Agente Sol (Día/Noche)
        self.sol = Sol(self)

        # 7) Contenedor general
        self.all_agents = ap.AgentList(self)
        self.all_agents += self.domeAgents
        self.all_agents += self.agents
        self.all_agents += self.semaforos
        self.all_agents += self.oxygen_points
        self.all_agents += self.peatones
        self.all_agents.append(self.sol)

    def step(self):
        # a) Semáforos
        for s in self.semaforos:
            s.step()

        # b) Mover MovingAgents
        for agent in self.agents:
            agent.step()

        # c) Mover Peatones
        for p in self.peatones:
            p.step()

        # d) Domes generan
        for dome in self.domeAgents:
            dome.action()

        # e) Acciones de MovingAgents
        for agent in self.agents:
            agent.action()

        # f) Acciones de Peatones (colisión)
        for p in list(self.peatones):
            p.action()

        # g) El Sol alterna día/noche
        self.sol.step()

    def remove_agent(self, agent):
        # Elimina un agente de todos los listados donde aparezca
        if agent in self.all_agents:
            self.all_agents.remove(agent)
        if agent in self.agents:
            self.agents.remove(agent)
        if agent in self.domeAgents:
            self.domeAgents.remove(agent)
        if agent in self.peatones:
            self.peatones.remove(agent)
        if agent in self.semaforos:
            self.semaforos.remove(agent)
        if agent in self.oxygen_points:
            self.oxygen_points.remove(agent)


# ============= EJECUCIÓN DE LA SIMULACIÓN =============

model = CityModel()
model.setup()

# Guardamos posiciones en cada paso para animar
agent_positions = []
dome_positions = []
oxygen_positions = []
peaton_positions = []
semaforo_positions = []

for _ in range(N_STEPS):
    model.step()
    agent_positions.append([(a.x, a.y) for a in model.agents])
    dome_positions.append([(d.x, d.y) for d in model.domeAgents])
    oxygen_positions.append([(o.x, o.y) for o in model.oxygen_points])
    peaton_positions.append([(p.x, p.y) for p in model.peatones])
    semaforo_positions.append([(s.x, s.y, s.state) for s in model.semaforos])

# --- Animación ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(city_grid, cmap="tab10", alpha=0.6)

# Scatter de cada tipo de agente
scat_agents = ax.scatter([], [], c='red', s=50, label='Moving Agents')
scat_domes = ax.scatter([], [], c='blue', s=100, marker='s', label='Domes')
scat_oxy = ax.scatter([], [], c='green', s=30, marker='^', label='Oxygen')
scat_peaton = ax.scatter([], [], c='black', s=30, marker='o', label='Peaton')
scat_semaforo_green = ax.scatter([], [], c='lime', s=80, marker='X', label='Semaforo GREEN')
scat_semaforo_red = ax.scatter([], [], c='tomato', s=80, marker='X', label='Semaforo RED')

def update(frame):
    # 1) Moving Agents
    positions_agents = agent_positions[frame]
    if positions_agents:
        x_agents, y_agents = zip(*positions_agents)
    else:
        x_agents, y_agents = [], []
    scat_agents.set_offsets(np.c_[y_agents, x_agents])

    # 2) Domes
    positions_domes = dome_positions[frame]
    if positions_domes:
        x_domes, y_domes = zip(*positions_domes)
    else:
        x_domes, y_domes = [], []
    scat_domes.set_offsets(np.c_[y_domes, x_domes])

    # 3) OxigenPoints
    positions_oxy = oxygen_positions[frame]
    if positions_oxy:
        x_oxy, y_oxy = zip(*positions_oxy)
    else:
        x_oxy, y_oxy = [], []
    scat_oxy.set_offsets(np.c_[y_oxy, x_oxy])

    # 4) Peatones
    positions_p = peaton_positions[frame]
    if positions_p:
        xp, yp = zip(*positions_p)
    else:
        xp, yp = [], []
    scat_peaton.set_offsets(np.c_[yp, xp])

    # 5) Semaforos
    positions_s = semaforo_positions[frame]
    green_xy = [(sx, sy) for (sx, sy, st) in positions_s if st == 'GREEN']
    red_xy   = [(sx, sy) for (sx, sy, st) in positions_s if st == 'RED']

    if green_xy:
        gx, gy = zip(*green_xy)
    else:
        gx, gy = [], []
    if red_xy:
        rx, ry = zip(*red_xy)
    else:
        rx, ry = [], []
    scat_semaforo_green.set_offsets(np.c_[gy, gx])
    scat_semaforo_red.set_offsets(np.c_[ry, rx])

    ax.set_title(f"Step {frame + 1}")

ani = animation.FuncAnimation(fig, update, frames=N_STEPS, interval=300)
plt.legend()
plt.show()
