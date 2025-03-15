# Mejor implementacion del algoritmo Q-learning
# En este codigo ya se implementarion todos los agentes de forma correcta
# En este codigo ya se implemento el resto de la API

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import agentpy as ap
import os
import requests
from dotenv import load_dotenv

#from main import N_SOLAR_PANELS, SolarPanels

# --------------

load_dotenv()
API_KEY_FROM_FILE = os.getenv("API_KEY")

def send_whole_simulation(simulation_data):
    API_URL = "http://127.0.0.1:5000/simulation_data"
    API_KEY = API_KEY_FROM_FILE
    HEADERS = {"Content-Type": "application/json", "X-API-KEY": API_KEY}
    try:
        response = requests.post(url=API_URL, json=simulation_data, headers=HEADERS)
        response.raise_for_status()
        print("Simulación enviada con éxito al servidor Flask.")
        print("Respuesta del servidor:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar simulacion{e}")

# --- Parámetros del modelo ---
SPAWN_RATE_EXPLORER = 1
GRID_SIZE = 1
PEATON_NUMBER = 2
DOME_NUMBER = 2
N_STEPS = 5
DAY_NIGHT_TIME = 2
OXYGEN_N = 1
N_ANUNCIOS = 2
N_DEFENSESYSTMES = 2
N_SOLAR_PANELS = 2
N_ALIENPLANTS = 3
N_FOODTRUCK = 2
N_PUBLICLAMP = 3
N_SPACEEQUIPMENT = 1
DOMES_Z_VALUE = 0
OXYGEN_Z_VALUE = 1
PEATON_Z_VALUE = 0
EXPLORERS_Z_VALUE = 1

# --- 1. Define la tabla de color-a-número ---
color_mapping = {
    (255, 0, 0): 1,
    (255, 180, 0): 2,
    (0, 255, 38): 3,
    (255, 0, 157): 4,
    (0, 49, 255): 5,
    (0, 255, 253): 6,
    (0, 0, 0): 7
}

# --- 2. Carga la imagen base de la ciudad ---
image_path = "citygrid.png"
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
        self.oxigenLevel = 3
        self.dome_id = None
        self.is_active = True
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_state(self):
        neighborhood = city_grid[max(0, self.x - 1):min(city_size[0], self.x + 2),
                                max(0, self.y - 1):min(city_size[1], self.y + 2)].flatten()
        proximity_to_oxygen = self.check_proximity_to_oxygen()
        return (tuple(neighborhood), self.oxigenLevel, proximity_to_oxygen)

    def check_proximity_to_oxygen(self):
        min_distance = float('inf')
        for oxygen_point in self.model.oxygen_points:
            distance = abs(self.x - oxygen_point.x) + abs(self.y - oxygen_point.y)
            min_distance = min(min_distance, distance)
        return min_distance < 10

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            directions = [(6, 0), (-6, 0), (0, 6), (0, -6)]
            return directions[np.random.randint(len(directions))]
        else:
            if state in self.q_table and self.q_table[state]:
                return max(self.q_table[state], key=self.q_table[state].get)
            else:
                directions = [(6, 0), (-6, 0), (0, 6), (0, -6)]
                return directions[np.random.randint(len(directions))]

    def step(self):
        state = self.get_state()
        action = self.choose_action(state)
        dx, dy = action
        new_x, new_y = self.x + dx, self.y + dy

        if (0 <= new_x < city_size[0] and 0 <= new_y < city_size[1] and
                all(city_grid[self.x + i * np.sign(dx), self.y + i * np.sign(dy)] in (2, 4, 7)
                    for i in range(1, 6 + 1))):
            old_x, old_y = self.x, self.y = new_x, new_y
            new_state = self.get_state()
            reward = self.get_reward()
            self.update_q_table(state, action, new_state, reward)
        else:
            reward = -5
            new_state = self.get_state()
            self.update_q_table(state, action, new_state, reward)

        self.oxigenLevel = max(0, self.oxigenLevel - 1)
        if self.oxigenLevel <= 0:
            self.model.remove_agent(self)
            self.is_active = False

    def get_reward(self):
        reward = -1
        if self.check_proximity_to_oxygen():
            reward += 5
        for peaton in self.model.peatones:
            if self.x == peaton.x and self.y == peaton.y:
                reward -= 10
        return reward

    def update_q_table(self, state, action, new_state, reward):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        old_value = self.q_table[state][action]
        next_max = max(self.q_table.get(new_state, {0: 0}).values())
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

    def action(self):
        pass



class Advertisement(ap.Agent):

    def setup(self):
        self.anouncement_id = self.id
        valid_positions = np.argwhere(city_grid == 2)

        if len(valid_positions) == 0:
            raise ValueError("No valid spawn points (1) found!")
        self.x, self.y = map(int, valid_positions[np.random.randint(len(valid_positions))])

        self.advertisement_number = np.random.randint(1,5)


    def step(self):
        self.advertisement_number = np.random.randint(1,5)

    def action(self):
        #print(f"anuncio en la posicion {self.x}, {self.y}")
        pass

class DefenseSystmes(ap.Agent):

    def setup(self):
        self.defenseSystems = self.id
        valid_positions = np.argwhere(city_grid == 1)

        if len(valid_positions) == 0:
            raise ValueError("No valid spawn points (1) found!")
        self.x, self.y = map(int, valid_positions[np.random.randint(len(valid_positions))])


class SolarPanels(ap.Agent):

    def setup(self):
        self.solarPanels = self.id
        valid_positions = np.argwhere(city_grid == 1)

        if len(valid_positions) == 0:
            raise ValueError("No valid spawn points (1) found!")
        self.x, self.y = map(int, valid_positions[np.random.randint(len(valid_positions))])


class AlienPlants(ap.Agent):

    def setup(self):
        self.alienPlants = self.id
        valid_positions = np.argwhere(city_grid == 1)

        if len(valid_positions) == 0:
            raise ValueError("No valid spawn points (1) found!")
        self.x, self.y = map(int, valid_positions[np.random.randint(len(valid_positions))])


class FoodTruck(ap.Agent):

    def setup(self):
        self.foodTruck= self.id
        valid_positions = np.argwhere(city_grid == 1)

        if len(valid_positions) == 0:
            raise ValueError("No valid spawn points (1) found!")
        self.x, self.y = map(int, valid_positions[np.random.randint(len(valid_positions))])

class SpaceEquipment(ap.Agent):

    def setup(self):
        self.spaceEquipment= self.id
        valid_positions = np.argwhere(city_grid == 1)

        if len(valid_positions) == 0:
            raise ValueError("No valid spawn points (1) found!")
        self.x, self.y = map(int, valid_positions[np.random.randint(len(valid_positions))])

class PublicLamp(ap.Agent):

    def setup(self):
        self.publicLamp= self.id
        valid_positions = np.argwhere(city_grid == 1)

        if len(valid_positions) == 0:
            raise ValueError("No valid spawn points (1) found!")
        self.x, self.y = map(int, valid_positions[np.random.randint(len(valid_positions))])



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
            #print(f"Dome {self.id} ha desaparecido.")

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
            #print(f"*** SOL => Ahora es {self.estado.upper()} ***")



# ============= MODELO DE LA CIUDAD =============
class CityModel(ap.Model):
    def setup(self):
        self.grid = city_grid

        # 1) Domes
        self.domeAgents = ap.AgentList(self, DOME_NUMBER, Dome)

        self.anuncios = ap.AgentList(self, N_ANUNCIOS, Advertisement)

        self.defenseSystems = ap.AgentList(self, N_DEFENSESYSTMES, DefenseSystmes)
        self.solarPanels = ap.AgentList(self, N_SOLAR_PANELS, SolarPanels)
        self.alienPlants = ap.AgentList(self, N_ALIENPLANTS, AlienPlants)
        self.foodTruck= ap.AgentList(self, N_FOODTRUCK, FoodTruck)
        self.spaceEquipment= ap.AgentList(self, N_SPACEEQUIPMENT, SpaceEquipment)
        self.publicLamp= ap.AgentList(self, N_PUBLICLAMP, PublicLamp )


        # 2) MovingAgents (creados por Dome.action())
        self.agents = ap.AgentList(self)

        # 3) Creamos semáforos (uno por cada celda 5)
        positions_5 = np.argwhere(city_grid == 5)
        self.semaforos = ap.AgentList(self, len(positions_5), Semaforo)
        for s, pos in zip(self.semaforos, positions_5):
            s.x, s.y = pos

        # 4) OxigenPoints SOLO en celdas 2
        positions_2 = np.argwhere(city_grid == 2)
        n_oxy = OXYGEN_N
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
        self.all_agents +=  self.anuncios
        self.all_agents += self.defenseSystems
        self.all_agents += self.solarPanels
        self.all_agents += self.alienPlants
        self.all_agents += self.foodTruck
        self.all_agents += self.spaceEquipment
        self.all_agents += self.publicLamp

        self.steps_counter = 0
        # DATA for api

        semaforos_positions_dics = [
            {

                "x": int(semaforo_agent.x),
                "y": int(DOMES_Z_VALUE),
                "Z": int(semaforo_agent.y),
                "id":int(semaforo_agent.id),
                "state":semaforo_agent.state,
            }for semaforo_agent,pos in zip(self.semaforos, [(semaforo.x,semaforo.y) for semaforo in self.semaforos])
        ]

        domes_positions_dics = [
            {
                "id": int(dome_agent.id),
                "x": int(pos[0]),
                "y": int(DOMES_Z_VALUE),
                "z": int(pos[1]),
            }
            for dome_agent, pos in zip(self.domeAgents, [(dome.x,dome.y) for dome in self.domeAgents])
        ]
        oxygen_positions_dics = [
            {
                "id": int(oxygen_agent.id),
                "x": int(pos[0]),
                "y": int(OXYGEN_Z_VALUE),
                "z": int(pos[1]),
            }
            for oxygen_agent, pos in zip(self.oxygen_points, [(oxygen.x,oxygen.y) for oxygen in self.oxygen_points])
        ]
        advertisement_positions_dics = [
            {
                "id":int(advertisement_agent.id),
                "x": int(pos[0]),
                "y":int(DOMES_Z_VALUE),
                "z": int(pos[1]),
            }
            for advertisement_agent, pos in zip(self.anuncios, [(anuncio.x,anuncio.y) for anuncio in self.anuncios])
        ]

        defenseSystem_positions_dics = [
            {
                "id": int(defense_system_agent.id),
                "x": int(pos[0]),
                "y": int(DOMES_Z_VALUE),
                "z": int(pos[1]),
            }
            for defense_system_agent, pos in zip(self.defenseSystems, [(defense.x, defense.y) for defense in self.defenseSystems])
        ]

        solarPanels_positions_dics = [
            {
                "id": int(solar_panel_agent.id),
                "x": int(pos[0]),
                "y": int(DOMES_Z_VALUE),
                "z": int(pos[1]),
            }
            for solar_panel_agent, pos in zip(self.solarPanels, [(solar.x, solar.y) for solar in self.solarPanels])
        ]

        alien_plants_positions_dics = [
            {
                "id": int(alien_plants_agent.id),
                "x": int(pos[0]),
                "y": int(DOMES_Z_VALUE),
                "z": int(pos[1]),
            }
            for alien_plants_agent, pos in zip(self.alienPlants, [(alien.x, alien.y) for alien in self.alienPlants])
        ]

        food_truks_dics = [
            {
                "id": int(food_trucks_agent.id),
                "x": int(pos[0]),
                "y": int(DOMES_Z_VALUE),
                "z": int(pos[1]),
            }
            for food_trucks_agent, pos in zip(self.foodTruck, [(food.x, food.y) for food in self.foodTruck])
        ]

        space_equipment_dics = [
            {
                "id": int(space_equipment_agent.id),
                "x": int(pos[0]),
                "y": int(DOMES_Z_VALUE),
                "z": int(pos[1]),
            }
            for space_equipment_agent, pos in zip(self.spaceEquipment, [(space.x, space.y) for space in self.spaceEquipment])
        ]

        public_lamp_dics = [
            {
                "id": int(public_lamp_agent.id),
                "x": int(pos[0]),
                "y": int(DOMES_Z_VALUE),
                "z": int(pos[1]),
            }
            for public_lamp_agent, pos in zip(self.publicLamp, [(lamp.x, lamp.y) for lamp in self.publicLamp])
        ]





        self.simulation_data = {}
        self.simulation_data["grid_size"] = GRID_SIZE
        self.simulation_data["shelters_n"] = DOME_NUMBER
        self.simulation_data["spawn_rate"] = SPAWN_RATE_EXPLORER
        self.simulation_data["oxygen_endpoint_n"] = OXYGEN_N
        self.simulation_data["simulation_steps"] = N_STEPS
        self.simulation_data["explorers_steps"] = []
        self.simulation_data["oxygen_positions"] = oxygen_positions_dics
        self.simulation_data["domes_positions"] = domes_positions_dics
        self.simulation_data["city_grid"] = city_grid.tolist()
        self.simulation_data["semaforos"] = semaforos_positions_dics
        self.simulation_data["peatones_positions"] = []
        self.simulation_data["advertisement"] = advertisement_positions_dics
        self.simulation_data["defense_systems"] = defenseSystem_positions_dics
        self.simulation_data["solar_panels"] = solarPanels_positions_dics
        self.simulation_data['alien_plants'] = alien_plants_positions_dics
        self.simulation_data["food_trucks"] = food_truks_dics
        self.simulation_data['space_equipment'] = space_equipment_dics
        self.simulation_data['public_lamp'] = public_lamp_dics

    def finalize_simulation_data(self):

        send_whole_simulation(self.simulation_data)

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

        for anuncio in self.anuncios:
            anuncio.step()
            anuncio.action()


        # g) El Sol alterna día/noche
        self.sol.step()
        self.collect_step_data()
        self.steps_counter += 1
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

    def collect_step_data(self):
        explorer_data_dict = {
            "step" :int(self.steps_counter),
            "agents":[

                {
                    "id":int(explorer.id),
                    "x":int(explorer.x),
                    "y":int(EXPLORERS_Z_VALUE),
                    "z":int(explorer.y),
                    "is_active":int(explorer.is_active),

                }
                for explorer in self.agents
            ]
        }

        peatones_data_dict = {
        "step" :int(self.steps_counter),
        "agents":[
            {
                "id":int(peatone.id),
                "x":int(peatone.x),
                "y":int(EXPLORERS_Z_VALUE),
                "z":int(peatone.y),
                
            }
            for peatone in self.peatones
        ]


        }

        self.simulation_data["explorers_steps"].append(explorer_data_dict)

        self.simulation_data["peatones_positions"].append(peatones_data_dict)





    def end(self):
        # This method is automatically called by AgentPy when the simulation ends
        super().end()
        self.finalize_simulation_data()

    # ... (tu código existente) ...

    # Después de que la simulación ha terminado y antes de la animación:
    # ... (tu código de animación) ...




# ============= EJECUCIÓN DE LA SIMULACIÓN =============

model = CityModel()
model.setup()

# Guardamos posiciones en cada paso para animar
agent_positions = []
dome_positions = []
oxygen_positions = []
peaton_positions = []
semaforo_positions = []
anuncio_positions = []
defenseSystems_positions = []
solarPanels_positions = []
alienPlants_positions= []
foodTrucks_positions = []
spaceEquipments_positions = []
publicLamp_positions = []


for _ in range(N_STEPS):
    model.step()
    agent_positions.append([(a.x, a.y) for a in model.agents])
    dome_positions.append([(d.x, d.y) for d in model.domeAgents])
    oxygen_positions.append([(o.x, o.y) for o in model.oxygen_points])
    peaton_positions.append([(p.x, p.y) for p in model.peatones])
    semaforo_positions.append([(s.x, s.y, s.state) for s in model.semaforos])
    anuncio_positions.append([(a.x,a.y) for a in model.anuncios])
    defenseSystems_positions.append([(a.x, a.y) for a in model.defenseSystems])
    solarPanels_positions.append([(a.x, a.y) for a in model.solarPanels])
    alienPlants_positions.append([(a.x, a.y) for a in model.alienPlants])
    foodTrucks_positions.append([(a.x, a.y) for a in model.foodTruck])
    spaceEquipments_positions.append([(a.x, a.y) for a in model.spaceEquipment])
    publicLamp_positions.append([(a.x, a.y) for a in model.publicLamp])


model.end()


# Imprimir posiciones de peatones DESPUÉS de model.end()
for step_data in model.simulation_data["peatones_positions"]:
    step_number = step_data["step"]
    print(f"Posiciones de peatones en el paso {step_number}:")
    for peaton_data in step_data["agents"]:
        peaton_id = peaton_data["id"]
        x = peaton_data["x"]
        z = peaton_data["z"]
        print(f"  Peatón ID: {peaton_id}, Posición: (x={x}, z={z})")


# --- Animación ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(city_grid, cmap="tab10", alpha=0.6)

# Scatter de cada tipo de agente
scat_agents = ax.scatter([], [], c='red', s=50, label='Moving Agents')
scat_anouncements = ax.scatter([], [], c='purple', s=120, label='Anouncements')
scat_defenseSystems = ax.scatter([], [], c='pink', s=120, label='defense systems')
scat_solarPanel= ax.scatter([], [], c='yellow', s=120, label='solar panels')
scat_alienPlants= ax.scatter([], [], c='green', s=120, label='alien plants')
scat_foodTrucks= ax.scatter([], [], c='gray', s=120, label='food trucks')
scat_spaceEquipment = ax.scatter([], [], c='white', s=120, label='space equipment')
scat_PublicLamp = ax.scatter([], [], c='orange', s=120, label='public lamp')
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

    # 3) Anouncements
    positions_anouncements = anuncio_positions[frame] #correccion
    if positions_anouncements:
        x_anouncements, y_anouncements = zip(*positions_anouncements)
    else:
        x_anouncements, y_anouncements = [], []
    scat_anouncements.set_offsets(np.c_[y_anouncements, x_anouncements])

    positions_defenseSystmes = defenseSystems_positions[frame]  # correccion
    if positions_defenseSystmes:
        x_anouncements2, y_anouncements2 = zip(*positions_defenseSystmes)
    else:
        x_anouncements2, y_anouncements2 = [], []
    scat_defenseSystems.set_offsets(np.c_[y_anouncements2, x_anouncements2])

    positions_solarPanel = solarPanels_positions[frame]  # correccion
    if positions_solarPanel:
        x_anouncements2, y_anouncements2 = zip(*positions_solarPanel)
    else:
        x_anouncements2, y_anouncements2 = [], []
    scat_solarPanel.set_offsets(np.c_[y_anouncements2, x_anouncements2])

    positions_alienPlants = alienPlants_positions[frame]  # correccion
    if positions_alienPlants:
        x_anouncements2, y_anouncements2 = zip(*positions_alienPlants)
    else:
        x_anouncements2, y_anouncements2 = [], []
    scat_alienPlants.set_offsets(np.c_[y_anouncements2, x_anouncements2])

    positions_foodTrucks = foodTrucks_positions[frame]  # correccion
    if positions_foodTrucks:
        x_anouncements2, y_anouncements2 = zip(*positions_foodTrucks)
    else:
        x_anouncements2, y_anouncements2 = [], []
    scat_foodTrucks.set_offsets(np.c_[y_anouncements2, x_anouncements2])

    positions_spaceEquipment = spaceEquipments_positions[frame]  # correccion
    if positions_spaceEquipment:
        x_anouncements2, y_anouncements2 = zip(*positions_spaceEquipment)
    else:
        x_anouncements2, y_anouncements2 = [], []
    scat_spaceEquipment.set_offsets(np.c_[y_anouncements2, x_anouncements2])

    positions_publicLamp = publicLamp_positions[frame]  # correccion
    if positions_publicLamp:
        x_anouncements2, y_anouncements2 = zip(*positions_publicLamp)
    else:
        x_anouncements2, y_anouncements2 = [], []
    scat_PublicLamp.set_offsets(np.c_[y_anouncements2, x_anouncements2])

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


#
