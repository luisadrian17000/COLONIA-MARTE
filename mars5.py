#GENERACION DE DOMOS SPAWNERS DE AGENTES.

import agentpy as ap
import matplotlib.pyplot as plt
import random
import os
import requests
from dotenv import load_dotenv
from psutil import SUNOS

load_dotenv()
API_KEY_FROM_FILE = os.getenv("API_KEY")

def send_whole_simulation(simulation_data):
    #API_URL = "https://tc2008b-rest-api.onrender.com/simulation_data"
    API_URL = "http://127.0.0.1:5000/simulation_data"
    API_KEY = API_KEY_FROM_FILE
    HEADERS = {"Content-Type": "application/json",
               "X-API-KEY": API_KEY
               }
    try:
        response = requests.post(url=API_URL, json=simulation_data, headers=HEADERS)
        response.raise_for_status()
        print("Simulación enviada con éxito al servidor Flask.")
        print("Respuesta del servidor:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error al enviar simulacion{e}")
#-----------------------
# Our Constants

EXPLORERS_Z_VALUE = 23
DOMES_Z_VALUE = 23
OXYGEN_ENDPOINT_Z_VALUE = 23

#------------------------

GRID_SIZE = 25
SHELTERS_N= 4
EXPLORERS_N= 0
OXYGEN_ENDPOINTS_N = 50
SIMULATION_STEPS = 100



class Explorer(ap.Agent):
    def setup(self):
        self.oxigenLevel = 20
        self.dome_id = None
        self.x, self.y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        self.is_active = True

    def action(self):
        if self.oxigenLevel > 0:
            self.x, self.y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            self.model.grid.move_to(self, (self.x,self.y))
            self.oxigenLevel -= 5
        else:
            self.is_active = False


class Dome(ap.Agent):
    def setup(self):
        self.spawned_explorers = []
        self.spawn_rate = 5

    def action(self):
        new_explorers = []
        for _ in range(self.spawn_rate):
            explorer = Explorer(self.model)
            explorer.dome_id = self.id
            self.spawned_explorers.append(explorer)
            new_explorers.append(explorer)
            self.model.add_explorer(explorer)

        self.model.grid.add_agents(new_explorers, random=True)

        if all(explorer.oxigenLevel <= 0 for explorer in self.spawned_explorers):
            self.model.grid.remove_agents(self)
            self.model.domeAgents.remove(self)
            print(f"Dome {self.id} ha desaparecido.")


class OxigenPoint(ap.Agent):
    def setup(self):
        pass


class Model(ap.Model):
    def setup(self):
        super().setup()#si truena algo, borrar esto
        self.grid = ap.Grid(self, (GRID_SIZE, GRID_SIZE), track_empty=False)
        self.explorerAgents = ap.AgentList(self, 0, Explorer)
        self.domeAgents = ap.AgentList(self, SHELTERS_N, Dome)
        self.oxigenAgents = ap.AgentList(self, OXYGEN_ENDPOINTS_N, OxigenPoint)
        self.all_explorers = []
        domes_random_positions = [
            (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            for _ in range(SHELTERS_N)
        ]

        oxygen_endpoints_random_positions = [
            (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            for _ in range(OXYGEN_ENDPOINTS_N)
        ]

        self.grid.add_agents(self.domeAgents, domes_random_positions)
        self.grid.add_agents(self.oxigenAgents, oxygen_endpoints_random_positions)
        self.steps_counter = 0

        domes_positions_dics = [
            {
                "id": int(dome_agent.id),
                "x": int(pos[0]),
                "y": int(pos[1]),
                "z": int(DOMES_Z_VALUE),
            }
            for dome_agent, pos in zip(self.domeAgents, domes_random_positions)
        ]
        oxygen_positions_dics = [
            {
                "id": int(oxygen_agent.id),
                "x": int(pos[0]),
                "y": int(pos[1]),
                "z": int(OXYGEN_ENDPOINT_Z_VALUE),
            }
            for oxygen_agent, pos in zip(self.oxigenAgents, oxygen_endpoints_random_positions)
        ]

        self.simulation_data ={

        "grid_size": GRID_SIZE,
        "shelters_n":SHELTERS_N,
        "explorers_n":EXPLORERS_N,
        "oxygen_endpoints_n":OXYGEN_ENDPOINTS_N,
        "simulation_steps":SIMULATION_STEPS,
        "explorers_steps":[],
        "oxygen_positions":oxygen_positions_dics,
        "domes_positions":domes_positions_dics,

        }

    def step(self):
        # Domos generan exploradores
        for dome in self.domeAgents:
            dome.action()

        # Movimiento y reducción de oxígeno
        for explorer in list(self.explorerAgents):  # Se usa list() para evitar modificación de iterador
            explorer.action()

        # Exploradores consumen oxígeno y pueden morir
        dead_explorers = [e for e in self.explorerAgents if e.oxigenLevel <= 0]
        for explorer in dead_explorers:
            self.grid.remove_agents(explorer)
            self.explorerAgents.remove(explorer)
            self.all_explorers.remove(explorer)

        # Exploradores recargan oxígeno si están en un punto de oxígeno
        oxigen_positions = {self.grid.positions[o] for o in self.oxigenAgents}
        for explorer in self.explorerAgents:
            if self.grid.positions[explorer] in oxigen_positions:
                explorer.oxigenLevel = min(explorer.oxigenLevel + 2, 100)

        # Mostrar estado de exploradores
        for explorer in self.explorerAgents:
            print(f"Agente Explorer {explorer.id}: oxigenLevel = {explorer.oxigenLevel}")


        self.collect_step_data()
        # Actualizar gráfico
        self.plot_grid()

        self.steps_counter += 1



    def collect_step_data(self):
        explorer_data_dict = {
            "step" :int(self.steps_counter),
            "agents":[

                {
                    "id":int(explorer.id),
                    "x":int(explorer.x),
                    "y":int(explorer.y),
                    "z":int(EXPLORERS_Z_VALUE),
                    "is_active":int(explorer.is_active),

                }
                for explorer in self.explorerAgents
            ]
        }
        self.simulation_data["explorers_steps"].append(explorer_data_dict)

    def finalize_simulation_data(self):
        """
        Guarda en disco (una sola vez al final) el JSON completo con todos los pasos.
        En el futuro, aquí podrías reemplazar la lógica para hacer un PUT/POST a tu API.
        """
        send_whole_simulation(self.simulation_data)


    def plot_grid(self):
        plt.figure(figsize=(6, 6))
        plt.xticks(range(0, 100, 10))
        plt.yticks(range(0, 100, 10))
        plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

        explorerPositions = [self.grid.positions[i] for i in self.explorerAgents]
        oxigenPositions = [self.grid.positions[i] for i in self.oxigenAgents]
        domePositions = [self.grid.positions[i] for i in self.domeAgents]

        for pos in explorerPositions:
            plt.scatter(pos[1], 99 - pos[0], color="red", s=50, alpha=0.35)

        for pos in oxigenPositions:
            plt.scatter(pos[1], 99 - pos[0], color="blue", s=20)

        for pos in domePositions:
            plt.scatter(pos[1], 99 - pos[0], color="green", s=70)

        plt.title(f"Step {self.steps_counter}")
        plt.show()

    def add_explorer(self, explorer):
        if explorer not in self.all_explorers:
            self.all_explorers.append(explorer)
            self.explorerAgents.append(explorer)
            self.grid.add_agents([explorer], random=True)
    def end(self):
        # This method is automatically called by AgentPy when the simulation ends
        super().end()
        self.finalize_simulation_data()

parameters = {"steps": SIMULATION_STEPS}
model = Model(parameters)
model.run()

