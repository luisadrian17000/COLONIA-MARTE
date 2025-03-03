#GENERACION DE DOMOS SPAWNERS DE AGENTES.

import agentpy as ap
import matplotlib.pyplot as plt
import random


class Explorer(ap.Agent):
    def setup(self):
        self.oxigenLevel = 20
        self.dome_id = None

    def action(self):
        if self.oxigenLevel > 0:
            X, Y = random.randint(0, 99), random.randint(0, 99)
            self.model.grid.move_to(self, (X, Y))
            self.oxigenLevel -= 5


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
        self.grid = ap.Grid(self, (100, 100), track_empty=False)
        self.explorerAgents = ap.AgentList(self, 0, Explorer)
        self.domeAgents = ap.AgentList(self, 4, Dome)
        self.oxigenAgents = ap.AgentList(self, 50, OxigenPoint)
        self.all_explorers = []

        self.grid.add_agents(self.domeAgents, random=True)
        self.grid.add_agents(self.oxigenAgents, random=True)
        self.steps_counter = 0

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

        # Actualizar gráfico
        self.plot_grid()

        self.steps_counter += 1

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


parameters = {"steps": 10}
model = Model(parameters)
model.run()
