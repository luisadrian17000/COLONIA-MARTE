#LO MISMO QUE LA VERSION ANTERIO, AHORA EL AGENTE PUEDE MORIR SI SU VIDA LLEGA A 0, Y SE RETIRA DEL GRID

import agentpy as ap
import matplotlib.pyplot as plt
import random

class Explorer(ap.Agent):

    def setup(self):
        self.oxigenLevel = 20

    def action(self):
        X = random.randint(0, 9)
        Y = random.randint(0, 9)
        self.model.grid.move_to(self, (X,Y))

class OxigenPoint(ap.Agent):
    def action(self):
        pass
        #X = random.randint(0, 9)
        #Y = random.randint(0, 9)
        #self.model.grid.move_to(self, (X, Y))

class Model(ap.Model):
    def setup(self):

        self.grid = ap.Grid(self, (10,10), track_empty=False)
        self.explorerAgents = ap.AgentList(self, 3, Explorer)
        self.oxigenAgents = ap.AgentList(self, 50, OxigenPoint)

        self.grid.add_agents(self.explorerAgents, random=True)
        self.grid.add_agents(self.oxigenAgents, random=True)
        self.steps_counter = 0



    def step(self):

        for i in self.explorerAgents:
            i.action()
            i.oxigenLevel -= 5

        for j in self.oxigenAgents:
            j.action()

        agents_to_remove = []
        for i in self.explorerAgents:
            if i.oxigenLevel <= 0:
                agents_to_remove.append(i)

        for agent in agents_to_remove:
            self.grid.remove_agents(agent)
            self.explorerAgents.remove(agent)

        self.steps_counter += 1
        self.plot_grid()

        self.grid.record_positions(self.explorerAgents)
        self.grid.record_positions(self.oxigenAgents)

        # Guardar posiciones de agentes exploradores y puntos de oxigeno por separado
        self.explorerAgents_positions = {E_agent: self.grid.positions[E_agent] for E_agent in self.explorerAgents}
        #print(self.explorerAgents_positions)

        self.oxigenAgents_positions = {OP_agent: self.grid.positions[OP_agent] for OP_agent in self.oxigenAgents}
        #print(self.oxigenAgents_positions)

        if self.explorerAgents_positions and self.oxigenAgents_positions:
            for explorer_agent, explorer_pos in self.explorerAgents_positions.items():
                for oxigen_pos in self.oxigenAgents_positions.values():
                    if explorer_pos == oxigen_pos:
                        explorer_agent.oxigenLevel += 2
                        explorer_agent.oxigenLevel = min(explorer_agent.oxigenLevel, 100)  # Limitar a 100

        for explorer_agent in self.explorerAgents:
            print(f"Agente Explorer {explorer_agent.id}: oxigenLevel = {explorer_agent.oxigenLevel}")


    # Funcion para graficar las posciciones de los agentes en el grid.
    def plot_grid(self):
        plt.figure(figsize=(6, 6))
        plt.xticks(range(10))
        plt.yticks(range(10))
        plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

        explorerPositions = [self.grid.positions[i] for i in self.explorerAgents]
        oxigenPositions = [self.grid.positions[i] for i in self.oxigenAgents]

        for pos in explorerPositions:
            plt.scatter(pos[1], 9 - pos[0], color="red", s=500, alpha=0.35)  # Ajustar coordenadas

        for pos in oxigenPositions:
            plt.scatter(pos[1], 9 - pos[0], color="blue", s=200)  # Ajustar coordenadas

        plt.title(f"Step {self.steps_counter}")
        plt.show()


parameters = {"steps": 10}
model = Model(parameters)
model.run()

