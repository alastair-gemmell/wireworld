import wireworld.engine as engine
import wireworld.simulation as simulation


def main():
    eng = engine.WireworldEngine()
    sim = simulation.WireworldSimulation(eng)
    sim.start()


if __name__ == "__main__":
    main()