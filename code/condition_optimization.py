import sys
sys.path.insert(0, "/opt/src")
from main import *
from results_analysis import *
from deap import base, creator, tools, algorithms
import random
import logging
import warnings
# Suppress logging for specific libraries
logging.getLogger("setuptools").setLevel(logging.CRITICAL)
logging.getLogger("distutils").setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore", module="setuptools.*")


def evaluate_conditions(individual):
    nitrogen, phosphorus = individual
    matrix.conditions["[N] mmol"].loc["TC"] = nitrogen
    matrix.conditions["[P] mmol"].loc["TC"] = phosphorus
    initial_parameters = json.load(open(f"{RESULTS_PATH}/parameters/optimized_parameters.json", "r"))
    concentrations_tc, trajectories_tc = create_dfba_model("TC", initial_parameters, True)
    return round(concentrations_tc['Carotene'].iloc[-1], 5), round(concentrations_tc['ActiveBiomass'].iloc[-1], 5)

def check_bounds(individual):
    if individual[0] < 0:
        individual[0] = 0
    elif individual[0] > 50:
        individual[0] = 50
    if individual[1] < 0:
        individual[1] = 0
    elif individual[1] > 0.5:
        individual[1] = 0.5
    return individual

def condition_optimization_mo():
    # Define fitness and individual types
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # Maximize both objectives
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Define the toolbox
    toolbox = base.Toolbox()

    # Decision variables: nitrogen (0-50), phosphorus (0-0.5)
    toolbox.register("attr_nitrogen", random.uniform, 0, 50)
    toolbox.register("attr_phosphorus", random.uniform, 0, 0.5)

    # Create individuals and populations
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_nitrogen, toolbox.attr_phosphorus), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation function
    toolbox.register("evaluate", evaluate_conditions)

    # Register genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    # Create an initial population
    population = toolbox.population(n=4)

    # Number of generations
    ngen = 2
    cxpb = 0.7  # Crossover probability
    mutpb = 0.2  # Mutation probability

    # Add progress bar using tqdm
    progress_bar = tqdm(total=ngen, desc="Running GA", unit="generation")

    # Evaluate the initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    # Run the algorithm generation by generation
    for gen in range(ngen):
        # Generate offspring using crossover and mutation
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)

        # Evaluate offspring
        for ind in offspring:
            check_bounds(ind)
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # Select the next generation
        population[:] = toolbox.select(population + offspring, len(population))
        progress_bar.update(1)

    progress_bar.close()

    # Extract Pareto-optimal solutions
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    # Print the Pareto-optimal solutions
    print("Pareto-optimal solutions:")
    for ind in pareto_front:
        print(ind, ind.fitness.values)


if __name__ == "__main__":
    condition_optimization_mo()



