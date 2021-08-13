import random
from deap import creator
from deap import base
import math
from deap import tools
import numpy as np


def decode(x_list):
    x_ = [str(i) for i in x_list]
    x = "".join(x_)
    y = 0+int(x, 2)/(2**17-1)*9
    return y


def aimFunction(x):
    y = x+5*math.sin(5*x)+2*math.cos(3*x)
    return y


def evaluate(population):
    value = []
    for i in range(len(population)):
        value.append(aimFunction(decode(population[i])))
        if value[i] < 0:
            value[i] = 0
    return value


# 这里这个base.Fitness是干嘛的？？？
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# creator.create("Individual", list, fitness=creator.FitnessMax)  # 这里的list，fitness是参数，干嘛的？？？
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

IND_SIZE = 17

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)  # 包含了0,1的随机整数。

toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, n=IND_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("population", tools.initRepeat, np.ndarray, toolbox.individual)

toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(63)
    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=300)
    print(pop)

    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    #
    # NGEN  is the number of generations for which the
    #       evolution runs   进化运行的代数！果然，运行40代之后，就停止计算了
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40
    print("Start of evolution")

    fitnesses = toolbox.evaluate(pop)
    print(fitnesses)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = (fit,)

    print("Evaluated %i individuals" % len(pop))  # 这时候，pop的长度还是300呢

    for g in range(NGEN):
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # print("population:")
        # print(offspring)

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values
        # print(offspring)
        # print("mate:")
        # print(offspring)

        # print(offspring[0] is offspring[1])
        # print(offspring[0] is offspring[2])
        # print(offspring[3] is offspring[4])
        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # print("mutant:")
        # print(offspring)
        # print(offspring)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.evaluate(invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)
        fitnesses = [ind.fitness.values for ind in offspring]
        # print(fitnesses)
        print("Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == '__main__':
    main()
