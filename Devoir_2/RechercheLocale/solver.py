from uflp import UFLP
import random
from typing import List, Tuple
""" 
    Binome 1 : Shimbi Masengo Wa Umba Papa Levi (2119138)
    Binome 2 : Tiomo EPongo Einstein Franck (2093771)
    Description succinte de l'implementation :

    L'implémentation de recherche locale que nous avons réalisée se base sur la métaheuristique des restarts.
    Lorsque l'algorithme atteint un minimum local, il redémarre à partir d'une nouvelle solution initiale aléatoire.

    Les solutions initiales sont obtenues en utilisant deux heuristiques différentes:
        - La première heuristique est se base sur le coût total de la solution, c'est-à-dire le coût d'ouverture des
          gares principales et le coût d'association des gares satellites aux gares principales.
        - La deuxième heuristique est quant à elle basée uniquement sur le cout d'association des gares satellites
          aux gares principales.
    ...
"""

NUMBER_OF_RESTARTS = 6

def solve(problem: UFLP) -> Tuple[List[int], List[int]]:
    """
    Votre implementation, doit resoudre le probleme via recherche locale.

    Args:
        problem (UFLP): L'instance du probleme à résoudre

    Returns:
        Tuple[List[int], List[int]]: 
        La premiere valeur est une liste représentant les stations principales ouvertes au format [0, 1, 0] qui indique que seule la station 1 est ouverte
        La seconde valeur est une liste représentant les associations des stations satellites au format [1 , 4] qui indique que la premiere station est associée à la station pricipale d'indice 1 et la deuxieme à celle d'indice 4
    """

    random_main_opened, random_satellite_association = random_solution(problem)
    
    # Variables to store the two initial solutions
    assoc_sol_heuristic = random_main_opened.copy(), random_satellite_association.copy()
    total_cost_sol_heuristic = random_main_opened.copy(), random_satellite_association.copy()

    total_cost_sol_heuristic = find_solution_by_heuristic(problem, total_cost_sol_heuristic, heuristic='total_cost')
    assoc_sol_heuristic = find_solution_by_heuristic(problem, assoc_sol_heuristic, heuristic='association_cost')

    assoc_cost_heuristic = problem.calculate_cost(assoc_sol_heuristic[0], assoc_sol_heuristic[1])
    total_cost_heuristic = problem.calculate_cost(total_cost_sol_heuristic[0], total_cost_sol_heuristic[1])

    heuristic_costs = [assoc_cost_heuristic, total_cost_heuristic]
    heuristic_solutions = [assoc_sol_heuristic, total_cost_sol_heuristic]

    # Store the current lowest cost and its solution
    best_design_cost = min(heuristic_costs)
    best_design_sol = heuristic_solutions[heuristic_costs.index(best_design_cost)]

    current_cost = 0
    current_solution = []

    # Local Search with restarts
    for i in range(NUMBER_OF_RESTARTS):

        # Use each initial solution on half of the restarts
        if max(heuristic_costs) > (min(heuristic_costs) * 2):
            current_cost = min(heuristic_costs)
            current_solution = heuristic_solutions[heuristic_costs.index(current_cost)]
        else:
            current_cost = heuristic_costs[i % 2]
            current_solution = heuristic_solutions[i % 2]

        local_min_found = False

        # Iterate until a local minima is found
        while not local_min_found:

            # Boolean variable for neighbor with better cost found
            updated_solution = False

            neighbours = get_neighbours(problem, current_solution)

            # Update cost and solution if better neighbor found
            for main_neighbour, satellite_neighbour in neighbours:
                cost = problem.calculate_cost(main_neighbour, satellite_neighbour)
                
                if cost < current_cost:
                    current_cost = cost
                    current_solution = main_neighbour, satellite_neighbour
                    updated_solution = True
            
            # If a lower cost is not found, we have reached a local minima
            if not updated_solution:
                local_min_found = True

        # Store the optimal solution found
        if current_cost < best_design_cost:
            best_design_cost = current_cost
            best_design_sol = current_solution[0], current_solution[1]

    return best_design_sol

def get_neighbours(
    problem: UFLP,
    current_solution: Tuple[List[int], List[int]]
) -> List[Tuple[List[int], List[int]]]:
    _, satellite_association = current_solution
    neighbours = []

    for sat_index, main_index in enumerate(satellite_association):

        random_sat_index_1 = random.choice([i for i in range(len(satellite_association)) if sat_index != i])
        random_sat_index_2 = random.choice([i for i in range(len(satellite_association)) if sat_index != i and random_sat_index_1 != i])

        for main_i in range(problem.n_main_station):
            if main_index != main_i:
                # Change main station of the current satellite station
                sat_neighbour = satellite_association.copy()
                sat_neighbour[sat_index] = main_i
                if sat_neighbour not in neighbours:
                    main_neighbour = [1 if i in sat_neighbour else 0 for i in range(problem.n_main_station)]
                    neighbours.append((main_neighbour, sat_neighbour))

                # Change main station of random_sat_index_1
                random_main_i = random.choice([i for i in range(problem.n_main_station) if satellite_association[random_sat_index_1] != i])
                random_sat_neighbour = satellite_association.copy()
                random_sat_neighbour[sat_index] = main_i
                random_sat_neighbour[random_sat_index_1] = random_main_i
                if random_sat_neighbour not in neighbours:
                    random_main_neighbour = [1 if i in random_sat_neighbour else 0 for i in range(problem.n_main_station)]
                    neighbours.append((random_main_neighbour, random_sat_neighbour))

                # Change main station of random_sat_index_2
                random_main_i_2 = random.choice([i for i in range(problem.n_main_station) if satellite_association[random_sat_index_2] != i])
                random_sat_neighbour_2 = satellite_association.copy()
                random_sat_neighbour_2[sat_index] = main_i
                random_sat_neighbour_2[random_sat_index_1] = random_main_i
                random_sat_neighbour_2[random_sat_index_2] = random_main_i_2
                if random_sat_neighbour_2 not in neighbours:
                    random_main_neighbour_2 = [1 if i in random_sat_neighbour_2 else 0 for i in range(problem.n_main_station)]
                    neighbours.append((random_main_neighbour_2, random_sat_neighbour_2))

    return neighbours


def find_solution_by_heuristic(
    problem: UFLP,
    solution: Tuple[List[int], List[int]],
    heuristic: str = 'total_cost'
) -> Tuple[List[int], List[int]]:
    """
    Trouve la solution initiale guidée par le coût total ou seulement par le coût d'association.

    Args:
        problem (UFLP): L'instance du problème.
        solution (Tuple[List[int], List[int]]): La solution initiale.
        heuristic (bool): Indique si la solution doit être guidée par le coût total (True) ou seulement par le coût d'association (False).

    Returns:
        Tuple[List[int], List[int]]: La solution initiale mise à jour.
    """
    main_stations_opened, satellite_station_association = solution

    for satellite_i, main_station in enumerate(satellite_station_association):
        old_cost = problem.get_opening_cost(main_station) + problem.get_association_cost(main_station, satellite_i)
        if heuristic == 'total_cost':
            old_cost = problem.get_opening_cost(main_station) + problem.get_association_cost(main_station, satellite_i)
        else:
            old_cost = problem.get_association_cost(main_station, satellite_i)

        for main_i in range(problem.n_main_station):
            if main_station != main_i:
                new_cost = problem.get_opening_cost(main_i) + problem.get_association_cost(main_i, satellite_i) if heuristic == 'total_cost' else problem.get_association_cost(main_i, satellite_i)

                if new_cost < old_cost:
                    satellite_station_association[satellite_i] = main_i
                    main_stations_opened = [1 if i in satellite_station_association else 0 for i in range(problem.n_main_station)]
                    solution = main_stations_opened, satellite_station_association
                    old_cost = new_cost

    return solution



def random_solution(problem: UFLP) -> Tuple[List[int], List[int]]:
    """
    Retourne une solution aléatoire au probleme : à battre pour avoir la moyenne.

    Args:
        problem (UFLP): L'instance du probleme à résoudre

    Returns:
        Tuple[List[int], List[int]]: 
        La premiere valeur est une liste représentant les stations principales ouvertes au format [0, 1, 0] qui indique que seule la station 1 est ouverte
        La seconde valeur est une liste représentant les associations des stations satellites au format [1 , 4] qui indique que la premiere station est associée à la station pricipale d'indice 1 et la deuxieme à celle d'indice 4
    """

    # Ouverture aléatoire des stations principales
    main_stations_opened = [random.choice([0,1]) for _ in range(problem.n_main_station)]

    # Si, par hasard, rien n'est ouvert, on ouvre une station aléatoirement
    if sum(main_stations_opened) == 0:
        main_stations_opened[random.choice(range(problem.n_main_station))] = 1

    # Association aléatoire des stations satellites aux stations principales ouvertes
    indices = [i for i in range(len(main_stations_opened)) if main_stations_opened[i] == 1]
    satellite_station_association = [random.choice(indices) for _ in range(problem.n_satellite_station)]

    return main_stations_opened, satellite_station_association