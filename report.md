---
title: "Computational Intelligence Report"
author: "Domenico Scalera s333304"
geometry: margin=0.8in
fontsize: 10pt
header-includes:
  - \usepackage{fvextra}
  - \usepackage{titlesec}
  - \titleformat{\section}{\Large\bfseries}{\thesection}{1em}{}
  - \titleformat{\subsection}{\large\bfseries}{\thesubsection}{1em}{}
  - \titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{1em}{}
  - \DefineVerbatimEnvironment{verbatim}{Verbatim}{
      breaklines=true,
      breakanywhere=true,
      fontsize=\footnotesize,
      frame=single,
      framesep=2mm
    }
---

# Computational Intelligence Report

## Laboratory 1: Set Cover Problem

**Objective:**  
To implement and compare different hill climbing algorithms for solving the Set Cover Problem

### Activities Performed
- Implemented multiple variations of hill climbing algorithms:
  - Basic Hill Climber
  - Steepest Step with Restart
  - Simple Random Heavy Climb Mutation (RHCM)
  - Improved RHCM with adaptive strength
  - Improved RHCM v2 with maximum coverage preservation
- Tested each algorithm on six problem instances with varying parameters
- Visualized performance using fitness progression plots
- Analyzed solution quality
- Compared algorithms to identify the most effective approach

### Problem Definition
The Set Cover Problem involves selecting a minimum-cost subset of sets that cover all elements in a universe. Formally:

- Given a universe U of n elements
- A collection S of m sets containing elements from U
- A cost function c assigning costs to each set in S
- Find a minimum-cost subcollection of S that covers all elements in U

### Functions Developed

#### Instance Generation
```python
def generate_instance(universe_size, num_sets, density):
    sets = np.random.random((num_sets, universe_size)) < density
    for s in range(universe_size):
        if not np.any(sets[:, s]):
            sets[np.random.randint(num_sets), s] = True
    costs = np.power(sets.sum(axis=1), 1.1)
    return sets, costs
```
**Explanation:**  
This function creates random problem instances with specified parameters:

-  `universe_size`: The number of elements in the universe
-  `num_sets`: The number of sets available to choose from
-  `density`: The probability of an element being included in a set

The function ensures every element in the universe is covered by at least one set. Costs are assigned to each set proportionally to its size raised to the power of 1.1, creating a slight superlinear penalty for larger sets.

#### Solution Validation and Evaluation
```python
def valid(solution, sets):
    phenotype = np.logical_or.reduce(sets[solution])
    return np.all(phenotype)

@counter
def cost(solution, costs):
    return costs[solution].sum()

def fitness(solution, sets, costs):
    return (valid(solution, sets), -cost(solution, costs))
```
**Explanation:**  

- `valid()`: Checks if a solution covers all elements in the universe
- `cost()`: Calculates the total cost of a solution (decorated with a counter to track function calls)
- `fitness()`: Returns a tuple combining validity and negative cost (negative because we maximize fitness but minimize cost)

#### Mutation Operations
```python
def tweak(solution, num_sets, rng):
    new_solution = solution.copy()
    i = rng.integers(0, num_sets)
    new_solution[i] = not new_solution[i]
    return new_solution

def multiple_mutation(solution, num_sets, rng):
    mask = rng.random(num_sets) < 0.01
    new_solution = np.logical_xor(solution, mask)
    return new_solution

def multiple_mutation_improve(solution, strength, rng, num_sets):
    mask = rng.random(num_sets) < strength
    if not np.any(mask):
        mask[np.random.randint(num_sets)] = True
    
    new_solution = np.logical_xor(solution, mask)
    return new_solution

```
**Explanation:**  

- `tweak()`: Performs a single-bit mutation by flipping exactly one randomly selected bit in the solution. This creates small local changes for fine-tuning.
- `multiple_mutation()`: Performs a heavy mutation where each bit has a 1% probability of being flipped. This allows for larger jumps in the solution space and helps escape local optima.
- `multiple_mutation_improve()`: An enhanced version of multiple mutation with adaptive strength. The mutation probability is controlled by the `strength` parameter, and it guarantees at least one bit is flipped to ensure the solution changes.

#### Basic Hill Climber
```python
def solve_instance_hc(instance, max_iterations=10_000):
    sets, costs = generate_instance(instance["UNIVERSE_SIZE"], instance["NUM_SETS"], instance["DENSITY"])
    rng = np.random.Generator(np.random.PCG64([instance["UNIVERSE_SIZE"], instance["NUM_SETS"], int(10_000 *instance["DENSITY"])]))
    
    solution = rng.random(instance["NUM_SETS"]) < 1
    solution_fitness = fitness(solution, sets, costs)
    history = [solution_fitness[1]]
    
    for _ in range(max_iterations):
        new_solution = tweak(solution, instance["NUM_SETS"], rng)
        f = fitness(new_solution, sets, costs)
        history.append(f[1])
        if f > solution_fitness:
            solution = new_solution
            solution_fitness = fitness(solution, sets, costs)
    
    return solution, solution_fitness, history
```
**Explanation:**  
This implements a simple hill climbing algorithm that:

1. Generates a random initial solution
2. Iteratively applies small modifications (single-bit tweaks) to the solution
3. Accepts modifications that improve the fitness
4. Continues for a fixed number of iterations

#### Steepest Step 
```python
def solve_instance(instance, max_iterations=10_000):
    sets, costs = generate_instance(instance["UNIVERSE_SIZE"], instance["NUM_SETS"], instance["DENSITY"])
    rng = np.random.Generator(np.random.PCG64([instance["UNIVERSE_SIZE"], instance["NUM_SETS"], int(10_000 *instance["DENSITY"])]))
    
    solution = rng.random(instance["NUM_SETS"]) < 1
    solution_fitness = fitness(solution, sets, costs)
    
    best_solution = solution.copy()
    best_fitness = solution_fitness
    history = [solution_fitness[1]]
    
    iterations_without_improvement = 0
    max_iterations_without_improvement = 1000
    
    for _ in range(max_iterations):
        if iterations_without_improvement < max_iterations_without_improvement:
            new_solution = tweak(solution, instance["NUM_SETS"], rng)
        else:
            new_solution = multiple_mutation(solution, instance["NUM_SETS"], rng)
        
        new_fitness = fitness(new_solution, sets, costs)
        history.append(new_fitness[1])
        
        if new_fitness > solution_fitness:
            solution = new_solution
            solution_fitness = new_fitness 
            iterations_without_improvement = 0
            
            if solution_fitness > best_fitness:
                best_solution = solution.copy()
                best_fitness = solution_fitness
        else:
            iterations_without_improvement += 1
    
    return best_solution, best_fitness, history
```
**Explanation:**  
This enhanced version:

1. Keeps track of the best solution found so far
2. Counts iterations without improvement
3. Applies a more significant mutation(multiple tweak) when stuck (after 1000 iterations without improvement)


#### Simple RHCM (Random Heavy Climb Mutation)
```python

def solve_instance_rhcm(instance, max_iterations=10_000):
    sets, costs = generate_instance(instance["UNIVERSE_SIZE"], instance["NUM_SETS"], instance["DENSITY"])
    rng = np.random.Generator(np.random.PCG64([instance["UNIVERSE_SIZE"], instance["NUM_SETS"], int(10_000 *instance["DENSITY"])]))
    
    solution = rng.random(instance["NUM_SETS"]) < 1
    solution_fitness = fitness(solution, sets, costs)
    history = [solution_fitness[1]]
    
    for _ in range(max_iterations):
        new_solution = multiple_mutation(solution, instance["NUM_SETS"], rng)
        f = fitness(new_solution, sets, costs)
        history.append(f[1])
        if f > solution_fitness:
            solution = new_solution
            solution_fitness = fitness(solution, sets, costs)
    
    return solution, solution_fitness, history
```
**Explanation:**  
This approach:

1. Uses random heavy mutations that can flip multiple bits simultaneously
2. Each bit has a fixed 1% probability of flipping in each iteration
3. Allows the algorithm to make larger jumps in the solution space

#### Improved RHCM with Adaptive Strength
```python
def multiple_mutation_improve(solution, strength, rng, num_sets):
    mask = rng.random(num_sets) < strength
    if not np.any(mask):
        mask[np.random.randint(num_sets)] = True
    
    new_solution = np.logical_xor(solution, mask)
    return new_solution

def solve_instance_with_adaptive_strength(instance, max_iterations=10_000, buffer_size=500):
    sets, costs = generate_instance(instance["UNIVERSE_SIZE"], instance["NUM_SETS"], instance["DENSITY"])
    rng = np.random.Generator(np.random.PCG64([instance["UNIVERSE_SIZE"], instance["NUM_SETS"], int(10_000 *instance["DENSITY"])]))
    
    if instance["UNIVERSE_SIZE"] <= 1000:
        solution = rng.random(instance["NUM_SETS"]) < 1
    else:
        solution = rng.random(instance["NUM_SETS"]) < 0.5
    solution_fitness = fitness(solution, sets, costs)
    history = [solution_fitness[1]]
    
    strength = 0.5  
    buffer = []  
    
    for _ in range(max_iterations):
        new_solution = multiple_mutation_improve(solution, strength, rng, instance["NUM_SETS"])
        f = fitness(new_solution, sets, costs)
        history.append(f[1])
        
        buffer.append(f > solution_fitness)
        buffer = buffer[-buffer_size:]  
        
        if sum(buffer) > buffer_size/2:
            strength *= 1.3  
        elif sum(buffer) < buffer_size/2:
            strength /= 1.3  
        
        if f > solution_fitness:
            solution = new_solution
            solution_fitness = f
    
    return solution, solution_fitness, history
```
**Explanation:**  
This version introduces several improvements:

1. Dynamic adjustment of mutation strength based on recent performance
2. A buffer that tracks successful improvements over a window of iterations
3. Increases mutation strength when improvements are rare (exploration)
4. Decreases mutation strength when improvements are frequent (exploitation)
5. Guarantees at least one bit is flipped in each mutation
6. Different initialization strategies based on problem size

#### Improved RHCM v2 with Maximum Coverage Preservation
```python
def improved_multiple_mutation(solution, strength, rng, num_sets, sets):
    # Find the set with maximum coverage
    set_coverages = [len(s) for s in sets]
    max_coverage_index = np.argmax(set_coverages)
    
    # Create the mask for mutation, ensuring the max coverage set is not mutated
    mask = rng.random(num_sets) < strength
    mask[max_coverage_index] = False
    
    # Ensure at least one mutation occurs (excluding the max coverage set)
    if not np.any(mask):
        available_indices = [i for i in range(num_sets) if i != max_coverage_index]
        mask[rng.choice(available_indices)] = True
    
    # Apply the mutation
    new_solution = np.logical_xor(solution, mask)
    
    # Ensure the set with maximum coverage is always included
    new_solution[max_coverage_index] = True
    
    return new_solution

def solve_instance_with_adaptive_strength2(instance, max_iterations=20_000, buffer_size=500):
    sets, costs = generate_instance(instance["UNIVERSE_SIZE"], instance["NUM_SETS"], instance["DENSITY"])
    rng = np.random.Generator(np.random.PCG64([instance["UNIVERSE_SIZE"], instance["NUM_SETS"], int(10_000 *instance["DENSITY"])]))
    
    if instance["UNIVERSE_SIZE"] <= 1000:
        solution = rng.random(instance["NUM_SETS"]) < 1
    else:
        solution = rng.random(instance["NUM_SETS"]) < 0.5
    solution_fitness = fitness(solution, sets, costs)
    history = [solution_fitness[1]]
    
    strength = 0.5  
    buffer = []  
    
    for _ in range(max_iterations):
        new_solution = improved_multiple_mutation(solution, strength, rng, instance["NUM_SETS"], sets)
        f = fitness(new_solution, sets, costs)
        history.append(f[1])
        
        buffer.append(f > solution_fitness)
        buffer = buffer[-buffer_size:]  
        
        if sum(buffer) > buffer_size/2:
            strength *= 1.3  
        elif sum(buffer) < buffer_size/2:
            strength /= 1.3  
        
        if f > solution_fitness:
            solution = new_solution
            solution_fitness = f
    
    return solution, solution_fitness, history
```
**Explanation:**  
The final refined version adds these key improvements:

1. Identifies the set with maximum coverage
2. Prevents this high-value set from being removed in mutations
3. Always includes this set in the solution
4. Ensures mutations still occur in other parts of the solution
5. Doubles the maximum iterations to 20,000 for better convergence

### Results
The algorithms were tested on six different problem instances with varying parameters:

| Instance | Universe Size | Number of Sets | Density |
|----------|---------------|----------------|---------|
| 1        | 100           | 10             | 0.2     |
| 2        | 1,000         | 100            | 0.2     |
| 3        | 10,000        | 1,000          | 0.2     |
| 4        | 100,000       | 10,000         | 0.1     |
| 5        | 100,000       | 10,000         | 0.2     |
| 6        | 100,000       | 10,000         | 0.3     |

Performance was measured in terms of:

- Solution validity (coverage of all elements)
- Convergence behavior (fitness improvement over iterations)

The **Improved RHCM v2** algorithm produced the best results. Key factors contributing to its success:

1. **Adaptive mutation strength**: The dynamic adjustment of mutation strength based on recent performance allows the algorithm to balance exploration and exploitation effectively.

2. **Intelligent initialization**: Different initialization strategies for different problem sizes improve the starting point for the search.

3. **Guaranteed mutation**: Ensuring that at least one bit is flipped in each iteration prevents stagnation.

#### Improved RHCM Results (10,000 iterations)

| Instance | Best Fitness (Valid, Cost) |
|----------|----------------------------|
| 1        | (True, -279.27) |
| 2        | (True, -6,960.54) |
| 3        | (True, -124,682.61) |
| 4        | (True, -45,224,974.68) |
| 5        | (True, -94,720,184.03) |
| 6        | (True, -156,359,019.70) |

#### Improved RHCM v2 Results (20,000 iterations)

| Instance | Best Fitness (Valid, Cost) | Cost Function Calls |
|----------|----------------------------|---------------------|
| 1        | (True, -286.43) | 294,986 |
| 2        | (True, -7412.05) | 314,988 |
| 3        | (True, -128,212.00) | 334,990 |
| 4        | (True, -16,506,708.51) | 354,992 | 
| 5        | (True, -34,892,582.44) | 374,994 |
| 6        | (True, -57,611,475.77) | 394,996 |



#### Visualizations

#### Improved RHCM v2

![Improved RHCM v2](C:/Users/domen/Desktop/CI2024_project-work/report-image/image.png)

#### Imroved RHCM 

![ Imroved RHCM ](C:/Users/domen/Desktop/CI2024_project-work/report-image/image-1.png)

### Peer Review

From Gpir0:
> "You've done a good job testing various algorithms and writing clean code. The RHCM v2 algorithm you have chosen takes an intelligent approach: selecting the set with the greatest coverage and excluding it from tweaks. The idea of a dynamic buffer that adjusts the algorithm's strength based on how 'stuck' it is in finding a better solution is brilliant. It resolves many issues found in other proposed algorithms (including mine). As for the results, you've achieved an improvement compared to the classic hill climbing algorithm. For some instances, you achieved values above the student average, while for others, you were at the average. A possible improvement would be to define a ranking of the set coverage and use that directly, instead of searching for the maximum in each tweak. Since the evaluation considers the number of calls to the 'cost function', this improvement is not strictly necessary."

From s331345:
> "The Improved RHCM v2 is a further improvement to RHCM that ensures the set with maximum coverage is not only preserved but also enforces mutation to avoid stagnation... The algorithms demonstrate the effectiveness of adaptive techniques for improving performance over time, with Improved RHCM v2 yielding the best results across most instances."

### My Peer Reviwes

To LorenzoFormentin:
> "The solution reported, is an good approach to the set covering problem. The use of the hill climber with a simple tweak that goes has changing only one value in the solution, allows to concert on the exploitation, tending however very little towards the exploration this can then lead to the possibility of getting stuck in possible local maxima.
Overall ,Great job!!!"

To XhoanaShkajoti:
> "The reported solution is an excellent approach to the set coverage problem. The use of multiple mutation tweaks in combination with simulated Annealing allows a wide movement between the different solutions. The approach tends toward both exploitation and exploration making use of multiple fixed mutation tweaks and simulated Annealing. Thanks to the latter in particular, it is able to accept with a given temperature-dependent probability (which increases each time a worse solution is not accepted) worse solutions, thus allowing to move out of possible local minima.
Good job!!!"


### Reflections after laboratory

While I initially selected Improved RHCM v2 as the best algorithm based on the experimental results, further reflection revealed an important insight: the approach of always preserving the set with maximum coverage may not actually be optimal for all instances. A set that covers many elements might also have a disproportionately high cost, whereas multiple smaller sets might cover the same elements at a lower total cost.
In reality, the Improved RHCM v2 is not a significant improvement over the basic Improved RHCM(It's improved beacuse i used more iteration 20000 vs 10000), which achieved almost identical results without making assumptions about which sets should always be included. The original adaptive strength mechanism in Improved RHCM provides sufficient flexibility to explore the solution space effectively without explicitly forcing the inclusion of specific sets.


## Laboratory 2: Traveling Salesman Problem

**Objective:**  
To develop an efficient evolutionary algorithm for solving the Traveling Salesman Problem (TSP) with various datasets representing different countries, focusing on achieving high-quality routes while balancing exploration and exploitation.

### Activities Performed
- Implemented a Nearest Neighbor algorithm for generating initial solutions
- Developed a Simulated Annealing approach for population initialization
- Created a specialized Inver-Over crossover operator for the TSP
- Implemented an adaptive mutation strategy with dynamic rate adjustment
- Integrated a restart mechanism using Simulated Annealing
- Tested the algorithm on multiple country datasets (Vanuatu, Italy, Russia, US, China)
- Analyzed performance across different problem sizes
- Compared two algorithm variants and selected the more efficient one

### Problem Definition
The Traveling Salesman Problem involves finding the shortest possible route that visits each city exactly once and returns to the starting city. Formally:

- Given a set of cities and the distances between them
- Find a tour (a closed path) that visits each city exactly once
- Minimize the total distance traveled

### Functions Developed

#### Route Validation and Cost Calculation
```python
def valid(tsp):
    """
    It verifies the validity of a TSP route.
    """
    tsp = np.array(tsp)
    
    if tsp[0] != tsp[-1]:
        return False
    
    # Verify that all cities are visited exactly once.
    cities_visited = tsp[:-1]
    expected_cities = np.arange(len(CITIES))
    
    return (len(np.unique(cities_visited)) == len(CITIES) and 
            np.all(np.isin(expected_cities, cities_visited)))

@counter
def tsp_cost(route, dist_matrix):
    """It calculates the cost of the route and its validity."""
    route = np.array(route)
    # Calculates distances between consecutive cities.
    total_distance = np.sum([dist_matrix[route[i], route[i+1]] 
                           for i in range(len(route)-1)])
    return (total_distance, valid(route))
```
**Explanation:**  

- `valid()`: Checks if a TSP solution is valid by ensuring that it starts and ends at the same city and visits all cities exactly once.
- `tsp_cost()`: Calculates the total distance of a route and checks its validity. The `@counter` decorator keeps track of the number of function calls for performance analysis.

#### Nearest Neighbor Algorithm
```python
def nearest_neighbor_tsp(start_city_index=0):
    """
    Solves the TSP using the Nearest Neighbour Greedy algorithm.
    :param start_city_index: Index of the starting city
    :return: List of the order of the visited cities and total length of the route
    """
    dist_matrix=DIST_MATRIX.copy()
    visited = np.full(len(CITIES), False)  
    city = start_city_index  
    visited[city] = True

    tsp = [city]  

    while not np.all(visited):
        dist_matrix[:, city] = np.inf
        closest_city = np.argmin(dist_matrix[city])
        visited[closest_city] = True
        tsp.append(int(closest_city))
        city = closest_city

    tsp.append(start_city_index)
    
    total_distance = tsp_cost(tsp,DIST_MATRIX)
    
    return tsp, total_distance
```
**Explanation:**  
This greedy algorithm constructs a TSP tour by repeatedly visiting the nearest unvisited city. While it doesn't guarantee an optimal solution, it provides a good starting point for the evolutionary algorithm.

#### Simulated Annealing for Population Initialization
```python
def simulated_annealing_population(dist_matrix, initial_temp, cooling_rate, stop_temp, population_size=100):
    """Simulated Annealing algorithm to generate an initial population"""
    dist_matrix = np.array(dist_matrix)
     # Find an initial solution with the Greedy method.
    current_solution, current_cost = nearest_neighbor_tsp(start_city_index=0)
    best_solution = current_solution[:]
    best_cost = current_cost

    population = [(best_solution, best_cost)]
    temperature = initial_temp

    while temperature > stop_temp:
        new_solution = tweak(current_solution[:])
        new_cost = tsp_cost(new_solution, dist_matrix)

        if (new_cost[0] < current_cost[0] or 
            np.random.random() < np.exp((current_cost[0] - new_cost[0]) / temperature)):
            current_solution = new_solution
            current_cost = new_cost

            if new_cost[0] < best_cost[0]:
                best_solution = new_solution
                best_cost = new_cost
    
            population.append((current_solution, current_cost))
            
        temperature *= cooling_rate
    # Sort the population by cost and return the best 'population_size' individuals    
    population = np.array(population, dtype=object)
    sorted_indices = np.argsort([cost[0] for _, cost in population])
    population = population[sorted_indices]
    
    return population[:population_size].tolist()
```
**Explanation:**  
Instead of generating a completely random initial population, this function uses Simulated Annealing to create a set of high-quality diverse solutions. It starts with a solution from the Nearest Neighbor algorithm and generates variations by:

1. Gradually reducing the temperature
2. Accepting improving solutions and occasionally worse solutions (with probability based on temperature)
3. Building a population of solutions encountered during the search
4. Returning the best solutions found

#### Inversion Mutation
```python
def tweak(tsp):
    """Performs an inversion mutation on a TSP solution"""
    tsp = np.array(tsp)
    # Select two random points in the path to perform the inversion.
    a, b = sorted(np.random.choice(range(1, len(tsp) - 1), 2, replace=False))
    # Reverses the sub-path between the two selected points.
    tsp[a:b+1] = np.flip(tsp[a:b+1])
    return tsp.tolist()
```
**Explanation:**  
This mutation operator:

1. Selects two random positions in the route (excluding the start/end city)
2. Reverses the segment between these positions
3. Maintains tour validity (because reversing a segment preserves connectivity)
This is one of the most effective mutation operators for TSP because it makes meaningful changes to the solution structure.

#### Inver-Over Crossover
```python
def inver_over_crossover(parent1, parent2, num_iterations=1):
    """Apply inver-over crossover """
    parent1, parent2 = np.array(parent1), np.array(parent2)
    child = parent1.copy()
 # Perform a series of crossover iterations to create the child
    for _ in range(num_iterations):
        start_index = np.random.randint(1, len(child) - 2)
        start_city = child[start_index]
        # Randomly select a destination city from one of the parents.
        if np.random.random() < 0.5:
            valid_cities = child[1:-1][child[1:-1] != start_city]
            end_city = np.random.choice(valid_cities)
        else:
            valid_cities = parent2[1:-1][parent2[1:-1] != start_city]
            end_city = np.random.choice(valid_cities)
        # Get start and destination city locations in the son.
        start_pos = np.where(child == start_city)[0][0]
        end_pos = np.where(child == end_city)[0][0]
        
        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos
          # Reverse the sub-path between the two positions.
        child[start_pos:end_pos + 1] = np.flip(child[start_pos:end_pos + 1])
    # The child ends up in the same city as the starting city.
    if child[0] != child[-1]:
        child = np.append(child, child[0])
    
    return child.tolist()
```
**Explanation:**  
This specialized crossover operator for TSP:

1. Creates a child that inherits characteristics from both parents
2. Uses inversion operations as the basis for crossover
3. Selects a city from the first parent, then a destination city from either parent
4. Reverses the segment between them to create new route structures
5. Can perform multiple iterations of this process
6. Ensures the resulting child is a valid tour

#### Tournament Selection
```python
def tournament_selection(population, k=5):
    """Tournament selection using NumPy."""
    # Randomly selects 'k' individuals from the population.
    selected_indices = np.random.choice(len(population), k, replace=False)
    selected = np.array(population, dtype=object)[selected_indices]
    # Returns the individual with the lowest cost among those selected.
    return min(selected, key=lambda x: x[1][0])  # x[1][0] accede al costo
```
**Explanation:**  
Tournament selection provides selection pressure while maintaining diversity:

1. Randomly selects k individuals from the population
2. Returns the best individual from this tournament
3. Allows both good and occasionally average solutions to be selected
4. Provides a good balance between exploitation (selecting good solutions) and exploration (giving chances to diverse solutions)

#### Main Evolutionary Algorithm
```python
def evolutionary_algorithm_tsp_adaptive_mutation(
    
    dist_matrix,
    population_size=100, 
    tournament_size=10, 
    num_generations=10000, 
    initial_mutation_rate=0.1, 
    crossover_iterations=1,
    max_generations_without_improvement=500,
    mutation_increase_factor=1.5,
    mutation_decrease_factor=0.7,
    mutation_threshold=1.3, # Threshold for restarting
    initial_temp=1000,      
    cooling_rate=0.992,
    stop_temp=1e-200,

):
    """It solves the TSP using a Evolutionary Algorithm with adaptive mutation."""
    dist_matrix = np.array(dist_matrix)
    population = initial_population(population_size)
    current_mutation_rate = initial_mutation_rate

    best_solution, best_cost = min(population, key=lambda x: x[1][0])
    all_time_best_solution = best_solution
    all_time_best_cost = best_cost
    generations_without_improvement = 0
    

    for generation in tqdm(range(num_generations)):
        # Check if it is necessary to reboot with SA.
        if current_mutation_rate > mutation_threshold:
            print(f"\nMutation rate ({current_mutation_rate:.3f}) exceeded threshold ({mutation_threshold})")
            print("Restarting with Simulated Annealing...")
            
             # Run SA and get new population
            new_population = simulated_annealing_population(
                dist_matrix, 
                initial_temp, 
                cooling_rate, 
                stop_temp, 
                population_size
            )
            population = sorted(new_population, key=lambda x: x[1][0])[:population_size]
             # Mutation rate reset
            current_mutation_rate = initial_mutation_rate
            print("Population refreshed and mutation rate reset")

        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            child = inver_over_crossover(parent1[0], parent2[0], crossover_iterations)
            
            if np.random.random() < current_mutation_rate:
                child = inversion_mutation(child)

            child_cost = tsp_cost(child, dist_matrix)
            new_population.append((child, child_cost))

        combined_population = np.array(new_population + population, dtype=object)
        sorted_indices = np.argsort([cost[0] for _, cost in combined_population])
        population = combined_population[sorted_indices][:population_size].tolist()

        new_best_solution, new_best_cost = min(population, key=lambda x: x[1][0])

         # Update the best result ever if needed
        if new_best_cost[0] < all_time_best_cost[0]:
            all_time_best_solution = new_best_solution
            all_time_best_cost = new_best_cost

        if new_best_cost[0] < best_cost[0]:
            best_solution, best_cost = new_best_solution, new_best_cost
            generations_without_improvement = 0
            current_mutation_rate = max(initial_mutation_rate, 
                                     current_mutation_rate * mutation_decrease_factor)
        else:
            generations_without_improvement += 1

        if generations_without_improvement > max_generations_without_improvement:
            current_mutation_rate *= mutation_increase_factor
            generations_without_improvement = 0

        if generation % 1000 == 0:
            print(f"Generation {generation}: "
                  f"Best current cost = {new_best_cost[0]:.2f} | "
                  f"Absolute best cost = {all_time_best_cost[0]:.2f} | "
                  f"Mutation rate = {current_mutation_rate:.3f} | "
                  f"Valid = {best_cost[1]}")
    
    return all_time_best_cost
```
**Explanation:**  
This is the main evolutionary algorithm that integrates all components into a complete solution. Its key features include:

1. **Adaptive Mutation Rate**: 
   - Starts with an initial mutation rate (0.1)
   - Decreases when improvements are found (multiply by 0.7)
   - Increases when stuck for 500 generations (multiply by 1.5)
   - Balances exploration and exploitation dynamically

2. **Simulated Annealing Restart**:
   - Triggered when mutation rate exceeds a threshold (1.3)
   - Completely refreshes the population using SA
   - Resets the mutation rate to its initial value
   - Helps escape local optima and explore new regions of the search space

3. **Selection and Reproduction**:
   - Uses tournament selection to pick parents
   - Applies the Inver-Over crossover to create offspring
   - Applies inversion mutation based on the current mutation rate
   - Combines offspring with parents and selects the best solutions

4. **Memory of Best Solution**:
   - Maintains the best solution found across all generations
   - Allows tracking progress even after restarts
   - Returns the overall best solution at the end

### Results

#### Vanuatu
- **Best route length**: 1,345.54 km
- **Function evaluations**: 58,196
- **Notes**: For this smaller dataset, applying just Simulated Annealing without the full evolutionary algorithm was sufficient.

#### Italy
- **Best route length**: 4,172.76 km
- **Function evaluations**: 10,872,940
- **Parameters**: Mutation increase factor = 1.3, decrease factor = 0.7
- **Notes**: The algorithm triggered one SA restart at around generation 5100.

#### Russia
- **Best route length**: 33,882.63 km
- **Function evaluations**: 2,290,980
- **Parameters**: Mutation threshold = 1.0
- **Notes**: The algorithm triggered one SA restart at around generation 8800.

#### United States
- **Best route length**: 39,614.05 km
- **Function evaluations**: 4,465,568
- **Generations**: 20,000
- **Notes**: The algorithm triggered two SA restarts, at generations 13600 and 17150.

#### China
- **Best route length**: 52,353.58 km
- **Function evaluations**: 13,931,136
- **Generations**: 30,000
- **Notes**: Required the longest run time due to dataset size, but achieved steady improvement throughout execution.

### Alternative Version
I developed a second version of the algorithm that included a population diversity check. This variant would insert new random solutions when the population became too homogeneous. Testing showed that this additional mechanism did not lead to significant improvements over the original version and added computational overhead. Therefore, the first version was selected as it proved to be faster while achieving equally good results.


### Peer Review
From LucianaColella7:
> "The Genetic Algorithm is well-implemented with a good mix of techniques, obtaining very good results. The adaptive mutation rate is a great addition, as it adjusts to balance exploration and exploitation, especially when it detects a lack of improvement. This is a smart way to prevent the algorithm from getting stuck in a local optimum. What I appreciated the most is the use of the Simulated Annealing restart. Another effective way to avoid local minima and give the algorithm the chance to keep exploring
In the README, it would have been helpful to include a summary of results for each country, as it would provide a clearer picture of the best outcomes from your implementation."

### My Peer Reviwes

To GiovanniTagliaferri:
> "Your work on the travelling salesman problem (TSP) with an approach combining the Greedy algorithm and a Genetic Algorithm is well done and shows a good understanding of both the theoretical foundations and their practical application.
One aspect I noticed concerns the calculation of distances. You chose to calculate them manually using the Euclidean formula, and the method works well for small sets of cities. However, it would be useful to explore the use of a library such as Geopy, especially for larger datasets: these libraries are optimised to calculate distances between geographical coordinates, saving computational resources and improving the efficiency of the algorithm.
Regarding the mutation, you have opted for a scramble mutation in both the implementation of the evolutionary algorithm and the elite solutions. This approach certainly introduces variety, but it can also lead to the loss of good gene segments, making it more difficult to maintain valid solutions between generations. I would suggest experimenting with an inversion mutation: instead of randomly rearranging genes, this mutation reverses a section of genes, maintaining blocks that may contain good solutions. The scramble mutation, on the other hand, is good for avoiding the risk of stagnation and lack of diversity.
Furthermore, as far as crossover is concerned, the partially mapped crossover (PMX) method you implemented is a good starting point. However, it can mix genes too much and break up sections of solutions that could be useful. You could try an inver-over crossover, which allows you to keep larger sections of good genes and preserve significant sequences, improving the maintenance of promising substructures between generations.
Overall, this is solid work. Some minor adjustments such as the use of a distance library and slightly different mutation and crossover strategies could make your algorithm more efficient and further improve your results."

To michepaolo:
> "The code addresses the travelling salesman problem by combining a greedy approach (fast algorithm) with a genetic algorithm (slow algorithm) to find optimised paths. The use of mutation and crossover functions is well thought out, especially with the implementation of inversion_mutation, which preserves significant sections of genes by inverting their order. This approach is effective in maintaining good solutions while avoiding altering the pathways found too much.
Inversion_over_crossover is also interesting because, unlike other crossovers, it does not drastically change genes and preserves effective sections of the pathway. In addition, the choice of dynamically varying the probability of using inver_over_crossover versus mutation and vice versa is remarkable, maintaining a good balance between exploration and exploitation of the solution.
The addition of a local_search on the first 10 individuals further improves the solutions, avoiding letting the algorithm stagnate. In addition, controlling the best_cost_unchanged variable, which triggers the random reintroduction of new individuals when no improvement is observed, is a good attempt (surely there are better ones) to reshuffle the population and overcome any blockages caused by lack of diversity. However, since parental selection only takes place among the first half of the population, there is a risk that these new random solutions, due to their high initial cost, are never really considered, thus limiting the contribution to population diversity.
In summary, the combined approach of greedy, targeted crossover, local search and dynamic management of crossover and mutation probability led to excellent results, balancing solution quality and genetic diversity effectively. Excellent work!"

## Laboratory 3: N-Puzzle Solver


**Objective:**  
To implement an efficient A* search algorithm for solving the N-puzzle problem, focusing on developing advanced heuristics to improve search efficiency.

### Activities Performed
- Implemented the A* search algorithm with careful state representation
- Developed advanced combined heuristics for more accurate distance estimation
- Created functions for state manipulation and action generation
- Implemented a puzzle generator for creating test instances
- Analyzed algorithm performance in terms of nodes expanded and solution quality


### Problem Definition
The N-puzzle consists of (N²-1) numbered tiles and one empty space arranged in an N×N grid. The goal is to rearrange the tiles from an initial scrambled state to a target configuration by sliding tiles into the empty space.

Formally:

- Initial state: A random configuration of tiles
- Actions: Moving a tile adjacent to the empty space into that space
- Goal state: Tiles arranged in order with the empty space in the bottom-right corner
- Cost: Each move has a cost of 1, and we want to minimize the total number of moves

### Functions Developed

#### Action and State Representation
```python
PUZZLE_DIM = 9
Action = namedtuple('Action', ['pos1', 'pos2'])

# Pre-calculates target positions for heuristics
TARGET_POSITIONS = {
    value: divmod(value-1, PUZZLE_DIM) 
    for value in range(1, PUZZLE_DIM**2)
}
```
**Explanation:**  

- `PUZZLE_DIM`: Defines the puzzle size (9×9)
- `Action`: A named tuple representing a move between two positions (the empty space and an adjacent tile)
- `TARGET_POSITIONS`: A pre-calculated dictionary mapping each tile value to its target position, optimizing heuristic calculations

#### Available Actions and State Transitions
```python
def available_actions(state: np.ndarray)->list[Action]:
    # Position of 0 (empty space)
    x, y = [int(_[0]) for _ in np.where(state == 0)]
    actions = []
    # List of possible directions
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < PUZZLE_DIM and 0 <= new_y < PUZZLE_DIM:
            actions.append(Action((x, y), (new_x, new_y)))
    return actions        

def do_action(state: np.ndarray, act: Action)-> np.ndarray:
    new_state = state.copy()
    new_state[act.pos1], new_state[act.pos2] = new_state[act.pos2], new_state[act.pos1]
    return new_state
```
**Explanation:**  

- `available_actions()`: Finds all valid moves by identifying adjacent positions to the empty space
- `do_action()`: Applies a move by swapping the empty space with an adjacent tile

#### Advanced Heuristic Function
```python
def heuristic(state: np.ndarray)->int:
    """ 
    Advanced heuristics combining:
    1. Manhattan distance
    2. Linear conflicts (rows and columns)
    3. Corner tiles detection
    4. Last moves detection
    """
    distance = 0
    row_conflicts = 0
    col_conflicts = 0
    
    # Manhattan distance
    for x in range(PUZZLE_DIM):
        for y in range(PUZZLE_DIM):
            value = state[x, y]
            if value != 0:
                target_x, target_y = TARGET_POSITIONS[value]
                distance += abs(x - target_x) + abs(y - target_y)

    # Linear conflicts - Rows check
    for row in range(PUZZLE_DIM):
        for i in range(PUZZLE_DIM):
            tile_i = state[row, i]
            if tile_i == 0:
                continue
            target_row_i, _ = TARGET_POSITIONS[tile_i]
            # Check if the tile is in the correct row
            if target_row_i == row:
                for j in range(i + 1, PUZZLE_DIM):
                    tile_j = state[row, j]
                    if tile_j != 0 and TARGET_POSITIONS[tile_j][0] == row and tile_i > tile_j:
                        row_conflicts += 2  # Add penalty for linear conflict
    
    # Linear conflicts - Column check
    for col in range(PUZZLE_DIM):
        for i in range(PUZZLE_DIM - 1):
            tile_i = state[i, col]
            if tile_i == 0:
                continue
            target_row_i, target_col_i = TARGET_POSITIONS[tile_i]
            
            if target_col_i == col:
                for j in range(i + 1, PUZZLE_DIM):
                    tile_j = state[j, col]
                    if tile_j != 0:
                        target_row_j, target_col_j = TARGET_POSITIONS[tile_j]
                        if target_col_j == col and target_row_i > target_row_j:
                            col_conflicts += 2
    
    # Corner tiles detection
    corner_penalty = 0
    corners = [(0, 0), (0, PUZZLE_DIM-1), (PUZZLE_DIM-1, 0), (PUZZLE_DIM-1, PUZZLE_DIM-1)]
    for x, y in corners:
        value = state[x, y]
        if value != 0:
            target_x, target_y = TARGET_POSITIONS[value]
            if (x, y) != (target_x, target_y) and (target_x, target_y) in corners:
                corner_penalty += 2
    
    # Last moves pattern detection
    last_moves_penalty = 0
    if (PUZZLE_DIM >= 4):
        # Checking whether the last two tiles of the final row/column are interchanged
        if state[PUZZLE_DIM-1, PUZZLE_DIM-2] == PUZZLE_DIM**2 - 1 and \
            state[PUZZLE_DIM-1, PUZZLE_DIM-1] == PUZZLE_DIM**2 - 2:
            last_moves_penalty += 6
            
        if state[PUZZLE_DIM-2, PUZZLE_DIM-1] == PUZZLE_DIM**2 - 1 and \
            state[PUZZLE_DIM-1, PUZZLE_DIM-1] == PUZZLE_DIM**2 - 3:
                last_moves_penalty += 6

    return distance + row_conflicts + col_conflicts + corner_penalty + last_moves_penalty
```
**Explanation:**  
This heuristic function combines four different metrics to estimate the distance to the goal state:

1. **Manhattan Distance**: Calculates the sum of horizontal and vertical distances each tile must move to reach its target position. This is the foundation of the heuristic.

2. **Linear Conflicts**: Identifies situations where two tiles are in their correct row or column but in the wrong order relative to each other. Each conflict adds a penalty of 2 moves, reflecting the additional steps needed to resolve these situations.

3. **Corner Tiles Detection**: Adds penalties when corner tiles are misplaced but their targets are also corners. Corner tiles are particularly difficult to position correctly in sliding puzzles.

4. **Last Moves Pattern Detection**: Adds penalties for specific patterns involving the last few tiles, which are often the most difficult to arrange correctly in the final stages of solving.



#### A* Search Implementation
```python
def a_star(initial_state: np.ndarray):
    open_set = []
    counter = itertools.count()
    goal_state = np.array([i for i in range(1, PUZZLE_DIM**2)] + [0]).reshape((PUZZLE_DIM, PUZZLE_DIM))
    
    # Set for visited states 
    visited = set()
    
    # Initial estimation of heuristics 
    initial_h = heuristic(initial_state)
    heapq.heappush(open_set, (initial_h, next(counter), initial_state, []))
    
    nodes_expanded = 0
    
    while open_set:
        f_score, _, state, path = heapq.heappop(open_set)
        nodes_expanded += 1
        
        if np.array_equal(state, goal_state):
            print(f"Cost: {nodes_expanded}")
            return path, state
        
        state_key = str(state.flatten().tolist())
        if state_key in visited:
            continue
            
        visited.add(state_key)
        
        actions = available_actions(state)
        for act in actions:
            new_state = do_action(state, act)
            new_state_key = str(new_state.flatten().tolist())
            
            if new_state_key not in visited:
                g_score = len(path) + 1
                h_score = heuristic(new_state)
                f_score = g_score + h_score
                heapq.heappush(open_set, (f_score, next(counter), new_state, path + [act]))
    
    return None
```
**Explanation:**  
This is a standard implementation of A* search algorithm:

1. **Priority Queue**: Uses a min-heap to efficiently select the state with the lowest f-score (g + h) for exploration

2. **Duplicate Detection**: Maintains a set of visited states to avoid exploring the same state multiple times

3. **State Representation**: Uses a flattened string representation of the state for efficient hashing and set operations

4. **Path Tracking**: Maintains the sequence of actions leading to each state for solution reconstruction

5. **Counter Tie-Breaking**: Uses a monotonically increasing counter to break ties between states with the same f-score, ensuring that older nodes are explored first

6. **Performance Tracking**: Counts the number of nodes expanded during the search

#### Puzzle Generator
```python
def generate_puzzle(num_moves=50):
    state = np.array([i for i in range(1, PUZZLE_DIM**2)] + [0]).reshape((PUZZLE_DIM, PUZZLE_DIM))
    
    for _ in range(num_moves):
        acts = available_actions(state)
        state = do_action(state, choice(acts))
    
    return state
```
**Explanation:**  
This function creates test instances by starting from the goal state and applying a series of random valid moves. This ensures that the generated puzzles are solvable, which is important because not all arrangements of tiles form valid N-puzzle instances.


### Results
The algorithm was tested on a 9×9 puzzle (80-puzzle) scrambled with 160 random moves.

- **Solution found in 145.75 seconds**
- **Nodes expanded: 274,068**
- **Solution length: 70 moves**


### Peer Review
From GioSilve:
> "The solution of the problem correctly implements an A* search algorithm, with the most important feature being the heuristic function, which combines four different types of metrics, and it allows the algorithm to be more informed and thus to solve much more efficiently the problem with even a large dimension puzzle, such as the 9x9 in less than 3 minutes that is reported in the output, that's excellent!"

From YaelNaor11:
> "Wow, it's impressive that you worked on the 9-puzzle and managed to implement the solution in such a short time! Your code is clear and demonstrates a good understanding of path-searching algorithms. Great work!"

### My Peer Reviwes

To graicidem:
> "Good job on implementing, in addition to A*, a bi-directional version of it. This approach significantly reduces the number of nodes explored, thanks to the two parallel searches: one starting from the initial state and one from the target state, meeting halfway.
The combined use of two heuristics reinforces the effectiveness of the algorithm:

>1. **Manhattan distance,** which measures how far each tile is from the final position.
>2. **Linear conflicts**, which penalise tiles that block each other in the same row or column.
>These choices ensure optimal solutions with fewer computational resources. To further improve, new heuristics could be introduced especially with higher values of N, such as the detection of misplaced corner tiles or necessary end moves, increasing accuracy without compromising efficiency.

>1. **Corner tiles detection**: detects and penalises situations where corner tiles are stuck in wrong positions, making their placement more complex.
>2. **Last moves detection**: penalises configurations that require additional moves to correctly position the last pieces, including the empty tile and the final pieces of the puzzle.

>Overall, your code is clear and demonstrates a good understanding of path-searching algorithms. Great work!"

To fedefortu8:
> "Excellent work in implementing the A* search algorithm to solve the N^2-1 puzzle problem. This approach demonstrates a sound understanding of informed search, effectively exploiting the Manhattan distance as a heuristic function. This choice is well motivated for the context of the puzzle, as it measures the overall distance between each tile and its target position, guiding the search optimally.
The implementation lacks a required element in the delivery: a counter that tracks the total number of states (or actions) evaluated during execution. (Recommendation to add it)
To further improve the algorithm, it would also be interesting to include additional heuristic criteria, such as:

>1. **Linear Conflict:** Add penalties for tiles in the correct row or column but in the wrong order, improving the accuracy of the estimation.
>2. **Corner Tiles Detection:** Evaluate problematic configurations with tiles in the wrong corners, a common situation in puzzles.
>3. **Last Move Detection:** Introduce a check to avoid repetitive moves that do not make significant progress.

>The last two especially with larger N-values
Overall, your code is clear and demonstrates a good understanding of path-searching algorithms. Great work!"

## Project work: Symbolic Regression with Genetic Programming

**Objective:**  
To implement an efficient Genetic Programming (GP) algorithm for symbolic regression, focusing on discovering mathematical expressions that approximate underlying functions based on observed data points.

### Activities Performed

- Designed and implemented a complete tree-based Genetic Programming framework
- Created an expression representation system with function and terminal nodes
- Developed advanced genetic operators (crossover, mutation, selection)
- Implemented Semantic Fitness Sharing and solution aging for maintaining population diversity
- Implemented an island model for maintaining population diversity
- Added adaptive mutation rates to balance exploration and exploitation
- Incorporated symbolic simplification of expressions
- Built a bloat control mechanism to prevent excessive expression growth
- Tested the system on nine different regression problems of varying complexity


### Problem Definition

Symbolic regression involves finding a mathematical expression that best fits a given dataset of input-output pairs, without assuming a specific model structure. Formally:

- Given a dataset of observations (X, y)
- Find a symbolic expression f such that f(X) approximates y as closely as possible

### Summary of the main points behind the algorithm

#### Function and Terminal Weighting

The algorithm implements a weighting system that allows for strategic biasing of the search process by assigning different selection probabilities to functions and terminals during tree generation. This mechanism enables the algorithm to focus exploration on more promising mathematical components while reducing the likelihood of selecting less relevant operations for specific problem domains.

#### Individual Representation

The solutions are represented as expression trees. Each tree consists of two primary node types: function nodes and terminal nodes. Function nodes represent mathematical operations such as addition, multiplication, division, trigonometric functions, and logarithms. These function nodes have a specific arity (number of arguments) and operate on their child nodes. Terminal nodes represent either variables (inputs from the dataset) or constants, serving as the leaves of the tree.

This representation creates a direct mapping between the tree structure and mathematical expressions.

Tree initialization employs a technique called "ramped half-and-half," which combines two tree-generation methods. The "full" method creates trees with all leaf nodes at the same depth, while the "grow" method allows for more varied structures where some branches may terminate earlier than others. By combining these approaches and randomizing depths within a predefined range, the initialization produces a structurally diverse population—a critical factor for effective exploration of the solution space.

#### Semantic Fitness Sharing

Traditional fitness evaluation in genetic programming often leads to premature convergence on structurally similar solutions. Semantic fitness sharing addresses this limitation by considering the behavioral characteristics of solutions rather than merely their structural differences.

The semantic distance between two expressions is calculated by evaluating them on a representative sample of data points and measuring the difference between their output patterns. Mathematically, this distance is computed as the mean squared error between the normalized outputs:

```
semantic_distance(tree1, tree2) = mean((normalized_output1 - normalized_output2)²)
```

Fitness sharing penalizes individuals that behave similarly to others in the population. For each individual, a sharing factor is calculated based on how many other expressions produce similar outputs and how close these outputs are. This factor increases as more semantically similar neighbors are found within a specified radius (sigma).

The adjusted fitness is then calculated by multiplying the original fitness by this sharing factor:

```
adjusted_fitness = original_fitness * sharing_factor
```

Since the algorithm minimizes fitness values, this adjustment makes semantically common solutions less attractive, creating selective pressure toward behavioral diversity. 

#### Crossover and Mutation Techniques

The genetic operators in this approach facilitate both exploration of the search space and exploitation of promising solutions.

Subtree crossover exchanges genetic material between two parent trees by selecting random crossover points in each parent and swapping the entire subtrees rooted at those points. This operation preserves syntactic correctness while creating new combinations of mathematical components. To prevent excessive growth, the algorithm validates that the resulting offspring do not exceed a maximum depth, attempting alternative crossover points if necessary.

Two distinct mutation operators modify existing solutions:

1. Subtree mutation replaces a randomly selected subtree with an entirely new randomly generated one. This operation can introduce novel mathematical structures, facilitating exploration of the search space.

2. Point mutation makes more targeted changes without altering the overall tree structure. For function nodes, it substitutes the function with another compatible one of the same arity. For terminal nodes, it either replaces a variable with another variable or slightly perturbs a constant value. Point mutation provides refinement capability, enabling fine-tuning of promising solutions.

#### Adaptive Mutation Strength

The algorithm implements an adaptive mutation mechanism that dynamically adjusts mutation rates based on evolutionary progress. The adaptation follows these principles:

When improvements occur (finding better solutions), mutation strength decreases while during stagnation (no improvement for several generations), mutation strength increases

This dynamic adjustment creates an automatic balance between exploration and exploitation. When progress stalls, the increased mutation encourages larger jumps in the search space. When promising areas are found, decreased mutation allows for more refined improvements. Each island maintains its own mutation strength, allowing different regions of the search space to be explored with different strategies simultaneously.

#### Migration with Adaptive Mutation

The island model divides the population into semi-isolated subpopulations that evolve independently most of the time. This approach leverages the benefits of parallel evolution—different islands can explore different regions of the search space and develop distinct solution strategies.

Periodically, migration occurs where individuals move from one island to another. Migration follows these steps:

1. For each source island, migrants are selected (a combination of the best individuals and some random selections).

2. The semantic diversity of both source and destination islands is calculated.

3. The mutation strength for migrants is adjusted based on the diversity relationship:
   - When moving from a more diverse island to a less diverse one, minimal mutation is applied to preserve the valuable diversity being introduced.
   - When moving from a less diverse island to a more diverse one, stronger mutation is applied to help migrants adapt to the new environment.

4. Selected migrants undergo mutation with the calculated strength.

5. The migrants replace the worst individuals in the destination island.


### Functions Developed

This section provides an overview of the key functions and classes developed for the Genetic Programming symbolic regression algorithm.

#### Expression Tree Classes

```python
class Node:
    """Base class for representing a node in the expression tree"""
    def __init__(self):
        self.depth = 0  # Node depth
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluates the node given an input X"""
        raise NotImplementedError("You must implement the evaluate method in the subclass")
        #[.... other fuction]
```

**Explanation:**  
The `Node` class serves as the abstract base class for all nodes in the expression tree. It defines common properties like depth and abstract methods that all nodes must implement, such as evaluation, copying, string representation, and complexity calculation.

```python
class FunctionNode(Node):
    """Class to represent a function node in the expression tree"""
    def __init__(self, function: Callable, arity: int, symbol: str, children: List[Node] = None):
        super().__init__()
        self.function = function # Function to be applied
        self.arity = arity # Number of arguments the function takes
        self.symbol = symbol # Symbol to represent the function
        self.children = children if children is not None else []

    def evaluate(self, X: np.ndarray)->np.ndarray:
        """Evaluate the function by applying it to the children's results"""
        args= [child.evaluate(X) for child in self.children]
        return self.function(*args)
    
        #[.... other fuction]
```

**Explanation:**  
The `FunctionNode` class represents operations in the expression tree. Each function node stores:

- The actual mathematical function to be applied (e.g., numpy's add, subtract)
- The arity (number of arguments) the function requires
- A symbolic representation for display (e.g., '+', '-', '*')
- A list of child nodes representing the function's arguments

The evaluate method recursively evaluates all children and then applies the function to their results.

```python
class TerminalNode(Node):
    """Represents an end node in the tree (variable or constant)"""
    def __init__(self, value, is_variable: bool = False, var_index: int = None):
        super().__init__()
        self.value = value
        self.is_variable = is_variable
        self.var_index = var_index  # only used if is_variable is True
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the terminal node"""
        if self.is_variable:
            # If it is a variable, we take the value from the input X
            if X.ndim == 1 and self.var_index == 0:
                return X  # special case for 1D 
            else:
                return X[:, self.var_index]
        else:
            # If it is a constant, we return the value (broadcast on all samples)
            return np.full(X.shape[0] if X.ndim > 1 else len(X), self.value)

    #[.... other fuction]
```

**Explanation:**  
The `TerminalNode` class represents leaf nodes in the expression tree—either variables or constants. For variables, it stores the index of the feature it represents. For constants, it stores the actual numeric value. The evaluate method either returns the corresponding feature values from the input data X (for variables) or the constant value broadcasted to match the input size.

```python
class ExpressionTree:
    """It represents a complete expression tree"""
    
    def __init__(self, root: Node):
        self.root = root
        self.update_node_depths()
        self.fitness = None
        self.adjusted_fitness = None  # for fitness sharing
        self.age = 0  # for age of the tree
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """Evaluates the expression tree on input data"""
        return self.root.evaluate(X)

    #[.... other fuction]
```

**Explanation:**  
The `ExpressionTree` class encapsulates a complete mathematical expression represented as a tree. It maintains a reference to the root node and tracks important metadata including fitness values and age (used for diversity maintenance). The evaluate method delegates to the root node, initiating the recursive evaluation of the entire expression tree.



#### Tree Initialization Methods

```python
def grow_tree(config: GPConfig, max_depth: int, min_depth: int = 1, current_depth: int = 0) -> Node:
    """
    Grow' method for generating a tree with variable depth
    
    Args:
        config: GP configuration
        max_depth: Maximum depth of the tree
        min_depth: Minimum depth of the tree
        current_depth: Current depth of the node
        
    Returns:
        Root node of the generated tree
    """
    # If we are at the maximum depth, we can only create terminal nodes
    if current_depth >= max_depth:
        terminal_info = config.get_random_terminal()
        if terminal_info['is_variable']:
            return TerminalNode(None, is_variable=True, var_index=terminal_info['var_index'])
        else:
            return TerminalNode(terminal_info['value'], is_variable=False)
    
    # If we have not yet reached the minimum depth, we only create function nodes
    if current_depth < min_depth:
        function_info = config.get_random_function()
        children = [grow_tree(config, max_depth, min_depth, current_depth + 1) for _ in range(function_info['arity'])]
        return FunctionNode(function_info['function'], function_info['arity'], function_info['symbol'], children)
    
    # Otherwise, we randomly choose between functions and terminals
    if random.random() < 0.5:  # 50% probability for functions or terminals
        function_info = config.get_random_function()
        children = [grow_tree(config, max_depth, min_depth, current_depth + 1) for _ in range(function_info['arity'])]
        return FunctionNode(function_info['function'], function_info['arity'], function_info['symbol'], children)
    else:
        terminal_info = config.get_random_terminal()
        if terminal_info['is_variable']:
            return TerminalNode(None, is_variable=True, var_index=terminal_info['var_index'])
        else:
            return TerminalNode(terminal_info['value'], is_variable=False)
```

**Explanation:**  
The `grow_tree` method generates expression trees with variable shapes. It follows these rules:

1. At the maximum depth, only terminal nodes (variables or constants) are created
2. Before the minimum depth, only function nodes are created to ensure sufficient complexity
3. Between min_depth and max_depth, nodes are created with a 50% probability of being a function or terminal

This approach creates trees with diverse structures and varying depths, important for initial population diversity.

```python
def full_tree(config: GPConfig, max_depth: int, current_depth: int = 0) -> Node:
    """
    Full' method for generating a tree with all branches at the same depth
    
    Args:
        config: GP configuration
        max_depth: Maximum depth of the tree
        current_depth: Current depth of the node
        
    Returns:
        Root node of the generated tree
    """
    # If we are at the maximum depth, we can only create terminal nodes
    if current_depth >= max_depth:
        terminal_info = config.get_random_terminal()
        if terminal_info['is_variable']:
            return TerminalNode(None, is_variable=True, var_index=terminal_info['var_index'])
        else:
            return TerminalNode(terminal_info['value'], is_variable=False)
    
    # Otherwise, we create only function nodes
    function_info = config.get_random_function()
    children = [full_tree(config, max_depth, current_depth + 1) for _ in range(function_info['arity'])]
    return FunctionNode(function_info['function'], function_info['arity'], function_info['symbol'], children)

```

**Explanation:**

The `full_tree` method generates a maximally dense expression tree with all leaf nodes at the same depth. Unlike the grow_tree method which creates varied structures,





```python
def ramped_half_and_half(config: GPConfig, min_depth: int, max_depth: int) -> ExpressionTree:
    """
    Ramped half-and-half' initialisation method
    Combines grow and full for greater diversity
    
    Args:
        config: GP configuration
        min_depth: Minimum tree depth
        max_depth: Maximum depth of trees
        
    Returns:
        A new expression tree
    """
    # Choose a random depth between min_depth and max_depth
    depth = random.randint(min_depth, max_depth)
    
    # Choose randomly between ‘grow’ and ‘full’.
    if random.random() < 0.5:
        root = grow_tree(config, depth, min_depth)
    else:
        root = full_tree(config, depth)
    
    return ExpressionTree(root)
```

**Explanation:**  
The `ramped_half_and_half` method is the primary initialization technique used by the algorithm. It:

1. Randomly selects a depth value from a range (typically 2-6)
2. Randomly chooses between the 'grow' and 'full' methods
3. Wraps the resulting tree in an ExpressionTree object

This combined approach generates a highly diverse initial population with varied tree shapes, depths, and structures, which is essential for effective exploration of the search space.

#### Protected Mathematical Functions
```python
def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Protected division: returns a/b or 1 when b is close to zero"""
    return np.divide(a, b, out=np.ones_like(a), where=np.abs(b) > 1e-8)

def safe_log(a: np.ndarray) -> np.ndarray:
    """Protected logarithm: returns log(|a|) or 0 for a close to zero"""
    return np.log(np.abs(a), out=np.zeros_like(a), where=np.abs(a) > 1e-10)

def safe_sqrt(a: np.ndarray) -> np.ndarray:
    """Square root protected: returns sqrt(|a|)"""
    return np.sqrt(np.abs(a))

def safe_exp(a: np.ndarray) -> np.ndarray:
    """Protected exponential: limits input to avoid overflow"""
    return np.exp(np.clip(a, -200, 200))

def safe_sin(a: np.ndarray) -> np.ndarray:
    """Protected sin"""
    return np.sin(np.clip(a, -1000, 1000))

def safe_cos(a: np.ndarray) -> np.ndarray:
    """Protected cos"""
    return np.cos(np.clip(a, -1000, 1000))

def safe_tan(a: np.ndarray) -> np.ndarray:
    """Protected tangent: limits outputs to avoid extreme values"""
    return np.clip(np.tan(a), -200, 200)

```

**Explanation:**  
These protected mathematical functions ensure robustness during expression evaluation by handling edge cases gracefully:

- `safe_div` avoids division by zero by returning 1 when the denominator is near zero
- `safe_log` handles negative and zero inputs by taking the absolute value and returning 0 for values near zero
- `safe_sqrt` accepts negative inputs by taking the absolute value first
- `safe_exp` prevents overflow by clipping extreme input values
- `safe_sin` and `safe_cos` clip inputs to reasonable ranges to prevent numerical issues
- `safe_tan` prevents extreme outputs by clipping the results to avoid overflow

This approach allows the evolutionary process to explore expressions without being derailed by mathematical errors, even during early generations when random expressions might contain problematic operations.


#### Weighted Function and Terminal Set Construction

```python
def create_function_set(use_trig: bool = True, use_exp_log: bool = True) -> List[Dict[str, Any]]:
    """
    Creates a set of functions to be used in the expression tree.
    
    Args:
        use_trig: Whether to include trigonometric functions.
        use_exp_log: Whether to include exponential and logarithmic functions.
        
    Returns:
        List of dictionaries, each containing:
            - function: the Python function to call.
            - arity: the number of arguments required.
            - symbol: the symbol for display.
            - weight: selection weight (relative probability).
    """
    # Basic arithmetic functions (always included)
    functions = [
        {'function': np.add, 'arity': 2, 'symbol': '+', 'weight': 1.0},
        {'function': np.subtract, 'arity': 2, 'symbol': '-', 'weight': 1.0},
        {'function': np.multiply, 'arity': 2, 'symbol': '*', 'weight': 1.0},
        {'function': safe_div, 'arity': 2, 'symbol': '/', 'weight': 0.7},  # Lower weight for division
    ]
    
    # Trigonometric functions (optional)
    if use_trig:
        functions.extend([
            {'function': safe_sin, 'arity': 1, 'symbol': 'sin', 'weight': 0.6},
            {'function': safe_cos, 'arity': 1, 'symbol': 'cos', 'weight': 0.6},
            {'function': safe_tan, 'arity': 1, 'symbol': 'tan', 'weight': 0.5},  
        ])
    
    # Exponential and logarithmic functions (optional)
    if use_exp_log:
        functions.extend([
            {'function': safe_exp, 'arity': 1, 'symbol': 'exp', 'weight': 0.4},
            {'function': safe_log, 'arity': 1, 'symbol': 'log', 'weight': 0.5},
            {'function': safe_sqrt, 'arity': 1, 'symbol': 'sqrt', 'weight': 0.6},
        ])
    
    return functions

def create_variable_terminals(n_features: int, variable_weight: float = 1.0) -> List[Dict[str, Any]]:
    """
    Create terminals for input variables
    
    Args:
        n_features: Number of input variables
        variable_weight: Weight assigned to the variables
        
    Returns:
        List of dictionaries for variable terminals
    """
    return [
        {
            'is_variable': True, 
            'var_index': i, 
            'weight': variable_weight
        } for i in range(n_features)
    ]

def create_constant_terminals(const_range: float, n_constants: int = 10, 
                             standard_weight: float = 0.3,
                             zero_weight: float = 0.5,
                             one_weight: float = 0.5,
                             minus_one_weight: float = 0.3,
                             pi_weight: float = 0.2,
                             e_weight: float = 0.2) -> List[Dict[str, Any]]:
    """
    Creates terminals for constants
    
    Args:
        const_range: Range for random constants
        n_constants: Number of pre-generated constants
        standard_weight: Weight for random constants
        zero_weight: Weight for the constant 0
        one_weight: Weight for the constant 1
        minus_one_weight: Weight for the constant -1
        pi_weight: Weight for the constant π
        e_weight: Weight for the constant e
        
    Returns:
        List of dictionaries for constant terminals
    """
    # Important fixed constants
    fixed_constants = [
        {'is_variable': False, 'value': 0.0, 'weight': zero_weight},
        {'is_variable': False, 'value': 1.0, 'weight': one_weight},
        {'is_variable': False, 'value': -1.0, 'weight': minus_one_weight},
        {'is_variable': False, 'value': np.pi, 'weight': pi_weight},
        {'is_variable': False, 'value': np.e, 'weight': e_weight},
    ]
    
    # Random constants pre-generated
    random_constants = [
        {
            'is_variable': False, 
            'value': random.uniform(-const_range, const_range) if abs(random.uniform(-const_range, const_range)) > 1e-8 else 1.0,
            'weight': standard_weight
        } for _ in range(n_constants)
    ]
    
    return fixed_constants + random_constants

def generate_ephemeral_constant(const_range: float) -> float:
    """
    Generates an ephemeral random constant
    avoiding values too close to zero
    """
    value = random.uniform(-const_range, const_range)
    # Avoid values too close to zero
    if abs(value) < 1e-8:
        if random.random() < 0.5:
            value = 1e-8
        else:
            value = -1e-8
    return value
```

**Explanation:**  

- `create_function_set` assigns higher weights to basic arithmetic operations (1.0) due to their fundamental importance, while division receives a reduced weight (0.7) to account for potential numerical instability
- Trigonometric functions receive moderate weights (0.5-0.6) making them suitable for problems with periodic characteristics, while maintaining lower probability than basic operations
- Exponential and logarithmic functions are assigned conservative weights (0.4-0.5) as they are typically relevant to specialized problem domains
(This are the deafult weight)
- `create_variable_terminals` allows uniform or custom weighting of input variables, enabling domain knowledge to guide the search toward more significant features
- `create_constant_terminals` implements a stratified weighting strategy that distinguishes between mathematically important constants (0, 1, π, e) and arbitrary numerical values
- Important mathematical constants receive specialized weights reflecting their prevalence in natural expressions, while random constants receive uniform lower weights to maintain diversity
- `generate_ephemeral_constant` ensures numerical stability by avoiding values too close to zero, preventing potential mathematical errors during evolution

This weighting mechanism influences all phases of the algorithm, from initial population generation to crossover and mutation operations, allowing the search process to naturally favor problem-appropriate mathematical constructs .


#### Fitness Evaluation

```python
def calculate_fitness(tree: ExpressionTree, X: np.ndarray, y: np.ndarray, 
                      parsimony_coef: float = 0.001) -> float:
    """
    Calculates the fitness of an individual
    
    Args:
        tree: The expression tree to be evaluated
        X: Input features
        y: Output target
        parsimony_coef: Penalty coefficient for the complexity of the tree
        
    Returns:
        Fitness value (the lower the bett
    """
    try:
        # Evaluates the tree on input data        
        predictions = tree.evaluate(X)        # In the case of NaN or infinite values, it assigns a very high (bad) fitness
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return float('inf')
        
        # Calculate the mean square error (MSE)
        mse = np.mean((predictions - y) ** 2)
        # add a small penalty to encourage more complex solutions
        complexity = tree.get_complexity()
        complexity_penalty = 0.0
        if complexity < 1:
            complexity_penalty = parsimony_coef * (complexity - 2)
        
        # The final fitness is MSE + penalty (lower is better)
        fitness = mse + complexity_penalty
        return fitness
    
    except Exception as e:
        # In case of errors during the evaluation, it assigns a very high fitness
        print(f"Errore durante la valutazione: {e}")
        return float('inf')
```

**Explanation:**  
The `calculate_fitness` function evaluates how well an expression fits the training data:

1. It evaluates the expression tree on the input features X
2. Handles invalid outputs (NaN or infinite values) by assigning the worst possible fitness
3. Calculates the mean square error (MSE) between predictions and targets
4. Adds a small complexity penalty to prevent oversimplification
5. Returns the combined fitness score (lower is better)


#### Semantic Fitness Sharing

```python
def apply_semantic_fitness_sharing(population: List[ExpressionTree], X_sample: np.ndarray, 
                                sigma: float = 0.5) -> None:
    """
    Apply fitness sharing based on semantic behavior (output patterns)
    
    Args:
        population: List of expression trees already evaluated
        X_sample: Sample of input data to evaluate behavior on
        sigma: Radius of the sharing kernel (similarity threshold)
    """
    n = len(population)
    
    # Pre-compute outputs for all individuals on sample data
    outputs = []
    valid_indices = []
    
    for i, tree in enumerate(population):
        try:
            # Evaluate tree on sample data
            output = tree.evaluate(X_sample)
            
            # Check for invalid outputs
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                outputs.append(None)
                continue
                
            # Normalize the output for more fair comparison
            if np.std(output) > 0:
                output = (output - np.mean(output)) / np.std(output)
            
            outputs.append(output)
            valid_indices.append(i)
        except Exception as e:
            # In case of evaluation errors, skip this individual
            outputs.append(None)
    
    
    # Calculate sharing factors based on output similarity
    for i in range(n):
        if i not in valid_indices:
            population[i].adjusted_fitness = float('inf')
            continue
            
        sharing_factor = 1.0
        
        for j in valid_indices:
            if i != j:
                # Calculate semantic distance (MSE between normalized outputs)
                distance = np.mean((outputs[i] - outputs[j]) ** 2)
                
                # Apply sharing if within the radius
                if distance < sigma:
                    # Sharing function: linear kernel
                    sharing_contribution = 1.0 - (distance / sigma)
                    sharing_factor += sharing_contribution
        
        # Limit and apply the sharing factor
        sharing_factor = min(10.0, max(1.0, sharing_factor))
        
        # Adjust fitness
        if population[i].fitness != float('inf'):
            population[i].adjusted_fitness = population[i].fitness * sharing_factor
        else:
            population[i].adjusted_fitness = float('inf')
```

**Explanation:**  
The `apply_semantic_fitness_sharing` function promotes behavioral diversity in the population:

1. It evaluates and normalizes all expressions on a sample of the input data
2. Calculates the semantic distance (difference in outputs) between each pair of individuals
3. Increases the fitness (worse) of expressions that produce similar outputs to others
4. Uses a linear sharing kernel with a radius parameter sigma to control the effect strength

#### Selection Method
```python
def tournament_selection(population: List[ExpressionTree], tournament_size: int, 
                         use_adjusted_fitness: bool = False) -> ExpressionTree:
    """
    Select an individual by tournament selection
    
    Args:
        population: List of expression trees
        tournament_size: Number of tournament participants
        use_adjusted_fitness: Whether to use diversity-adjusted fitness
        
    Returns:
        The selected individual
    """
    # Randomly selects tournament_size individuals from the population
    contestants = random.sample(population, min(tournament_size, len(population)))
    
    # Find the individual with the best (lowest) fitness
    if use_adjusted_fitness:
        # Use fitness adjusted for diversity, if available
        best = min(contestants, key=lambda x: float('inf') if x.adjusted_fitness is None else x.adjusted_fitness)
    else:
        best = min(contestants, key=lambda x: float('inf') if x.fitness is None else x.fitness)
    
    return best


```

**Explanation:** 
The `tournament_selection` function works by:

1. Randomly sampling a subset of individuals (the "tournament participants") from the population
2. Selecting the individual with the best fitness (lowest value in this minimization problem) from this subset
3. Optionally(always true) using diversity-adjusted fitness instead of raw fitness to promote diversity

```python
def age_weighted_selection(population: List[ExpressionTree], 
                          max_age: int = 10, 
                          young_advantage: float = 0.3) -> ExpressionTree:
    """
    Selection favouring younger individuals (with less age)
    
    Args:
        population: List of expression trees
        max_age: Maximum age considered for advantage
        young_advantage: Percentage advantage for young individuals
        
    Returns:
        The selected individual
    """
    # Calculates weights based on age
    age_weights = [max(0.1, 1.0 - (tree.age / max_age) * young_advantage) 
                  for tree in population]
    
    # Normalise weights
    total_weight = sum(age_weights)
    if total_weight > 0:
        norm_weights = [w / total_weight for w in age_weights]
    else:
        norm_weights = [1.0 / len(population)] * len(population)
    
    # Weighted selection
    return random.choices(population, weights=norm_weights, k=1)[0]

```

**Explanation:** 

The `age_weighted_selection` function implements a selection strategy that favors younger individuals, providing a mechanism to counterbalance the dominance of older, well-optimized solutions

```python
def select_parents(population: List[ExpressionTree], config: GPConfig, 
                  X_sample: np.ndarray) -> Tuple[ExpressionTree, ExpressionTree]:
    """
    Select two parents from the population
    
    Args:
        population: List of expression trees
        config: GP configuration
        X_sample: Data sample to calculate semantic diversity
        
    Returns:
        Parent pair
    """
    # We randomly choose the selection method
    selection_r = random.random()
    
    if selection_r < 0.9:  # 90% chance of using the standard tournament with adjusted fitness
        parent1 = tournament_selection(population, config.tournament_size, use_adjusted_fitness=True)
        parent2 = tournament_selection(population, config.tournament_size, use_adjusted_fitness=True)
    else:  # 10% probability of using age-based selection
        parent1 = age_weighted_selection(population)
        parent2 = age_weighted_selection(population)
    
    # Make sure the parents are different
    attempts = 0
    while parent1 == parent2 and attempts < 5:
        parent2 = tournament_selection(population, config.tournament_size)
        attempts += 1
    
    return parent1, parent2
```

**Explanation:** 
The `select_parents` function combines multiple selection strategies to choose parent pairs for reproduction:

1. It employs a probabilistic approach, using tournament selection with diversity-adjusted fitness 90% of the time
2. It uses age-weighted selection 10% of the time to promote genetic diversity
3. It attempts to ensure the parents are different individuals (up to 5 tries)



#### Mutation and Crossover

```python
def subtree_crossover(parent1: ExpressionTree, parent2: ExpressionTree, 
                     max_tries: int = 5, max_depth: int = 10) -> Tuple[ExpressionTree, ExpressionTree]:
    """
    Crossover for trees: exchanging subtrees between parents
    
    Args:
        parent1, parent2: Parent trees
        max_tries: Maximum number of attempts to find valid crossover points
        max_depth: Maximum depth allowed for the resulting tree
        
    Returns:
        Two child trees generated by the crossover
    """
    # We create copies of parents
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Get all nodes in trees
    nodes1 = child1.get_nodes()
    nodes2 = child2.get_nodes()
    
    if not nodes1 or not nodes2:
        return child1, child2  # We cannot crossover if one of the trees is empty
    
    # Attempts crossover a limited number of times
    for _ in range(max_tries):
        # Randomly choose crossover points
        crossover_point1 = random.randrange(len(nodes1))
        crossover_point2 = random.randrange(len(nodes2))
        
        # Obtain the subtrees to be exchanged
        subtree1 = nodes1[crossover_point1]
        subtree2 = nodes2[crossover_point2]
        
        # Create copies of the subtrees to avoid modifying the originals
        subtree1_copy = subtree1.copy()
        subtree2_copy = subtree2.copy()
        
        # Replace the subtrees in the children
        child1.replace_subtree_at_index(crossover_point1, subtree2_copy)
        child2.replace_subtree_at_index(crossover_point2, subtree1_copy)
        
        # Update the depths of the nodes in the children
        child1.update_node_depths()
        child2.update_node_depths()
        
        # Check if the resulting trees are within the allowed depth
        if child1.get_height() <= max_depth and child2.get_height() <= max_depth:
            break
        else:
            # Restore children from parental copies
            child1 = parent1.copy()
            child2 = parent2.copy()
            nodes1 = child1.get_nodes()
            nodes2 = child2.get_nodes()
    
    # Increase age
    child1.age = 0
    child2.age = 0
    
    return child1, child2
```

**Explanation:** 
The `subtree_crossover` function implements the primary recombination operator for tree-based genetic programming:

1. It randomly selects crossover points in both parent trees
2. Exchanges subtrees between parents to create two new offspring
3. Verifies that the resulting trees don't exceed the maximum allowed depth
4. Makes multiple attempts (up to max_tries) to find valid crossover points if initial attempts produce oversized trees
5. Resets the age of the offspring to zero, treating them as new individuals


```python
def subtree_mutation(tree: ExpressionTree, config: GPConfig, 
                    max_depth: int = 10) -> ExpressionTree:
    """
    Subtree mutation: replaces a random subtree with a new one
    
    Args:
        tree: Tree to be mutated
        config: GP configuration
        max_depth: Maximum depth allowed for the resulting tree
        
    Returns:
        Shaft mutated
    """
    # Create a copy of the tree
    mutated = tree.copy()
    
    # Get all nodes in the tree
    nodes = mutated.get_nodes()
    
    if not nodes:
        return mutated  # We cannot mutate an empty tree
    
    # Randomly select a mutation point
    mutation_point = random.randrange(len(nodes))
    
    # Calculate the maximum depth for the new subtree
    node_depth = nodes[mutation_point].depth
    remaining_depth = max_depth - node_depth
    
    if remaining_depth < 1:
        return mutated  # We cannot change if there is no room to grow
    
    # Generates a new random subtree
    new_subtree = grow_tree(config, remaining_depth, min_depth=1)
    
    # Replace the subtree
    mutated.replace_subtree_at_index(mutation_point, new_subtree)
    
    # Updates node depths
    mutated.update_node_depths()
    
    # Reset age
    mutated.age = 0
    
    return mutated
```

**Explanation:** 
The `subtree_mutation` function introduces larger-scale changes to an expression:

1. It randomly selects a node in the tree as the mutation point
2. Calculates the allowable depth for a replacement subtree based on the node's position and maximum depth constraint
3. Generates a completely new random subtree using the grow method
4. Replaces the selected node and its descendants with this new subtree
5. Resets the age of the mutated tree to zero


```python
def point_mutation(tree: ExpressionTree, config: GPConfig) -> ExpressionTree:
    """
    Point mutation: changes a single node while maintaining the tree structure
    
    Args:
        tree: Tree to be mutated
        config: GP configuration
        
    Returns:
        Tree mutated
    """
    # Create a copy of the tree
    mutated = tree.copy()
    
    # Obtain all nodes in the tree
    nodes = mutated.get_nodes()
    
    if not nodes:
        return mutated  # We cannot mutate an empty tree
    
    # Randomly select a mutation point
    mutation_point = random.randrange(len(nodes))
    node = nodes[mutation_point]
    
    # Mutation based on node type
    if isinstance(node, FunctionNode):
        # Replace with another function of the same arity
        compatible_functions = [f for f in config.function_set if f['arity'] == node.arity]
        if compatible_functions:
            function_info = random.choice(compatible_functions)
            new_node = FunctionNode(function_info['function'], 
                                   function_info['arity'], 
                                   function_info['symbol'],
                                   node.children.copy())  # reuses the same children
            
            # Replace the node
            mutated.replace_subtree_at_index(mutation_point, new_node)
    
    elif isinstance(node, TerminalNode):
        if node.is_variable:
            # Replace with another variable
            if len(config.variable_terminals) > 1:
                terminal_info = config.get_random_variable()
                while terminal_info['var_index'] == node.var_index:
                    terminal_info = config.get_random_variable()
                
                new_node = TerminalNode(None, True, terminal_info['var_index'])
                mutated.replace_subtree_at_index(mutation_point, new_node)
        else:
            # We could replace it with another constant or slightly modify the valu
            if random.random() < 0.5:  # 50% probability of changing the value
                # Change existing value (small perturbation)
                new_value = node.value * (1.0 + random.uniform(-0.1, 0.1))
                new_node = TerminalNode(new_value, False)
                #Avoid zero values
                if abs(new_value) < 1e-8:
                    new_value = 1e-8 if new_value >= 0 else -1e-8
            else:
                # Replace with a new constant
                terminal_info = config.get_random_constant()
                new_node = TerminalNode(terminal_info['value'], False)
            
            mutated.replace_subtree_at_index(mutation_point, new_node)
    
    # Updates node depths
    mutated.update_node_depths()
    
    # Reset age
    mutated.age = 0
    
    return mutated

```

**Explanation:** 
The `point_mutation` function implements a more conservative mutation operator that preserves most of the tree structure:

1. It randomly selects a single node in the tree as the mutation point
2. Applies different mutation strategies based on the node type:
   - For function nodes, it replaces the function with another compatible one of the same arity
   - For variable nodes, it replaces it with another input variable
   - For constant nodes, it either slightly perturbs the value or replaces it with a new constant
3. Maintains the overall tree structure by keeping the same children for function nodes
4. Resets the age of the mutated tree to zero


#### Island Model and Evolution

```python
    class Island:
    """Representing an island in the Island Model"""
    
    def __init__(self, population: List[ExpressionTree], config: GPConfig, island_id: int):
        self.population = population
        self.config = config
        self.id = island_id
        self.best_individual = None
        self.best_fitness = float('inf')
        self.generations_without_improvement = 0
        self.mutation_strength = 1.0  # Starting mutation strength
    #[.... other fuction]
```

**Explanation:**  
The `Island` class implements the island model for population distribution in genetic programming, serving as a semi-isolated evolutionary environment. Key features include:

1. It maintains a separate population of expression trees with its own configuration parameters, allowing for parallel but distinct evolutionary trajectories.

2. It tracks island-specific metrics including the best individual found, generations without improvement, and a local mutation strength parameter.

3. The evolve method handles the progression of the island's population through one generation

4. The adaptive mutation mechanism enables each island to independently shift between exploration and exploitation based on its own evolutionary progress, creating diversity in search strategies across islands.

```python
    def evolve(self, X: np.ndarray, y: np.ndarray, generation: int,
               use_adaptive_mutation: bool = False,
               min_mutation_strength: float = 0.5,
               max_mutation_strength: float = 3.0,
               adaptation_rate: float = 0.1) -> None:
        """
        Evolving the island for a generation
        
        Args:
            X, y: Training data
            generation: Number of the current generation
            use_adaptive_mutation: Whether to use adaptive mutation
            min_mutation_strength: Minimum mutation strength multiplier
            max_mutation_strength: Maximum mutation strength multiplier
            adaptation_rate: How quickly mutation strength changes
        """
        # Store original mutation probability
        original_mutation_prob = self.config.mutation_prob
        
        # Apply adaptive mutation if enabled
        if use_adaptive_mutation:
            # Adjust mutation probability based on current strength
            self.config.mutation_prob = original_mutation_prob * self.mutation_strength

        # Apply genetic operators
        self.population = apply_genetic_operators(self.population, X, y, self.config)
        
        # Update the best solution
        current_best = min(self.population, key=lambda x: float('inf') if x.adjusted_fitness is None else x.adjusted_fitness)
       
    
        if current_best.adjusted_fitness < self.best_fitness:
            self.best_individual = current_best.copy()
            self.best_fitness = current_best.adjusted_fitness
            self.generations_without_improvement = 0

            # Decrease mutation strength when improvement found
            if use_adaptive_mutation:
                self.mutation_strength = max(min_mutation_strength, 
                                           self.mutation_strength * (1 - adaptation_rate))


        else:
            self.generations_without_improvement += 1

            # Increase mutation strength during stagnation
            if use_adaptive_mutation:
                self.mutation_strength = min(max_mutation_strength, 
                                           self.mutation_strength * (1 + adaptation_rate))
                
        # Reset mutation probability to original value
        self.config.mutation_prob = original_mutation_prob
        
        # Progress log
        if generation % 100 == 0:  # Every 100 generations
            mutation_info = f" | Mutation Strength: {self.mutation_strength:.2f}" if use_adaptive_mutation else ""
            print(f"Island {self.id} | Generation {generation} | Best Fitness: {self.best_fitness}{mutation_info}")
```

**Explanation:**  
The `evolve` method handles the progression of an island's population through one generation:

1. It temporarily adjusts the mutation probability based on the island's adaptive mutation strength
2. Applies genetic operators (selection, crossover, mutation) to create a new population
3. Updates the island's best solution if an improvement is found
4. Adjusts the mutation strength based on whether progress is being made
   - Decreases mutation when improvements are found (exploitation)
   - Increases mutation during stagnation periods (exploration)
5. Tracks generations without improvement to guide adaptation

This method implements the core adaptive control mechanism that balances exploration and exploitation dynamically during the evolution process.

#### Migration Between Islands


```python

def calculate_semantic_diversity(population: List[ExpressionTree], X_sample: np.ndarray, 
                               distance_threshold: float = 0.01) -> float:
    """
    Calculate population diversity based on semantic behavior
    
    Args:
        population: List of expression trees
        X_sample: Sample of input data to evaluate semantic behavior
        distance_threshold: Threshold to consider two behaviors as distinct
        
    Returns:
        Diversity ratio (semantically unique individuals / total population)
    """
    # Calculate outputs for each individual on the sample data
    outputs = []
    valid_indices = []
    
    for i, tree in enumerate(population):
        try:
            output = tree.evaluate(X_sample)
            # Normalize output to make comparisons more meaningful
            if np.std(output) > 0:
                output = (output - np.mean(output)) / np.std(output)
            outputs.append(output)
            valid_indices.append(i)
        except Exception:
            continue
    
    if not outputs:
        return 0.0  # No valid outputs
    
    # Cluster individuals based on semantic similarity
    unique_behaviors = 1  # Start with the first individual
    
    
    # For each subsequent individual, check if it's semantically unique
    for i in range(1, len(outputs)):
        is_unique = True
        for j in range(i):
            semantic_distance = np.mean((outputs[i] - outputs[j]) ** 2)
            if semantic_distance < distance_threshold:
                is_unique = False
                break
        
        if is_unique:
            unique_behaviors += 1
    
    return unique_behaviors / len(population)

```
**Explanation:**  
The `calculate_semantic_diversity` function measures population diversity based on behavioral differences rather than structural differences in the expressions. This semantic approach to diversity assessment offers several advantages:

The returned diversity measure is the ratio of unique behaviors to total population size, providing a normalized metric between 0 (all expressions behave identically) and 1 (every expression behaves uniquely).

```python
def migration(islands: List[Island], migration_rate: float = 0.2, X: np.ndarray = None, y: np.ndarray = None, X_sample: np.ndarray = None) -> None:
    """
    Migration with adaptive mutation based on source-destination diversity relationship
    
    Args:
        islands: List of Island objects
        migration_rate: Percentage of population migrating
        X_sample: Sample of input data to evaluate diversity
    """
    if len(islands) <= 1:
        return
    
    print("Performing inter-island migration...")

     # Calculate diversity for each island if sample data is provided
    island_diversities = []
    if X_sample is not None:
        for island in islands:
            island_div = calculate_semantic_diversity(island.population, X_sample)
            island_diversities.append(island_div)
            print(f"  Island {island.id} diversity: {island_div:.3f}")
    else:
        # Default to medium diversity if we can't calculate
        island_diversities = [0.5] * len(islands)
            
    for i, source_island in enumerate(islands):
        # Calculates the destination island (the next, or the first if it is the last)
        dest_idx = (i + 1) % len(islands)
        dest_island = islands[dest_idx]

        # Get diversity values
        source_diversity = island_diversities[i]
        dest_diversity = island_diversities[dest_idx]

         # If source is more diverse than destination, use lower mutation strength
        if source_diversity > dest_diversity:
            # Minimal mutation - just enough to avoid exact duplicates
            mutation_strength = 0.5
        else:
            # Stronger mutation to introduce novelty
            # The larger the diversity gap, the stronger the mutation
            diversity_gap = max(0, dest_diversity - source_diversity)
            mutation_strength = 1.0 + (diversity_gap * 3.0)  # Scale up based on gap
            mutation_strength = min(3.0, max(0.5, mutation_strength))  # Limit range
        
        # Number of individuals to migrate
        n_migrants = max(1, int(source_island.config.pop_size * migration_rate))
        
        # Select migrants (half best, half random)
        n_best = n_migrants // 2
        n_random = n_migrants - n_best
        
        # Sort by fitness
        sorted_pop = sorted(source_island.population, 
                          key=lambda x: float('inf') if x.adjusted_fitness is None else x.adjusted_fitness)
        
        # Get the best
        migrants_best = [ind.copy() for ind in sorted_pop[:n_best]]
        
        # Take some random
        migrants_random = [ind.copy() for ind in random.sample(source_island.population, n_random)]
        
        migrants = migrants_best + migrants_random
        # Mutation of migrants if requested
        
        for j, migrant in enumerate(migrants):
            was_mutated = False
            # Apply mutation with increased probability
            if random.random() < source_island.config.mutation_prob * mutation_strength:
                
                # Randomly choose the mutation type
                mutation_choice = random.random()
                
                if mutation_choice < 0.7:  # 70% subtree mutation
                    migrants[j] = subtree_mutation(migrant, source_island.config, 
                                                 max_depth=source_island.config.max_depth)
                    was_mutated = True
                else:  # 30% point mutation
                    migrants[j] = point_mutation(migrant, source_island.config)
                    was_mutated = True
                
                # Rcompute the fitness of the mutated individual
                if was_mutated and X is not None and y is not None:                    
                        migrants[j].fitness = calculate_fitness(migrants[j], X, y, source_island.config.parsimony_coef)

        if X_sample is not None and len(migrants) > 1:
            apply_semantic_fitness_sharing(migrants, X_sample)
        
        # Replace the worst in the destination island
        dest_sorted = sorted(dest_island.population, 
                           key=lambda x: float('inf') if x.adjusted_fitness is None else x.adjusted_fitness, 
                           reverse=True)  # decrescent order
        
        # Remove the worst individuals from the destination island
        for j in range(min(n_migrants, len(dest_sorted))):
            dest_island.population.remove(dest_sorted[j])
        
        # Add the migrants to the destination island
        dest_island.population.extend(migrants)
        
        print(f"  Migration: {n_migrants} individuals from island {i} to island {dest_idx}" + 
              f"(mutation strength: {mutation_strength:.2f})")
```

**Explanation:**  
The `migration` function manages the periodic exchange of individuals between islands:

1. It first calculates the semantic diversity of each island's population
2. For each island pair, it determines the appropriate mutation strength based on diversity relationship
   - Uses lower mutation when moving from more diverse to less diverse islands
   - Uses higher mutation when moving from less diverse to more diverse islands
3. Selects migrants using a combination of best individuals and random selection
4. Applies mutation to migrants with strength based on the diversity relationship
5. Replaces the worst individuals in the destination island with the mutated migrants

This adaptive migration strategy helps maintain global diversity while allowing effective information exchange between subpopulations, preventing premature convergence.

#### Expression Simplification

```python
def sympy_simplify_expression(expression: str) -> str:
    """
    Simplifies an expression using the sympy library for symbolic calculation.
    """
    try:
        # Prepare the expression for sympy (replacing x[0] with x_0, etc.)
        prepared_expr = re.sub(r'x\[(\d+)\]', r'x_\1', expression)
        
        # Define Symbols
        symbol_names = set(re.findall(r'x_(\d+)', prepared_expr))
        symbols = {f'x_{i}': sp.Symbol(f'x_{i}', real=True) for i in symbol_names}
        
        # Analyses and simplifies the expression
        parsed_expr = parse_expr(prepared_expr, local_dict=symbols)
        simplified = sp.sympify(parsed_expr)
        
        # Checks whether the expression contains complex numbers or special symbols such as zoo
        if "zoo" in str(simplified) or "I" in str(simplified) or "oo" in str(simplified):
            # Fallback to basic simplification if we obtain problematic results
            return simplify_expression(expression)
        
        # Convert back to original format
        result = str(simplified)
        result = re.sub(r'x_(\d+)', r'x[\1]', result)
        return result
    except Exception as e:
        # Fallback to basic simplification if sympy is not available or there is an error
        print(f"Error in simplification sympy: {str(e)}")
        return simplify_expression(expression)
```

**Explanation:**  
The `sympy_simplify_expression` function uses symbolic mathematics to transform evolved expressions into more human-readable forms:

1. It converts the expression from the GP's format to sympy's format
2. Creates symbolic variables for each input feature
3. Parses and simplifies the expression using sympy's powerful simplification engine
4. Checks for problematic symbols or complex numbers and falls back to basic simplification if needed
5. Converts the result back to the original format




#### GP Configuration

```python
class GPConfig:
    """Class for managing the configuration of the GP algorithm"""
    
    def __init__(self, 
                 n_features: int,
                 const_range: float,
                 use_trig: bool = True,
                 use_exp_log: bool = True,
                 min_depth: int = 2,
                 max_depth: int = 6,
                 pop_size: int = 500,
                 generations: int = 50,
                 tournament_size: int = 5,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 elitism_rate: float = 0.1,
                 max_tree_size: int = 50,
                 parsimony_coef: float = 0.01,
                function_weights: dict = None,
                terminal_weights: dict = None):
        
        # Function Set Configuration and Terminals
        self.function_set = create_function_set(use_trig, use_exp_log)
        self.variable_terminals = create_variable_terminals(n_features)
        self.constant_terminals = create_constant_terminals(const_range)
        
        # Apply any custom weights to functions
        if function_weights:
            self._apply_function_weights(function_weights)
            
        # Apply any customised weights to the terminals
        if terminal_weights:
            self._apply_terminal_weights(terminal_weights)

        # Calculates cumulative weights for weighted selection
        self._calculate_weights()
        
        # Tree size limits
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_tree_size = max_tree_size
        
        # Parameters of the evolutionary algorithm
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_rate = elitism_rate
        
        # Bloat control 
        self.parsimony_coef = parsimony_coef  # penalty for complexity
        
        # Other parameters
        self.n_features = n_features
        self.const_range = const_range

    #[.... other fuction]
```

**Explanation:**  
The `GPConfig` class centralizes all configuration parameters for the Genetic Programming algorithm:

1. It manages the set of available functions (arithmetic, trigonometric, exponential) and terminals (variables, constants)
2. Allows custom weighting of functions and terminals to bias the search process
3. Controls tree size constraints (min/max depth, maximum size) to prevent bloat
4. Defines evolutionary parameters (population size, tournament size, genetic operator probabilities)
5. Sets bloat control parameters to balance accuracy and complexity

#### Main Genetic Programming Algorithm

```python
def genetic_programming(X: np.ndarray, y: np.ndarray, config: GPConfig, 
                       # Parameters for the island model
                       use_islands: bool = False,
                       n_islands: int = 5, 
                       migration_interval: int = 10,
                       migration_rate: float = 0.1,
                       
                       # Parameters for bloat control
                       bloat_control_interval: int = 5,

                        # Parameters for adaptive mutation
                       use_adaptive_mutation: bool = True,
                       base_mutation_rate: float = None,  # If None, use config.mutation_prob
                       min_mutation_strength: float = 0.5,
                       max_mutation_strength: float = 3.0,
                       adaptation_rate: float = 0.1,  # How quickly mutation strength changes
                       ) -> ExpressionTree:
    """
    Main Genetic Programming Algorithm for Symbolic Regression
    
    Args:
        X: Input features
        y: Output target
        config: GP configuration
        
        # Parameters for the island model
        use_islands: Whether to use the island model
        n_islands: Number of islands (if use_islands is True)
        migration_interval: Interval of generations between migrations
        migration_rate: Percentage of population migrating
        
        # Parameters for bloat control
        bloat_control_interval: Interval of generations for bloat control
       
        # Parameters for adaptive mutation
        use_adaptive_mutation: Whether to use adaptive mutation
        base_mutation_rate: Base mutation rate (if None, use config.mutation_prob)
        min_mutation_strength: Minimum mutation strength
        max_mutation_strength: Maximum mutation strength
        adaptation_rate: Rate of adaptation for mutation strength
        
    Returns:
        Best expression tree found
    """


    #Debug info
    X_original = X.copy()
    y_original = y.copy()
    
    print(f"Prima di qualsiasi elaborazione:")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X mean: {np.mean(X)}, y mean: {np.mean(y)}")
    print(f"X std: {np.std(X)}, y std: {np.std(y)}")
    print(f"X range: [{np.min(X)}, {np.max(X)}], y range: [{np.min(y)}, {np.max(y)}]")


    start_time = time.time()
    print(f"Starting Genetic Programming for Symbolic Regression...")
    print(f"Configuration: pop_size={config.pop_size}, max_depth={config.max_depth}, "
          f"generations={config.generations}")
    
    # Initialize adaptive mutation parameters
    if use_adaptive_mutation:
        if base_mutation_rate is None:
            base_mutation_rate = config.mutation_prob
        current_mutation_strength = 1.0
        print(f"Using adaptive mutation with strength range [{min_mutation_strength:.2f}, {max_mutation_strength:.2f}]")
    
    if use_islands:
        print(f"Island model: {n_islands} islands, migration every {migration_interval} generations")
        config.print_function_weights()

    X_sample = X[:min(len(X), 500)]  # Use a sample for efficiency
    
    # Initialise the population
    initial_population = initialize_population(config)
    
    # Assess the initial population
    evaluate_population(initial_population, X, y, config)
    
    # Apply fitness sharing for diversity
    apply_semantic_fitness_sharing(initial_population, X_sample)
    
    # Initialise islands or single population
    if use_islands:
        islands = initialize_islands(initial_population, config, n_islands)
        best_individual = min([island.best_individual for island in islands if island.best_individual], 
                            key=lambda x: x.adjusted_fitness, default=None)
        best_fitness = float('inf') if best_individual is None else best_individual.adjusted_fitness
    else:
        population = initial_population
        best_individual = min(population, key=lambda x: float('inf') if x.adjusted_fitness is None else x.adjusted_fitness)
        best_fitness = best_individual.adjusted_fitness
    
    # Statistics for monitoring
    stats = {
        'best_fitness': [],
        'avg_fitness': [],
        'avg_size': [],
        'best_size': []
        #'diversity': []
    }
    
    # Principal loop of the algorithm
    generations_without_improvement = 0
    for generation in tqdm(range(config.generations)): 
        
        if use_islands:
            # Evolve each island separately
            for island in islands:
                island.evolve(X, y, generation, 
                         use_adaptive_mutation=use_adaptive_mutation,
                         min_mutation_strength=min_mutation_strength,
                         max_mutation_strength=max_mutation_strength,
                         adaptation_rate=adaptation_rate)
            
            # Collect all individuals for statistics and migration
            all_individuals = []
            for island in islands:
                all_individuals.extend(island.population)
            
            # Calculate semantic diversity occasionally
            #if generation % 5 == 0 or generation == 0:  # First gen and every 5 gens
            #    diversity = calculate_semantic_diversity(all_individuals, X_sample)
            #else:
            #   # Use previous value
            #    diversity = stats['diversity'][-1] if stats['diversity'] else 0
            
            # Periodic migration
            if (generation + 1) % migration_interval == 0:
                    migration(islands, migration_rate=migration_rate, X=X, y=y, X_sample=X_sample)
            
           
            #  Periodic bloat control
            if generation % bloat_control_interval == 0:
                for island in islands:
                    apply_bloat_control(island.population,X,y, config)

            # Calculate the best overall individual
            current_best = min([island.best_individual for island in islands if island.best_individual], 
                             key=lambda x: x.adjusted_fitness)
  
            
            # Calculate other statistics
            avg_fitness = np.mean([tree.adjusted_fitness for tree in all_individuals if tree.adjusted_fitness != float('inf')])
            avg_size = np.mean([tree.get_complexity() for tree in all_individuals])
            
        else:

            if use_adaptive_mutation:
               original_mutation_prob = config.mutation_prob
            
            # Adjust mutation based on stagnation
            if generations_without_improvement > 5:
                # Increase mutation strength
                current_mutation_strength = min(max_mutation_strength, 
                                              current_mutation_strength * (1 + adaptation_rate))
                config.mutation_prob = original_mutation_prob * current_mutation_strength
            else:
                # Decrease mutation strength
                current_mutation_strength = max(min_mutation_strength, 
                                              current_mutation_strength * (1 - adaptation_rate))
                config.mutation_prob = original_mutation_prob * current_mutation_strength
            
            
            # Apply genetic operators
            population = apply_genetic_operators(population, X, y, config)
            
            # Periodic bloat control
            if generation % bloat_control_interval == 0:
                apply_bloat_control(population,X, y, config)
            
            # Calculate the best individual
            current_best = min(population, key=lambda x: float('inf') if x.adjusted_fitness is None else x.adjusted_fitness)
        
            # Calculate statistics
            avg_fitness = np.mean([tree.adjusted_fitness for tree in population if tree.adjusted_fitness != float('inf')])
            avg_size = np.mean([tree.get_complexity() for tree in population])
            
            # Calculate semantic diversity occasionally
            #if generation % 5 == 0 or generation == 0:  # First gen and every 5 gens
            #    diversity = calculate_semantic_diversity(population, X_sample)
            #else:
                # Use previous value
            #    diversity = stats['diversity'][-1] if stats['diversity'] else 0
        
            if use_adaptive_mutation:
                config.mutation_prob = original_mutation_prob

        # Upgrade the best global individual
        if current_best.adjusted_fitness < best_fitness:
            best_individual = current_best.copy()
            best_fitness = current_best.adjusted_fitness
            generations_without_improvement = 0
            print("Upgrade the best global individual:")
            print(f"New best solution found:")
            print(f"  Expression: {best_individual.to_string()}")
            print(f"  Simplified Expression: {sympy_simplify_expression(best_individual.to_string())}")
            print(f"  Fitness: {best_fitness}")
            print(f"  Complexity: {best_individual.get_complexity()} nodes")
        else:
            generations_without_improvement += 1
        
        # Store statistics
        stats['best_fitness'].append(best_fitness)
        stats['avg_fitness'].append(avg_fitness)
        stats['avg_size'].append(avg_size)
        stats['best_size'].append(best_individual.get_complexity())
        #stats['diversity'].append(diversity)
        
        # Generation log
        if generation % 10 == 0 or generation == config.generations - 1:
           print(f"Generation {generation}, Best Fitness: {best_fitness}")
        
       
    
    total_time = time.time() - start_time
    print(f"Algorithm completed in {total_time:.2f} seconds")
    print(f"Best solution found:")
    print(f"  Simplified Expression: {sympy_simplify_expression(best_individual.to_string())}")
    print(f"  Expression: {best_individual.to_string()}")
    print(f"  Fitness: {best_fitness}")
    print(f"  Complexity: {best_individual.get_complexity()} nodes")
    
    # Visualizza statistiche
    plot_statistics(stats)
    
    return best_individual
```

**Explanation:**  
The `genetic_programming` function is the main algorithm that coordinates the entire symbolic regression process:

1. **Initialization Phase**:
   - Sets up parameters for adaptive mutation
   - Initializes the population using ramped half-and-half
   - Evaluates initial fitness and applies sharing

2. **Island Model Setup (Used in each problem)**:
   - Distributes the population across multiple islands
   - Prepares for periodic migration between islands

3. **Main Evolutionary Loop**:
   - For each generation:
     - Evolves each island independently (or the single population)
     - Performs periodic migration between islands if using the island model
     - Applies bloat control at specified intervals
     - Tracks the best overall solution and updates statistics

4. **Adaptive Control Mechanisms**:
   - Dynamically adjusts mutation rates based on progress
   - Uses the generations without improvement counter to guide adaptation
   - Applies semantic fitness sharing to maintain diversity

5. **Output and Visualization**:
   - Simplifies the best expression found for better interpretability
   - Reports performance metrics and total runtime
   - Generates plots of fitness and complexity over time

```python
def apply_genetic_operators(population: List[ExpressionTree], X: np.ndarray, y: np.ndarray, 
                           config: GPConfig) -> List[ExpressionTree]:
    """
    Apply genetic operators to create a new population
    
    Args:
        population: List of expression trees
        X, y: Evaluation data
        config: GP configuration
        
    Returns:
        New population
    """
    # Sort population by fitness (best first)
    sorted_population = sorted(population, key=lambda x: float('inf') if x.adjusted_fitness is None else x.adjusted_fitness)
    
    # Number of individuals to be selected by elitism
    n_elite = int(config.pop_size * config.elitism_rate)
    
    # Sample to calculate semantic diversity
    X_sample = X[:min(len(X), 500)]  # Use a sample for efficiency
    
    # Select elites
    new_population = [tree.copy() for tree in sorted_population[:n_elite]]
    
    # Increase the age of each elite individual
    for tree in new_population:
        tree.age += 1
    
    # Complete the population with new individual
    while len(new_population) < config.pop_size:
        # Select genetic operation (crossover or mutation)
        op_choice = random.random()
        
        if op_choice < config.crossover_prob:
            # Crossover
            parent1, parent2 = select_parents(population, config, X_sample)
            child1, child2 = subtree_crossover(parent1, parent2, max_depth=config.max_depth)
            
            # Apply post-crossover mutation based on mutation probability 
            # The higher the mutation_prob, the more likely this happens
            if random.random() < config.mutation_prob * 0.5:  # 50% of mutation_prob chance
                mutation_choice = random.random()
                if mutation_choice < 0.7:  # 70% subtree mutation
                    child1 = subtree_mutation(child1, config, max_depth=config.max_depth)
                else:  # 30% point mutation
                    child1 = point_mutation(child1, config)
                    
            if random.random() < config.mutation_prob * 0.5:  # 50% of mutation_prob chance
                mutation_choice = random.random()
                if mutation_choice < 0.7:  # 70% subtree mutation
                    child2 = subtree_mutation(child2, config, max_depth=config.max_depth)
                else:  # 30% point mutation
                    child2 = point_mutation(child2, config)

            # Use deterministic crowding to decide which individuals to keep
            selected = deterministic_crowding(parent1, parent2, child1, child2, X, y, config)
            
            # Add to new population
            new_population.extend(selected)
            if len(new_population) > config.pop_size:
                new_population = new_population[:config.pop_size]
        
        elif op_choice < config.crossover_prob + config.mutation_prob:
            # Mutation
            parent = tournament_selection(population, config.tournament_size)
            
            # Rnandomly choose between subtree mutation and point mutation
            mutation_choice = random.random()
            
            if mutation_choice < 0.7:  # 70% subtree mutation
                child = subtree_mutation(parent, config, max_depth=config.max_depth)
            else:  # 30% point mutation
                child = point_mutation(parent, config)
            
            # Calculate fitness of the child
            child.fitness = calculate_fitness(child, X, y, config.parsimony_coef)
            temp_pop = [parent, child]
            apply_semantic_fitness_sharing(temp_pop, X_sample)
            # Compare parent and child
            if child.adjusted_fitness <= parent.adjusted_fitness:
                new_population.append(child)
            else:
                # Still add the son with some probability
                if random.random() < 0.1:  # 10% probability
                    new_population.append(child)
                else:
                    new_population.append(parent.copy())
        
        else:
            # Playback (direct copy)
            parent = tournament_selection(population, config.tournament_size)
            offspring = parent.copy()
            offspring.age += 1  # Increase age
            new_population.append(offspring)
    
    # Make sure the population is exactly the right size
    if len(new_population) > config.pop_size:
        new_population = new_population[:config.pop_size]
    
    # Fitness adjusted for diversity
    apply_semantic_fitness_sharing(new_population, X_sample)
    
    return new_population
```

**Explanation:**  
The `apply_genetic_operators` function creates a new population from the current one by applying selection, crossover, and mutation:

1. **Elitism**: The best individuals are copied directly to the new population to preserve good solutions
2. **Genetic Operations**: For the remainder of the population, it applies:
   - **Crossover**:  Creates children by exchanging subtrees between parents, with optional post-crossover mutation. The probability of mutation increases/decreases adaptively based on stagnation calculated by the number of generations without improvement. 
   - **Mutation**: Creates new individuals by modifying existing ones, either by replacing a subtree or changing a single node
   - **Reproduction**: Directly copies some individuals, increasing their age counter
3. **Deterministic Crowding**: Uses a parent-child competition strategy to determine which individuals survive
4. **Fitness Sharing**: Applies semantic fitness sharing to maintain diversity in the new population

The function carefully balances exploitation of good solutions (through elitism and selective pressure) with exploration of new solutions (through mutation and diversity mechanisms).



### Results

#### Input data

```python
problems = [

    {"file_path": "../data/problem_0.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 20
     },
     "use_islands": True,
     "n_islands": 5,
     "migration_interval": 40,
     "migration_rate": 0.1,
     "tournament_size": 100,
    },
    

    {"file_path": "../data/problem_1.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 10000, 
         "generations": 100,
         "max_tree_size": 20
     },
     "use_islands": True,
     "n_islands": 5,
     "migration_interval": 40,
     "migration_rate": 0.1
    },
    

    {"file_path": "../data/problem_2.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 30000, 
         "generations": 500,
         "max_tree_size": 60,
        "function_weights": {
         "sin": 0.3,
         "cos": 0.3,
         "tan": 0.3,
         "log": 0.4,
         "sqrt": 0.4,
        }
     },
     "use_islands": True,
     "n_islands": 10,
     "migration_interval": 40,
     "migration_rate": 0.2
    },
    
 
    {"file_path": "../data/problem_3.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 65,
        "function_weights": {   
         "sin": 0.0,
         "cos": 0.0,
         "tan": 0.0,
         "log": 0.0,
         "sqrt": 0.0,
         "exp": 0.0,
        }
     },
    
     "use_islands": True,
     "n_islands": 10,
     "migration_interval": 40,
     "migration_rate": 0.2
    },

    {"file_path": "../data/problem_4.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 100,
     },
     "use_islands": True,
     "n_islands": 5,  
     "migration_interval": 40,
     "migration_rate": 0.08
    },

    {"file_path": "../data/problem_5.npz", 
     "config": {
         "max_depth": 8,  
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 80,
        "function_weights": {
         "sin": 0.3,
         "cos": 0.3,
         "tan": 0.3,
         "log": 0.4,
         "sqrt": 0.4,
        }
     }, 

     "use_islands": True,
     "n_islands": 5,
     "migration_interval": 40,
     "migration_rate": 0.15
    },
    

    {"file_path": "../data/problem_6.npz", 
     "config": {
         "max_depth": 8,  
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 40,
         "parsimony_coef": 0.15, 
     },
     "use_islands": True,
     "n_islands": 5,
     "migration_interval": 40, 
    },
    

    {"file_path": "../data/problem_7.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 60,
         "function_weights": {
         "sin": 0.1,
         "cos": 0.1,
         "tan": 0.1,
         "log": 0.25,
         "sqrt": 0.25,
         "exp": 0.1,
        },
     },
    
     "use_islands": True,
     "n_islands": 10,
     "migration_interval": 40,
     "migration_rate": 0.25  
    },
    
    {"file_path": "../data/problem_8.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 10000, 
         "generations": 1000,
         "max_tree_size": 50,
        "function_weights": {
         "sin": 0.1,
         "cos": 0.1,
         "tan": 0.1,
         "log": 0.1,
         "sqrt": 0.1,
         "exp": 0.1,
        }
     },
   
     "use_islands": True,
     "n_islands": 10,
     "migration_interval": 40,
     "migration_rate": 0.15
    }
]
```



#### Result Table


| Problem | Population Size | Generations | Islands | Expression | Fitness (MSE) |
|---------|----------------|-------------|---------|----------------------|---------------|
| 0 | 10,000 | 500 | 5 | `x[0] + (-2.8636943484015354 * cos(x[1] - -1.583968308262229)) * exp(-3.2082386888331254) + sin(x[1]) * exp(-3.154932922313467) + sin(x[1] + sqrt(tan(3.141592653589793)) * (3.141592653589793 / exp(-3.2082386888331254))) * exp(-3.154932922313467) - sqrt(sqrt(tan(3.141592653589793)) * (-3.154932922313467 / 0.2607832440731248) / 0.2532699192266387 * (x[1] + (x[1] - -1.0329760905609568) - -1.1123733897283157) * (sin(sin(x[1])) - -1.0439752897500634 + cos(-1.9944929251132986 * (x[1] - 3.1410124598253333))))` | 2.633315e-10 |
| 1 | 10,000 | 100 | 5 | `sin(x[0])` | 7.125941e-34 |
| 2 | 30,000 | 500 | 10 | `(-1301573.6081735631 + -1667888.905645238) * (sin((x[2] + x[0] + x[0] + x[1]) * log(tan(-6476959.681381762))) + sin((x[2] + x[0] + x[1] + x[0]) * log(tan(-3096338.9650774663))) + (x[2] + x[0] + x[0] + x[1]) * exp(tan(4397592.634281375 / -4354655.722764039))) * sin(-1230797.8088492597 - (x[1] + x[0] + x[2] + x[1] - sqrt(-1230797.8088492597) - sqrt(-3096338.9650774663) * sqrt(x[0]) * (x[0] + x[0])² - sqrt(-3272060.4581361106) * log(-7355106.67995647) * (x[1] + x[2]) * (x[0] + x[0])))` | 3.107689e+11 |
| 3 | 10,000 | 500 | 10 | `(1.0 + 1.0 + x[0]² + 1.0 + x[0]/x[0] + x[0]²) - (x[1] - x[0] + x[0]) - ((x[0] + x[2] - x[1]) * -1.0) - (x[1] * x[2]/x[1] - x[0]) - x[2]/((x[0] + 1.0 - x[0] + x[0] + 1.0 - x[0]) * (x[2] - -1.0 - x[2])) + x[2] - x[1]³` | 3.445432e-29 |
| 4 | 10,000 | 500 | 5 | `cos(exp(-8.23509920982603 - cos(x[0] + x[1]) * x[0]) + x[1] + tan(3.141592653589793) + tan(3.141592653589793) + tan(3.141592653589793) - tan(exp(-10.613659135997567))) + sqrt(-10.822240096384276 + x[0] * exp(-0.9405033382101015) + x[0] * cos(-0.9916749730176175) * exp(-0.9916749730176175)) + sin(7.871174350285515) * sqrt(7.871174350285515) * cos(exp(-7.795847756522948) - x[1]) + sqrt(-10.205488998391495) * cos(exp(-8.36973460432399) - x[1] - tan(3.141592653589793) + tan(3.141592653589793))` | 7.296206e-05 |
| 5 | 10,000 | 500 | 5 | `(x[0] * x[0] * x[1] + x[0] * log(exp(x[1]) + x[1]) * 4.092327718778701) * sin(sin(3.141592653589793) * sqrt(log(exp(x[1]) + 3.0580787826247215 + x[1]))) * (exp(x[1] + x[1]) - (exp(6.5609367685064) - 5.939230066817374 * x[0] * exp(x[0])) / exp(cos(5.939230066817374 + x[1])) - exp(5.083623282487277) * ((5.547345449854117 * 6.5609367685064 + 4.9787556043961265 - exp(5.083623282487277) / (x[0] / 5.547345449854117)) / x[1]) - ((5.939230066817374 + 5.939230066817374) * 4.818438684907152 * 5.547345449854117 * exp(4.9787556043961265) - exp(5.083623282487277 + x[1]) - exp(x[1]) * exp(x[0]) + x[0] * x[0] * x[1] * x[0] * exp(3.0580787826247215 + x[1])) / 4.032984299426399)` | 2.511431e-21 |
| 6 | 10,000 | 500 | 5 | `sqrt(sqrt     (8.254213750314564)) * (x[1] + (-10.225547483951923 * 9.025417700203663 * 9.590571516689572 * 8.318079817950606 * x[0] * 8.254213750314564 * tan(3.141592653589793) - x[1] * tan(3.141592653589793) * -9.347525141879945 * 9.590571516689572 * -11.05365123789452 * 9.590571516689572) - (-10.225547483951923 * 8.254213750314564² * 6.944557806606948 * -9.892836400798545 * 8.254213750314564² * 6.944557806606948 * x[0] * 8.254213750314564 * tan(3.141592653589793) - -9.347525141879945 * 8.318079817950606 * tan(3.141592653589793) * -9.36898251373954 * 8.254213750314564 * x[1] * -9.570992156231187 * 8.254213750314564 * 6.944557806606948 * -9.36898251373954 * 8.254213750314564)) - (exp(-7.99007532984518) * 2.0257904697066818 * (x[0] * cos(1.0567228657700947) + x[1]) + x[0] - exp(-9.36898251373954 * 2.0257904697066818) * x[1] - x[1] * tan(3.141592653589793) * -9.347525141879945 * 8.318079817950606 * -10.225547483951923 * 9.590571516689572) / sqrt(2.074545847057164)` | 1.472670e-24 |
| 7 | 10,000 | 500 | 10 | `(sqrt(x[1]) / ((x[1] - x[0]) / (x[1] / 384.8962882269058)) / ((-6.862665872131174 - x[1]) * 359.84784980813555) + x[1]) * log(tan(0.9724434591834495) / x[1] / (log(x[0]) * (x[1] - x[0]))) * (-332.6904684986283 - x[0] * x[1] - x[0] * x[1]) * ((-163.24116507533665 + x[1]) / 493.73440938907135) / (77.07105818030101 - x[0]) * x[0] * x[0] * (log(x[0] / x[1] / (x[1] - x[0] + 0.9724434591834495 / -261.97840724970973)) + 1.1430702439215117) * (43.26877293866266 / x[0]² / (x[1] * 223.57402074165518) + sqrt(x[1]) / ((sqrt(2.718281828459045) + x[1]) * (493.73440938907135 + x[1] * 245.95435231808204)) + x[1])` | 5.742911e+01 |
| 8 | 10,000 | 1000 | 10 | `(x[5] * log(14171.661837288537) - exp(x[5]) * cos(x[5]) * exp(x[5] + x[5]) + exp(x[5] + x[3]) - (x[5] + 5.22424095200899 + 4.919255252102357) + exp(x[5] + 4.919255252102357) / exp(cos(x[5])) + (cos(x[5]) + x[5]) * log(15361.32198836566) - log(-5373.432650868888 * -20631.50274792795) * exp(x[5] + x[4]) + exp(x[5] - x[4]) * (x[4] - 4.671905536726776 + x[4] - 4.982397747782802) + x[5] + x[5] + exp(x[5] + x[3])) / exp(x[5]) + exp(x[5] + x[5] + 263.1009235265916 / (-4534.039441136765 - (x[5] + x[3]) * exp(x[5])))` | 2.902945e+04 |





