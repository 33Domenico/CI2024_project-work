# Computational Intelligence Report

## Laboratory 1: Set Cover Problem

**Objective:**  
To implement and compare different hill climbing algorithms for solving the Set Cover Problem, with a focus on finding an optimal balance between solution quality and computational efficiency.

#### Activities Performed
- Implemented multiple variations of hill climbing algorithms:
  - Basic Hill Climber
  - Steepest Step with Restart
  - Simple Random Heavy Climb Mutation (RHCM)
  - Improved RHCM with adaptive strength
  - Improved RHCM v2 with maximum coverage preservation
- Tested each algorithm on six problem instances with varying parameters
- Visualized performance using fitness progression plots
- Analyzed convergence behavior and solution quality
- Compared algorithms to identify the most effective approach

#### Problem Definition
The Set Cover Problem involves selecting a minimum-cost subset of sets that cover all elements in a universe. Formally:
- Given a universe U of n elements
- A collection S of m sets containing elements from U
- A cost function c assigning costs to each set in S
- Find a minimum-cost subcollection of S that covers all elements in U

#### Functions Developed

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
- `universe_size`: The number of elements in the universe
- `num_sets`: The number of sets available to choose from
- `density`: The probability of an element being included in a set

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

#### Results
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
- Solution cost (sum of costs of selected sets)
- Convergence behavior (fitness improvement over iterations)

The **Improved RHCM v2** algorithm produced the best results (comparable with the Improved RHCM). Key factors contributing to its success:

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
![alt text](/report-image/image.png)

#### Imroved RHCM 

![alt text](/report-image/image-1.png)

### Peer Review
From Gpir0:
> "You've done a good job testing various algorithms and writing clean code. The RHCM v2 algorithm you have chosen takes an intelligent approach: selecting the set with the greatest coverage and excluding it from tweaks. The idea of a dynamic buffer that adjusts the algorithm's strength based on how 'stuck' it is in finding a better solution is brilliant. It resolves many issues found in other proposed algorithms (including mine). As for the results, you've achieved an improvement compared to the classic hill climbing algorithm. For some instances, you achieved values above the student average, while for others, you were at the average. A possible improvement would be to define a ranking of the set coverage and use that directly, instead of searching for the maximum in each tweak. Since the evaluation considers the number of calls to the 'cost function', this improvement is not strictly necessary."

From s331345:
> "The Improved RHCM v2 is a further improvement to RHCM that ensures the set with maximum coverage is not only preserved but also enforces mutation to avoid stagnation... The algorithms demonstrate the effectiveness of adaptive techniques for improving performance over time, with Improved RHCM v2 yielding the best results across most instances."

### Reflections after laboratory

While I initially selected Improved RHCM v2 as the best algorithm based on the experimental results, further reflection revealed an important insight: the approach of always preserving the set with maximum coverage may not actually be optimal for all instances. A set that covers many elements might also have a disproportionately high cost, whereas multiple smaller sets might cover the same elements at a lower total cost.
In reality, the Improved RHCM v2 is not a significant improvement over the basic Improved RHCM, which achieved almost identical results without making assumptions about which sets should always be included. The original adaptive strength mechanism in Improved RHCM provides sufficient flexibility to explore the solution space effectively without explicitly forcing the inclusion of specific sets.


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

### Route Validation and Cost Calculation
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

### Nearest Neighbor Algorithm
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

### Simulated Annealing for Population Initialization
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

### Inversion Mutation
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

### Inver-Over Crossover
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

### Tournament Selection
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

### Main Evolutionary Algorithm
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

### Vanuatu
- **Best route length**: 1,345.54 km
- **Function evaluations**: 58,196
- **Notes**: For this smaller dataset, applying just Simulated Annealing without the full evolutionary algorithm was sufficient.

### Italy
- **Best route length**: 4,172.76 km
- **Function evaluations**: 10,872,940
- **Parameters**: Mutation increase factor = 1.3, decrease factor = 0.7
- **Notes**: The algorithm triggered one SA restart at around generation 5100.

### Russia
- **Best route length**: 33,882.63 km
- **Function evaluations**: 2,290,980
- **Parameters**: Mutation threshold = 1.0
- **Notes**: The algorithm triggered one SA restart at around generation 8800.

### United States
- **Best route length**: 39,614.05 km
- **Function evaluations**: 4,465,568
- **Generations**: 20,000
- **Notes**: The algorithm triggered two SA restarts, at generations 13600 and 17150.

### China
- **Best route length**: 52,353.58 km
- **Function evaluations**: 13,931,136
- **Generations**: 30,000
- **Notes**: Required the longest run time due to dataset size, but achieved steady improvement throughout execution.

## Alternative Version
I developed a second version of the algorithm that included a population diversity check. This variant would insert new random solutions when the population became too homogeneous. Testing showed that this additional mechanism did not lead to significant improvements over the original version and added computational overhead. Therefore, the first version was selected as it proved to be faster while achieving equally good results.


## Peer Review
From LucianaColella7:
> "The Genetic Algorithm is well-implemented with a good mix of techniques, obtaining very good results. The adaptive mutation rate is a great addition, as it adjusts to balance exploration and exploitation, especially when it detects a lack of improvement. This is a smart way to prevent the algorithm from getting stuck in a local optimum. What I appreciated the most is the use of the Simulated Annealing restart. Another effective way to avoid local minima and give the algorithm the chance to keep exploring
In the README, it would have been helpful to include a summary of results for each country, as it would provide a clearer picture of the best outcomes from your implementation."


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

### Action and State Representation
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

### Available Actions and State Transitions
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

### Advanced Heuristic Function
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



### A* Search Implementation
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

### Puzzle Generator
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


## Results
The algorithm was tested on a 9×9 puzzle (80-puzzle) scrambled with 160 random moves.

- **Solution found in 145.75 seconds**
- **Nodes expanded: 274,068**
- **Solution length: 70 moves**


## Peer Review
From GioSilve:
> "The solution of the problem correctly implements an A* search algorithm, with the most important feature being the heuristic function, which combines four different types of metrics, and it allows the algorithm to be more informed and thus to solve much more efficiently the problem with even a large dimension puzzle, such as the 9x9 in less than 3 minutes that is reported in the output, that's excellent!"

From YaelNaor11:
> "Wow, it's impressive that you worked on the 9-puzzle and managed to implement the solution in such a short time! Your code is clear and demonstrates a good understanding of path-searching algorithms. Great work!"


