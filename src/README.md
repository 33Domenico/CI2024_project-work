## Project work: Symbolic Regression with Genetic Programming

**Objective:**  
To implement an efficient Genetic Programming (GP) algorithm for symbolic regression, focusing on discovering mathematical expressions that approximate underlying functions based on observed data points.

### Activities Performed

- Designed and implemented a complete tree-based Genetic Programming framework
- Created an expression representation system with function and terminal nodes
- Developed advanced genetic operators (crossover, mutation, selection)
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

Fitness sharing penalizes individuals that behave similarly to others in the population. For each individual, a sharing factor is calculated based on how many other expressions produce similar outputs and how close these outputs are. This factor increases as more semantically similar neighbors are found within a specified radius (σ).

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
        children = [grow_tree(config, max_depth, min_depth, current_depth + 1) 
                   for _ in range(function_info['arity'])]
        return FunctionNode(function_info['function'], function_info['arity'], 
                           function_info['symbol'], children)
    
    # Otherwise, we randomly choose between functions and terminals
    if random.random() < 0.5:  # 50% probability for functions or terminals
        function_info = config.get_random_function()
        children = [grow_tree(config, max_depth, min_depth, current_depth + 1) 
                   for _ in range(function_info['arity'])]
        return FunctionNode(function_info['function'], function_info['arity'], 
                           function_info['symbol'], children)
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
    'Ramped half-and-half' initialisation method
    Combines grow and full for greater diversity
    """
    # Choose a random depth between min_depth and max_depth
    depth = random.randint(min_depth, max_depth)
    
    # Choose randomly between 'grow' and 'full'
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
4. Uses a linear sharing kernel with a radius parameter σ to control the effect strength

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
def migration(islands: List[Island], migration_rate: float = 0.2, X_sample: np.ndarray = None) -> None:
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
            # Apply mutation with increased probability
            if random.random() < source_island.config.mutation_prob * mutation_strength:
                # Randomly choose the mutation type
                mutation_choice = random.random()
                
                if mutation_choice < 0.7:  # 70% subtree mutation
                    migrants[j] = subtree_mutation(migrant, source_island.config, 
                                                 max_depth=source_island.config.max_depth)
                else:  # 30% point mutation
                    migrants[j] = point_mutation(migrant, source_island.config)
        
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
    """
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
            
            # Collect all individuals for statistics
            all_individuals = []
            for island in islands:
                all_individuals.extend(island.population)
            
            # Periodic migration
            if (generation + 1) % migration_interval == 0:
                    migration(islands, migration_rate=migration_rate, X_sample=X_sample)
            
            # Calculate the best overall individual
            current_best = min([island.best_individual for island in islands if island.best_individual], 
                             key=lambda x: x.adjusted_fitness)
            
            #  Periodic bloat control
            if generation % bloat_control_interval == 0:
                for island in islands:
                    apply_bloat_control(island.population, config)

            # Calculate other statistics
            avg_fitness = np.mean([tree.adjusted_fitness for tree in all_individuals if tree.adjusted_fitness != float('inf')])
            avg_size = np.mean([tree.get_complexity() for tree in all_individuals])
            
        else:
            # Single population evolution
            if use_adaptive_mutation:
               original_mutation_prob = config.mutation_prob
            
            # Adjust mutation based on stagnation
            if generations_without_improvement > 0:
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
                apply_bloat_control(population, config)
            
            # Calculate the best individual
            current_best = min(population, key=lambda x: float('inf') if x.adjusted_fitness is None else x.adjusted_fitness)
            
            # Calculate statistics
            avg_fitness = np.mean([tree.adjusted_fitness for tree in population if tree.adjusted_fitness != float('inf')])
            avg_size = np.mean([tree.get_complexity() for tree in population])
            
            if use_adaptive_mutation:
                config.mutation_prob = original_mutation_prob

        # Update the best global individual
        if current_best.adjusted_fitness < best_fitness:
            best_individual = current_best.copy()
            best_fitness = current_best.adjusted_fitness
            generations_without_improvement = 0
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
        
        # Generation log
        if generation % 5 == 0 or generation == config.generations - 1:
           print(f"Generation {generation}, Best Fitness: {best_fitness}")
    
    total_time = time.time() - start_time
    print(f"Algorithm completed in {total_time:.2f} seconds")
    print(f"Best solution found:")
    print(f"  Simplified Expression: {sympy_simplify_expression(best_individual.to_string())}")
    print(f"  Expression: {best_individual.to_string()}")
    print(f"  Fitness: {best_fitness}")
    print(f"  Complexity: {best_individual.get_complexity()} nodes")
    
    # Visualiza statistics
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
    
    # Complete the population with new individuals
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
            
            # Randomly choose between subtree mutation and point mutation
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
            # Reproduction (direct copy)
            parent = tournament_selection(population, config.tournament_size)
            offspring = parent.copy()
            offspring.age += 1  # Increase age
            new_population.append(offspring)
    
    # Make sure the population is exactly the right size
    if len(new_population) > config.pop_size:
        new_population = new_population[:config.pop_size]
    
    # Apply fitness sharing for diversity
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
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 60
     },
     "use_islands": True,
     "n_islands": 5,
     "migration_interval": 40,
     "migration_rate": 0.12
    },
    
 
    {"file_path": "../data/problem_3.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 65,
     },
     "use_islands": True,
     "n_islands": 4,
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
         "max_depth": 10,  
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 80
     },
     "use_islands": True,
     "n_islands": 5,
     "migration_interval": 40,
     "migration_rate": 0.15
    },
    

    {"file_path": "../data/problem_6.npz", 
     "config": {
         "max_depth": 10,  
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 40,
         "parsimony_coef": 0.05, 
     },
     "use_islands": True, 
    },
    

    {"file_path": "../data/problem_7.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 60,
         "tournament_size": 100, 
         "elitism_rate": 0.2,    
     },
     "use_islands": True,
     "n_islands": 5,
     "migration_interval": 40,
     "migration_rate": 0.25  
    },
    
    {"file_path": "../data/problem_8.npz", 
     "config": {
         "max_depth": 8, 
         "pop_size": 10000, 
         "generations": 500,
         "max_tree_size": 50,
     },
     "use_islands": True,
     "n_islands": 5,
     "migration_interval": 40,
     "migration_rate": 0.15
    }
]
```



#### Result Table

| Problem | Population Size | Generations | Simplified Expression | Fitness (MSE) |
|---------|----------------|-------------|----------------------|---------------|
| 0 | 10,000 | 500 | tan(x[0] - x[1] + tan(x[0])) + 0.15322121158548999*sin(x[1])/x[1] | 3.16172454411196e-11 |
| 1 | 10,000 | 100 | 0.3782860327383275*x[0]/(x[0] - 0.7458863362755191) | 7.125940794232773e-34 |
| 2 | 10,000 | 500 | (-1112267.7584117618*x[0] - 1112267.7584117618*sin(1.9842202415071949*sin(x[0])))*sqrt(-sqrt(x[1] + x[2])*sin(sin(x[0])) + sin(x[2]))*sqrt(x[1] - x[2]/sin(sin(x[0])) - tan(sin(x[0]))) + (x[0] + sin(1.50996183095054*sin(x[0])))*(-2189.267872464636*x[2]*(x[1] + x[2] - tan(sin(x[0]))) + 2742717.0591339385) | 5497243377966.15 |
| 3 | 10,000 | 500 | 2*x[0]**2 + 1.888011443473933*x[0] - 5.751754953796507*x[1]**2 + x[1]*(-x[0]*x[1] - tan(tan(x[2])) + 7.960067489897881) - 1.866842958998818*x[1] - 3.499972972522648*x[2] + 3.343913409679408 | 5.3895077493714596e-05 |
| 4 | 10,000 | 500 | sqrt(0.59442506135513196*x[0] - 10.843735353774745) + (0.01074365698385143*exp(0.50406178400871155*x[0]**3 - sin(2*cos(x[1]))) + 6.990332232812374)*cos(x[1]) | 6.244233514458511e-05 |
| 5 | 10,000 | 500 | -6.226751041691537e-16*x[0]*x[1]*(374.93213046082964*x[0]**2*x[1]**3 + (-x[0] - x[1] + exp(x[0]))*exp(2*x[1]) - 0.2477463143761617*exp(0.4441536940414833*x[0] + x[1]) - 34390.847090009492*exp(x[0] + sin(sin(x[0]))) - exp(exp(x[1])) - 24175.980509968689) | 7.074053721003686e-21 |
| 6 | 10,000 | 500 | tan(sqrt(-1.0268830067559338)) * (0.26623240720729413 * sqrt(sqrt(sqrt(-1.8923799511965012 + -1.6226422471794368) / x[1]))) | 3.430645922923297e-06 |
| 7 | 10,000 | 500 | 9.431522071392338*sqrt(sqrt(x[0])*(x[1] + 1.605583320053832)*log(-x[0])) | 20.3946149322377 |
| 8 | 10,000 | 500 | -x[1] - x[4] + sqrt(x[5]) - 3*x[5] - 39.027296247012435*sqrt(-x[4]) - (-x[4] + 39.027296247012435*sqrt(-x[5]) + 1523.1298523520709)*exp(-x[0]) - 5*exp(x[4]) + exp(2*x[5]) - 2*exp(x[5]) + 11691.94197593936 | 1042712.1291460795 |





