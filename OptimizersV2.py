# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:59:35 2025

@author: karim Bechiri
"""

import numpy as np
import random
import math 

def _sigmoid(self, x):
    """Sigmoid function for binary conversion"""
    return 1 / (1 + np.exp(-x))

def _binary_to_float(self, binary, lb, ub):
    """Convert binary string to float within bounds"""
    num = int(''.join(map(str, binary)), 2)
    max_val = 2**len(binary) - 1
    return lb + (ub - lb) * num / max_val

def _float_to_binary(self, x, lb, ub, bits):
    """Convert float to binary string representation"""
    x_norm = (x - lb) / (ub - lb)
    num = round(x_norm * (2**bits - 1))
    return [int(b) for b in format(num, f'0{bits}b')]

class optimizers:
    def __init__(self,parameters):
        # Fixed properties  
        self.params=parameters
        # Progress callback for tracking iterations
        self.progress_callback = parameters.get('progress_callback', None)
        self.show_progress = parameters.get('show_progress', False)
        
        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
            self.tqdm_available = True
        except ImportError:
            self.tqdm_available = False
    
    def _update_progress(self, iteration, max_iter, best_fitness, method_name="Optimizer"):
        """Update progress bar or callback"""
        if self.progress_callback is not None:
            self.progress_callback(iteration, max_iter, best_fitness)
        if hasattr(self, 'pbar') and self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({'fitness': f'{best_fitness:.4f}'})
    
    def _create_progress_bar(self, max_iter, method_name):
        """Create progress bar if enabled"""
        if self.show_progress and self.tqdm_available:
            self.pbar = self.tqdm(total=max_iter, desc=f"    {method_name}", 
                                 unit="iter", ncols=80, leave=False,
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        else:
            self.pbar = None
    
    def _close_progress_bar(self):
        """Close progress bar"""
        if hasattr(self, 'pbar') and self.pbar is not None:
            self.pbar.close()
            self.pbar = None
    
    def _sigmoid(self, x):
        """Sigmoid function for binary conversion"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def GWO(self):
        """
        Gray Wolf Optimizer (GWO)
        
        Parameters:
            obj_func (function): The objective function to optimize
            lb (list): Lower bounds for each dimension
            ub (list): Upper bounds for each dimension
            dim (int): Number of dimensions
            search_agents_no (int): Number of search agents (population size)
            max_iter (int): Maximum number of iterations
            
        Returns:
            tuple: (best solution, best fitness value, convergence curve)
        """
        dim=self.params["dim"]
        lb=self.params["lb"]
        ub=self.params["ub"]
        nbr=self.params["agents_nbr"]
        max_iter=self.params["max_iter"]
        
        # Initialize alpha, beta, and delta positions
        alpha_pos = np.zeros(dim) ;    alpha_score = float("inf")    
        beta_pos  = np.zeros(dim) ;    beta_score  = float("inf")
        delta_pos = np.zeros(dim) ;    delta_score = float("inf")
        
        # Initialize the positions of search agents
        positions = np.zeros((nbr, dim))
        for i in range(nbr):
            for j in range(dim):
                positions[i,j] = random.uniform(lb[j], ub[j])
        
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "GWO")
        
        # Main loop
        for t in range(max_iter):
            for i in range(nbr):
                # Return back the search agents that go beyond the boundaries
                for j in range(dim):
                    positions[i,j] = np.clip(positions[i,j], lb[j], ub[j])
                
                # Calculate objective function for each search agent
                fitness = self.params["obj_func"](positions[i,:])
                
                # Update alpha, beta, and delta
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha_pos = positions[i,:].copy()
                
                if (fitness > alpha_score and fitness < beta_score):
                    beta_score = fitness
                    beta_pos = positions[i,:].copy()
                
                if (fitness > alpha_score and fitness > beta_score and fitness < delta_score):
                    delta_score = fitness
                    delta_pos = positions[i,:].copy()
            
            # a decreases linearly from 2 to 0 (decreasive step)
            a = 3 - t * (3 / max_iter)
            
            # Update the position of search agents
            for i in range(nbr):
                for j in range(dim):
                    
                    # r1, r2 are a random number in [0,1]
                    r1 = random.random()  ;   r2 = random.random()  
                    
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i,j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1 = random.random()  ;   r2 = random.random()  
                    
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    D_beta = abs(C2 * beta_pos[j] - positions[i,j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1 = random.random()  ;   r2 = random.random()  
                    
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    D_delta = abs(C3 * delta_pos[j] - positions[i,j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    positions[i,j] = (X1 + X2 + X3) / 3
            
            convergence_curve[t] = alpha_score
            
            # Update progress bar
            self._update_progress(t, max_iter, alpha_score, "GWO")
        
        # Close progress bar
        self._close_progress_bar()
        
        return alpha_pos, alpha_score, convergence_curve


    def DE(self, F=0.8, CR=0.9):
        """
        Differential Evolution (DE) optimizer
        
        Returns:
            tuple: (best solution, best fitness value, convergence curve)
        """
        dim = self.params["dim"]
        lb = self.params["lb"]
        ub = self.params["ub"]
        nbr = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        
        # Initialize population
        population = np.zeros((nbr, dim))
        fitness = np.zeros(nbr)
        
        for i in range(nbr):
            population[i,:] = np.array([random.uniform(lb[j], ub[j]) for j in range(dim)])
            fitness[i] = obj_func(population[i,:])
        
        # Initialize best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx,:].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "DE")
        
        for t in range(max_iter):
            for i in range(nbr):
                # Mutation: select three distinct random individuals
                candidates = [idx for idx in range(nbr) if idx != i]
                a, b, c = random.sample(candidates, 3)
                
                # Generate mutant vector
                mutant = population[a,:] + F * (population[b,:] - population[c,:])
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover: create trial vector
                trial = population[i,:].copy()
                cross_point = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == cross_point:
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = obj_func(trial)
                if trial_fitness < fitness[i]:
                    population[i,:] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            convergence_curve[t] = best_fitness
            
            # Update progress bar
            self._update_progress(t, max_iter, best_fitness, "DE")
        
        # Close progress bar
        self._close_progress_bar()
        
        return best_solution, best_fitness, convergence_curve

    def PSO(self, w=0.7, c1=1.5, c2=1.5):
        """
        Particle Swarm Optimization (PSO)
        
        Returns:
            tuple: (best solution, best fitness value, convergence curve)
        """
        dim = self.params["dim"]
        lb = self.params["lb"]
        ub = self.params["ub"]
        nbr = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        
        # Initialize particles
        positions = np.zeros((nbr, dim))
        velocities = np.zeros((nbr, dim))
        personal_best_pos = np.zeros((nbr, dim))
        personal_best_fit = np.full(nbr, float('inf'))
        
        for i in range(nbr):
            positions[i,:] = np.array([random.uniform(lb[j], ub[j]) for j in range(dim)])
            personal_best_pos[i,:] = positions[i,:].copy()
            personal_best_fit[i] = obj_func(positions[i,:])
        
        # Initialize global best
        global_best_idx = np.argmin(personal_best_fit)
        global_best_pos = personal_best_pos[global_best_idx,:].copy()
        global_best_fit = personal_best_fit[global_best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "PSO")
        
        for t in range(max_iter):
            for i in range(nbr):
                # Update velocity
                r1, r2 = random.random(), random.random()
                velocities[i,:] = (w * velocities[i,:] + 
                                  c1 * r1 * (personal_best_pos[i,:] - positions[i,:]) + 
                                  c2 * r2 * (global_best_pos - positions[i,:]))
                
                # Update position
                positions[i,:] = np.clip(positions[i,:] + velocities[i,:], lb, ub)
                
                # Evaluate fitness
                current_fit = obj_func(positions[i,:])
                
                # Update personal best
                if current_fit < personal_best_fit[i]:
                    personal_best_pos[i,:] = positions[i,:].copy()
                    personal_best_fit[i] = current_fit
                    
                    # Update global best
                    if current_fit < global_best_fit:
                        global_best_pos = positions[i,:].copy()
                        global_best_fit = current_fit
            
            convergence_curve[t] = global_best_fit
            
            # Update progress bar
            self._update_progress(t, max_iter, global_best_fit, "PSO")
        
        # Close progress bar
        self._close_progress_bar()
        
        return global_best_pos, global_best_fit, convergence_curve


    def GA(self, pc=0.9, pm=0.1, elitism=True):
        """
        Genetic Algorithm (GA) optimizer
        
        Parameters:
            pc (float): Probability of crossover [0, 1]
            pm (float): Probability of mutation [0, 1]
            elitism (bool): Whether to use elitism
            
        Returns:
            tuple: (best solution, best fitness value, convergence curve)
        """
        dim = self.params["dim"]
        lb = self.params["lb"]
        ub = self.params["ub"]
        nbr = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        
        # Initialize population
        population = np.zeros((nbr, dim))
        fitness = np.zeros(nbr)
        
        for i in range(nbr):
            population[i,:] = np.array([random.uniform(lb[j], ub[j]) for j in range(dim)])
            fitness[i] = obj_func(population[i,:])
        
        # Initialize best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx,:].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "GA")
        
        for t in range(max_iter):
            # Selection (Tournament selection)
            new_population = np.zeros((nbr, dim))
            new_fitness = np.zeros(nbr)
            
            if elitism:
                # Keep the best individual
                new_population[0,:] = best_solution.copy()
                new_fitness[0] = best_fitness
            
            start_idx = 1 if elitism else 0
            
            for i in range(start_idx, nbr):
                # Tournament selection (size 2)
                a, b = random.sample(range(nbr), 2)
                if fitness[a] < fitness[b]:
                    winner = a
                else:
                    winner = b
                new_population[i,:] = population[winner,:].copy()
                new_fitness[i] = fitness[winner]
            
            # Crossover (Single-point crossover)
            for i in range(start_idx, nbr, 2):
                if i+1 < nbr and random.random() < pc:
                    # Select crossover point
                    cross_point = random.randint(1, dim-1)
                    # Perform crossover
                    temp = new_population[i,cross_point:].copy()
                    new_population[i,cross_point:] = new_population[i+1,cross_point:]
                    new_population[i+1,cross_point:] = temp
            
            # Mutation (Gaussian mutation)
            for i in range(start_idx, nbr):
                for j in range(dim):
                    if random.random() < pm:
                        # Gaussian mutation with std dev proportional to search range
                        sigma = 0.1 * (ub[j] - lb[j])
                        new_population[i,j] += random.gauss(0, sigma)
                        new_population[i,j] = np.clip(new_population[i,j], lb[j], ub[j])
            
            # Update population and fitness
            population = new_population.copy()
            for i in range(start_idx, nbr):
                fitness[i] = obj_func(population[i,:])
            
            # Update best solution
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution = population[current_best_idx,:].copy()
                best_fitness = fitness[current_best_idx]
            
            convergence_curve[t] = best_fitness
            
            # Update progress bar
            self._update_progress(t, max_iter, best_fitness, "GA")
        
        # Close progress bar
        self._close_progress_bar()
        
        return best_solution, best_fitness, convergence_curve
    
    def TSA(self, Pr_exp=0.7, Pr_esc=0.2):
        """
        Tangent Search Algorithm (TSA)
        
        Parameters:
            Pr_exp (float): Probability of exploration [0, 1]
            Pr_esc (float): Probability of escape [0, 1]
            
        Returns:
            tuple: (best solution, best fitness value, convergence curve)
        """
        dim = self.params["dim"]
        lb = np.array(self.params["lb"])
        ub = np.array(self.params["ub"])
        nbr = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        
        # Initialize population
        population = np.zeros((nbr, dim))
        fitness = np.zeros(nbr)
        
        for i in range(nbr):
            population[i,:] = np.array([random.uniform(lb[j], ub[j]) for j in range(dim)])
            fitness[i] = obj_func(population[i,:])
        
        # Initialize best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx,:].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "TSA")
        
        for t in range(max_iter):
            for i in range(nbr):
                X = population[i,:].copy()
                
                if random.random() <= Pr_exp:
                    # Exploration phase
                    if random.random() <= 0.25:  # High exploration
                        X = X + np.tan(np.random.randn(dim) * np.pi)  # Large tangent flight
                    else:  # Low exploration
                        if np.array_equal(best_solution, X):
                            step = 0.5 * np.sign(np.random.rand(dim) - 0.5)
                        else:
                            step = 0.5 * np.sign(np.random.rand(dim) - 0.5) * np.linalg.norm(best_solution - X)
                        theta = np.random.rand(dim) * np.pi / 2.5
                        X = X + np.multiply(step, np.tan(theta))  # Small tangent flight
                else:
                    # Intensification phase
                    theta = np.random.rand(dim) * np.pi / 2.5
                    step = 5 * np.sign(np.random.rand(dim) - 0.5) * np.linalg.norm(best_solution) * np.log(1 + 15 * dim / (t+1))
                    
                    if np.array_equal(best_solution, X):
                        X = best_solution + step * np.tan(theta) * (random.random() * (best_solution - X))
                    else:
                        if random.random() <= 0.3:
                            X = best_solution + step * np.tan(theta) * (best_solution - X)
                        else:
                            f1 = -1 + 2 * random.random()  # Random number in [-1, 1]
                            step = 15 * f1 / np.log(1 + (t+1))
                            X = X + step * (best_solution - random.random() * (best_solution - X))
                
                # Local escape mechanism
                if random.random() <= Pr_esc:
                    if random.random() <= 0.8:  # Large local escape
                        theta = np.random.randn(dim) * np.pi
                        X = X + np.tan(theta) * (ub - lb)
                    else:  # Small local escape
                        idx = random.randint(0, dim-1)
                        mask = np.random.rand(dim) <= (0.05/dim)
                        mask[idx] = True
                        X[mask] = population[i,mask].copy()
                
                # Boundary handling
                X = np.clip(X, lb, ub)
                
                # Evaluate new solution
                new_fitness = obj_func(X)
                
                # Update if better
                if new_fitness < fitness[i]:
                    population[i,:] = X.copy()
                    fitness[i] = new_fitness
                    
                    if new_fitness < best_fitness:
                        best_solution = X.copy()
                        best_fitness = new_fitness
            
            convergence_curve[t] = best_fitness
            
            # Update progress bar
            self._update_progress(t, max_iter, best_fitness, "TSA")
        
        # Close progress bar
        self._close_progress_bar()
        
        return best_solution, best_fitness, convergence_curve

    def CACO(self, q=0.5, xi=0.85):
        """
        Continuous Ant Colony Optimization (CACO / ACOr)
        Socha, K., & Dorigo, M. (2008). "Ant colony optimization for
        continuous domains." European Journal of Operational Research.

        Classic ACO deposits pheromone on discrete graph edges. ACOr adapts
        this to continuous search spaces by replacing the pheromone table
        with a solution archive: the k best solutions found so far act as
        the "pheromone trail." Each ant builds a new candidate by
        probabilistically choosing an archive member (better-ranked members
        are chosen more often, like stronger pheromone) and sampling each
        dimension from a Gaussian centered on that member, with spread
        proportional to how scattered the archive is along that dimension.
        The archive is then refreshed by keeping the best k solutions among
        (old archive U new ants), which plays the role of pheromone
        evaporation (worse trails fade out) and reinforcement (good trails
        persist) in a single step.

        Parameters:
            q (float): Locality/exploitation parameter. Smaller q biases
                sampling more strongly toward the best-ranked archive
                members (faster convergence, less diversity). Typical
                range [0.01, 1.0].
            xi (float): Controls how fast the sampling spread shrinks as
                the archive converges (evaporation-rate analog). Smaller
                xi narrows the search faster. Default 0.85, as recommended
                by Socha & Dorigo.

        Returns:
            tuple: (best solution, best fitness value, convergence curve)
        """
        dim = self.params["dim"]
        lb = np.array(self.params["lb"], dtype=float)
        ub = np.array(self.params["ub"], dtype=float)
        k = self.params["agents_nbr"]          # archive size = colony size
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]

        # --- Initialize solution archive (continuous pheromone table) ---
        archive = np.zeros((k, dim))
        for i in range(k):
            archive[i, :] = np.array([random.uniform(lb[j], ub[j]) for j in range(dim)])
        archive_fitness = np.array([obj_func(archive[i, :]) for i in range(k)])

        # Sort archive by fitness ascending (rank 1 = best, minimization)
        sort_idx = np.argsort(archive_fitness)
        archive = archive[sort_idx, :]
        archive_fitness = archive_fitness[sort_idx]

        best_solution = archive[0, :].copy()
        best_fitness = archive_fitness[0]

        # --- Precompute rank-based selection weights (Gaussian over rank) ---
        ranks = np.arange(1, k + 1)
        weights = (1.0 / (q * k * np.sqrt(2 * np.pi))) * \
                  np.exp(-((ranks - 1) ** 2) / (2 * (q ** 2) * (k ** 2)))
        probs = weights / np.sum(weights)

        convergence_curve = np.zeros(max_iter)

        # Create progress bar
        self._create_progress_bar(max_iter, "CACO")

        for t in range(max_iter):
            new_ants = np.zeros((k, dim))
            new_fitness = np.zeros(k)

            for i in range(k):
                # Roulette-wheel pick of guiding archive member (pheromone trail)
                l = np.random.choice(k, p=probs)

                ant = np.zeros(dim)
                for j in range(dim):
                    # Average distance from guide to rest of archive -> spread
                    sigma_j = xi * np.sum(np.abs(archive[:, j] - archive[l, j])) / max(k - 1, 1)
                    if sigma_j <= 0:
                        sigma_j = 1e-6 * (ub[j] - lb[j] + 1e-9)
                    ant[j] = np.random.normal(archive[l, j], sigma_j)

                ant = np.clip(ant, lb, ub)
                fit = obj_func(ant)

                new_ants[i, :] = ant
                new_fitness[i] = fit

                if fit < best_fitness:
                    best_solution = ant.copy()
                    best_fitness = fit

            # Pheromone update: merge archive + new ants, keep k best
            # (implicit evaporation of weaker trails, reinforcement of strong ones)
            combined = np.vstack([archive, new_ants])
            combined_fitness = np.concatenate([archive_fitness, new_fitness])
            keep_idx = np.argsort(combined_fitness)[:k]

            archive = combined[keep_idx, :]
            archive_fitness = combined_fitness[keep_idx]

            convergence_curve[t] = best_fitness

            # Update progress bar
            self._update_progress(t, max_iter, best_fitness, "CACO")

        # Close progress bar
        self._close_progress_bar()

        return best_solution, best_fitness, convergence_curve


    """
    Enhanced Binary Optimization Library
    Includes:
    - Binary Genetic Algorithm (BGA)
    - Binary Particle Swarm Optimization (BPSO)
    - Binary Simulated Annealing (BSA)
    - Binary Grey Wolf Optimizer (BGWO)
    - Binary Differential Evolution (BDE)
    """



    def BGA(self, pc=0.9, pm=0.05, elitism=True):
        """
        Binary Genetic Algorithm (BGA)
        
        Parameters:
            pc (float): crossover probability [0,1]
            pm (float): mutation probability [0,1]
            elitism (bool): whether to keep best solution
            
        Returns:
            tuple: (best_solution, best_fitness, convergence_curve)
        """
        dim = self.params["dim"]
        nbr = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        
        # Initialize population
        population = np.random.randint(0, 2, (nbr, dim))
        fitness = np.array([obj_func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "BGA")
        
        for t in range(max_iter):
            # Selection (Tournament selection)
            new_pop = np.zeros((nbr, dim), dtype=int)
            new_fitness = np.zeros(nbr)
            
            if elitism:
                new_pop[0] = best_solution.copy()
                new_fitness[0] = best_fitness
            
            start_idx = 1 if elitism else 0
            
            for i in range(start_idx, nbr):
                # Tournament selection (size 2)
                a, b = random.sample(range(nbr), 2)
                if fitness[a] < fitness[b]:
                    winner = a
                else:
                    winner = b
                new_pop[i] = population[winner].copy()
                new_fitness[i] = fitness[winner]
            
            # Crossover (Single-point)
            for i in range(start_idx, nbr, 2):
                if i+1 < nbr and random.random() < pc:
                    pt = random.randint(1, dim-1)
                    # Swap tails
                    temp = new_pop[i, pt:].copy()
                    new_pop[i, pt:] = new_pop[i+1, pt:]
                    new_pop[i+1, pt:] = temp
            
            # Mutation (Bit-flip)
            for i in range(start_idx, nbr):
                for j in range(dim):
                    if random.random() < pm:
                        new_pop[i,j] = 1 - new_pop[i,j]
            
            # Update population
            population = new_pop.copy()
            for i in range(start_idx, nbr):
                fitness[i] = obj_func(population[i])
            
            # Update best solution
            current_best = np.argmin(fitness)
            if fitness[current_best] < best_fitness:
                best_solution = population[current_best].copy()
                best_fitness = fitness[current_best]
            
            convergence_curve[t] = best_fitness
            
            # Update progress bar
            self._update_progress(t, max_iter, best_fitness, "BGA")
        
        # Close progress bar
        self._close_progress_bar()
        
        return best_solution, best_fitness, convergence_curve
    
    def BPSO(self, w=0.7, c1=1.5, c2=1.5):
        """
        Binary Particle Swarm Optimization (BPSO)
        
        Parameters:
            w (float): inertia weight
            c1 (float): cognitive coefficient
            c2 (float): social coefficient
            
        Returns:
            tuple: (best_solution, best_fitness, convergence_curve)
        """
        dim = self.params["dim"]
        nbr = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        
        # Initialize particles and velocities
        particles = np.random.randint(0, 2, (nbr, dim))
        velocities = np.random.uniform(-1, 1, (nbr, dim))
        
        pbest = particles.copy()
        pbest_fitness = np.array([obj_func(p) for p in particles])
        gbest = pbest[np.argmin(pbest_fitness)]
        gbest_fitness = np.min(pbest_fitness)
        
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "BPSO")
        
        for t in range(max_iter):
            for i in range(nbr):
                # Update velocity
                r1, r2 = random.random(), random.random()
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (pbest[i] - particles[i]) + 
                               c2 * r2 * (gbest - particles[i]))
                
                # Update position using sigmoid
                sig_vel = self._sigmoid(velocities[i])
                particles[i] = (sig_vel > np.random.rand(dim)).astype(int)
                
                # Evaluate
                current_fit = obj_func(particles[i])
                
                # Update personal best
                if current_fit < pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = current_fit
                    
                    # Update global best
                    if current_fit < gbest_fitness:
                        gbest = particles[i].copy()
                        gbest_fitness = current_fit
            
            convergence_curve[t] = gbest_fitness
            
            # Update progress bar
            self._update_progress(t, max_iter, gbest_fitness, "BPSO")
        
        # Close progress bar
        self._close_progress_bar()
        
        return gbest, gbest_fitness, convergence_curve
    
    def SA(self, initial_temp=1000, cooling_rate=0.95, min_temp=1e-10):
        """
        Continuous Simulated Annealing (CSA) optimizer
        
        Parameters:
            initial_temp (float): Initial temperature
            cooling_rate (float): Cooling rate (0.8-0.99 typical)
            min_temp (float): Minimum temperature threshold
            
        Returns:
            tuple: (best_solution, best_fitness, convergence_curve)
        """
        dim = self.params["dim"]
        lb = self.params["lb"]
        ub = self.params["ub"]
        obj_func = self.params["obj_func"]
        max_iter = self.params["max_iter"]
        
        # Initialize random solution
        current_solution = np.array([random.uniform(lb[i], ub[i]) for i in range(dim)])
        current_fitness = obj_func(current_solution)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        convergence_curve = np.zeros(max_iter)
        temperature = initial_temp
        
        # Create progress bar
        self._create_progress_bar(max_iter, "SA")
        
        for t in range(max_iter):
            # Generate neighbor solution
            neighbor = current_solution + np.random.normal(0, 1, dim) * temperature
            neighbor = np.clip(neighbor, lb, ub)  # Keep within bounds
            
            # Evaluate neighbor
            neighbor_fitness = obj_func(neighbor)
            
            # Calculate energy difference
            delta_energy = neighbor_fitness - current_fitness
            
            # Acceptance criteria
            if delta_energy < 0 or random.random() < np.exp(-delta_energy / temperature):
                current_solution = neighbor.copy()
                current_fitness = neighbor_fitness
                
                # Update best solution
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            
            # Update convergence curve
            convergence_curve[t] = best_fitness
            
            # Cool down
            temperature *= cooling_rate
            
            # Early stopping if temperature too low
            if temperature < min_temp:
                convergence_curve[t:] = best_fitness  # Fill remaining iterations
                break
                
            # Update progress bar
            self._update_progress(t, max_iter, best_fitness, "SA")
        
        # Close progress bar
        self._close_progress_bar()
        
        return best_solution, best_fitness, convergence_curve
    
    def BSA(self):
        """
        Binary Simulated Annealing (BSA)
        
        Returns:
            tuple: (best_solution, best_fitness, convergence_curve)
        """
        dim = self.params["dim"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        initial_temp = self.params.get("initial_temp", 100)
        cooling_rate = self.params.get("cooling_rate", 0.95)
        
        current_solution = np.random.randint(0, 2, dim)
        current_fitness = obj_func(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        temp = initial_temp
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "BSA")
        
        for t in range(max_iter):
            # Generate neighbor by flipping one random bit
            neighbor = current_solution.copy()
            flip_pos = random.randint(0, dim-1)
            neighbor[flip_pos] = 1 - neighbor[flip_pos]
            neighbor_fitness = obj_func(neighbor)
            
            # Calculate energy difference
            delta = neighbor_fitness - current_fitness
            
            # Acceptance probability
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            
            # Cooling schedule
            temp *= cooling_rate
            convergence_curve[t] = best_fitness
            
            # Update progress bar
            self._update_progress(t, max_iter, best_fitness)
        
        # Close progress bar
        self._close_progress_bar()
        
        return best_solution, best_fitness, convergence_curve
    
    def BGWO(self):
        """
        Binary Grey Wolf Optimizer (BGWO)
        
        Returns:
            tuple: (best_solution, best_fitness, convergence_curve)
        """
        dim = self.params["dim"]
        nbr = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        
        # Initialize wolves
        wolves = np.random.randint(0, 2, (nbr, dim))
        fitness = np.array([obj_func(w) for w in wolves])
        
        # Initialize alpha, beta, delta
        sorted_idx = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_idx[:3]]
        alpha_fit, beta_fit, delta_fit = fitness[sorted_idx[:3]]
        
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "BGWO")
        
        for t in range(max_iter):
            a = 2 - t * (2 / max_iter)  # Decreases linearly from 2 to 0
            
            for i in range(nbr):
                for j in range(dim):
                    # Calculate distances to alpha, beta, delta
                    r1, r2 = random.random(), random.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolves[i,j])
                    X1 = alpha[j] - A1 * D_alpha
                    
                    r1, r2 = random.random(), random.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - wolves[i,j])
                    X2 = beta[j] - A2 * D_beta
                    
                    r1, r2 = random.random(), random.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - wolves[i,j])
                    X3 = delta[j] - A3 * D_delta
                    
                    # Update position using majority vote
                    wolves[i,j] = 1 if (X1 + X2 + X3) > 1.5 else 0
            
            # Evaluate new solutions
            fitness = np.array([obj_func(w) for w in wolves])
            
            # Update alpha, beta, delta
            sorted_idx = np.argsort(fitness)
            current_alpha, current_beta, current_delta = wolves[sorted_idx[:3]]
            current_alpha_fit, current_beta_fit, current_delta_fit = fitness[sorted_idx[:3]]
            
            if current_alpha_fit < alpha_fit:
                alpha = current_alpha.copy()
                alpha_fit = current_alpha_fit
            
            if current_beta_fit < beta_fit:
                beta = current_beta.copy()
                beta_fit = current_beta_fit
            
            if current_delta_fit < delta_fit:
                delta = current_delta.copy()
                delta_fit = current_delta_fit
            
            convergence_curve[t] = alpha_fit
            
            # Update progress bar
            self._update_progress(t, max_iter, alpha_fit, "BGWO")
        
        # Close progress bar
        self._close_progress_bar()
        
        return alpha, alpha_fit, convergence_curve
    
    def BDE(self, F=0.8, CR=0.9):
        """
        Binary Differential Evolution (BDE)
        
        Parameters:
            F (float): scaling factor [0,2]
            CR (float): crossover probability [0,1]
            
        Returns:
            tuple: (best_solution, best_fitness, convergence_curve)
        """
        dim = self.params["dim"]
        nbr = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        
        # Initialize population
        population = np.random.randint(0, 2, (nbr, dim))
        fitness = np.array([obj_func(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "BDE")
        
        for t in range(max_iter):
            for i in range(nbr):
                # Select three distinct random individuals
                candidates = [idx for idx in range(nbr) if idx != i]
                a, b, c = random.sample(candidates, 3)
                
                # Generate mutant vector (binary XOR operation)
                mutant = population[a] ^ (population[b] & population[c])
                
                # Crossover
                trial = population[i].copy()
                cross_point = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == cross_point:
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = obj_func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            convergence_curve[t] = best_fitness
            
            # Update progress bar
            self._update_progress(t, max_iter, best_fitness)
        
        # Close progress bar
        self._close_progress_bar()
        
        return best_solution, best_fitness, convergence_curve

    def ACO(self, n_select=None, alpha=1.0, beta=2.0, rho=0.1, Q=1.0,
            elite_weight=2.0, use_heuristic=True):
        """
        Ant Colony Optimization (ACO) -- original combinatorial form

        Classic (discrete) Ant Colony Optimization, applied natively to
        the client-selection subset problem rather than continuous
        search. Client selection is naturally combinatorial -- choose k
        of N clients -- so this mirrors classic ACO's construction-graph
        mechanism directly instead of routing through a continuous
        pheromone archive (see the CACO method for that variant).

        Mechanism:
            - One pheromone value tau_i is kept per CLIENT (node-based
              pheromone, since client subset selection has no meaningful
              edge/order structure -- unlike TSP).
            - Each ant constructs a candidate subset by sequentially
              picking clients without replacement. The probability of
              picking client c from the remaining pool is proportional
              to (tau_c ** alpha) * (eta_c ** beta), exactly the classic
              ACO transition rule.
            - eta_c is a static heuristic desirability per client,
              estimated once up front via a "leave-one-in" probe (select
              only client c and evaluate). This costs `dim` extra
              obj_func calls made only once before the main loop --
              report this fixed overhead alongside per-iteration cost in
              any timing comparison (Table 5-style analysis). Set
              use_heuristic=False to skip this and rely on pheromone
              alone (pure Ant System, zero extra evaluations).
            - After all ants in an iteration are evaluated: pheromone
              evaporates on every client (tau *= (1-rho)), then every
              ant deposits pheromone on the clients it selected,
              proportional to 1/fitness (better subsets -> stronger
              trail). The best-so-far subset gets an additional elitist
              deposit, which speeds convergence without requiring
              exact-best-ant tracking each round.

        Parameters:
            n_select (int or None): Number of clients to select (the
                subset size k). If None, falls back to
                self.params["n_select"], then to
                self.params["selection_ratio"] * dim, then to 30% of
                dim -- matching the selection_ratio convention already
                used by RSELECT.
            alpha (float): Pheromone influence exponent.
            beta (float): Heuristic influence exponent.
            rho (float): Pheromone evaporation rate in [0, 1].
            Q (float): Pheromone deposit scaling constant.
            elite_weight (float): Extra deposit multiplier applied to
                the best-so-far solution each iteration (elitist Ant
                System reinforcement).
            use_heuristic (bool): Whether to spend `dim` one-time
                obj_func calls building a per-client heuristic (eta).
                If False, eta is uniform and only pheromone guides
                search.

        Returns:
            tuple: (best_solution, best_fitness, convergence_curve)
                best_solution is a length-dim binary array (1 = client
                selected), matching the same encoding used by BGA,
                BPSO, BSA, BGWO, and BDE.
        """
        dim = self.params["dim"]
        n_ants = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]

        # --- Resolve subset size (number of clients selected per round) ---
        if n_select is None:
            n_select = self.params.get("n_select", None)
        if n_select is None:
            sel_ratio = self.params.get("selection_ratio", 0.3)
            n_select = round(sel_ratio * dim)
        n_select = int(min(max(n_select, 1), dim))

        # --- Pheromone table: one trail per client (node-based, not edge-based) ---
        tau = np.ones(dim)

        # --- One-time heuristic desirability per client (optional) ---
        if use_heuristic:
            eta = np.zeros(dim)
            for i in range(dim):
                probe = np.zeros(dim, dtype=int)
                probe[i] = 1
                f = obj_func(probe)
                eta[i] = 1.0 / (f + 1e-9)  # lower fitness (minimization) -> more desirable
        else:
            eta = np.ones(dim)

        best_solution = None
        best_fitness = float("inf")
        convergence_curve = np.zeros(max_iter)

        # Create progress bar
        self._create_progress_bar(max_iter, "ACO")

        for t in range(max_iter):
            ants_solutions = np.zeros((n_ants, dim), dtype=int)
            ants_fitness = np.zeros(n_ants)

            for a in range(n_ants):
                available = list(range(dim))
                chosen = []

                # Sequential construction: pick n_select clients without replacement
                for _ in range(n_select):
                    weights = np.array([(tau[c] ** alpha) * (eta[c] ** beta) for c in available])
                    total = weights.sum()
                    if total <= 0 or not np.isfinite(total):
                        probs = np.ones(len(available)) / len(available)
                    else:
                        probs = weights / total
                    pick = np.random.choice(available, p=probs)
                    chosen.append(pick)
                    available.remove(pick)

                solution = np.zeros(dim, dtype=int)
                solution[chosen] = 1
                fit = obj_func(solution)

                ants_solutions[a, :] = solution
                ants_fitness[a] = fit

                if fit < best_fitness:
                    best_fitness = fit
                    best_solution = solution.copy()

            # --- Pheromone evaporation ---
            tau *= (1.0 - rho)
            tau = np.maximum(tau, 1e-6)  # keep trails from collapsing to zero

            # --- Pheromone deposit: every ant reinforces the clients it selected ---
            for a in range(n_ants):
                deposit = Q / (ants_fitness[a] + 1e-9)
                idx = np.where(ants_solutions[a] == 1)[0]
                tau[idx] += deposit

            # --- Elitist reinforcement from best-so-far solution ---
            if best_solution is not None:
                elite_deposit = elite_weight * Q / (best_fitness + 1e-9)
                idx = np.where(best_solution == 1)[0]
                tau[idx] += elite_deposit

            convergence_curve[t] = best_fitness

            # Update progress bar
            self._update_progress(t, max_iter, best_fitness, "ACO")

        # Close progress bar
        self._close_progress_bar()

        return best_solution, best_fitness, convergence_curve
    
    def RSELECT(self, selection_ratio=0.3, replacement_rate=0.2):
        """
        Random Selection (RSELECT) optimizer that randomly selects clients to be added
        to the optimizer library.
        
        Parameters:
        -----------
        selection_ratio: float
            Ratio of clients to be selected in each iteration (default: 0.3)
        replacement_rate: float
            Rate at which to replace existing clients with new ones (default: 0.2)
        
        Returns:
        --------
        tuple: (best solution, best fitness value, convergence curve)
        """
        # Extract parameters
        dim = self.params["dim"]
        lb = self.params["lb"]
        ub = self.params["ub"]
        nbr = self.params["agents_nbr"]
        max_iter = self.params["max_iter"]
        obj_func = self.params["obj_func"]
        
        # Initialize client population
        population = np.zeros((nbr, dim))
        fitness = np.zeros(nbr)
        
        # Initialize random clients
        for i in range(nbr):
            population[i,:] = np.array([random.uniform(lb[j], ub[j]) for j in range(dim)])
            fitness[i] = obj_func(population[i,:])
        
        # Initialize optimizer library (initially empty)
        library_size = max(1, int(nbr * selection_ratio))
        library = np.zeros((library_size, dim))
        library_fitness = np.zeros(library_size)
        library_filled = 0  # Track how many slots are filled
        
        # Find initial best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx,:].copy()
        best_fitness = fitness[best_idx]
        
        # Add best solution to library
        library[0,:] = best_solution
        library_fitness[0] = best_fitness
        library_filled = 1
        
        # Track convergence
        convergence_curve = np.zeros(max_iter)
        
        # Create progress bar
        self._create_progress_bar(max_iter, "RSELECT")
        
        # Main optimization loop
        for t in range(max_iter):
            # Randomly select clients to add to library
            selected_indices = random.sample(range(nbr), min(library_size, nbr))
            
            # Check if selected clients are better than existing library members
            for idx in selected_indices:
                # If library isn't full yet, add the client
                if library_filled < library_size:
                    library[library_filled,:] = population[idx,:].copy()
                    library_fitness[library_filled] = fitness[idx]
                    library_filled += 1
                else:
                    # Find worst client in library
                    worst_idx = np.argmax(library_fitness)
                    
                    # If selected client is better than worst client in library, replace it
                    if fitness[idx] < library_fitness[worst_idx]:
                        library[worst_idx,:] = population[idx,:].copy()
                        library_fitness[worst_idx] = fitness[idx]
            
            # Generate new population influenced by library
            for i in range(nbr):
                if random.random() < replacement_rate:
                    # Replace with randomly generated client
                    population[i,:] = np.array([random.uniform(lb[j], ub[j]) for j in range(dim)])
                else:
                    # Select random library member to influence this client
                    if library_filled > 0:  # Make sure library has members
                        lib_idx = random.randint(0, library_filled - 1)
                        
                        # Move client towards library member
                        move_strength = random.uniform(0.1, 0.5)
                        population[i,:] = population[i,:] + move_strength * (library[lib_idx,:] - population[i,:])
                        
                        # Ensure bounds
                        population[i,:] = np.clip(population[i,:], lb, ub)
                
                # Evaluate fitness
                fitness[i] = obj_func(population[i,:])
                
                # Update best solution if needed
                if fitness[i] < best_fitness:
                    best_solution = population[i,:].copy()
                    best_fitness = fitness[i]
                    
                    # Also update library with new best
                    worst_idx = np.argmax(library_fitness)
                    library[worst_idx,:] = best_solution
                    library_fitness[worst_idx] = best_fitness
            
            # Update convergence history
            convergence_curve[t] = best_fitness
            
            # Update progress bar
            self._update_progress(t, max_iter, best_fitness, "RSELECT")
        
        # Close progress bar
        self._close_progress_bar()
        
        return best_solution, best_fitness, convergence_curve