"""
Phylogenetic Tree Reconstruction using OR-Tools CP-SAT Solver

This module implements a branch-based phylogenetic tree reconstruction algorithm
using Google's OR-Tools CP-SAT solver. The algorithm minimizes the total number
of mutations required to explain the evolutionary relationships between sequences.

Author: [Jiawei Zhang](https://github.com/JiaweiZhang6)
"""

import time
import random
import numpy as np
from threading import Timer
from ortools.sat.python import cp_model


class PhylogeneticTreeConfig:
    """Configuration class for phylogenetic tree reconstruction parameters."""

    def __init__(self, leaf_nodes=10, parsimony_penalty=10, timer_limit=500, max_solve_time=180):
        self.P = parsimony_penalty
        self.leaf_nodes = leaf_nodes
        self.reference_node = {0}
        self.internal_nodes = set(range(leaf_nodes - 2))
        self.all_nodes = set(range(2 * leaf_nodes - 2))
        self.timer_limit = timer_limit  # Timer limit for solution callback
        self.max_solve_time = max_solve_time  # Maximum solver time in seconds

        # Generate random nucleotide sequences for leaf nodes
        # 0: A, 1: T, 2: G, 3: C, 4: Gap/Unknown
        self.leaves_nucleotides = [random.randint(0, 4) for _ in range(leaf_nodes)]

        # Substitution step matrix (cost matrix)
        self.substitution_matrix = np.array([
            [0, 2, 1, 2, 4],
            [2, 0, 2, 1, 4],
            [1, 2, 0, 2, 4],
            [2, 1, 2, 0, 4],
            [4, 4, 4, 4, 0]
        ])


class SolutionMutationCounter(cp_model.CpSolverSolutionCallback):
    """
    Custom solution callback to count mutations and track solution progress.

    This callback monitors the solver's progress, calculates mutation counts
    for each solution, and implements a timeout mechanism to stop search
    after a specified time without improvement.
    """

    def __init__(self, e_vars, n_vars, config):
        """
        Initialize the solution callback.

        Args:
            e_vars: Dictionary of binary variables for tree structure
            n_vars: Dictionary of binary variables for nucleotide assignments
            config: PhylogeneticTreeConfig instance
        """
        super(SolutionMutationCounter, self).__init__()
        self.__solution_count = 0
        self.__start_time = time.time()
        self._timer = None

        # Store variables and configuration
        self.e_vars = e_vars
        self.n_vars = n_vars
        self.config = config

        # Initialize timer
        self._reset_timer()

    def _reset_timer(self):
        """Reset the timeout timer."""
        if self._timer:
            self._timer.cancel()
        self._timer = Timer(self.config.timer_limit, self.StopSearch)
        self._timer.start()

    def on_solution_callback(self):
        """Called when a new solution is found."""
        current_time = time.time()
        objective = self.ObjectiveValue()
        best_bound = self.BestObjectiveBound()
        self.__solution_count += 1

        # Calculate mutation count for current solution
        mutation_count = self._calculate_mutations()

        print(f"Solution {self.__solution_count}, time = {current_time - self.__start_time:.2f}s, "
              f"objective = {objective}, best_bound = {best_bound}, mutations = {mutation_count}")

        # Reset timer for next solution
        self._reset_timer()

    def _calculate_mutations(self):
        """
        Calculate the total number of mutations in the current solution.

        Returns:
            int: Total number of mutations in the phylogenetic tree
        """
        # Build tree structure from e variables
        tree_structure = {}
        for u in self.config.all_nodes - self.config.reference_node:
            for v in self.config.internal_nodes:
                if u > v and self.Value(self.e_vars[(u, v)]) == 1:
                    tree_structure[u] = v

        # Build nucleotide assignments for all nodes
        node_nucleotides = {}

        # Internal nodes from z variables
        for u in self.config.internal_nodes:
            for p in range(5):
                if self.Value(self.n_vars[(u, p)]) == 1:
                    node_nucleotides[u] = p

        # Leaf nodes from configuration
        for u in self.config.all_nodes - self.config.internal_nodes:
            node_nucleotides[u] = self.config.leaves_nucleotides[u - len(self.config.internal_nodes)]

        # Count mutations along edges
        edge_mutations = 0
        for node, parent in tree_structure.items():
            node_nuc = node_nucleotides[node]
            parent_nuc = node_nucleotides[parent]

            # Count mutation if nucleotides differ and neither is a gap
            if node_nuc != parent_nuc and node_nuc != 4 and parent_nuc != 4:
                edge_mutations += 1

        return edge_mutations

    def StopSearch(self):
        """Stop the search due to timeout."""
        print(f"{self.config.timer_limit} seconds without improvement. Stopping search.")
        super().StopSearch()

    def total_num_solutions(self):
        """Return the total number of solutions found."""
        return self.__solution_count


class PhylogeneticTreeSolver:
    """
    Main solver class for phylogenetic tree reconstruction.

    This class implements the constraint programming formulation for
    phylogenetic tree reconstruction using the CP-SAT solver.
    """

    def __init__(self, config):
        """
        Initialize the solver with configuration.

        Args:
            config: PhylogeneticTreeConfig instance
        """
        self.config = config
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        # Decision variables
        self.e_vars = {}  # Tree structure variables
        self.n_vars = {}  # Nucleotide assignment variables

        self._create_variables()
        self._add_constraints()
        self._set_objective()

    def _create_variables(self):
        """Create decision variables for the model."""
        for v in self.config.all_nodes - self.config.reference_node:
            for i in self.config.internal_nodes:
                if v > i:
                    self.e_vars[(v, i)] = self.model.NewBoolVar(f'e_{v}_{i}')

        for i in self.config.internal_nodes:
            for p in range(5):
                self.n_vars[(i, p)] = self.model.NewBoolVar(f'n_{i}_{p}')

    def _add_constraints(self):
        """Add constraints to the model."""
        # Each non-reference node connects to exactly one internal node
        for v in self.config.all_nodes - self.config.reference_node:
            self.model.Add(sum(self.e_vars[(v, i)] for i in self.config.internal_nodes if i < v) == 1)

        # Each non-reference internal node connects to exactly two nodes
        for i in self.config.internal_nodes - self.config.reference_node:
            self.model.Add(
                sum(self.e_vars[(v, i)] for v in self.config.all_nodes - self.config.reference_node if i < v) == 2)

        # Each internal node has exactly one nucleotide
        for i in self.config.internal_nodes:
            self.model.Add(sum(self.n_vars[(i, p)] for p in range(5)) == 1)

    def _set_objective(self):
        """Set the objective function to minimize total evolutionary cost."""
        objective_terms = []

        # Cost for leaf nodes
        for v in self.config.all_nodes - self.config.internal_nodes:
            for i in self.config.internal_nodes:
                for p in range(5):
                    w = self.model.NewIntVar(0, 1, f'w_{v}_{i}_{p}')
                    self.model.AddMultiplicationEquality(w, [self.e_vars[(v, i)], self.n_vars[(i, p)]])

                    # Add cost term
                    leaf_idx = v - len(self.config.internal_nodes)
                    cost = self.config.substitution_matrix[self.config.leaves_nucleotides[leaf_idx], p]
                    objective_terms.append(cost * w)

        # Cost for internal nodes
        for v in self.config.internal_nodes - self.config.reference_node:
            for i in self.config.internal_nodes:
                if i < v:
                    for p in range(5):
                        for q in range(5):
                            aux = self.model.NewIntVar(0, 1, f'aux_{v}_{i}_{q}')
                            self.model.AddMultiplicationEquality(aux, [self.e_vars[(v, i)], self.n_vars[(v, q)]])

                            w = self.model.NewIntVar(0, 1, f'w_{v}_{i}_{p}_{q}')
                            self.model.AddMultiplicationEquality(w, [aux, self.n_vars[(i, p)]])

                            # Add cost term
                            cost = self.config.substitution_matrix[p, q]
                            objective_terms.append(cost * w)

        self.model.Minimize(sum(objective_terms))

    def solve(self):
        """
        Solve the phylogenetic tree reconstruction problem.

        Returns:
            dict: Solution information including tree structure and statistics
        """
        start_time = time.time()

        # Create solution callback
        solution_counter = SolutionMutationCounter(
            e_vars=self.e_vars,
            n_vars=self.n_vars,
            config=self.config
        )

        # Set solver parameters
        self.solver.parameters.max_time_in_seconds = self.config.max_solve_time

        # Solve with callback
        status = self.solver.SolveWithSolutionCallback(self.model, solution_counter)

        # Process results
        return self._process_results(status, start_time, solution_counter)

    def _process_results(self, status, start_time, solution_counter):
        """
        Process and display solver results.

        Args:
            status: Solver status
            start_time: Start time of solving
            solution_counter: Solution callback instance

        Returns:
            dict: Solution information
        """
        end_time = time.time()
        solve_time = end_time - start_time

        # Print status
        if status == cp_model.OPTIMAL:
            print(f"Optimal solution found, objective value = {self.solver.ObjectiveValue()}")
        elif status == cp_model.FEASIBLE:
            print(f"Feasible solution found, objective value = {self.solver.ObjectiveValue()}")
        else:
            print("No feasible solution found.")
            return {"status": "INFEASIBLE", "solve_time": solve_time}

        # Extract solution
        tree_structure, node_nucleotides = self._extract_solution()

        # Calculate final statistics
        mutation_count = self._calculate_final_mutations(tree_structure, node_nucleotides)

        # Print results
        self._print_results(tree_structure, node_nucleotides, mutation_count, solve_time)

        # Calculate optimality gap
        best_solution = self.solver.ObjectiveValue()
        best_bound = self.solver.BestObjectiveBound()
        gap = (best_solution - best_bound) / float(abs(best_solution)) if best_solution != 0 else 0

        return {
            "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
            "objective_value": best_solution,
            "best_bound": best_bound,
            "gap": gap,
            "mutation_count": mutation_count,
            "solve_time": solve_time,
            "tree_structure": tree_structure,
            "node_nucleotides": node_nucleotides
        }

    def _extract_solution(self):
        """Extract tree structure and nucleotide assignments from solution."""
        # Extract tree structure
        tree_structure = {}
        for u in self.config.all_nodes - self.config.reference_node:
            for v in self.config.internal_nodes:
                if u > v and self.solver.Value(self.e_vars[(u, v)]) == 1:
                    tree_structure[u] = v

        # Extract nucleotide assignments
        node_nucleotides = {}

        # Internal nodes
        for u in self.config.internal_nodes:
            for p in range(5):
                if self.solver.Value(self.n_vars[(u, p)]) == 1:
                    node_nucleotides[u] = p

        # Leaf nodes
        for u in self.config.all_nodes - self.config.internal_nodes:
            node_nucleotides[u] = self.config.leaves_nucleotides[u - len(self.config.internal_nodes)]

        return tree_structure, node_nucleotides

    def _calculate_final_mutations(self, tree_structure, node_nucleotides):
        """Calculate total mutations in final solution."""
        mutation_count = 0

        for node, parent in tree_structure.items():
            node_nuc = node_nucleotides[node]
            parent_nuc = node_nucleotides[parent]

            if node_nuc != parent_nuc and node_nuc != 4 and parent_nuc != 4:
                mutation_count += 1

        return mutation_count

    def _print_results(self, tree_structure, node_nucleotides, mutation_count, solve_time):
        """Print detailed results."""
        print("\n" + "=" * 60)
        print("PHYLOGENETIC TREE RECONSTRUCTION RESULTS")
        print("=" * 60)

        print(f"Final node nucleotide mapping: {node_nucleotides}")
        print(f"Total mutations in phylogenetic tree: {mutation_count}")
        print(f"Solving time: {solve_time:.2f} seconds")

        print("\nMutation details:")
        for node, parent in tree_structure.items():
            node_nuc = node_nucleotides[node]
            parent_nuc = node_nucleotides[parent]

            if node_nuc != parent_nuc and node_nuc != 4 and parent_nuc != 4:
                nucleotide_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'Gap'}
                print(f"Edge {parent} -> {node}: {nucleotide_map[parent_nuc]} -> {nucleotide_map[node_nuc]}")


def main():
    """Main function to run the phylogenetic tree reconstruction."""
    # Create configuration
    config = PhylogeneticTreeConfig(
        leaf_nodes=10,
        parsimony_penalty=10,
        timer_limit=500,
        max_solve_time=180
    )

    print("PHYLOGENETIC TREE RECONSTRUCTION USING OR-TOOLS CP-SAT")
    print("=" * 60)
    print(f"Number of leaf nodes: {config.leaf_nodes}")
    print(f"Leaf node nucleotides: {config.leaves_nucleotides}")
    print(f"Nucleotide mapping: 0=A, 1=T, 2=G, 3=C, 4=Gap")
    print("=" * 60)

    # Create and solve
    solver = PhylogeneticTreeSolver(config)
    results = solver.solve()

    # Print final statistics
    if results["status"] != "INFEASIBLE":
        print(f"\nFinal Statistics:")
        print(f"- Status: {results['status']}")
        print(f"- Objective value: {results['objective_value']}")
        print(f"- Best bound: {results['best_bound']}")
        print(f"- Optimality gap: {results['gap']:.6f}")
        print(f"- Total mutations: {results['mutation_count']}")
        print(f"- Solve time: {results['solve_time']:.2f}s")


if __name__ == '__main__':
    main()
