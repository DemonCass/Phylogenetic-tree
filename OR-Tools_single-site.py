import time
import random
import numpy as np
from threading import Timer
from ortools.sat.python import cp_model


class PhylogeneticTreeConfig:
    def __init__(self, leaf_nodes=10, parsimony_penalty=10, timer_limit=500, max_solve_time=180):
        self.P = parsimony_penalty
        self.leaf_nodes = leaf_nodes
        self.reference_node = {0}
        self.internal_nodes = set(range(leaf_nodes - 2))
        self.all_nodes = set(range(2 * leaf_nodes - 2))
        self.timer_limit = timer_limit
        self.max_solve_time = max_solve_time
        self.leaves_nucleotides = [random.randint(0, 4) for _ in range(leaf_nodes)]
        self.substitution_matrix = np.array([
            [0, 2, 1, 2, 4],
            [2, 0, 2, 1, 4],
            [1, 2, 0, 2, 4],
            [2, 1, 2, 0, 4],
            [4, 4, 4, 4, 0]
        ])


class SolutionMutationCounter(cp_model.CpSolverSolutionCallback):
    def __init__(self, e_vars, n_vars, config):
        super(SolutionMutationCounter, self).__init__()
        self.__solution_count = 0
        self.__start_time = time.time()
        self._timer = None
        self.e_vars = e_vars
        self.n_vars = n_vars
        self.config = config
        self._reset_timer()

    def _reset_timer(self):
        if self._timer:
            self._timer.cancel()
        self._timer = Timer(self.config.timer_limit, self.StopSearch)
        self._timer.start()

    def on_solution_callback(self):
        current_time = time.time()
        objective = self.ObjectiveValue()
        best_bound = self.BestObjectiveBound()
        self.__solution_count += 1

        mutation_count = self._calculate_mutations()

        print(f"Solution {self.__solution_count}, time = {current_time - self.__start_time:.2f}s, "
              f"objective = {objective}, best_bound = {best_bound}, mutations = {mutation_count}")

        self._reset_timer()

    def _calculate_mutations(self):
        tree_structure = {}
        for u in self.config.all_nodes - self.config.reference_node:
            for v in self.config.internal_nodes:
                if u > v and self.Value(self.e_vars[(u, v)]) == 1:
                    tree_structure[u] = v

        node_nucleotides = {}

        for u in self.config.internal_nodes:
            for p in range(5):
                if self.Value(self.n_vars[(u, p)]) == 1:
                    node_nucleotides[u] = p

        for u in self.config.all_nodes - self.config.internal_nodes:
            node_nucleotides[u] = self.config.leaves_nucleotides[u - len(self.config.internal_nodes)]

        edge_mutations = 0
        for node, parent in tree_structure.items():
            node_nuc = node_nucleotides[node]
            parent_nuc = node_nucleotides[parent]

            if node_nuc != parent_nuc and node_nuc != 4 and parent_nuc != 4:
                edge_mutations += 1

        return edge_mutations

    def StopSearch(self):
        print(f"{self.config.timer_limit} seconds without improvement. Stopping search.")
        super().StopSearch()

    def total_num_solutions(self):
        return self.__solution_count


class PhylogeneticTreeSolver:
    def __init__(self, config):
        self.config = config
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.e_vars = {}
        self.n_vars = {}
        self._create_variables()
        self._add_constraints()
        self._set_objective()

    def _create_variables(self):
        for v in self.config.all_nodes - self.config.reference_node:
            for i in self.config.internal_nodes:
                if v > i:
                    self.e_vars[(v, i)] = self.model.NewBoolVar(f'e_{v}_{i}')

        for i in self.config.internal_nodes:
            for p in range(5):
                self.n_vars[(i, p)] = self.model.NewBoolVar(f'n_{i}_{p}')

    def _add_constraints(self):
        for v in self.config.all_nodes - self.config.reference_node:
            self.model.Add(sum(self.e_vars[(v, i)] for i in self.config.internal_nodes if i < v) == 1)

        for i in self.config.internal_nodes - self.config.reference_node:
            self.model.Add(
                sum(self.e_vars[(v, i)] for v in self.config.all_nodes - self.config.reference_node if i < v) == 2)

        for i in self.config.internal_nodes:
            self.model.Add(sum(self.n_vars[(i, p)] for p in range(5)) == 1)

    def _set_objective(self):
        objective_terms = []

        for v in self.config.all_nodes - self.config.internal_nodes:
            for i in self.config.internal_nodes:
                for p in range(5):
                    w = self.model.NewIntVar(0, 1, f'w_{v}_{i}_{p}')
                    self.model.AddMultiplicationEquality(w, [self.e_vars[(v, i)], self.n_vars[(i, p)]])

                    leaf_idx = v - len(self.config.internal_nodes)
                    cost = self.config.substitution_matrix[self.config.leaves_nucleotides[leaf_idx], p]
                    objective_terms.append(cost * w)

        for v in self.config.internal_nodes - self.config.reference_node:
            for i in self.config.internal_nodes:
                if i < v:
                    for p in range(5):
                        for q in range(5):
                            aux = self.model.NewIntVar(0, 1, f'aux_{v}_{i}_{q}')
                            self.model.AddMultiplicationEquality(aux, [self.e_vars[(v, i)], self.n_vars[(v, q)]])

                            w = self.model.NewIntVar(0, 1, f'w_{v}_{i}_{p}_{q}')
                            self.model.AddMultiplicationEquality(w, [aux, self.n_vars[(i, p)]])

                            cost = self.config.substitution_matrix[p, q]
                            objective_terms.append(cost * w)

        self.model.Minimize(sum(objective_terms))

    def solve(self):
        start_time = time.time()

        solution_counter = SolutionMutationCounter(
            e_vars=self.e_vars,
            n_vars=self.n_vars,
            config=self.config
        )

        self.solver.parameters.max_time_in_seconds = self.config.max_solve_time

        status = self.solver.SolveWithSolutionCallback(self.model, solution_counter)

        return self._process_results(status, start_time, solution_counter)

    def _process_results(self, status, start_time, solution_counter):
        end_time = time.time()
        solve_time = end_time - start_time

        if status == cp_model.OPTIMAL:
            print(f"Optimal solution found, objective value = {self.solver.ObjectiveValue()}")
        elif status == cp_model.FEASIBLE:
            print(f"Feasible solution found, objective value = {self.solver.ObjectiveValue()}")
        else:
            print("No feasible solution found.")
            return {"status": "INFEASIBLE", "solve_time": solve_time}

        tree_structure, node_nucleotides = self._extract_solution()

        mutation_count = self._calculate_final_mutations(tree_structure, node_nucleotides)

        self._print_results(tree_structure, node_nucleotides, mutation_count, solve_time)

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
        tree_structure = {}
        for u in self.config.all_nodes - self.config.reference_node:
            for v in self.config.internal_nodes:
                if u > v and self.solver.Value(self.e_vars[(u, v)]) == 1:
                    tree_structure[u] = v

        node_nucleotides = {}

        for u in self.config.internal_nodes:
            for p in range(5):
                if self.solver.Value(self.n_vars[(u, p)]) == 1:
                    node_nucleotides[u] = p

        for u in self.config.all_nodes - self.config.internal_nodes:
            node_nucleotides[u] = self.config.leaves_nucleotides[u - len(self.config.internal_nodes)]

        return tree_structure, node_nucleotides

    def _calculate_final_mutations(self, tree_structure, node_nucleotides):
        mutation_count = 0

        for node, parent in tree_structure.items():
            node_nuc = node_nucleotides[node]
            parent_nuc = node_nucleotides[parent]

            if node_nuc != parent_nuc and node_nuc != 4 and parent_nuc != 4:
                mutation_count += 1

        return mutation_count

    def _print_results(self, tree_structure, node_nucleotides, mutation_count, solve_time):
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

    solver = PhylogeneticTreeSolver(config)
    results = solver.solve()

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
