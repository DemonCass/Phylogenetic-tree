import time
import random
import numpy as np
import collections
from threading import Timer
from ortools.sat.python import cp_model


class SequencePhylogeneticConfig:
    def __init__(self, num_leaves=20, sequence_length=200, parsimony_penalty=10,
                 timer_limit=3000, max_solve_time=100000, num_workers=12):
        self.A = parsimony_penalty
        self.num_leaves = num_leaves
        self.sequence_length = sequence_length
        self.timer_limit = timer_limit
        self.max_solve_time = max_solve_time
        self.num_workers = num_workers

        self.reference_node = {0}
        self.internal_nodes = set(range(num_leaves - 2))
        self.all_nodes = set(range(2 * num_leaves - 2))
        self.leaf_nodes = self.all_nodes - self.internal_nodes

        self.leaf_sequences = {}
        for leaf in self.leaf_nodes:
            self.leaf_sequences[leaf] = [random.randint(0, 4) for _ in range(sequence_length)]

        self.substitution_matrix = np.array([
            [0, 2, 1, 2, 4],
            [2, 0, 2, 1, 4],
            [1, 2, 0, 2, 4],
            [2, 1, 2, 0, 4],
            [4, 4, 4, 4, 0]
        ])


class SequenceSolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, Y_vars, X_vars, config):
        super(SequenceSolutionCallback, self).__init__()
        self.__solution_count = 0
        self.__start_time = time.time()
        self._timer = None

        self.Y_vars = Y_vars
        self.X_vars = X_vars
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

    def _calculate_mutations(self):
        tree_structure = {}
        for u in self.config.all_nodes - self.config.reference_node:
            for v in self.config.internal_nodes:
                if u > v and self.Value(self.Y_vars[(u, v)]) == 1:
                    tree_structure[u] = v

        node_sequences = collections.defaultdict(list)
        for u in self.config.all_nodes:
            for s in range(self.config.sequence_length):
                for p in range(5):
                    if self.Value(self.X_vars[(u, s, p)]) == 1:
                        node_sequences[u].append(p)

        total_mutations = 0
        for node, parent in tree_structure.items():
            node_seq = node_sequences[node]
            parent_seq = node_sequences[parent]

            edge_mutations = 0
            for i in range(len(node_seq)):
                if (node_seq[i] != parent_seq[i] and
                        node_seq[i] != 4 and parent_seq[i] != 4):
                    edge_mutations += 1

            total_mutations += edge_mutations

        return total_mutations

    def StopSearch(self):
        print(f"{self.config.timer_limit} seconds without improvement. Stopping search.")
        super().StopSearch()

    def total_num_solutions(self):
        return self.__solution_count


class SequencePhylogeneticSolver:
    def __init__(self, config):
        self.config = config
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        self.Y_vars = {}
        self.X_vars = {}

        self._create_variables()
        self._add_constraints()
        self._set_objective()
        self._configure_solver()

    def _create_variables(self):
        for u in self.config.all_nodes - self.config.reference_node:
            for v in self.config.internal_nodes:
                if u > v:
                    self.Y_vars[(u, v)] = self.model.NewBoolVar(f'Y_{u}_{v}')

        for u in self.config.all_nodes:
            for s in range(self.config.sequence_length):
                for p in range(5):
                    self.X_vars[(u, s, p)] = self.model.NewBoolVar(f'X_{u}_{s}_{p}')

    def _add_constraints(self):
        for leaf in self.config.leaf_nodes:
            for s in range(self.config.sequence_length):
                nucleotide = self.config.leaf_sequences[leaf][s]
                for p in range(5):
                    if p == nucleotide:
                        self.model.Add(self.X_vars[(leaf, s, p)] == 1)
                    else:
                        self.model.Add(self.X_vars[(leaf, s, p)] == 0)

        for u in self.config.all_nodes - self.config.reference_node:
            self.model.Add(sum(self.Y_vars[(u, v)] for v in self.config.internal_nodes if v < u) == 1)

        for v in self.config.internal_nodes - self.config.reference_node:
            self.model.Add(
                sum(self.Y_vars[(u, v)] for u in self.config.all_nodes - self.config.reference_node if v < u) == 2)

        for u in self.config.internal_nodes:
            for s in range(self.config.sequence_length):
                self.model.Add(sum(self.X_vars[(u, s, p)] for p in range(5)) == 1)

    def _set_objective(self):
        objective_terms = []

        for u in self.config.all_nodes - self.config.internal_nodes:
            for v in self.config.internal_nodes:
                for s in range(self.config.sequence_length):
                    for p in range(5):
                        for q in range(5):
                            term = self.model.NewIntVar(0, 100, f'term1_{u}_{v}_{s}_{p}_{q}')
                            self.model.AddMultiplicationEquality(
                                term, [self.Y_vars[(u, v)], self.X_vars[(u, s, p)], self.X_vars[(v, s, q)]]
                            )
                            cost = self.config.substitution_matrix[p, q]
                            objective_terms.append(cost * term)

        for u in self.config.internal_nodes - self.config.reference_node:
            for v in self.config.internal_nodes:
                if v < u:
                    for s in range(self.config.sequence_length):
                        for p in range(5):
                            for q in range(5):
                                term = self.model.NewIntVar(0, 100, f'term2_{u}_{v}_{s}_{p}_{q}')
                                self.model.AddMultiplicationEquality(
                                    term, [self.Y_vars[(u, v)], self.X_vars[(u, s, p)], self.X_vars[(v, s, q)]]
                                )
                                cost = self.config.substitution_matrix[p, q]
                                objective_terms.append(cost * term)

        self.model.Minimize(sum(objective_terms))

    def _configure_solver(self):
        self.solver.parameters.max_time_in_seconds = self.config.max_solve_time
        self.solver.parameters.num_search_workers = self.config.num_workers
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.cp_model_presolve = True
        self.solver.parameters.cp_model_probing_level = 2
        self.solver.parameters.linearization_level = 2
        self.solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

    def solve(self):
        start_time = time.time()

        solution_callback = SequenceSolutionCallback(
            Y_vars=self.Y_vars,
            X_vars=self.X_vars,
            config=self.config
        )

        status = self.solver.SolveWithSolutionCallback(self.model, solution_callback)

        return self._process_results(status, start_time, solution_callback)

    def _process_results(self, status, start_time, solution_callback):
        end_time = time.time()
        solve_time = end_time - start_time

        if status == cp_model.OPTIMAL:
            print(f"Optimal solution found, objective value = {self.solver.ObjectiveValue()}")
        elif status == cp_model.FEASIBLE:
            print(f"Feasible solution found, objective value = {self.solver.ObjectiveValue()}")
        else:
            print("No feasible solution found.")
            return {"status": "INFEASIBLE", "solve_time": solve_time}

        tree_structure, node_sequences = self._extract_solution()

        total_mutations, edge_mutations = self._calculate_final_mutations(tree_structure, node_sequences)

        self._print_results(tree_structure, node_sequences, total_mutations, edge_mutations, solve_time)

        best_solution = self.solver.ObjectiveValue()
        best_bound = self.solver.BestObjectiveBound()
        gap = (best_solution - best_bound) / float(abs(best_solution)) if best_solution != 0 else 0

        return {
            "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
            "objective_value": best_solution,
            "best_bound": best_bound,
            "gap": gap,
            "total_mutations": total_mutations,
            "edge_mutations": edge_mutations,
            "solve_time": solve_time,
            "tree_structure": tree_structure,
            "node_sequences": node_sequences
        }

    def _extract_solution(self):
        tree_structure = {}
        for u in self.config.all_nodes - self.config.reference_node:
            for v in self.config.internal_nodes:
                if u > v and self.solver.Value(self.Y_vars[(u, v)]) == 1:
                    tree_structure[u] = v

        node_sequences = collections.defaultdict(list)
        for u in self.config.all_nodes:
            for s in range(self.config.sequence_length):
                for p in range(5):
                    if self.solver.Value(self.X_vars[(u, s, p)]) == 1:
                        node_sequences[u].append(p)

        return tree_structure, node_sequences

    def _calculate_final_mutations(self, tree_structure, node_sequences):
        total_mutations = 0
        edge_mutations = {}

        for node, parent in tree_structure.items():
            node_seq = node_sequences[node]
            parent_seq = node_sequences[parent]

            edge_mutation_count = 0
            for i in range(len(node_seq)):
                if (node_seq[i] != parent_seq[i] and
                        node_seq[i] != 4 and parent_seq[i] != 4):
                    edge_mutation_count += 1

            edge_mutations[(parent, node)] = edge_mutation_count
            total_mutations += edge_mutation_count

        return total_mutations, edge_mutations

    def _print_results(self, tree_structure, node_sequences, total_mutations, edge_mutations, solve_time):
        print("\n" + "=" * 80)
        print("SEQUENCE-BASED PHYLOGENETIC TREE RECONSTRUCTION RESULTS")
        print("=" * 80)

        print(f"Total mutations in phylogenetic tree: {total_mutations}")
        print(f"Solving time: {solve_time:.2f} seconds")

        print("\nMutation details by edge:")
        nucleotide_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'Gap'}

        for (parent, node), mutations in edge_mutations.items():
            print(f"  Edge {parent} -> {node}: {mutations} mutations")

            if mutations > 0:
                parent_seq = node_sequences[parent]
                node_seq = node_sequences[node]
                differences = []

                for i in range(min(len(parent_seq), 10)):  # Show first 10 positions
                    if (parent_seq[i] != node_seq[i] and
                            parent_seq[i] != 4 and node_seq[i] != 4):
                        differences.append(f"pos{i}: {nucleotide_map[parent_seq[i]]}->{nucleotide_map[node_seq[i]]}")

                if differences:
                    print(f"    Example mutations: {', '.join(differences[:5])}")
                    if len(differences) > 5:
                        print(f"    ... and {len(differences) - 5} more")

        print(f"\nSequence statistics:")
        print(f"- Number of sequences: {len(self.config.leaf_nodes)}")
        print(f"- Sequence length: {self.config.sequence_length}")
        print(f"- Total possible positions: {len(self.config.leaf_nodes) * self.config.sequence_length}")


def main():
    config = SequencePhylogeneticConfig(
        num_leaves=20,
        sequence_length=200,
        parsimony_penalty=10,
        timer_limit=3000,
        max_solve_time=100000,
        num_workers=12
    )

    print("SEQUENCE-BASED PHYLOGENETIC TREE RECONSTRUCTION USING OR-TOOLS CP-SAT")
    print("=" * 80)
    print(f"Number of leaf nodes: {config.num_leaves}")
    print(f"Sequence length: {config.sequence_length}")
    print(f"Nucleotide mapping: 0=A, 1=T, 2=G, 3=C, 4=Gap")
    print(f"Number of search workers: {config.num_workers}")
    print("=" * 80)

    print("\nSample leaf sequences (first 5 sequences, first 20 positions):")
    nucleotide_map = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: '-'}
    sample_leaves = list(config.leaf_nodes)[:5]

    for leaf in sample_leaves:
        seq_str = ''.join(nucleotide_map[nuc] for nuc in config.leaf_sequences[leaf][:20])
        print(f"Leaf {leaf}: {seq_str}...")

    print("=" * 80)

    solver = SequencePhylogeneticSolver(config)
    results = solver.solve()

    if results["status"] != "INFEASIBLE":
        print(f"\nFinal Statistics:")
        print(f"- Status: {results['status']}")
        print(f"- Objective value: {results['objective_value']}")
        print(f"- Best bound: {results['best_bound']}")
        print(f"- Optimality gap: {results['gap']:.6f}")
        print(f"- Total mutations: {results['total_mutations']}")
        print(f"- Solve time: {results['solve_time']:.2f}s")
        print(f"- Average mutations per edge: {results['total_mutations'] / len(results['edge_mutations']):.2f}")


if __name__ == '__main__':
    main()
