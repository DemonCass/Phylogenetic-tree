import numpy as np
import sympy
import time
from scipy.optimize import minimize
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import random


class PhylogeneticTreeReconstruction:
    def __init__(self, num_leaves= , penalty_weight= , random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)

        self.penalty_weight = penalty_weight
        self.num_leaves = num_leaves
        self.datum_node = {0}
        self.interior_node = set(range(num_leaves - 2))
        self.all_node = set(range(2 * num_leaves - 2))
        self.leaves_nuc = [random.randint(0, 4) for _ in range(num_leaves)]

        self.cost_matrix = np.array([
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0]
        ])

        self.y_size = (2 * num_leaves - 3) * (num_leaves - 2) - (num_leaves - 2) * (num_leaves - 3) // 2
        self.z_size = (num_leaves - 2) * 5

        self.hami_symbols = self._get_hamiltonian_symbols()
        self.nqubits = len(self.hami_symbols)
        self.hami_expr = self._get_simplified_hamiltonian()
        self.hamiltonian = self._build_pennylane_hamiltonian()

    def _get_hamiltonian_symbols(self):
        symbols = []
        for v in self.all_node - self.datum_node:
            for i in self.interior_node:
                if v > i:
                    symbols.append(sympy.symbols(f"Zy{v}{i}"))
        for i in self.interior_node:
            for p in range(5):
                symbols.append(sympy.symbols(f"Zz{i}{p}"))
        return symbols

    def _y_variable(self, v, i):
        return (1 - sympy.symbols(f"Zy{v}{i}")) / 2

    def _z_variable(self, i, p):
        return (1 - sympy.symbols(f"Zz{i}{p}")) / 2

    def _get_score_expression(self):
        score_expr = 0

        for v in self.all_node - self.interior_node:
            for i in self.interior_node:
                for p in range(5):
                    score_expr += (self.cost_matrix[self.leaves_nuc[v - len(self.interior_node)], p] *
                                   self._y_variable(v, i) * self._z_variable(i, p))

        for v in self.interior_node - self.datum_node:
            for i in self.interior_node:
                if i < v:
                    for p in range(5):
                        for q in range(5):
                            score_expr += (self.cost_matrix[p, q] *
                                           self._y_variable(v, i) *
                                           self._z_variable(v, p) *
                                           self._z_variable(i, q))
        return score_expr

    def _get_penalty_expression(self):
        penalty_expr = 0

        for v in self.all_node - self.datum_node:
            constraint_sum = sum(self._y_variable(v, i) for i in self.interior_node if i < v)
            penalty_expr += (1 - constraint_sum) ** 2

        for i in self.interior_node - self.datum_node:
            constraint_sum = sum(self._y_variable(v, i) for v in self.all_node - self.datum_node if i < v)
            penalty_expr += (2 - constraint_sum) ** 2

        for i in self.interior_node:
            constraint_sum = sum(self._z_variable(i, p) for p in range(5))
            penalty_expr += (1 - constraint_sum) ** 2

        return self.penalty_weight * penalty_expr

    def _get_hamiltonian_expression(self):
        return self._get_score_expression() + self._get_penalty_expression()

    def _get_simplified_hamiltonian(self):
        expr = sympy.expand(self._get_hamiltonian_expression())
        for symbol in self.hami_symbols:
            expr = expr.subs(symbol ** 2, 1)
        return expr

    def _build_pennylane_hamiltonian(self):
        expr_dict = self.hami_expr.as_coefficients_dict()
        coeffs = []
        observables = []

        for qubit in range(self.nqubits):
            coeff = expr_dict.get(self.hami_symbols[qubit], 0)
            if coeff != 0:
                coeffs.append(float(coeff))
                obs = qml.Identity(wires=list(range(self.nqubits)))
                obs @= qml.PauliZ(wires=qubit)
                observables.append(obs)

        for qubit_1 in range(self.nqubits):
            for qubit_2 in range(qubit_1 + 1, self.nqubits):
                coeff = expr_dict.get(self.hami_symbols[qubit_1] * self.hami_symbols[qubit_2], 0)
                if coeff != 0:
                    coeffs.append(float(coeff))
                    obs = qml.Identity(wires=list(range(self.nqubits)))
                    obs @= qml.PauliZ(wires=qubit_1)
                    obs @= qml.PauliZ(wires=qubit_2)
                    observables.append(obs)

        for qubit_1 in range(self.nqubits):
            for qubit_2 in range(qubit_1 + 1, self.nqubits):
                for qubit_3 in range(qubit_2 + 1, self.nqubits):
                    coeff = expr_dict.get(
                        self.hami_symbols[qubit_1] * self.hami_symbols[qubit_2] * self.hami_symbols[qubit_3], 0)
                    if coeff != 0:
                        coeffs.append(float(coeff))
                        obs = qml.Identity(wires=list(range(self.nqubits)))
                        obs @= qml.PauliZ(wires=qubit_1)
                        obs @= qml.PauliZ(wires=qubit_2)
                        obs @= qml.PauliZ(wires=qubit_3)
                        observables.append(obs)

        return qml.Hamiltonian(coeffs, observables)

    def get_exact_solution(self):
        eigenvals = qml.eigvals(self.hamiltonian)
        ground_state_energy = eigenvals[0]
        return ground_state_energy

    def evaluate_bitstring(self, bitstring):
        execute_expr = self.hami_expr

        if isinstance(bitstring, np.ndarray):
            psi = 1 - 2 * bitstring
        elif isinstance(bitstring, str):
            psi = 1 - 2 * np.array([int(digit) for digit in bitstring])
        else:
            raise ValueError("Bitstring must be numpy array or string")

        for i, symbol in enumerate(self.hami_symbols):
            execute_expr = execute_expr.subs(symbol, psi[i])

        return float(execute_expr)


class QAOASolver:
    def __init__(self, problem_instance, num_layers=1):
        self.problem = problem_instance
        self.num_layers = num_layers
        self.device = qml.device("default.qubit", wires=problem_instance.nqubits)
        self.optimization_history = []

    def _qaoa_circuit(self, params):
        for i in range(self.problem.nqubits):
            qml.Hadamard(wires=i)

        for layer in range(self.num_layers):
            qml.evolve(self.problem.hamiltonian, params[layer])
            for qubit in range(self.problem.nqubits):
                qml.RX(2 * params[self.num_layers + layer], wires=qubit)

        return qml.expval(self.problem.hamiltonian)

    def optimize(self, max_iterations=1000):
        cost_function = qml.QNode(self._qaoa_circuit, self.device)

        init_params = np.concatenate([
            np.linspace(0, np.pi, self.num_layers),
            np.linspace(0, 2 * np.pi, self.num_layers)
        ])

        def objective(params):
            value = cost_function(params)
            self.optimization_history.append(value)
            return value

        result = minimize(objective, init_params, method='L-BFGS-B',
                          options={'maxiter': max_iterations})

        return result.x, result.fun

    def sample_solution(self, optimal_params, num_shots=8192):
        sample_device = qml.device("default.qubit", wires=self.problem.nqubits, shots=num_shots)

        @qml.qnode(sample_device)
        def sampling_circuit(params):
            for i in range(self.problem.nqubits):
                qml.Hadamard(wires=i)

            for layer in range(self.num_layers):
                qml.evolve(self.problem.hamiltonian, params[layer])
                for qubit in range(self.problem.nqubits):
                    qml.RX(2 * params[self.num_layers + layer], wires=qubit)

            return [qml.sample(wires=i) for i in range(self.problem.nqubits)]

        samples = sampling_circuit(optimal_params)
        samples = np.array(samples).T

        bitstrings = [''.join(map(str, sample)) for sample in samples]
        counts = Counter(bitstrings)

        return counts.most_common(10)


class VQESolver:
    def __init__(self, problem_instance, num_layers=1):
        self.problem = problem_instance
        self.num_layers = num_layers
        self.device = qml.device("default.qubit", wires=problem_instance.nqubits)
        self.optimization_history = []

    def _ansatz(self, params, wires):
        for i in wires:
            qml.Hadamard(wires=i)

        for layer in range(self.num_layers):
            for j, wire in enumerate(wires):
                qml.RY(params[layer * 2 * len(wires) + j], wires=wire)
                qml.RZ(params[layer * 2 * len(wires) + len(wires) + j], wires=wire)

            for j in range(len(wires) - 1):
                qml.CZ(wires=[wires[j], wires[j + 1]])
            if len(wires) > 2:
                qml.CZ(wires=[wires[-1], wires[0]])

    def _vqe_circuit(self, params):
        self._ansatz(params, range(self.problem.nqubits))
        return qml.expval(self.problem.hamiltonian)

    def optimize(self, max_iterations=1000):
        cost_function = qml.QNode(self._vqe_circuit, self.device)

        num_params = self.num_layers * 2 * self.problem.nqubits
        init_params = np.random.uniform(0, 2 * np.pi, num_params)

        def objective(params):
            value = cost_function(params)
            self.optimization_history.append(value)
            return value

        result = minimize(objective, init_params, method='COBYLA',
                          options={'maxiter': max_iterations})

        return result.x, result.fun

    def get_quantum_state(self, optimal_params):
        @qml.qnode(self.device)
        def state_circuit(params):
            self._ansatz(params, range(self.problem.nqubits))
            return qml.state()

        return state_circuit(optimal_params)

    def sample_solution(self, optimal_params, num_shots=1000):
        sample_device = qml.device("default.qubit", wires=self.problem.nqubits, shots=num_shots)

        @qml.qnode(sample_device)
        def sampling_circuit(params):
            self._ansatz(params, range(self.problem.nqubits))
            return [qml.sample(wires=i) for i in range(self.problem.nqubits)]

        samples = sampling_circuit(optimal_params)
        samples = np.array(samples).T

        bitstrings = [''.join(map(str, sample)) for sample in samples]
        counts = Counter(bitstrings)

        return counts.most_common(10)


def run_comparison_study():
    problem = PhylogeneticTreeReconstruction(num_leaves=3, penalty_weight=10)

    print(f"Number of qubits: {problem.nqubits}")
    print(f"Leaf nucleotides: {problem.leaves_nuc}")

    exact_energy = problem.get_exact_solution()
    print(f"Exact ground state energy: {exact_energy:.6f}")

    start_time = time.time()

    qaoa_solver = QAOASolver(problem, num_layers=30)
    qaoa_params, qaoa_energy = qaoa_solver.optimize()
    qaoa_samples = qaoa_solver.sample_solution(qaoa_params)

    print(f"QAOA energy: {qaoa_energy:.6f}")
    print("QAOA top solutions:")
    for bitstring, count in qaoa_samples[:5]:
        energy = problem.evaluate_bitstring(bitstring)
        print(f"  {bitstring}: count={count}, energy={energy:.6f}")

    vqe_solver = VQESolver(problem, num_layers=1)
    vqe_params, vqe_energy = vqe_solver.optimize()
    vqe_samples = vqe_solver.sample_solution(vqe_params)

    print(f"VQE energy: {vqe_energy:.6f}")
    print("VQE top solutions:")
    for bitstring, count in vqe_samples[:5]:
        energy = problem.evaluate_bitstring(bitstring)
        print(f"  {bitstring}: count={count}, energy={energy:.6f}")

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

    return {
        'exact_energy': exact_energy,
        'qaoa_energy': qaoa_energy,
        'vqe_energy': vqe_energy,
        'qaoa_history': qaoa_solver.optimization_history,
        'vqe_history': vqe_solver.optimization_history
    }


if __name__ == "__main__":
    results = run_comparison_study()
