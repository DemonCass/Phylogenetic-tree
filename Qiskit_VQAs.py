import random
import sympy
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from qiskit import Aer, transpile
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import StateFn, PauliSumOp
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms import NumPyMinimumEigensolver, QAOA, VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.visualization import plot_histogram

algorithm_globals.massive = True


class PhylogeneticTreeOptimizer:
    def __init__(self, num_leaves=3, penalty_weight=10, random_seed=170):
        self.num_leaves = num_leaves
        self.penalty_weight = penalty_weight
        self.random_seed = random_seed
        self.reference_node = {0}
        self.internal_nodes = set(range(num_leaves - 2))
        self.all_nodes = set(range(2 * num_leaves - 2))
        self.leaves_nuc = [random.randint(0, 4) for _ in range(num_leaves)]
        self.similarity_matrix = np.array([
            [0, 2, 1, 2, 4],
            [2, 0, 2, 1, 4],
            [1, 2, 0, 2, 4],
            [2, 1, 2, 0, 4],
            [4, 4, 4, 4, 0]
        ])
        self.hami_symbols = self._get_hamiltonian_symbols()
        self.nqubits = len(self.hami_symbols)
        self.hamiltonian = self._build_hamiltonian()

    def _get_hamiltonian_symbols(self):
        symbol_container = []
        for v in self.all_nodes - self.reference_node:
            for i in self.internal_nodes:
                if v > i:
                    symbol_container.append(sympy.symbols(f"Zy{v}{i}"))
        for i in self.internal_nodes:
            for p in range(5):
                symbol_container.append(sympy.symbols(f"Zz{i}{p}"))
        return symbol_container

    def _y_variable(self, v, i):
        return (1 - sympy.symbols(f"Zy{v}{i}")) / 2

    def _z_variable(self, i, p):
        return (1 - sympy.symbols(f"Zz{i}{p}")) / 2

    def _get_score_expression(self):
        score_expr = 0
        for v in self.all_nodes - self.internal_nodes:
            for i in self.internal_nodes:
                for p in range(5):
                    score_expr += (self.similarity_matrix[self.leaves_nuc[v - len(self.internal_nodes)], p] *
                                   self._y_variable(v, i) * self._z_variable(i, p))

        for v in self.internal_nodes - self.reference_node:
            for i in self.internal_nodes:
                if i < v:
                    for p in range(5):
                        for q in range(5):
                            score_expr += (self.similarity_matrix[p, q] *
                                           self._y_variable(v, i) *
                                           self._z_variable(v, p) *
                                           self._z_variable(i, q))
        return score_expr

    def _get_penalty_expression(self):
        penalty_expr = 0

        for v in self.all_nodes - self.reference_node:
            _sum = 0
            for i in self.internal_nodes:
                if i < v:
                    _sum += self._y_variable(v, i)
            penalty_expr += (1 - _sum) ** 2

        for i in self.internal_nodes - self.reference_node:
            _sum = 0
            for v in self.all_nodes - self.reference_node:
                if i < v:
                    _sum += self._y_variable(v, i)
            penalty_expr += (2 - _sum) ** 2

        for i in self.internal_nodes:
            _sum = 0
            for p in range(5):
                _sum += self._z_variable(i, p)
            penalty_expr += (1 - _sum) ** 2

        return self.penalty_weight * penalty_expr

    def _simplify_expression(self, expr):
        for symbol in self.hami_symbols:
            expr = expr.subs(symbol ** 2, 1)
        return expr

    def _build_hamiltonian(self):
        hamiltonian_expr = self._get_score_expression() + self._get_penalty_expression()
        simplified_expr = self._simplify_expression(sympy.expand(hamiltonian_expr))
        return self._convert_to_pauli_sum_op(simplified_expr)

    def _convert_to_pauli_sum_op(self, expr):
        expr_dict = expr.as_coefficients_dict()
        pauli_list = []

        for qubit in range(self.nqubits):
            pauli_z = np.zeros(self.nqubits, dtype=bool)
            pauli_x = np.zeros(self.nqubits, dtype=bool)
            pauli_z[qubit] = True
            coeff = expr_dict.get(self.hami_symbols[qubit], 0)
            if coeff != 0:
                pauli_list.append([coeff, Pauli((pauli_z, pauli_x))])

        for qubit_1 in range(self.nqubits):
            for qubit_2 in range(qubit_1 + 1, self.nqubits):
                pauli_z = np.zeros(self.nqubits, dtype=bool)
                pauli_x = np.zeros(self.nqubits, dtype=bool)
                pauli_z[qubit_1] = True
                pauli_z[qubit_2] = True
                coeff = expr_dict.get(self.hami_symbols[qubit_1] * self.hami_symbols[qubit_2], 0)
                if coeff != 0:
                    pauli_list.append([coeff, Pauli((pauli_z, pauli_x))])

        for qubit_1 in range(self.nqubits):
            for qubit_2 in range(qubit_1 + 1, self.nqubits):
                for qubit_3 in range(qubit_2 + 1, self.nqubits):
                    pauli_x = np.zeros(self.nqubits, dtype=bool)
                    pauli_z = np.zeros(self.nqubits, dtype=bool)
                    pauli_z[qubit_1] = True
                    pauli_z[qubit_2] = True
                    pauli_z[qubit_3] = True
                    coeff = expr_dict.get(self.hami_symbols[qubit_1] *
                                          self.hami_symbols[qubit_2] *
                                          self.hami_symbols[qubit_3], 0)
                    if coeff != 0:
                        pauli_list.append([coeff, Pauli((pauli_z, pauli_x))])

        pauli_list = [(pauli[1].to_label(), pauli[0]) for pauli in pauli_list]
        return PauliSumOp.from_list(pauli_list)

    def compute_reference_value(self):
        npme = NumPyMinimumEigensolver()
        result = npme.compute_minimum_eigenvalue(operator=self.hamiltonian)
        ref_value = result.eigenvalue.real
        print(f'Reference value: {ref_value:.5f}')
        return ref_value

    def solve_with_qaoa(self, reps=1, plot_convergence=True):
        counts = []
        values = []

        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)

        algorithm_globals.random_seed = self.random_seed
        backend = Aer.get_backend('statevector_simulator')
        qi = QuantumInstance(
            backend=backend,
            seed_simulator=self.random_seed,
            seed_transpiler=self.random_seed,
        )

        optimizer = COBYLA()
        qaoa = QAOA(
            reps=reps,
            optimizer=optimizer,
            callback=store_intermediate_result,
            quantum_instance=qi
        )

        result = qaoa.compute_minimum_eigenvalue(self.hamiltonian)
        print(f'QAOA value: {result.eigenvalue.real:.5f}')

        if plot_convergence:
            plt.figure(figsize=(10, 4))
            plt.plot(counts, values)
            plt.xlabel('Eval count')
            plt.ylabel('Energy')
            plt.axhline(y=-121, color='r', linestyle='--', label='Reference Energy')
            plt.legend()
            plt.show()

        final_params = result.optimal_point
        final_circuit = qaoa.ansatz.bind_parameters(final_params)
        print(f"QAOA circuit depth: {final_circuit.depth()}")

        self._analyze_quantum_state(final_circuit)
        final_circuit.measure_all()

        backend = Aer.get_backend('statevector_simulator')
        transpiled_circuit = transpile(final_circuit, backend, optimization_level=3)
        job = backend.run(transpiled_circuit, shots=1024)
        final_counts = job.result().get_counts()

        print("Final bitstring distribution:", final_counts)
        plot_histogram(final_counts).show()

        return self._sample_most_likely(result.eigenstate)

    def solve_with_vqe(self, reps=1, plot_convergence=True):
        counts = []
        values = []

        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)

        algorithm_globals.random_seed = self.random_seed
        backend = Aer.get_backend('qasm_simulator')
        qi = QuantumInstance(
            backend=backend,
            seed_simulator=self.random_seed,
            seed_transpiler=self.random_seed
        )

        ansatz = TwoLocal(
            rotation_blocks='ry',
            entanglement_blocks='cz',
            reps=reps
        )
        optimizer = COBYLA()

        vqe = VQE(
            ansatz,
            optimizer=optimizer,
            callback=store_intermediate_result,
            quantum_instance=qi
        )

        result = vqe.compute_minimum_eigenvalue(self.hamiltonian)
        print(f'VQE value: {result.eigenvalue.real:.5f}')

        if plot_convergence:
            plt.figure(figsize=(10, 4))
            plt.plot(counts, values, label='VQE Energy')
            plt.xlabel('Evaluation Count')
            plt.ylabel('Energy')
            ref_value = self.compute_reference_value()
            plt.axhline(y=ref_value, color='r', linestyle='--', label='Reference Energy')
            plt.legend()
            plt.show()

        final_params = result.optimal_point
        final_circuit = vqe.ansatz.bind_parameters(final_params)
        print(f"VQE circuit depth: {final_circuit.depth()}")

        final_circuit.measure_all()
        backend = Aer.get_backend('statevector_simulator')
        transpiled_circuit = transpile(final_circuit, backend)
        job = backend.run(transpiled_circuit, shots=1024)
        final_counts = job.result().get_counts()

        print("Final bitstring distribution:", final_counts)
        plot_histogram(final_counts).show()

        return self._sample_most_likely(result.eigenstate)

    def _analyze_quantum_state(self, circuit):
        backend_sv = Aer.get_backend('statevector_simulator')
        job_sv = backend_sv.run(transpile(circuit, backend_sv))
        statevector = job_sv.result().get_statevector()

        probabilities = np.abs(statevector.data) ** 2
        top_indices = np.argsort(-probabilities)[:10]

        print("\nTop 10 highest amplitude basis states:")
        for idx in top_indices:
            binary = format(idx, f'0{self.nqubits}b')[::-1]
            print(f"State |{binary}⟩: Amplitude = {statevector.data[idx]}, "
                  f"Probability = {probabilities[idx]:.6f}")

    def _sample_most_likely(self, state_vector):
        if isinstance(state_vector, (OrderedDict, dict)):
            binary_string = sorted(state_vector.items(), key=lambda kv: kv[1])[-1][0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        elif isinstance(state_vector, StateFn):
            binary_string = list(state_vector.sample().keys())[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        else:
            n = int(np.log2(state_vector.shape[0]))
            k = np.argmax(np.abs(state_vector))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x

    def evaluate_bitstring(self, bit_string):
        hamiltonian_expr = self._simplify_expression(
            sympy.expand(self._get_score_expression() + self._get_penalty_expression())
        )
        return self._evaluate_expression(bit_string, hamiltonian_expr)

    def _evaluate_expression(self, bit_string, expression):
        execute_expr = expression

        if len(bit_string) != len(self.hami_symbols):
            raise ValueError('Bitstring length does not match number of qubits.')

        if isinstance(bit_string, np.ndarray):
            psi = 1 - 2 * bit_string
        elif isinstance(bit_string, str):
            psi = 1 - 2 * np.array([int(digit) for digit in bit_string])
        else:
            raise ValueError('Unexpected type of bit_string.')

        for ii in range(len(psi)):
            execute_expr = execute_expr.subs(self.hami_symbols[ii], psi[ii])

        return execute_expr

    def compute_expectation_value(self, counts):
        avg = 0
        sum_count = 0
        for bitstring, count in counts.items():
            obj = self.evaluate_bitstring(bitstring)
            avg += obj * count
            sum_count += count
        return avg / sum_count


def main():
    optimizer = PhylogeneticTreeOptimizer(num_leaves=3, penalty_weight=10)

    print("Leaf nucleotide types:", optimizer.leaves_nuc)
    print(f"Number of qubits: {optimizer.nqubits}")

    start_time = time.time()

    ref_value = optimizer.compute_reference_value()
    qaoa_result = optimizer.solve_with_qaoa(reps=50)
    vqe_result = optimizer.solve_with_vqe(reps=1)

    end_time = time.time()

    print(f"QAOA result: {qaoa_result}")
    print(f"VQE result: {vqe_result}")
    print(f"Runtime: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()