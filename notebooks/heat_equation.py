import matplotlib.pyplot as plt
import numpy as np
import time

from ml_control.greedy_algorithm import greedy
from ml_control.problem_definitions.heat_equation import create_heat_equation_problem_complex
from ml_control.reduced_model import ReducedModel
from ml_control.machine_learning_models.kernel_reduced_model import KernelReducedModel
from ml_control.systems import solve_optimal_control_problem, get_control_from_final_time_adjoint
from ml_control.visualization import plot_final_time_adjoints, plot_controls


system_dimension = 50
T, nt, N, h, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, \
    R_chol, M, parameter_space = create_heat_equation_problem_complex(system_dimension)
spatial_norm = lambda x: np.linalg.norm(h * x)
temporal_norm = lambda u: np.linalg.norm(u * (T / nt))

k_train = 8
training_parameters = np.array(np.meshgrid(np.linspace(*parameter_space[0], k_train),
                                           np.linspace(*parameter_space[1], k_train))).T.reshape(-1, 2)
tol = 1e-6
max_basis_size = k_train ** 2

selected_indices, reduced_basis, non_orthonormalized_reduced_basis, \
            estimated_errors, training_data = greedy(training_parameters, N, T, nt, parametrized_A, parametrized_B,
                    parametrized_x0, parametrized_xT, R_chol, M, tol=tol, max_basis_size=max_basis_size,
                    return_errors_and_efficiencies=False, spatial_norm=spatial_norm)

rb_rom = ReducedModel(reduced_basis, N, T, nt, parametrized_A, parametrized_B, parametrized_x0, parametrized_xT, R_chol,
                      M, spatial_norm=spatial_norm)

ml_rom = KernelReducedModel(rb_rom, training_data, T, nt, parametrized_A, parametrized_B, parametrized_x0,
                            parametrized_xT, R_chol, M, spatial_norm=spatial_norm)
ml_rom.train()

print()
print()
print("======================= TEST RESULTS =======================")
mu = np.stack([np.random.uniform(np.array(parameter_space)[0,0], np.array(parameter_space)[0,1], 1),
               np.random.uniform(np.array(parameter_space)[1,0], np.array(parameter_space)[1,1], 1)]).T[0]
print(f"Test parameter: mu={mu}")

x0 = parametrized_x0(mu)
xT = parametrized_xT(mu)
A = parametrized_A(mu)
B = parametrized_B(mu)
phiT_init = np.zeros(N)
tic = time.perf_counter()
phi_opt = solve_optimal_control_problem(x0, xT, T, nt, A, B, R_chol, M, phiT_init)
u_opt = get_control_from_final_time_adjoint(phi_opt, T, nt, A, B, R_chol)
time_full = time.perf_counter() - tic

fig, axs = plt.subplots(2)
plot_final_time_adjoints([phi_opt], labels=["Optimal adjoint"], show_plot=False, ax=axs[0])
axs[0].set_title("Final time adjoint")
axs[0].legend()
plot_controls([u_opt], T, labels=["Optimal control"], show_plot=False, ax=axs[1])
axs[1].set_title("Controls")
axs[1].legend()
fig.suptitle(f"Results for parameter mu={mu}")
plt.show()

print()
print("Full model:")
print("===========")
print(f"Runtime: {time_full}")

print()
print("Reduced model:")
print("==============")
tic = time.perf_counter()
u_rb, phi_rb = rb_rom.solve(mu)
time_rb = time.perf_counter() - tic
print(f"Reduced basis size: {len(rb_rom.reduced_basis)}")
print(f"Runtime: {time_rb}")
print(f"Speedup: {time_full / time_rb}")
print(f"Relative error in final time adjoint: {spatial_norm(phi_opt - phi_rb) / spatial_norm(phi_opt)}")
print(f"Error in control: {temporal_norm(u_opt - u_rb)}")

print()
print("ML reduced model:")
print("=================")
tic = time.perf_counter()
u_ml, phi_ml = ml_rom.solve(mu)
time_ml = time.perf_counter() - tic
print(f"Runtime: {time_ml}")
print(f"Speedup: {time_full / time_ml}")
print(f"Relative error in final time adjoint: {spatial_norm(phi_opt - phi_ml) / spatial_norm(phi_opt)}")
print(f"Error in control: {temporal_norm(u_opt - u_ml)}")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.semilogy(np.arange(0, len(estimated_errors)), estimated_errors, 'tab:blue', label='Greedy estimated errors')
ax.plot(np.arange(0, len(estimated_errors)), [tol] * len(estimated_errors), 'tab:red', label='Greedy tolerance')
ax.set_xlim((0., len(selected_indices)))
ax.set_xlabel('greedy step')
ax.set_ylabel('maximum estimated error')
ax.set_xticks(np.arange(0, len(selected_indices)) + 1)
ax.legend()
plt.show()
