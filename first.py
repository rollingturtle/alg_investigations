import torch

size_mat = 64
n_matrices = 10
# generate initial bunch of matrices 
matrices = []
for _ in range(n_matrices):
    A = torch.randn(size_mat, size_mat)
    sym_A = (A + A.t()) / 2  # Make symmetric
    matrices.append(sym_A)


# compute eigenvector and eigenvalues
eigenvalues_list = []
eigenvectors_list = []

for matrix in matrices:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues_list.append(eigenvalues)
    eigenvectors_list.append(eigenvectors)


# extend the matrices with a row and col and keep them symmetric and non singular possibly (addin 1 in 4,4)
matrices_extended = []
for mat in matrices:
    n = mat.shape[0]
    mat_ext = torch.zeros(n + 1, n + 1, dtype=mat.dtype)
    mat_ext[:n, :n] = mat
    mat_ext[n, n] = 1
    matrices_extended.append(mat_ext)
matrices_extended

# loop to create the step towards some target matrices and show how eigevalues evolve after the adding and the incremental process
import matplotlib.pyplot as plt
import random
import numpy as np
Nstep = 10

# Generate step vectors with last element zero
random_vectors = []
step_vectors = []
for _ in matrices_extended:
    #vec = torch.empty(size_mat).uniform_(-1, 1)
    vec = torch.empty(size_mat + 1).normal_()
    #vec = torch.cat([vec, torch.tensor([0.0])])
    vec[-1] = vec[-1] - 1 
    random_vectors.append(vec)
    step_vectors.append(vec / Nstep)

# Select 3 random indices
selected_indices = random.sample(range(len(matrices_extended)), 3)

plt.figure(figsize=(12, 8))

for idx, mat_idx in enumerate(selected_indices):
    mat = matrices_extended[mat_idx]
    step_vec = step_vectors[mat_idx]
    eigs_over_steps = []

    perturbed = mat.clone()
    for step in range(Nstep + 1):
        # Compute eigenvalues at this step
        eigvals, _ = torch.linalg.eigh(perturbed)
        eigs_over_steps.append(eigvals.numpy())
        # Increment perturbation for next step (except last)
        if step < Nstep:
            # perturbed[:-1, -1] += step_vec[:-1]
            # perturbed[-1, :-1] += step_vec[:-1]
            perturbed += step_vec

    eigs_over_steps = np.array(eigs_over_steps)
    plt.figure(figsize=(8, 5))
    for eig_idx in range(eigs_over_steps.shape[1]):
        plt.plot(range(Nstep + 1), eigs_over_steps[:, eig_idx], label=f"Eigenvalue {eig_idx+1}", marker='o')
    plt.xlabel("Step")
    plt.ylabel("Eigenvalue")
    plt.title(f"Eigenvalue Evolution for Matrix {mat_idx} Size: {size_mat}x{size_mat} to {size_mat+1}x{size_mat+1}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"mat_{mat_idx}.png")
    plt.close()