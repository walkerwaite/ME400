## Walker’s test (checking separability of uniform random data):

#import data and libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import cvxpy as cp
from tqdm import tqdm

# Define uniform random data
X_robust = np.random.uniform(0, 1, (5000, 784))
X_non_robust = np.random.uniform(0, 1, (5000, 784))
print(np.array_equal(X_robust, X_non_robust))

# Check separability using the convex hull method
def check_sep(X_robust, X_nonrobust):
    A = np.array(X_robust)
    A = np.reshape(A, (A.shape[0], -1)).T

    print(X_robust.shape, X_nonrobust.shape)
    num_robust = X_robust.shape[0]
    one_vec = np.ones(num_robust)

    B = np.array(X_nonrobust)
    B = np.reshape(B, (B.shape[0], -1))

    for y in tqdm(B, desc="Checking separability"):
        lam = cp.Variable(num_robust)
        obj = cp.Minimize(1)  # Dummy objective
        const = [lam >= 0, one_vec @ lam == 1, A @ lam == y]
        prob = cp.Problem(obj, const)
        result = prob.solve(solver=cp.ECOS, verbose=True)  # Use ECOS solver
        
        if result is None:
            print("The point is not within the convex hull of the robust points.")
        else:
            print("The point is within the convex hull of the robust points.")
            random_indices = np.random.choice(num_robust, 5, replace=False)
            print("Random 5 Lambda values:", lam.value[random_indices])
            print("Objective value:", result)

check_sep(X_robust, X_non_robust)
