import control
from _PARAMETERS import *

def _get_D(B, P, R):
    D = np.linalg.inv(R + np.transpose(B) @ P @ B) @ np.transpose(B)
    return D

def _get_H(B, D):
    H = B @ D
    return H

def _get_F(A, P, H):
    F = A - np.matmul(H, np.matmul(P, A))
    return F

def _get_K(F, P, H):
    K = np.linalg.inv(P) - F @ np.linalg.inv(P) @ np.transpose(F) - H
    return K

P, _, _ = control.dare(A, B, Q, R)
D = _get_D(B, P, R)
H = _get_H(B, D)
F = _get_F(A, P, H)

def self_tuning_ftl(N, t, phi_true_past, phi_pred_past, P, F, H, eta_true_past, eta_pred_past):
    """
    Parameters:
    - phi_true_past: Array of true values phi_star_tau for tau = t - 1
                     Shape: (k,) if scalar, or (k, dim) if vector
    - phi_pred_past: Array of predictions phi_{t - 1 | t - tau} for tau = 1 to k
                     Shape: same as phi_true_past
    """
    if t < 1:
        return 0.5, [], []
    if t == 1:
        # t - 1 = 0
        eta_true_past.append(np.linalg.matrix_power(F.T, 0) @ P @ phi_true_past)
        eta_pred_past.append(np.linalg.matrix_power(F.T, 0) @ P @ phi_pred_past[0])
    else:
        for i in range(len(eta_true_past)):
            eta_true_past[i] += np.linalg.matrix_power(F.T, t - i - 1) @ P @ phi_true_past
            if i >= t - N:
                eta_pred_past[i] += np.linalg.matrix_power(F.T, t - i - 1) @ P @ phi_pred_past[t - i - 1]
        eta_true_past.append(np.linalg.matrix_power(F.T, 0) @ P @ phi_true_past)
        eta_pred_past.append(np.linalg.matrix_power(F.T, 0) @ P @ phi_pred_past[0])
    num = 0
    den = 0
    for i in range(t-1):
            num += np.transpose(eta_true_past[i]) @ H @ eta_pred_past[i]
            den += np.transpose(eta_pred_past[i]) @ H @ eta_pred_past[i]
    if den == 0:
        return 0.5, eta_true_past, eta_pred_past
    return np.clip(num/den, 0, 1), eta_true_past, eta_pred_past

# P, _, _ = control.dare(A, B, Q, R)