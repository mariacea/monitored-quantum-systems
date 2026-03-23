from typing import List

def calculate_errors(eigenvalues: List[float]) -> List[float]:
    """
    Calculates relative errors between consecutive eigenvalues.

    Args:
        eigenvalues (List[float]): List of eigenvalues from the iterations.
    
    Returns:
        relative_errors (List[float]): List of relative errors, with None for the first iteration
    """
    relative_errors = [None]
    for i in range(1, len(eigenvalues)):
        relative_error = abs(eigenvalues[i] - eigenvalues[i - 1]) / abs(eigenvalues[i - 1])
        relative_errors.append(relative_error)
    return relative_errors
