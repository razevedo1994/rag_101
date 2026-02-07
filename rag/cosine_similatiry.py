import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print(f"v1 and v2 similatiry: {cosine_similarity(v1, v2)}")

v3 = np.array([4, 5, 6])
v4 = np.array([-1, -7, -12])

print(f"v3 and v4 similatiry: {cosine_similarity(v3, v4)}")
