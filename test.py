import numpy as np

def generate_permutations_random(num_perms, num_shots):
    if not 0 < num_perms < np.math.factorial(num_shots):
        raise ValueError(f"num_perms out of range: {num_perms}")
    
    permutations = [list(range(num_shots))]
    while len(permutations) < num_perms:
        perm = list(range(num_shots))
        np.random.shuffle(perm)
        if not perm in permutations:
            permutations.append( list(perm) )

    return permutations


permutations = generate_permutations_random(10, 6)
for perm in permutations:
    print(perm)