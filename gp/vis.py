import numpy as np

def count_real_solutions(gamma, b_sq, delta):
    # Create the polynomial coefficients by clearing denominators
    # This is complex; a simpler way is to find local minima
    # and compare them to gamma.
    
    # Let's use a sampling/sign-change approach between poles
    poles = np.sort(-1 / delta)
    test_points = []
    
    # Add points between poles to check local minima
    for i in range(len(poles)-1):
        mid = (poles[i] + poles[i+1]) / 2
        test_points.append(mid)
        
    # Count how many interior branches have a minimum below gamma
    # Each such branch provides 2 solutions.
    interior_solutions = 0
    for p_mid in test_points:
        # Simplification: if the midpoint value < gamma, 
        # there are likely 2 solutions in that interval
        if secular_function(p_mid, b_sq, delta) < gamma:
            interior_solutions += 2
            
    return 2 + interior_solutions # 2 for the outer branches