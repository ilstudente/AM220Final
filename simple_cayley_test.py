"""
Very simple test for Cayley graph concept.
"""

import numpy as np
from collections import deque

def test_cayley_concept():
    """Test basic concept of Cayley graph generation."""
    print("Testing Cayley graph concept...")
    
    n = 2  # Small value for testing
    
    # Define generators for SL(2, Z_n)
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]]
    ])
    
    # Start with identity matrix
    identity = np.array([[1, 0], [0, 1]])
    
    # Manually track nodes and edges
    nodes = {}
    edges = []
    
    # Add identity as first node
    nodes[(1, 0, 0, 1)] = 0
    
    # Use a queue for BFS
    queue = deque([identity])
    next_id = 1
    
    # Basic BFS to build the Cayley graph
    while queue:
        current = queue.popleft()
        current_flat = (current[0, 0], current[0, 1], current[1, 0], current[1, 1])
        current_id = nodes[current_flat]
        
        # Apply each generator
        for i, gen in enumerate(generators):
            # Multiply matrices and take modulo n
            result = np.matmul(current, gen) % n
            result_flat = (result[0, 0], result[0, 1], result[1, 0], result[1, 1])
            
            # Check if this is a new node
            if result_flat not in nodes:
                nodes[result_flat] = next_id
                next_id += 1
                queue.append(result)
                
            # Add edge
            target_id = nodes[result_flat]
            edges.append((current_id, target_id))
    
    print(f"Cayley graph has {len(nodes)} nodes and {len(edges)} edges")
    
    # For n=2, we expect a very small graph
    print("Nodes:")
    for node, idx in nodes.items():
        print(f"  {idx}: {node}")
    
    print("Edges:")
    for edge in edges:
        print(f"  {edge}")

if __name__ == "__main__":
    test_cayley_concept()
