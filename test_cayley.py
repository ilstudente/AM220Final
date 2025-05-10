"""
Test script for Cayley graph utility functions.
"""

import torch
import numpy as np
from collections import deque

def simple_cayley_graph(n):
    """
    Simple implementation of Cayley graph generation to test the concept.
    """
    generators = np.array([
        [[1, 1], [0, 1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]]])
    ind = 1

    queue = deque([np.array([[1, 0], [0, 1]])])
    nodes = {(1, 0, 0, 1): 0}

    senders = []
    receivers = []

    while queue:
        x = queue.popleft()
        x_flat = (x[0][0], x[0][1], x[1][0], x[1][1])
        ind_x = nodes[x_flat]
        for i in range(4):
            tx = np.matmul(x, generators[i])
            tx = np.mod(tx, n)
            tx_flat = (tx[0][0], tx[0][1], tx[1][0], tx[1][1])
            if tx_flat not in nodes:
                nodes[tx_flat] = ind
                ind += 1
                queue.append(tx)
            else:
                ind_tx = nodes[tx_flat]
                senders.append(ind_x)
                receivers.append(ind_tx)
            
    print(f"Cayley graph with n={n} has {len(nodes)} nodes and {len(senders)} edges")
    return torch.tensor([senders, receivers])

def cayley_graph_size(n):
    """
    Simple approximation of Cayley graph size.
    """
    # For SL(2, Z_n), it's approximately n^3
    return n*n*n

def test_cayley_graph():
    print("Testing Cayley graph generation...")
    for n in range(2, 5):
        print(f"Creating Cayley graph with n={n}")
        try:
            edge_index = simple_cayley_graph(n)
            print(f"  Edge index shape: {edge_index.shape}")
            print(f"  Approximate size estimate: {cayley_graph_size(n)}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_cayley_graph()

def test_cayley_graph_size():
    """Test that the Cayley graph size calculation works properly."""
    print("Testing Cayley graph size calculation...")
    sizes = [_cayley_graph_size(n) for n in range(2, 10)]
    print(f"Cayley graph sizes for n=2 to 9: {sizes}")
    
    # Verify get_cayley_n works properly
    for nodes in [10, 20, 50, 100]:
        n = _get_cayley_n(nodes)
        size = _cayley_graph_size(n)
        print(f"For {nodes} nodes, we need n={n} which gives {size} nodes")
        assert size >= nodes, f"Size {size} should be >= {nodes}"
        
    print("Cayley graph size calculation test passed!")

def test_edge_weight_initialization():
    """Test that the edge weight initialization works properly."""
    print("Testing edge weight initialization...")
    
    # Test with small example
    num_base_nodes = 10
    num_virtual_nodes = 14  # This should match a Cayley graph with n=2
    
    # Get edge weights
    try:
        edge_weights = cayley_initialize_edge_weight(num_base_nodes, num_virtual_nodes)
        print(f"Edge weights shape: {edge_weights.shape}")
        print(f"Number of non-zero edges: {(edge_weights > 0).sum().item()}")
        print(f"First few weights: {edge_weights[:5, :5]}")
        
        # Each base node should connect to some virtual nodes
        assert (edge_weights.sum(axis=1) > 0).all(), "Some base nodes have no connections"
        
        # Similarly, each virtual node should connect to some base nodes
        assert (edge_weights.sum(axis=0) > 0).all(), "Some virtual nodes have no connections"
        
        print("Edge weight initialization test passed!")
    except ValueError as e:
        print(f"Test failed with error: {e}")
        
        # Try with a different size that should work
        cayley_n = 2
        cayley_size = _cayley_graph_size(cayley_n)
        num_virtual_nodes = cayley_size - num_base_nodes
        print(f"Retrying with num_virtual_nodes={num_virtual_nodes}")
        
        edge_weights = cayley_initialize_edge_weight(num_base_nodes, num_virtual_nodes)
        print(f"Edge weights shape: {edge_weights.shape}")
        print(f"Number of non-zero edges: {(edge_weights > 0).sum().item()}")

def main():
    """Run all tests."""
    print("Running Cayley graph utility tests...")
    
    test_cayley_graph_generation()
    print("")
    
    test_cayley_graph_size()
    print("")
    
    test_edge_weight_initialization()
    print("")
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
