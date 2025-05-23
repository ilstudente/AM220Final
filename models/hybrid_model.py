from typing import Union, List, Callable, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData

from samplers.gumbel_scheme import GumbelSampler
from samplers.imle_scheme import IMLESampler
from samplers.simple_scheme import SIMPLESampler

LARGE_NUMBER = 1.e10


class HybridModel(torch.nn.Module):
    def __init__(self,
                 scorer_model: torch.nn.Module,
                 list_num_centroids: List,
                 base2centroid_model: torch.nn.Module,
                 sampler: Union[IMLESampler, SIMPLESampler, GumbelSampler],
                 hetero_gnn: torch.nn.Module,

                 jk_func: Callable,
                 graph_pool_idx: str,
                 pred_head: torch.nn.Module,
                 auxloss_func: Optional[Callable],
                 edge_init_type: str = 'uniform'):
        super(HybridModel, self).__init__()

        self.scorer_model = scorer_model
        self.list_num_centroids = list_num_centroids
        self.base2centroid_model = base2centroid_model
        self.sampler = sampler
        self.hetero_gnn = hetero_gnn

        self.jk_func = jk_func
        self.graph_pool_idx = graph_pool_idx
        self.pred_head = pred_head
        self.auxloss_func = auxloss_func
        self.edge_init_type = edge_init_type

    def forward(self, data, for_plots_only=False):
        device = data.x.device
        plot_scores, plot_node_mask = None, None

        cumsum_nnodes = data._slice_dict['x']
        nnodes_list = (cumsum_nnodes[1:] - cumsum_nnodes[:-1]).to(device)
        n_graphs = data.num_graphs

        # get scores and samples
        scores = self.scorer_model(data)
        nnodes, max_n_centroids, n_ensemble = scores.shape
        if for_plots_only:
            plot_scores = scores.detach().clone().cpu().numpy()

        if min(self.list_num_centroids) < max_n_centroids:
            # this is a mask indexing which centroids are kept in each ensemble
            ens_idx_mask = torch.zeros(n_ensemble, max_n_centroids, dtype=torch.bool, device=device)
            for i, n_ct in enumerate(self.list_num_centroids):
                ens_idx_mask[i, :n_ct] = True

            # for sampling, if there are less n_centroids than max possible num, then pad with a bias
            scores[:, ~ens_idx_mask.t()] = scores[:, ~ens_idx_mask.t()] - LARGE_NUMBER

        # k-subset sampling is carried out as usual, in parallel
        node_mask, _ = self.sampler(scores) if self.training else self.sampler.validation(scores)
        if for_plots_only:
            plot_node_mask = node_mask.detach().cpu().numpy()
            return None, plot_node_mask, plot_scores, None

        # when combination with various cluster numbers, some clusters have 0s embedding
        n_samples, nnodes, n_centroids, n_ensemble = node_mask.shape
        repeats = n_samples * n_ensemble

        node_mask = node_mask.permute(0, 3, 2, 1).reshape(repeats, n_centroids, nnodes)

        # map clustered nodes into each centroid
        # add a dimension for multiply broadcasting
        # centx: repeats, n_centroids, n_graphs, features
        centroid_x = self.base2centroid_model(data, node_mask[..., None])
        centroid_x = centroid_x.permute(0, 2, 1, 3).reshape(-1, centroid_x.shape[-1])        # construct a heterogeneous hierarchical graph
        # low to high hierarchy
        src = torch.arange(data.num_nodes * repeats, device=device).repeat_interleave(n_centroids)
        dst = torch.arange(repeats * n_graphs, device=device).repeat_interleave(
            nnodes_list.repeat(repeats)) * n_centroids
        dst = dst[None] + torch.arange(n_centroids, device=device, dtype=torch.long)[:, None]
        dst = dst.t().reshape(-1)

        # edges intra super nodes, assume graphs are undirected
        # dumb edge index
        idx = np.hstack([np.vstack(np.triu_indices(n_centroids, k=1)),
                         np.vstack(np.tril_indices(n_centroids, k=-1))])
        idx = torch.from_numpy(idx).to(device)
          # Determine edge weights based on initialization type
        if self.edge_init_type == 'cayley':
            from models.cayley_utils import cayley_initialize_edge_weight
            # Handle edge weights for each individual graph in the batch
            edge_weight_list = []
            
            # Process each graph in the batch separately
            offset = 0
            for nodes_in_graph in nnodes_list:
                try:
                    # Initialize with Cayley graph for this specific graph
                    graph_weights = cayley_initialize_edge_weight(
                        num_base_nodes=int(nodes_in_graph), 
                        num_virtual_nodes=n_centroids
                    ).to(device)
                    
                    # Repeat the weights for each sample/ensemble
                    graph_weights = graph_weights.repeat(repeats, 1, 1)
                    
                    # Reshape to match expected format
                    edge_weight = graph_weights.permute(0, 2, 1).reshape(-1)
                    edge_weight_list.append(edge_weight)
                    
                except Exception as e:
                    # Fall back to uniform initialization if Cayley graph initialization fails
                    print(f"Warning: Falling back to uniform initialization for a graph: {e}")
                    uniform_weight = node_mask.permute(0, 2, 1).reshape(-1)[offset:offset+int(nodes_in_graph)*n_centroids]
                    edge_weight_list.append(uniform_weight)
                    
                offset += int(nodes_in_graph) * n_centroids
                
            # Combine all weights
            try:
                edge_weight = torch.cat(edge_weight_list)
            except Exception as e:
                print(f"Warning: Failed to concatenate edge weights: {e}. Falling back to uniform initialization.")
                edge_weight = node_mask.permute(0, 2, 1).reshape(-1)
        else:
            # Original uniform initialization
            edge_weight = node_mask.permute(0, 2, 1).reshape(-1)

        new_data = HeteroData(
            base={'x': data.x,  # will be later filled
                  'batch': data.batch.repeat(repeats) +
                           torch.arange(repeats, device=device).repeat_interleave(nnodes) * n_graphs},
            centroid={'x': centroid_x,
                      'batch': torch.arange(repeats * n_graphs, device=device).repeat_interleave(n_centroids)},

            base__to__base={
                'edge_index': data.edge_index.repeat(1, repeats) +
                              torch.arange(repeats, device=device).repeat_interleave(data.num_edges) * data.num_nodes,
                'edge_attr': (data.edge_attr.repeat(repeats) if data.edge_attr.dim() == 1 else
                              data.edge_attr.repeat(repeats, 1)) if data.edge_attr is not None else None,
                'edge_weight': None
            },
            base__to__centroid={
                'edge_index': torch.vstack([src, dst]),
                'edge_attr': None,
                'edge_weight': edge_weight
            },
            centroid__to__base={
                'edge_index': torch.vstack([dst, src]),
                'edge_attr': None,
                'edge_weight': edge_weight
            },
            centroid__to__centroid={
                'edge_index': idx.repeat(1, n_graphs * repeats) +
                              (torch.arange(n_graphs * repeats, device=device) * n_centroids).repeat_interleave(
                                  idx.shape[1]),
                'edge_attr': None,
                'edge_weight': None
            }
        )

        list_base_embeddings, list_centroid_embeddings = self.hetero_gnn(
            data,
            new_data,
            hasattr(data, 'edge_attr') and data.edge_attr is not None)

        base_embedding = self.jk_func(list_base_embeddings)
        centroid_embedding = self.jk_func(list_centroid_embeddings)

        if isinstance(self.pred_head, torch.nn.ModuleList):
            assert isinstance(base_embedding, list) and isinstance(centroid_embedding, list)
            graph_embedding = []
            for head, n_emb, c_emb in zip(self.pred_head, base_embedding, centroid_embedding):
                graph_embedding.append(
                    head(n_emb.reshape(repeats, nnodes, n_emb.shape[-1]),
                         c_emb.reshape(repeats, n_graphs * n_centroids, c_emb.shape[-1]),
                         getattr(data, self.graph_pool_idx),
                         new_data['centroid']['batch'][:n_graphs * n_centroids])
                )
        else:
            graph_embedding = self.pred_head(
                base_embedding.reshape(repeats, nnodes, base_embedding.shape[-1]),
                centroid_embedding.reshape(repeats, n_graphs * n_centroids, centroid_embedding.shape[-1]),
                getattr(data, self.graph_pool_idx),
                new_data['centroid']['batch'][:n_graphs * n_centroids]
            )

        # get auxloss
        auxloss = self.auxloss_func(scores=scores, data=data) if self.training and self.auxloss_func is not None else 0.

        return graph_embedding, plot_node_mask, plot_scores, auxloss
