import math
from operator import index
import numpy as np
import torch
import torch_geometric
from typing import Dict, Iterable, Callable, Tuple
from polymerlearn.utils import make_like_batch
from polymerlearn.explain.custom_gcam import LayerGradCam

# Source: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904

class FeatureExtractor(torch.nn.Module):
    '''
    Extracts inputs/outputs to each layer in the model

    Source: https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    
    '''
    
    layers = ['Asage', 'Gsage']

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self._features = {layer: torch.empty(0) for layer in self.layers}

        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, input_tup) -> Dict[str, torch.Tensor]:
        _ = self.model(*input_tup)
        return self._features

def parse_batches(
        batch: torch_geometric.data.Batch, 
        add_test: torch.Tensor):
    Abatch, Gbatch = make_like_batch(batch)

    A_X = Abatch.x
    A_edge_index = Abatch.edge_index
    A_batch = Abatch.batch

    G_X = Gbatch.x
    G_edge_index = Gbatch.edge_index
    G_batch = Gbatch.batch

    return (A_X, 
        A_edge_index,
        A_batch,
        G_X,
        G_edge_index,
        G_batch,
        torch.tensor(add_test).float())

def index_to_batch_mapper(batch, ratio = 0.5):
    '''
    Computes a backwards map from index in a SAGPool output
      to the original sample inputs.
    '''
    num_batches = max(batch).item() + 1
    batch_sizes = [torch.sum(batch == b) for b in range(num_batches)]

    # Multiply and take math.ceil for each batch
    final_sizes = [math.ceil(b * ratio) for b in batch_sizes]
    final_sizes = np.cumsum(final_sizes)

    # Now return dictionary mapping integer index to the given input sample:
    ind_map = {}
    for i in range(len(final_sizes)):
        bottom = 0 if i == 0 else final_sizes[i-1]
        for j in range(bottom, final_sizes[i]):
            ind_map[j] = i

    return ind_map


class PolymerGNNExplainer:
    '''
    Explainer for the PolymerGNN. Uses Grad CAM with Captum implementation.

    Similar to GXAI-Bench API
    '''

    def __init__(self, model: torch.nn.Module, explain_layer = 'fc1',
            pool_ratio = 0.5):
        
        self.model = model
        self.explain_layer = explain_layer
        self.ratio = pool_ratio
        self.gcam  = LayerGradCam(model, getattr(model, explain_layer))
        self.extractor = FeatureExtractor(model)

    def get_explanation(self, 
            batch: Tuple,
            add_test: torch.Tensor):
        '''
        Get explaination for a given sample from the dataset on the model.

        ..note:: Assumes max pooling. Would need to implement another expansion
            to work backwards through another pooling method.

        Args:
        '''
        # Parse the batches for captum usage
        batches_tup = parse_batches(batch, add_test)
        input_tup = tuple([batches_tup[j] for j in range(1, len(batches_tup))])

        # Compute the attribution from captum
        attribution = self.gcam.attribute(
            batches_tup[0],
            additional_forward_args = input_tup,
            attribute_to_layer_input = True
        )

        # Get features in a feedforward step
        features = self.extractor(batches_tup)

        def attr_scores(key = 'A', hc = 32):
            bind = 2 if key == 'A' else -2
            add_to_bottom = 0 if key == 'A' else 32
            ind_map = index_to_batch_mapper(batches_tup[bind], ratio = self.ratio)

            str_key = '{}sage'.format(key)

            feat_argmax = torch.argmax(features[str_key][0], dim = 0)

            # Expand scores backward from the max pooling:
            scores = torch.zeros((len(set(ind_map.values())), 32))
            for j in range(feat_argmax.shape[0]):
                score_ind = ind_map[feat_argmax[j].item()]
                scores[score_ind,j] = attribution[add_to_bottom + j] 

            return scores

        scores = {
            'A': attr_scores('A'),
            'G': attr_scores('G')
        }

        # Score individual attributes:
        num_add = add_test.shape[0]

        scores['add'] = attribution[-num_add:]

        return scores



