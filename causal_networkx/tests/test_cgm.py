import pytest 

import numpy as np
from causal_networkx.cgm import CausalGraph, CausalGraphicalModel


nodes = [
    'a', 'b', 'c', 'd', 'e'
]
cgm = CausalGraphicalModel()
for node in nodes:
    cgm.add_nodes_from(nodes, latent=False)

cgm.is_directed()
print(cgm)




def test_causal_graph_methods():
    incoming_latent_data = [('X', 'Y')]
    G = CausalGraph(incoming_latent_data=incoming_latent_data)

    
