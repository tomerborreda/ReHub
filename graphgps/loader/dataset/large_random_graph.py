import os.path as osp
import random

import torch
from networkx.generators.random_graphs import random_regular_graph
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx


class LargeRandomRegularGraphsDataset(InMemoryDataset):
    def __init__(self, root='datasets', name='large', num_graphs=48, num_nodes_in_graph=100000, transform=None, pre_transform=None):
        """
        Large random regular graphs

        Args:
            root (string): Root directory where the dataset should be saved.
        """

        self.original_root = root
        self.num_graphs = num_graphs
        self.num_nodes_in_graph = num_nodes_in_graph
        self.name = name
        self.folder = osp.join(root, 'random-large-regular-graphs', self.name)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f'LargeRandomRegularGraphsDataset-{self.name}'

    @property
    def processed_file_names(self):
        return f'LargeRandomRegularGraphsDataset-{self.name}'

    def download(self):
        open(osp.join(self.raw_dir, self.raw_file_names), 'a').close()

    def process(self):
        data_list = []
        for _ in range(self.num_graphs):
            data = from_networkx(random_regular_graph(d=3, n=self.num_nodes_in_graph))
            data.edge_attr = torch.rand((data.edge_index.shape[-1],)).to(torch.float32)
            data.x = torch.rand((data.num_nodes, 1)).to(torch.float32)
            data.y = random.randint(0, 5)
            data.num_nodes = torch.tensor(data.num_nodes, dtype=torch.int64)

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        return {'train': torch.arange(start=0, end=16, dtype=torch.long), 'val': torch.arange(start=16, end=32, dtype=torch.long), 'test': torch.arange(start=32, end=48, dtype=torch.long)}


if __name__ == '__main__':
    dataset = LargeRandomRegularGraphsDataset(root='../datasets')
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[5])
    print(dataset[5].y)
    print(dataset.get_idx_split())
