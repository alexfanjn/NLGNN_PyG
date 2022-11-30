import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class NLGNN(torch.nn.Module):
    def __init__(self, data, le, window_size, num_features, num_hidden, num_classes, dropout):
        super(NLGNN, self).__init__()

        if le == 'mlp':
            self.first_1 = torch.nn.Linear(num_features, num_hidden[0])
            self.first_2 = torch.nn.Linear(num_hidden[0], num_hidden[1])
        elif le == 'gcn':
            self.first_1 = GCNConv(num_features, num_hidden[0])
            self.first_2 = GCNConv(num_hidden[0], num_hidden[1])
        else: # 'gat'
            self.first_1 = GATConv(num_features, num_hidden[0])
            self.first_2 = GATConv(num_hidden[0], num_hidden[1])

        self.le = le
        self.attention_layer = torch.nn.Linear(num_hidden[1], 1)
        self.window_size = window_size
        self.conv1d1 = torch.nn.Conv1d(num_hidden[1], num_hidden[1], kernel_size=window_size, padding=int((self.window_size-1)/2))
        self.conv1d2 = torch.nn.Conv1d(num_hidden[1], num_hidden[1], kernel_size=window_size, padding=int((self.window_size-1)/2))

        self.final_layer = torch.nn.Linear(2 * num_hidden[1], num_classes)

        self.dropout = dropout
        self.data = data






    def forward(self):
        if self.le == 'mlp':
            h = self.first_1(self.data.x)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.first_2(h)
            # h = F.relu(h)
        else: #gcn or gat
            h = self.first_1(self.data.x, self.data.edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.first_2(h, self.data.edge_index)
            # h = F.relu(h)

        # h = F.dropout(h, p=self.dropout, training=self.training)

        before_h = h

        a = self.attention_layer(h)



        sort_index = torch.argsort(a.flatten(), descending=True)
        h = a * h

        h = h[sort_index].T.unsqueeze(0)
        h = self.conv1d1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv1d2(h)
        # h = F.relu(h)
        # h = F.dropout(h, p=self.dropout, training=self.training)

        h = h.squeeze().T
        arg_index = torch.argsort(sort_index)
        h = h[arg_index]

        final_h = torch.cat([before_h, h], dim=1)
        final_h = self.final_layer(final_h)

        return F.log_softmax(final_h, 1)