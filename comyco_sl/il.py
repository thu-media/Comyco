import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_NUM = 128


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.fc1_actor = nn.Linear(1, FEATURE_NUM)
        self.fc2_actor = nn.Linear(1, FEATURE_NUM)
        self.conv1_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv2_actor = nn.Linear(self.s_dim[1], FEATURE_NUM)
        self.conv3_actor = nn.Linear(self.a_dim, FEATURE_NUM)
        self.fc3_actor = nn.Linear(1, FEATURE_NUM)
        self.fc4_actor = nn.Linear(FEATURE_NUM * self.s_dim[0], FEATURE_NUM)
        self.pi_head = nn.Linear(FEATURE_NUM, action_dim)

    def forward(self, inputs):
        split_0 = F.relu(self.fc1_actor(inputs[:, 0:1, -1]))
        split_1 = F.relu(self.fc2_actor(inputs[:, 1:2, -1]))
        split_2 = F.relu(self.conv1_actor(inputs[:, 2:3, :]).view(-1, FEATURE_NUM))
        split_3 = F.relu(self.conv2_actor(inputs[:, 3:4, :]).view(-1, FEATURE_NUM))
        split_4 = F.relu(self.conv3_actor(inputs[:, 4:5, :self.a_dim]).view(-1, FEATURE_NUM))
        split_5 = F.relu(self.fc3_actor(inputs[:, 5:6, -1]))

        merge_net = torch.cat([split_0, split_1, split_2, split_3, split_4, split_5], 1)

        pi_net = F.relu(self.fc4_actor(merge_net))
        return self.pi_head(pi_net)  # one score per action


class Network:
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device('cpu')

        self.model = Actor(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _prepare_state(self, state):
        tensor = torch.from_numpy(state).float().to(self.device)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def predict(self, state):
        self.model.eval()
        with torch.no_grad():
            tensor = self._prepare_state(state)
            scores = self.model(tensor)
        return scores.cpu().numpy()[0]

    def train_step(self, s_batch, a_batch):
        self.model.train()

        states = self._prepare_state(s_batch)
        targets = torch.from_numpy(a_batch).float().to(self.device)

        scores = self.model(states)
        loss = F.mse_loss(scores, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        acc = (scores.argmax(1) == targets.argmax(1)).float().mean().item()
        return loss.item(), acc

    def weights(self):
        return {k: v.detach().cpu().numpy().astype('float32') for k, v in self.model.state_dict().items()}

    def save_model(self, path):
        torch.save({'model_state_dict': self.model.state_dict()}, path)
        return path

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        self.model.load_state_dict(state_dict)
