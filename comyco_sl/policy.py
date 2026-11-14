import numpy as np


def save_weights(weights, path):
    np.savez(path, **weights)


def load_weights(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _lin(w, name, x):
    return w[name + '.weight'] @ x + w[name + '.bias']


def forward_np(w, state):
    a_dim = w['pi_head.weight'].shape[0]
    relu = lambda v: np.maximum(v, 0.0)
    parts = [
        relu(_lin(w, 'fc1_actor', state[0, -1:])),
        relu(_lin(w, 'fc2_actor', state[1, -1:])),
        relu(_lin(w, 'conv1_actor', state[2, :])),
        relu(_lin(w, 'conv2_actor', state[3, :])),
        relu(_lin(w, 'conv3_actor', state[4, :a_dim])),
        relu(_lin(w, 'fc3_actor', state[5, -1:])),
    ]
    h = relu(_lin(w, 'fc4_actor', np.concatenate(parts)))
    return _lin(w, 'pi_head', h)  # one score per action, the best action is the argmax
