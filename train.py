import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from comyco_sl import il as network
from comyco_sl import pool, traces
from comyco_sl.common import load_movie
from comyco_sl.discount import Discount
from comyco_sl.env import Env
from comyco_sl.expert import expert_scores
from comyco_sl.policy import save_weights
from comyco_sl.state import S_INFO, S_LEN
from comyco_sl.valid import Valid

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_DIM = [S_INFO, S_LEN]
A_DIM = 10
ACTOR_LR_RATE = 1e-4
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 1000
MODEL_SAVE_INTERVAL = 10
RANDOM_SEED = 42
SUMMARY_DIR = './models/step05'
LOG_FILE = os.path.join(SUMMARY_DIR, 'log.tsv')
MODEL_PREFIX = 'nn_model_ep_'
MODEL_SUFFIX = '.pth'
SEEK_PROB = 0.5  # fraction of episodes with 1-3 random seeks (empty buffer, or 1-5 chunks kept)
PAPER_EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'paper_eval.py')

# expert: beam search over the next HORIZON chunks, one beam per first action, planning on a discounted bandwidth
# (the discount is adapted during training on a validation set, models/mse 0.9, a1 1.0, a08 0.8, a07 0.7 were fixed)
TOP_K = 3000
HORIZON = 15
TEMPERATURE = 0.25  # exploration while collecting data: scores are divided by this before Gumbel noise

os.makedirs(SUMMARY_DIR, exist_ok=True)

NN_MODEL = None


def start_eval(epoch, npz_path, stats):
    # B evaluation runs in a separate process (StreamLens' paper workflow) while training goes on
    out = os.path.abspath(os.path.join(SUMMARY_DIR, 'paper', 'ep%d' % epoch))
    score = out + '.json'
    proc = subprocess.Popen([sys.executable, PAPER_EVAL, os.path.abspath(npz_path), out, score],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dict(epoch=epoch, proc=proc, score=score, stats=stats)


def finish_eval(item, log_file, t0):
    item['proc'].wait()
    epoch, (loss, acc, train_reward, val, discount) = item['epoch'], item['stats']
    if os.path.exists(item['score']):
        s = json.load(open(item['score']))
        linear, nonlinear = s['linear'], s['nonlinear']
    else:
        linear = nonlinear = float('nan')
    print('epoch %3d loss=%.4f acc=%.3f train-reward=%6.3f val=%.3f discount=%.3f B-linear=%.3f B-nonlinear+prefetch=%.3f [%ds]' % (
        epoch, loss, acc, train_reward, val, discount, linear, nonlinear, time.time() - t0), flush=True)
    log_file.write('%d\t%f\t%f\t%f\t%f\t%f\t%d\t%f\t%f\n' % (epoch, loss, acc, train_reward, linear, nonlinear, time.time() - t0, val, discount))
    log_file.flush()


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    fams = traces.cached()
    flat, idx = [], []
    for f in fams.values():
        idx.append(list(range(len(flat), len(flat) + len(f))))
        flat += f
    env = Env(load_movie(), flat, idx, seed=RANDOM_SEED, seek_prob=SEEK_PROB)
    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)

    if NN_MODEL is not None:
        actor.load_model(NN_MODEL)
        print('Model restored.')

    actor_pool = pool.pool(obs_dim=S_DIM, act_dim=A_DIM)
    valid, discount = Valid(), Discount()
    pending = None
    t0 = time.time()
    with open(LOG_FILE, 'w') as log_file:
        log_file.write('epoch\tloss\tacc\ttrain_reward\tb_linear\tb_nonlinear\tsec\tval\tdiscount\n')
        for epoch in range(TRAIN_EPOCH + 1):
            obs = env.reset()
            losses, accs, rewards = [], [], []
            for step in range(TRAIN_SEQ_LEN):
                scores = actor.predict(obs)
                bit_rate = int(np.argmax(scores / TEMPERATURE + np.random.gumbel(size=len(scores))))

                actor_pool.submit(obs, expert_scores(env.sim, TOP_K, HORIZON, discount.value))

                s_batch, a_batch = actor_pool.get()
                loss, acc = actor.train_step(s_batch, a_batch)
                losses.append(loss)
                accs.append(acc)

                obs, rew, done, info = env.step(bit_rate)
                rewards.append(rew)
                if done:
                    break

            if epoch % MODEL_SAVE_INTERVAL == 0:
                save_path = os.path.join(SUMMARY_DIR, MODEL_PREFIX + str(epoch) + MODEL_SUFFIX)
                actor.save_model(save_path)
                npz_path = save_path.replace(MODEL_SUFFIX, '.npz')
                weights = actor.weights()
                save_weights(weights, npz_path)
                if pending is not None:
                    finish_eval(pending, log_file, t0)
                val = valid.score(weights)
                stats = (np.mean(losses), np.mean(accs), np.mean(rewards), val, discount.value)
                pending = start_eval(epoch, npz_path, stats)
                if discount.update(val):
                    actor_pool.clear()  # old labels were made with the old discount
        if pending is not None:
            finish_eval(pending, log_file, t0)
    print('done in %ds' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
