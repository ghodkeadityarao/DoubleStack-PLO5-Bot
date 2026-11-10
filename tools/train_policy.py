"""Train the advisor network (needs PyTorch: pip install torch; the bot itself only needs numpy).

    python tools/train_policy.py OUT.npz DATA.npz [DATA2.npz ...] [--epochs 14]

The network maps the decision features to the expected chip result (in units of
80 chips) of each of the four actions. Only the action that was actually taken is
trained on each row. OUT.npz holds the weights in the format advisor.py loads;
copy it to data/policy_net.npz to use it.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("data", nargs="+")
    p.add_argument("--epochs", type=int, default=14)
    a = p.parse_args()

    parts = [np.load(f) for f in a.data]
    X = torch.tensor(np.concatenate([d["X"] for d in parts]))
    A = torch.tensor(np.concatenate([d["A"] for d in parts])).long()
    Y = torch.tensor(np.concatenate([d["Y"] for d in parts]))
    n = len(X)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    nv = n // 10
    val, train = perm[:nv], perm[nv:]
    print(f"{n} rows, variance of the target {float(Y.var()):.4f}")

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 4))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    def loss_on(idx):
        q = net(X[idx])
        return ((q.gather(1, A[idx][:, None])[:, 0] - Y[idx]) ** 2).mean()

    for ep in range(a.epochs):
        order = train[torch.randperm(len(train))]
        for i in range(0, len(order), 4096):
            opt.zero_grad()
            loss_on(order[i:i + 4096]).backward()
            opt.step()
        sched.step()
        with torch.no_grad():
            print(f"epoch {ep}: validation mse {float(loss_on(val)):.5f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    np.savez(a.out, W1=sd["0.weight"].T, b1=sd["0.bias"], W2=sd["2.weight"].T, b2=sd["2.bias"],
             W3=sd["4.weight"].T, b3=sd["4.bias"])
    print("saved", a.out)


if __name__ == "__main__":
    main()
