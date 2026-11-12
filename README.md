# FedCM: Communication-Efficient Collaborative Mixture Learning for Model-Heterogeneous Personalized Federated Learning

## Overview

FedCM is a communication-efficient framework for model-heterogeneous personalized federated learning. Clients with different backbones learn through a shared representation space. FedCM identifies groups of clients with related data distributions and combines federation-wide, group-specific, and client-private knowledge for personalized prediction.

## Key Features

- **Heterogeneous representation alignment.** A client-local adapter and a shared projector map features from different backbones into a common space.
- **Multi-view signatures and reliability-aware clustering.** The server combines complementary distribution statistics and checks candidate partitions for separation, balance, and stability. If no partition passes, training proceeds with the generalist and private experts.
- **Hierarchical residual experts.** A shared generalist provides the base representation. Gated cluster and private residuals adapt it to group-level and client-specific patterns.
- **Masked expert exchange.** Each client accesses and exchanges the cluster experts in its assigned neighborhood, limiting the expert payload sent in either direction.

## Experimental Setup

- **Datasets.** The paper evaluates FashionMNIST, SVHN, CIFAR-10, and CIFAR-100 under label skew. It also uses Rotated CIFAR-10 to study controlled feature shift.
- **Baselines.** FedProx, FedRep, pFedAFM, FedRE, IFCA, SnapCFL, FedGWC, MSCFL, FedFew, FedPOE, and FedMoE.
- **Metrics.** Overall and per-client test accuracy, best test accuracy within a communication budget, and cumulative uplink and downlink traffic required to reach a target accuracy.

The standalone CIFAR-10 example uses rotated client domains and lightweight heterogeneous CNNs. It runs local training, client grouping, masked expert exchange, and server aggregation, then reports sample-weighted test accuracy. It uses a compact demonstration setup. Run it from this directory:

```sh
python -m pip install -r requirements.txt
python __main__.py
```

The first run downloads CIFAR-10. Training defaults to CPU. Use `--device cuda` for a CUDA-enabled installation, `--data-dir PATH` to select the dataset directory, or `--synthetic` to run an offline example.

FedCM was developed within [PFLlib](https://github.com/TsingZ0/PFLlib). See [LICENSE](LICENSE) for the license.
