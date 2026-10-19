from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.random import randint
from sklearn.ensemble import ExtraTreesRegressor

from ebpfn.priors.nanoprior.eta import PriorEta
from ebpfn.priors.nanoprior.mechanisms import Role, select
from ebpfn.priors.nanoprior.scalars import resolve

# ebpfn: smallest target spread, relative to the target's own offset, that float32 can
# still resolve. Below it the "variation" is rounding error rather than signal.
_MIN_TARGET_SPREAD = 16 * torch.finfo(torch.float32).eps

# ----- Dataset sampling -----


def rand_dataset_plain(
    x_cat_sizes: list[int], y_cat_sizes: list[int], n_samples: int, eta: PriorEta | None = None
) -> dict[str, torch.Tensor]:
    # categorical sizes: 0 for numericals, >0 for categoricals with this cardinality.
    # ----- Create computation graph -----
    n_nodes = randlogint(2, 32 + 1, "z.graph.n_nodes")
    graph = rand_cauchy_graph(n_nodes)
    node_cat_sizes = [{} for _ in range(n_nodes)]

    for feature_group, cat_sizes in [("x", x_cat_sizes), ("y", y_cat_sizes)]:
        feature_nodes = np.random.permutation(n_nodes)[: randint(1, n_nodes + 1)]
        feature_node_idxs = np.random.choice(feature_nodes, replace=True, size=len(cat_sizes))

        for idx, (node_idx, cat_size) in enumerate(zip(feature_node_idxs, cat_sizes, strict=False)):
            node_cat_sizes[node_idx][f"{feature_group}_{idx}"] = cat_size

    # ----- Evaluate computation graph -----
    node_values: list[Any] = [None for _ in range(n_nodes)]  # filled in topological order
    columns = {}
    for node_idx in range(n_nodes):
        parent_values = [node_values[parent] for parent in graph[node_idx]]
        node_values[node_idx], out_features = rand_node_func(node_cat_sizes[node_idx], parent_values, n_samples, eta)
        columns.update(out_features)
    for col in columns.values():
        col[~torch.isfinite(col)] = 0  # fill nan/inf with zero
    return columns


def rand_dataset_filtered(
    x_cat_sizes: list[int],
    y_cat_sizes: list[int],
    n_samples: int,
    eta: PriorEta | None = None,
    max_attempts: int | None = None,
) -> dict[str, torch.Tensor]:
    # ebpfn: max_attempts bounds upstream's unbounded retry. Corner etas can make most
    # draws unlearnable, and an unbounded loop there is indistinguishable from a hang.
    attempts = 0
    while max_attempts is None or attempts < max_attempts:
        attempts += 1
        tensors = rand_dataset_plain(x_cat_sizes, y_cat_sizes, n_samples, eta)

        # ----- ExtraTrees filtering check -----
        X_np = torch.cat([t.float() for name, t in tensors.items() if name.startswith("x")], dim=-1).numpy()
        y = tensors["y_0"]

        if y_cat_sizes[0] > 0:  # classification
            y = y.long().squeeze(-1)
            y = F.one_hot(y, num_classes=int(y.max().item() + 1)).float()  # (n, n_classes)
            y = y[:, :1] if y.shape[1] == 2 else y  # drop one dimension for binary to make it faster
        elif y.std() <= _MIN_TARGET_SPREAD * y.abs().max():
            # ebpfn: the check below cannot reject a constant regression target. It compares
            # (y - mean)^2 against (y - oob)^2 in float32, and for a constant y the rounding
            # error of the mean makes the first term strictly positive while the trees
            # reproduce y exactly -- so a task carrying no signal at all scores as learnable.
            continue

        Y_np = y.float().cpu().numpy()

        # random_state=0 doesn't give OOB scores for all samples for some 133 <= n <= 257
        et = ExtraTreesRegressor(
            n_estimators=25, bootstrap=True, oob_score=True, n_jobs=1, random_state=1, max_depth=6
        ).fit(X_np, Y_np[:, 0] if Y_np.shape[1] == 1 else Y_np)

        # compute improvement in MSE over mean prediction baseline per sample
        Yhat = et.oob_prediction_[:, None] if len(et.oob_prediction_.shape) == 1 else et.oob_prediction_  # (n, d)
        mask = ~np.isnan(Yhat).any(axis=1)  # keep only valid out-of-bag rows
        imp = ((Y_np[mask] - Y_np.mean(axis=0, keepdims=True)) ** 2 - (Y_np[mask] - Yhat[mask]) ** 2).sum(axis=1)
        idx = np.random.default_rng(0).integers(0, len(imp), size=(200, len(imp)))  # 200 bootstrap samples
        pval = float(np.mean(imp[idx].mean(axis=1) <= 0.0))  # vectorized bootstrap

        if pval < 0.05:
            return tensors
    raise RuntimeError(f"no learnable dataset in {max_attempts} attempts; this eta may be mostly unlearnable")


def rand_cat_sizes(n_features: int, max_cat_size: int = 100) -> list[int]:
    cat_fraction = np.clip(np.random.uniform(-0.5, 1.2), 0.0, 1.0)
    n_cat = round(n_features * cat_fraction)
    cat_size_limit = randlogint(2, max_cat_size + 1)
    return [0] * (n_features - n_cat) + [randlogint(2, cat_size_limit + 1) for _ in range(n_cat)]


# ----- Scalar sampling -----


def randlognum(low: float, high: float, site: str | None = None) -> float:
    # ebpfn: a named site lets eta reparameterize this interval; unnamed is upstream's own.
    if site is not None:
        low, high = resolve(site, low, high)
    return float(np.exp(np.random.uniform(np.log(low), np.log(high))))


def randlogint(low: float, high: float, site: str | None = None) -> int:
    # ebpfn: resolved here rather than passed down, so the interval is not shifted twice.
    # Upstream's low is a structural floor (a tree count or a width below it crashes the
    # mechanism), so a downward shift piles mass on it rather than crossing it.
    floor = low
    if site is not None:
        low, high = resolve(site, low, high)
    return int(np.clip(np.floor(randlognum(low, high)).astype(int), floor, max(floor, high - 1)))


def randbool() -> bool:
    return bool(np.random.uniform() < 0.5)


def randchoice(options: list) -> Any:
    return options[randint(0, len(options))]


# ----- Random graph -----


def rand_cauchy_graph(n_nodes: int) -> list[list[int]]:  # returns list of parent node idxs for each node
    output_importances = torch.empty(n_nodes).cauchy_()
    input_importances = torch.empty(n_nodes).cauchy_()
    logits = np.random.standard_cauchy() + output_importances[None, :] + input_importances[:, None]
    adjacency_matrix = torch.rand_like(logits) <= torch.sigmoid(logits)
    return [[i_in for i_in in range(i_out) if adjacency_matrix[i_in, i_out]] for i_out in range(n_nodes)]


# ----- Random node function -----


def rand_node_func(cat_sizes: dict[str, int], xs: list[torch.Tensor], n_samples: int, eta: PriorEta | None = None):
    n_features = sum(max(csz, 1) for csz in cat_sizes.values()) + randlogint(1, 32, "z.node.extra_width")
    x = rand_points(n_samples, n_features, eta) if len(xs) == 0 else rand_multi_func(xs, n_features, eta)
    weights = rand_weights(1, x.shape[1])
    x = l2_normalize(standardize(x) * (weights / weights.square().mean().sqrt()))

    out_features = {}
    start_idx = 0
    for name, cat_size in cat_sizes.items():
        end_idx = start_idx + max(cat_size, 1)
        x[:, start_idx:end_idx], out_features[name] = rand_converter(x[:, start_idx:end_idx], cat_size, eta)
        start_idx = end_idx

    return x * randlognum(0.1, 10.0, "z.node.output_scale"), out_features


# ----- Random converter -----


def rand_converter(x: torch.Tensor, cat_size: int, eta: PriorEta | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if cat_size <= 0:  # numerical case is easy
        return x, (x if randbool() else rand_kumaraswamy_act(x))

    mode = randchoice(
        ["neigh_id", "neigh_disc", "neigh_func", "neigh_int", "softmax_id", "softmax_disc", "softmax_int"]
    )

    if mode.startswith("softmax"):
        x = randlognum(0.1, 10, "g.converter.softmax_temperature") * standardize(x) + torch.log(
            rand_weights(1, x.shape[1]) + 1e-4
        )
        x[~torch.isfinite(x)] = 0.0  # this is a dirty fix to avoid errors in torch.multinomial
        x_idxs = torch.multinomial(torch.softmax(x, dim=-1), num_samples=1, replacement=True)
        x_disc = rand_points(cat_size, cat_size, eta)[x_idxs.squeeze(-1)]
    else:  # neighbor-based
        centers = x[torch.randperm(len(x))[:cat_size]]
        x_idxs = torch.cdist(x, centers, p=randlognum(0.5, 4.0, "g.converter.neighbor_metric")).argmin(dim=-1)[
            :, None
        ]  # nearest center idx
        x_disc = centers[x_idxs.squeeze(-1)]

    if mode.endswith("_disc"):  # transform x for further use in the graph
        x = x_disc
    elif mode.endswith("_func"):  # don't have this for softmax because x_disc is already random for softmax
        x = rand_func(x_disc, cat_size, only_cheap=True, eta=eta)
    elif mode.endswith("_int"):
        x = x_idxs.float()  # x_idxs.float() is 1 column, it's going to be broadcasted

    return x, x_idxs


# ----- Random multi-function -----


def rand_multi_func(xs: list[torch.Tensor], d_out: int, eta: PriorEta | None = None):
    if randbool():
        return rand_func(torch.cat(xs, dim=-1), d_out, eta=eta)  # concatenate before random function
    out_cat = torch.stack([rand_func(x, d_out, eta=eta) for x in xs], dim=0)
    agg = randchoice([torch.sum, torch.prod, torch.max, torch.logsumexp])(out_cat, dim=0)
    return agg.values if isinstance(agg, tuple) else agg  # torch.max returns a namedtuple


# ----- Random function -----


def rand_func(
    x: torch.Tensor,
    d_out: int,
    only_cheap: bool = False,
    eta: PriorEta | None = None,
    role: Role = "mechanism",
) -> torch.Tensor:
    cheap_funcs = [rand_lin_func, rand_quad_func, rand_gp_func, rand_tree_func, rand_discretization_func]
    candidates = cheap_funcs if only_cheap else [*cheap_funcs, rand_mlp_func, rand_em_func, rand_prod_func]
    # ebpfn: without an eta this is upstream's uniform draw, bit-for-bit.
    func = randchoice(candidates) if eta is None else select(candidates, eta.weights_for(role))
    return rand_prod_func(x, d_out, eta, role) if func is rand_prod_func else func(x, d_out)


def rand_lin_func(x: torch.Tensor, d_out: int) -> torch.Tensor:
    return x @ (rand_matrix(1, d_out, x.shape[1])[0].t())


def rand_quad_func(x: torch.Tensor, d_out: int) -> torch.Tensor:
    idxs = torch.randperm(x.shape[1])[:20] if x.shape[1] > 20 else torch.arange(x.shape[1])  # drop x columns if needed
    tensor_3d = rand_matrix(d_out, len(idxs) + 1, len(idxs) + 1)
    x = torch.cat([x[:, idxs], torch.ones(x.shape[0], 1)], dim=-1)  # append ones to get constant + linear terms
    return torch.einsum("oij,bi,bj->bo", tensor_3d, x, x)


def rand_mlp_func(x: torch.Tensor, d_out: int) -> torch.Tensor:
    hidden_width = randlogint(1, 128, "z.mlp.hidden_width")
    x = x if randbool() else rand_act(x)
    for _ in range(randlogint(1, 4, "z.mlp.depth") - 1):  # loop over layers except last one
        x = rand_act(rand_lin_func(x, hidden_width))
    x = rand_lin_func(x, d_out)
    return x if randbool() else rand_act(x)


def rand_tree_func(x: torch.Tensor, d_out: int) -> torch.Tensor:
    n_trees = randlogint(1, 128, "z.tree.n_trees")
    depth = randint(1, 8)
    feature_imp = torch.clamp(x.std(dim=0, correction=0), 1e-8)
    feature_imp[~torch.isfinite(feature_imp)] = 1e-8
    split_dims = torch.multinomial(feature_imp, n_trees * depth, replacement=True)
    split_points = x[torch.randint(x.shape[0], size=(n_trees * depth,)), split_dims]
    split_sides = (x[:, split_dims] > split_points).reshape(x.shape[0], n_trees, depth)
    leaf_idxs = torch.einsum("btd,d->bt", split_sides.long(), 2 ** torch.arange(depth, dtype=torch.long))
    tree_idxs = torch.arange(n_trees, dtype=torch.long).expand(x.shape[0], n_trees)
    leaf_values = torch.randn(n_trees, 2**depth, d_out)  # Gaussian points -> avoid recursion
    return leaf_values[tree_idxs, leaf_idxs].mean(dim=1)  # mean_tree leaf_values[tree, leaf_idxs[batch, tree], d]


def rand_discretization_func(x: torch.Tensor, d_out: int) -> torch.Tensor:
    n_centers = x.shape[0] if x.shape[0] <= 2 else randlogint(2, min(x.shape[0], 256), "z.discretization.n_centers")
    centers = x[torch.randperm(len(x), device=x.device)[:n_centers]]
    targets = rand_lin_func(centers, d_out)  # transformed version of centers, for output with d_out dimensions
    dists = torch.cdist(x, centers, p=randlognum(0.5, 4.0, "z.discretization.metric"))
    closest_idx = dists.argmin(dim=-1)
    return targets[closest_idx]


def rand_gp_func(x: torch.Tensor, d_out: int, n_freqs: int = 256) -> torch.Tensor:
    a = randlognum(2.0, 20.0, "z.gp.decay")  # global decay rate a > 1

    if randbool():  # use standard kernel
        input_tfm = torch.randn(x.shape[1], x.shape[1]) * rand_weights(1, x.shape[1]).t()
        u = torch.clamp(torch.rand(n_freqs), 1e-6, 1 - 1e-6)  # clip to avoid trouble with inverse CDF
        invcdf = torch.pow(u, 1 / (1 - a)) - 1.0  # see the paper
        freqs = torch.randn(x.shape[1], n_freqs)
        freqs *= invcdf[None, :] / freqs.norm(dim=0, keepdim=True)  # invcdf for the radial component
        freqs = (
            randlognum(0.5, 10.0, "z.gp.lengthscale") * input_tfm @ freqs
        )  # sample global lengthscale for the kernel
    else:  # use product kernel, no specialized input transform etc.
        u = torch.clamp(torch.rand(x.shape[1], n_freqs), 1e-6, 1 - 1e-6)  # avoid too much trouble with inverse CDF
        freqs = torch.pow(u, 1 / (1 - a)) - 1.0

    bias = 2 * np.pi * torch.rand(1, n_freqs)
    weights = torch.randn(n_freqs, d_out) / np.sqrt(n_freqs)
    return torch.cos(x @ freqs + bias) @ weights


def rand_em_func(x: torch.Tensor, d_out: int) -> torch.Tensor:
    n_ind = randlogint(
        2, max(16, 2 * d_out) + 1, "z.em.n_inducing"
    )  # need to have >= 2 outputs, otherwise softmax is constant
    x_ind = x[torch.randint(x.shape[0], size=(n_ind,))] + torch.randn(n_ind, x.shape[1])  # centers with some noise
    stds = torch.exp(torch.rand(1) * torch.randn(1, n_ind))  # random standard deviations
    consts = -torch.log(2 * torch.pi * stds**2) * (x.shape[-1] / 2)  # const term of Gaussian log-prob
    dists = torch.cdist(x, x_ind, p=randlognum(1.0, 4.0, "z.em.metric"))  # matrix of distances
    logits = consts - torch.clamp(dists / stds, min=0.0) ** np.random.uniform(1.0, 2.0)  # generalized exponent
    return rand_lin_func(torch.softmax(logits, dim=-1), d_out)  # transform to d_out with linear function


def rand_prod_func(x: torch.Tensor, d_out: int, eta: PriorEta | None = None, role: Role = "mechanism") -> torch.Tensor:
    # only_cheap -> no recursion. ebpfn: the operands keep the caller's role, or a
    # multiplicative root draw would silently pull them from the mechanism simplex.
    return rand_func(x, d_out, only_cheap=True, eta=eta, role=role) * rand_func(
        x, d_out, only_cheap=True, eta=eta, role=role
    )


# ----- Random activation -----


def rand_act(x: torch.Tensor) -> torch.Tensor:
    return standardize(rand_plain_act(rand_rescale(standardize(x))))


def rand_plain_act(x: torch.Tensor) -> torch.Tensor:
    return randchoice([lambda x: randchoice(acts)(x)] * 4 + [rand_power_relu_act, rand_power_act])(x)


acts = [  # TabICLv1
    lambda x: x,
    torch.tanh,
    F.leaky_relu,
    F.elu,
    F.selu,
    F.silu,
    F.relu,
    F.softplus,
    F.relu6,
    F.hardtanh,
    torch.sign,
    torch.exp,
    torch.sin,
    torch.square,
    torch.abs,
    lambda x: (x >= 0.0).float(),
    lambda x: torch.exp(-(x**2)),
    lambda x: (torch.abs(x) <= 1.0).float(),
    # extra TabPFNv2 (some are unclear: power?, smooth relu might be softplus)
    lambda x: torch.log(torch.clamp(torch.abs(x), min=1e-6)),  # unclear how to handle log, here we do clamp+abs
    F.sigmoid,
    torch.round,
    lambda x: x - torch.floor(x),  # modulo
    lambda x: torch.argsort(torch.argsort(x, dim=-1), dim=-1).float(),  # converts x to ranks
    # new in TabICLv2
    F.logsigmoid,
    lambda x: F.softmax(x, dim=-1),
    lambda x: (x == torch.max(x, dim=-1, keepdim=True).values).float(),  # argmax
    lambda x: torch.argsort(x, dim=-1).float(),
]


def rand_power_relu_act(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) ** randlognum(0.1, 10.0, "z.act.power_relu_exponent")


def rand_power_act(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (x.abs() ** randlognum(0.1, 10.0, "z.act.power_exponent"))


def standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True, correction=0) + 1e-4)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.square().sum(dim=-1).mean().sqrt() + 1e-8)


def rand_rescale(x: torch.Tensor) -> torch.Tensor:
    # take random datapoints for shifts, so that activations like ReLU are not always zero
    bias = -x[torch.randint(x.shape[0], size=(x.shape[1],)), torch.arange(x.shape[1])][None, :]
    return randlognum(1e-0, 1e1, "z.act.rescale") * (x + bias)


def rand_kumaraswamy_act(x: torch.Tensor) -> torch.Tensor:
    a, b = (randlognum(0.2, 5, "g.column.kumaraswamy_shape") for _ in range(2))
    min, max = x.min(dim=0).values, x.max(dim=0).values
    x = torch.clamp((x - min) / (max - min + 1e-30), 0.0, 1.0)
    return 1.0 - (1.0 - x**a) ** b


# ----- Random matrix -----


def row_normalize(matrix: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    return matrix / (eps + matrix.norm(dim=-1, keepdim=True))


def rand_matrix(n_batch: int, n: int, m: int, no_act: bool = False) -> torch.Tensor:
    no_act_types = [torch.randn, rand_weights_matrix, rand_singular_values_matrix, rand_kernel_matrix]
    matrix = randchoice(no_act_types if no_act else [*no_act_types, rand_activation_matrix])(n_batch, n, m)
    return row_normalize(matrix + 1e-6 * torch.randn(n_batch, n, m), eps=1e-6)


def rand_weights_matrix(n_batch: int, n: int, m: int) -> torch.Tensor:
    matrix = rand_weights(n_batch * n, m).reshape(n_batch, n, m)
    return row_normalize(matrix * torch.randn_like(matrix), eps=1e-6)  # multiply by Gaussian -> allow negative weights


def rand_singular_values_matrix(n_batch: int, n: int, m: int) -> torch.Tensor:
    U, D, V = torch.randn(n_batch, n, min(n, m)), rand_weights(n_batch, min(n, m)), torch.randn(n_batch, min(n, m), m)
    return (U * D[:, None, :]) @ V  # SVD, but Gaussian matrix is cheaper than orthogonal and still rotation-invariant


def rand_kernel_matrix(n_batch: int, n: int, m: int) -> torch.Tensor:
    # use Laplace kernel in d=3 (arbitrary choice)
    points = rand_gauss_mixture_points(n_batch * (n + m), 3).reshape(n_batch, n + m, 3)
    dists = randlognum(0.1, 10.0, "z.matrix.kernel_scale") * torch.cdist(points[:, :n], points[:, n:])
    return torch.exp(-dists) * torch.sign(torch.randn(n_batch, n, m))


def rand_activation_matrix(n_batch: int, n: int, m: int) -> torch.Tensor:
    # rand_plain_act: no standardize -> works with n_batch=1. no_act: avoids infinite recursion
    matrix = rand_plain_act(rand_matrix(n_batch, n, m, no_act=True).reshape(n_batch, n * m)).reshape(n_batch, n, m)
    return matrix + 1e-3 * torch.randn_like(matrix)


# ----- Random points -----


def rand_points(n_batch: int, n: int, eta: PriorEta | None = None) -> torch.Tensor:
    # ebpfn: the root-marginal role -- shaping latent points, not mapping parents to a child.
    points = randchoice([rand_cov_points, torch.randn, rand_unif_points, rand_circle_points])(n_batch, n)
    return rand_func(points, n, eta=eta, role="root")


def rand_cov_points(n_batch: int, n: int) -> torch.Tensor:
    return randchoice([rand_unif_points, torch.randn])(n_batch, n) @ (torch.randn(n, n) * rand_weights(1, n)).t()


def rand_unif_points(n_batch: int, n: int) -> torch.Tensor:
    return 2 * torch.rand(n_batch, n) - 1.0


def rand_circle_points(n_batch: int, n: int) -> torch.Tensor:
    # radial density is proportional to r^{n-1}, so radial CDF is F(r) = r^n, inverse CDF is r = u^{1/n}
    return (torch.rand(n_batch, 1) ** (1 / n)) * row_normalize(torch.randn(n_batch, n))  # radial * uniform angular


def rand_gauss_mixture_points(n_batch: int, n: int) -> torch.Tensor:
    n_centers = randlogint(1, 16, "z.points.n_centers")
    center_idxs = torch.multinomial(rand_weights(1, n_centers).squeeze(0), num_samples=n_batch, replacement=True)
    matrices = torch.randn(n_centers, n, n) * rand_weights(n_centers, n)[:, None, :]
    # multiply by random factors, which themselves have random mean and random std
    matrices = matrices * torch.exp(torch.randn(1) + torch.randn(1) * torch.randn(n_centers, 1, 1))
    # this can use a large amount of RAM, so it's not used in rand_points()
    return torch.randn(n_centers, n)[center_idxs] + (matrices[center_idxs] @ torch.randn(n_batch, n, 1)).squeeze(-1)


# ----- Random weights -----


def rand_weights(n_batch: int, n: int) -> torch.Tensor:
    # ebpfn: these two intervals are the feature-skew control and the most reused sites in
    # the generator. Drawn vectorized, so they resolve directly rather than through randlognum.
    decay_low, decay_high = resolve("z.weights.decay", 0.1 / np.log(1 + n), 6.0)
    decay_rate = torch.as_tensor(np.exp(np.random.uniform(np.log(decay_low), np.log(decay_high), size=n_batch))).float()
    base_weights = torch.linspace(1.0, n, n)
    log_weights = -decay_rate[:, None] * torch.log(base_weights)
    scale_low, scale_high = resolve("z.weights.dispersion", 1e-4, 10.0)
    std_scale = torch.as_tensor(np.exp(np.random.uniform(np.log(scale_low), np.log(scale_high), size=n_batch))).float()
    logits = log_weights + std_scale[:, None] * torch.randn(n_batch, n)
    logits = torch.stack([logits[i, torch.randperm(n)] for i in range(n_batch)], dim=0)  # no batch randperm available
    return np.sqrt(n) * row_normalize(torch.softmax(logits, dim=-1))
