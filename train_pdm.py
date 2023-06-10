import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.nn as dglnn
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl import AddSelfLoop
import argparse


def get_similarity(a, b, temperature=0.05, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, torch.zeros_like(a_n) + eps)
    b_norm = b / torch.max(b_n, torch.zeros_like(b_n) + eps)
    sim_mt = a_norm @ b_norm.T
    return sim_mt / temperature


class GCN(nn.Module):
    def __init__(self, in_size, rep_size, depth=3, dropout=0.1, activation=nn.GELU, last_activation=nn.GELU, norm=False):
        super().__init__()
        self.rep_size = rep_size
        self.layers = nn.ModuleList([
            dglnn.GraphConv(
                in_size if d == 0 else rep_size, rep_size, 
                # activation=activation if d < depth - 1 else last_activation,
            ) for d in range(depth)
        ])
        # self.layers.append(dglnn.GraphConv(in_size, rep_size, activation=activation))
        # for d in range(depth-1):
        #     self.layers.append(dglnn.GraphConv(rep_size, rep_size, activation=activation if d < depth - 2 else last_activation))

        self.norms = torch.nn.ModuleList([
            nn.BatchNorm1d(rep_size) if norm else nn.Identity() for _ in range(depth)
        ])
        self.activations = torch.nn.ModuleList([
            # nn.GELU() if i < len(hidden_channels) - 1 else nn.Tanh() for i in range(len(hidden_channels))
            activation() if d < depth - 1 else last_activation() for d in range(depth)
        ])
        self.dropout = nn.Dropout(dropout)

        self.params = nn.parameter.Parameter(torch.empty(rep_size, rep_size))
        self.proj = nn.Linear(rep_size, rep_size)
        nn.init.xavier_uniform_(self.params)

    def forward(self, g, features):
        h = features
        for l, layer in enumerate(self.layers):
            h = self.dropout(self.activations[l](self.norms[l](layer(g, h))))
        return h

    def forward_diffusion(self, g, features, idx, negative_features=None, mode='dp'):
        node_emb = self.forward(g, features)
        if negative_features is not None:
            negative_dp = (node_emb[idx] * self.forward(g, negative_features)[idx]).sum(dim=1)
            # ((node_emb, (node_emb * negative_emb).sum(dim=1)[:, None]))
        if mode == 'bilinear':
            return (node_emb if idx is None else node_emb[idx]) @ ((self.params + self.params.T) / 2) @ node_emb.T
        elif mode == 'projected_dp':
            node_emb = self.proj(node_emb)
            return (node_emb if idx is None else node_emb[idx]) @ node_emb.T
        elif mode == 'dp':
            if negative_features is None:
                return node_emb[idx] @ node_emb.T
            return torch.cat((node_emb[idx] @ node_emb.T, negative_dp[:, None]), dim=1)
        elif mode == 'cosine':
            return get_similarity(node_emb if idx is None else node_emb[idx], node_emb)


class RandIndexDataset(Dataset):
    def __init__(self, range):
        self.range = range

    def __len__(self):
        return self.range

    def __getitem__(self, idx):
        return idx
        
    
def evaluate(labels, mask, model_output, classifier):
    with torch.no_grad():
        logits = classifier(model_output)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class PairLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x0, x1):
        sim = get_similarity(x0, x1)
        targets = torch.arange(x0.shape[0]).long().to(x0.device)
        return self.loss(sim, targets)


def pair_loss(x, x_prime, graph, khop=3, degree_adj=False):
    khop_adj_mat = dgl.khop_adj(graph, khop).to(x.device)
    if not degree_adj:
        khop_adj_mat = torch.minimum(khop_adj_mat, torch.ones_like(khop_adj_mat))
    targets = khop_adj_mat / khop_adj_mat.sum(dim=1)
    sim = get_similarity(x, x_prime)
    return F.cross_entropy(sim, targets) # TODO: label smoothing?


def heat_loss(x, x_prime, targets):
    sim = get_similarity(x, x_prime)
    return F.cross_entropy(sim, targets) # TODO: label smoothing?


def ppr_matrix(adj_matrix, alpha = 0.1, add_self_loop = False):
    num_nodes = adj_matrix.shape[0]
    device = adj_matrix.device
    A_tilde = adj_matrix if not add_self_loop else adj_matrix + torch.eye(num_nodes).to(device)
    D_tilde = torch.diag(1 / torch.sqrt(A_tilde.sum(dim=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * torch.linalg.inv(torch.eye(num_nodes).to(device) - (1 - alpha) * H)


def heat_diffusion_matrix(adj_matrix, t = 5.0, add_self_loop = False):
    num_nodes = adj_matrix.shape[0]
    device = adj_matrix.device
    A_tilde = adj_matrix if not add_self_loop else adj_matrix + torch.eye(num_nodes).to(device)
    D_tilde = torch.diag(1 / torch.sqrt(A_tilde.sum(dim=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return torch.matrix_exp(-t * (torch.eye(num_nodes).to(device) - H))


def ss_train(g, features, data_loader, model, diffusion_matrix, optimizer, scheduler, device, start_epoch=0, end_epoch=200, mode='dp', log_interval=20, negative=None, negative_weight=2):
    model.train()

    # training loop
    for epoch in range(start_epoch, end_epoch):
        if negative == 'random':
            negative_features = torch.randn_like(features).to(device)
            print('Random negative features:', negative_features)
        elif negative == 'shuffle':
            negative_features = features[torch.randperm(features.shape[0])]
            # print('Shuffled negative features:', negative_features)
        elif negative == 'original':
            negative_features = features
            print('Shuffled negative features:', negative_features)
        else:
            negative_features = None

        sum_loss, num_batches = 0, 0
        for idx in data_loader:
            target_matrix = diffusion_matrix[idx].to(device)
            weight = None
            if isinstance(negative, str):
                target_matrix = torch.cat([target_matrix, torch.zeros(target_matrix.shape[0], 1).to(device)], dim=1)
                weight = torch.ones(target_matrix.shape[1]).to(device)
                weight[-1] = negative_weight
            loss = F.cross_entropy(model.forward_diffusion(g, features, negative_features=negative_features, idx=idx.to(device), mode=mode), target_matrix, weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            num_batches += 1

        scheduler.step()
        if epoch % log_interval == 0:
            print(f"Self-supervised Epoch {epoch:05d} | Loss {sum_loss / num_batches:.4f}")


def train(g, features, labels, masks, model, classifier, lr=1e-2, min_lr=1e-4, epochs=200, log_interval=20, normalize=False):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[2]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, min_lr / lr, total_iters=epochs, verbose=False)
    max_acc = 0

    model.eval()
    with torch.no_grad():
        model_output = model(g, features).detach()
        if normalize:
            model_output = F.normalize(model_output)

    for epoch in range(epochs):
        logits = classifier(model_output)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        acc = evaluate(labels, val_mask, model_output, classifier)
        max_acc = max(acc, max_acc)
        if epoch % log_interval == 0:
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Accuracy {acc:.4f}")

    return max_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', 'citeseer', 'pubmed', 'ogbn-arxiv'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--rep-size', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--ss-optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--min-lr', type=float, default=1e-4)
    parser.add_argument('--cls-lr', type=float, default=1e-2)
    parser.add_argument('--cls-min-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--ss-epochs', type=int, default=500)
    parser.add_argument('--train-interval', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--t', type=float, default=5.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--mode', type=str, default='dp', choices=['bilinear', 'projected_dp', 'dp', 'cosine'])
    parser.add_argument('--diffusion', type=str, default='heat', choices=['heat', 'ppr'])
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--negative', type=str, default=None, choices=['random', 'shuffle', 'original'])
    parser.add_argument('--negative-weight', type=float, default=2.0)
    parser.add_argument('--activation', type=str, default='leaky_relu', choices=['leaky_relu', 'gelu'])
    parser.add_argument('--last-activation', type=str, default='leaky_relu', choices=['leaky_relu', 'tanh', 'gelu'])
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--norm', action='store_true')

    args = parser.parse_args()

    print(args)
 
    # load and preprocess dataset
    transform = AddSelfLoop()  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == 'cora':
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset(transform=transform)
    elif args.dataset.startswith('ogbn'):
        data = DglNodePropPredDataset(name = args.dataset)
    g = data[0]
    device = torch.device(args.device)
    g = g.int().to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    if args.dataset.startswith('ogbn'):
        split_idx = data.get_idx_split()
        masks = split_idx["train"], split_idx["valid"], split_idx["test"]
    else:
        masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
    print(masks[0].sum(), masks[1].sum(), masks[2].sum())

    # compute heat diffusion matrix
    adj_mat = g.adj().to_dense().to(device)
    diffusion_mat = heat_diffusion_matrix(adj_mat, t=args.t) if args.diffusion == 'heat' else ppr_matrix(adj_mat, alpha=args.alpha)
        
    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5).to(device)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model    
    in_size = features.shape[1]
    out_size = data.num_classes
    activation = {'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}[args.activation]
    last_activation = {'leaky_relu': nn.LeakyReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}[args.last_activation]
    model = GCN(in_size, args.rep_size, depth=args.depth, dropout=args.dropout, activation=activation, last_activation=last_activation, norm=args.norm).to(device)
    print(model)

    dataset = RandIndexDataset(g.num_nodes())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    epoch = 0
    max_accu = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) \
        if args.ss_optimizer == 'adam' else torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, args.min_lr / args.lr, total_iters=args.ss_epochs, verbose=True)
    while epoch < args.ss_epochs:
        print('Self-supervised Training...')
        ss_train(
            g, features, loader, model, diffusion_mat, optimizer, scheduler, device,
            start_epoch=epoch, end_epoch=epoch + args.train_interval, mode=args.mode, log_interval=args.log_interval, negative=args.negative, negative_weight=args.negative_weight,
        )
        epoch += args.train_interval
        
        print('Training...')
        curr_accu = train(
            g, features, labels, masks, model, 
            nn.Linear(args.rep_size, out_size, bias=True).to(device),
            lr=args.cls_lr, min_lr=args.cls_min_lr, epochs=args.epochs, log_interval=args.log_interval, normalize=args.normalize,
        )
        max_accu = max(curr_accu, max_accu)
        print(f'*** Current Test Accuracy {curr_accu:.4f} | Best Test Accuracy {max_accu:.4f} ***\n')
