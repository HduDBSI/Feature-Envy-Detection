import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import models
import utils
import data_load
import copy
from tqdm import tqdm

def train():
    encoder.train()
    classifier.train()
    decoder.train()
    optimizer_en.zero_grad()
    optimizer_cls.zero_grad()
    optimizer_de.zero_grad()

    # extract embeddings from features
    embed = encoder(features, adj)

    # number of original labels
    ori_num = labels.shape[0]
    
    # generate synthetic nodes
    embed, labels_new, idx_train_new, adj_up = utils.recon_upsample(
        embed, labels, idx_train, adj=adj.detach().to_dense(), portion=args.up_scale, im_class_num=args.im_class_num)
    # restore graph from embeddings
    generated_G = decoder(embed)

    # calculate loss between orginal and new adjacency matrix
    loss_edge = utils.adj_mse_loss(generated_G[:ori_num, :][:, :ori_num], adj.detach().to_dense())
    
    # binarize calling relationship or keep it
    if args.binarize:
        adj_new = copy.deepcopy(generated_G.detach())
        threshold = 0.5
        adj_new[adj_new < threshold] = 0
        adj_new[adj_new >= threshold] = 1
    else:
        adj_new = generated_G

    adj_new = torch.mul(adj_up.to_dense(), adj_new)
    adj_new[:ori_num, :][:, :ori_num] = adj.detach().to_dense()
    adj_new = adj_new.detach()

    output = classifier(embed, adj_new)
    
    loss_node = F.cross_entropy(output[idx_train_new], labels_new[idx_train_new])
    
    if args.loss_settings == 'node_edge':
        loss = loss_node + loss_edge * args.edge_weight
    elif args.loss_settings == 'edge':
        loss = loss_edge
    else:
        loss = loss_node

    loss.backward()
    
    optimizer_en.step()
    optimizer_cls.step()
    optimizer_de.step()

    F1 = utils.print_metrics(output[idx_val], labels[idx_val], isPrint=False)

    return F1

# for test
from refactoring import saveRefactoringResults
def test():
    encoder.eval()
    classifier.eval()
    decoder.eval()

    embed = encoder(features, adj)
    output = classifier(embed, adj)
    
    utils.print_metrics(output[idx_test], labels[idx_test])

    preds = output.max(1)[1].type_as(labels).cpu().detach()
    calling_strength = decoder(embed)
    calling_strength = utils.sparse_dense_mul(calling_strength, adj)
    saveRefactoringResults(train_project, preds, calling_strength)


# Training setting
parser = utils.get_parser()
args = parser.parse_args()


# Model and optimizer
encoder = models.Sage_En(nfeat=args.feature_num,
        nhid=args.nhid, nembed=args.nhid, dropout=args.dropout)

classifier = models.Sage_Classifier(nembed=args.nhid,
        nhid=args.nhid, nclass=args.class_num, dropout=args.dropout)

decoder = models.Decoder(nembed=args.nhid, dropout=args.dropout)

optimizer_en = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_cls = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_de = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)


project_list = ['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java']
train_project = project_list[0] # select project

# load data
adj, features, labels = data_load.load_data_custom(train_project)

# get train, validatio, test data split
idx_train, idx_val, idx_test = utils.split_arti(labels)

encoder = encoder.cuda()
classifier = classifier.cuda()
decoder = decoder.cuda()

features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()

t_start = time.time()
F1_best = 40
pbar = tqdm(range(args.epochs), total=args.epochs, leave=False, 
            ncols=100, unit="epoch", unit_scale=False, colour="red")

for epoch in enumerate(pbar):
    F1_tmp = train() 
    if F1_tmp > F1_best:
        model_best = utils.keep_model(encoder, decoder, classifier)
        F1_best = F1_tmp
    pbar.set_postfix({"best F1 in val": F1_best})

utils.save_model(model_best, 'checkpoint/'+train_project+'.pth')
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

# Testing
encoder.load_state_dict(model_best['encoder'])
decoder.load_state_dict(model_best['decoder'])
classifier.load_state_dict(model_best['classifier'])
test()