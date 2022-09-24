import models
import utils
import data_load
from refactoring import saveRefactoringResults

def test(project):
    encoder.eval()
    classifier.eval()
    decoder.eval()

    embed = encoder(features, adj)
    output = classifier(embed, adj)

    print(project, end =":")
    utils.print_metrics(output, labels)

    preds = output.max(1)[1].type_as(labels).cpu().detach()
    calling_strength = decoder(embed)
    calling_strength = utils.sparse_dense_mul(calling_strength, adj)
    saveRefactoringResults(project, preds, calling_strength)

# Training setting
parser = utils.get_parser()
args = parser.parse_args()

encoder = models.Sage_En(nfeat=args.feature_num,
        nhid=args.nhid, nembed=args.nhid, dropout=args.dropout)

classifier = models.Sage_Classifier(nembed=args.nhid,
        nhid=args.nhid, nclass=args.class_num, dropout=args.dropout)

decoder = models.Decoder(nembed=args.nhid, dropout=args.dropout)

project_list = ['binnavi', 'activemq', 'kafka', 'alluxio', 'realm-java']
train_project = project_list[0]

encoder, decoder, classifier = utils.load_model('checkpoint/'+train_project+'.pth', 
    encoder, decoder, classifier)

project_list.remove(train_project)
for project in project_list:
    # Load data
    adj, features, labels = data_load.load_data_custom(project)
    test(project)
    print()