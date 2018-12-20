import pickle
import logging
import argparse
from common.evaluators.evaluator import Evaluator
from common.dataset import DatasetFactory
from common.evaluation import EvaluatorFactory
from utils.serialization import load_checkpoint

def other_evaluate(split_name, dataset_cls, model, embedding, loader, batch_size, device):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    ALL_PAIRS, scores, metric_names = saved_model_evaluator.final_evaluation()
    logger.info('Evaluation metrics for {}'.format(split_name))
    logger.info('\t'.join([' '] + metric_names))
    logger.info('\t'.join([split_name] + list(map(str, scores))))
    return ALL_PAIRS

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name of the dataset to be tested on')
parser.add_argument('--model', help='filename of the pre-trained model, pickle format')
parser.add_argument('--castor-dir', help='directory that saves out_file')
parser.add_argument('--word-vectors-dir', help='word vectors directory')
parser.add_argument('--word-vectors-file', help='word vectors filename')
parser.add_argument('--word-vectors-dim', type=int, default=200, help='number of dimensions of word vectors (default: 200)')
parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training (default: 8)')
parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')

args = parser.parse_args()

with open(args.model, 'rb') as file:  
    pickle_model = pickle.load(file)
print(pickle_model)

model= pickle_model.to(args.device)

print("Loading data...")
dataset_cls, embedding, _, test_loader, _ \
        = DatasetFactory.get_dataset(args.dataset, args.word_vectors_dir, args.word_vectors_file, args.batch_size, args.device, args.castor_dir)
embedding = embedding.to(args.device)
saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, test_loader, args.batch_size, args.device)
#conv_pairs, div_pairs = saved_model_evaluator.DIV_test()
print("EVALUATING.........")
ALL_PAIRS, SCORES, TITLES = saved_model_evaluator.final_evaluation()

print(TITLES)
print(SCORES)

print("Writing to file>>>")
with open(args.model +  "_" + args.dataset + "_classify.txt", 'w') as f_out:
        for pair in ALL_PAIRS:
            SENT_1 = pair[0][0]
            SENT_2 = pair[0][1]
            sco = pair[1]
            lbl = pair[2]
            f_out.write(SENT_1 + '\t' + SENT_2 + '\t' + str(lbl) + '\t' + str(sco) + '\n')
