from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F

from .evaluator import Evaluator

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

class SICKEvaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        num_classes = self.dataset_cls.NUM_CLASSES
        test_kl_div_loss = 0
        predictions = []
        true_labels = []

        for batch in self.data_loader:
            # Select embedding
            sent1, sent2 = self.get_sentence_embeddings(batch)

            output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
            #print(output.detach().exp())
            #print(batch.label)
            test_kl_div_loss += F.kl_div(output, batch.label, size_average=False).item()

            predict_classes = batch.label.new_tensor(torch.arange(1, num_classes + 1)).expand(self.batch_size, num_classes)
            # handle last batch which might have smaller size
            if len(predict_classes) != len(batch.sentence_1):
                predict_classes = batch.label.new_tensor(torch.arange(1, num_classes + 1)).expand(len(batch.sentence_1), num_classes)

            true_labels.append((predict_classes * batch.label.detach()).sum(dim=1))
            predictions.append((predict_classes * output.detach().exp()).sum(dim=1))

            del output

        predictions = torch.cat(predictions)
        true_labels = torch.cat(true_labels)
        mse = F.mse_loss(predictions, true_labels).item()
        test_kl_div_loss /= len(batch.dataset.examples)
        predictions = predictions.cpu().numpy()
        true_labels = true_labels.cpu().numpy()
        pearson_r = pearsonr(predictions, true_labels)[0]
        spearman_r = spearmanr(predictions, true_labels)[0]
        relations = [0,1]

        y_test = list(true_labels)
        pred = list(predictions)

        return [pearson_r, spearman_r, mse, test_kl_div_loss], ['pearson_r', 'spearman_r', 'mse', 'KL-divergence loss']

    def get_final_prediction_and_label(self, batch_predictions, batch_labels):
        num_classes = self.dataset_cls.NUM_CLASSES
        predict_classes = batch_labels.new_tensor(torch.arange(1, num_classes + 1)).expand(batch_predictions.size(0), num_classes)

        predictions = (predict_classes * batch_predictions.exp()).sum(dim=1)
        true_labels = (predict_classes * batch_labels).sum(dim=1)

        return predictions, true_labels

    def final_evaluation(self):
        predicts = []
        true_labels = []
        pair_list = []
        conv_pairs = []
        div_pairs = []
        div_score = []
        for batch in self.data_loader:
            sent1, sent2 = self.get_sentence_embeddings(batch)
            for SENTS in zip(batch.sentence_1_raw,batch.sentence_2_raw):
                pair_list.append(SENTS)
            output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
            #print(output.detach().exp())
            
            for item in output.exp():
                [Conv, Div] = [item[0].item(), item[1].item()]
                if item[0].item() == item.max():
                    predicts.append(1)
                    div_score.append(Conv*100)
                else:
                    predicts.append(0)
                    div_score.append((1 - Div)*100)
            
            for label in batch.label:
                if label[0].item() == label.max():
                    true_labels.append(1)
                else:
                    true_labels.append(0)

        pair_n_score = zip(pair_list,div_score,predicts)

        for k in range(len(predicts)):
            if predicts[k] == 1:
                conv_pairs.append(pair_n_score[k])
            else:
                div_pairs.append(pair_n_score[k])

        all_pairs = conv_pairs + div_pairs
            
 #       conv_pairs.sort(key=lambda tup: tup[1], reverse=True)
#        div_pairs.sort(key=lambda tup: tup[1], reverse=True)
        pair_n_score.sort(key=lambda tup: tup[1], reverse=True)
        
        
        prec = precision_score(true_labels, predicts)
        rec = recall_score(true_labels, predicts)
        f1_sco = f1_score(true_labels, predicts)
        
 #       for k in range(len(predicts)):
#            if predicts[k] == 1:
#                conv_pairs.append(pair_list[k])
#            else:
#                div_pairs.append(pair_list[k])
        
        return pair_n_score, [prec, rec, f1_sco], ['Prec', 'Rec', 'F1']
    
    def DIV_test(self):
        predicts = []
        pair_list = []
        conv_pairs = []
        div_pairs = []
        for batch in self.data_loader:
            sent1, sent2 = self.get_sentence_embeddings(batch)
            for SENTS in zip(batch.sentence_1_raw,batch.sentence_2_raw):
                pair_list.append(SENTS)
            output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
            
            for item in output.exp():
                if item[0].item() == item.max():
                    predicts.append(1)
                else:
                    predicts.append(0)
        
            for k in range(len(predicts)):
                if predicts[k] == 1:
                    conv_pairs.append(pair_list[k])
                else:
                    div_pairs.append(pair_list[k])
        
        return conv_pairs, div_pairs
