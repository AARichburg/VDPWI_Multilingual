import random
import argparse
import os
import pickle

#Code is an amalgamation of generate_synthetic_data.py from Yogarshi and others by me
#Generates synthetic examples for training

def fraction_of_words(s1,s2,trans_dict):
	count = 0
	for w1 in s1:
		try:
			for each_w in trans_dict[w1]:
				if each_w in s2:
					count += 1
					break
		except KeyError:
			continue
	return count

def word_filter(sp, dict_e2f, dict_f2e, len_ratio=2., trans_ratio=.5):
	"""
	Returns true for a sentence pair if it satisfies certain word overlap criteria
	:param sp: 			tuple of sentence pairs
	:param dict_e2f:	Translation dictionary from english to french
	:param dict_f2e:	Translation dictionary from french to english
	:param len_ratio: 	The maximum ratio between the lengths of the two sentences
	:param trans_ratio:	The minimum fraction of words that have to have a translation in the other sentence
	:return: true if both criteria are satisfied, else false
	"""

	s1, s2 = sp[0].split(), sp[1].split()
	l1, l2 = len(s1), len(s2)

	# Check length ratio
	if l1 > 2 * l2 or l2 > 2 * l1:
		return False

	# Check overlap
	if fraction_of_words(s1, s2, dict_e2f) * 1.0 / l1 < trans_ratio:
		return False

	if fraction_of_words(s2, s1, dict_f2e) * 1.0 / l2 < trans_ratio:
		return False

	return True

def neg_exam_gen(en_sents, fr_sents, dict_e2f, dict_f2e, ratio=5, gold=None):
    print "Creating negative examples ..."
    positive_pairs = zip(en_sents, fr_sents)
    negative_pairs = set()
    count = 0
    # Hacky way to generate balanced negative examples - sample a positive
	# sentence, a negative sentence. With high probability, we will not have
	# seen this sentence pair earlier. If it satisfies our criteria for a good
	# negative example, add it to our set. If not, discard and repeat.
    while len(negative_pairs) < ratio * len(positive_pairs):
        if count % 10000 == 0:
            print count
        count += 1
        sp = (random.choice(en_sents), random.choice(fr_sents))
        if sp in positive_pairs:
            continue
        if not word_filter(sp, dict_e2f, dict_f2e):
            continue
        if gold != None:
            if sp in gold:
                continue
        negative_pairs.add(sp)
        negative_paar = list(negative_pairs)
    print("Made negative pairs")
    pos_lbls = [1 for j in positive_pairs]
    neg_lbls = [0 for j in negative_pairs]
    return positive_pairs, negative_paar, pos_lbls, neg_lbls

def data_shuffler(a_list, b_list,sub=False,amo=300):
    indices = range(len(a_list))
    random.shuffle(indices)
    shuffled = [a_list[k] for k in indices]
    shuffled_lbls = [b_list[j] for j in indices]
    return shuffled, shuffled_lbls
        
parser = argparse.ArgumentParser()
parser.add_argument("--english_sent_file", help="English language file")
parser.add_argument("--foreign_sent_file", help="Foreign language file")
parser.add_argument("--data_dir", help="Directory to contain files")
parser.add_argument("--seed",default=1,type=int)
parser.add_argument("--true_test_path",default=None)
args = parser.parse_args()

def main():
    eng_file = args.english_sent_file
    for_file = args.foreign_sent_file
    sents_e = []
    sents_f = []
    print("Reading files...")
    with open(eng_file, 'r') as f_in_e:
            for lines in f_in_e:
                    sents_e.append(lines)
 #       text_e = f_in_e.read()
 #       sents_e = text_e.splitlines()
    with open(for_file, 'r') as f_in_f:
            for lines in f_in_f:
                    sents_f.append(lines)
#        text_f = f_in_f.read()
#        sents_f = text_f.splitlines()
        
    print("Collecting random sentence pairs...")       
    total = len(sents_e)
    print(total)
    print(len(sents_f))
    assert len(sents_e) == len(sents_f)
    indices = list(range(total))
    list_of_random_items = random.sample(indices, 5250)
    
    ind_dev = random.sample(list_of_random_items, 50)
    list_of_random_items = [i for i in list_of_random_items if i not in ind_dev]
    ind_test = random.sample(list_of_random_items, 200)
    ind_train = [i for i in list_of_random_items if i not in ind_test]

    #Training list
    sent_e_train = [sents_e[i] for i in ind_train]
    sent_f_train = [sents_f[i] for i in ind_train]
    print(len(sent_e_train))
    
    #Dev list
    sent_e_dev = [sents_e[j] for j in ind_dev]
    sent_f_dev = [sents_f[j] for j in ind_dev]
    print(len(sent_e_dev))
    
    #Test list
    sent_e_test = [sents_e[k] for k in ind_test]
    sent_f_test = [sents_f[k] for k in ind_test]
    print(len(sent_e_test))

	###
	# "data_dir" should contain the following files
	# dict.e2f		: 	Translation dictionary from English to French
	# dict.f2e		: 	Translation dictionary from French to English
	###

        # Set random seed for replicability - 42
    random.seed(args.seed)

	# Load dictionaries
    print "Loading dictionaries.."
    dict_e2f_path = os.path.join(args.data_dir, "dict.e2z")
    dict_f2e_path = os.path.join(args.data_dir, "dict.z2e")
    dict_e2f = pickle.load(open(dict_e2f_path))
    print "Loaded e2f dict with {0} entries".format(len(dict_e2f))
    dict_f2e = pickle.load(open(dict_f2e_path))
    print "Loaded f2e dict with {0} entries".format(len(dict_f2e))

	# Load positive training data
	#en_sents = [x.strip().replace('|',' ') for x in
	#			open(os.path.join(args.train_dir, "en_pos.tok.lc"))]
	#fr_sents = [x.strip().replace('|',' ') for x in
	#			open(os.path.join(args.train_dir, "fr_pos.tok.lc"))]
    en_sents = sent_e_train
    fr_sents = sent_f_train
    assert len(en_sents) == len(fr_sents)

    #Create negative examples
    gold_pairs = None
    if args.true_test_path != None:
            eng_gold_file = os.path.join(args.true_test_path,"a.toks")
            for_gold_file = os.path.join(args.true_test_path,"b.toks")
            sen_g_e = []
            sen_g_f = []
            with open(eng_gold_file, 'r') as f_g_e, open(for_gold_file, 'r') as f_g_f:
                    for lines in f_g_f:
                            sen_g_f.append(lines)
                    for lines in f_g_e:
                            sen_g_e.append(lines)
            gold_pairs = zip(sen_g_e, sen_g_f)
            print("Making sure no gold test overlap...")
    pos_train, neg_train, pos_lbl_train, neg_lbl_train = neg_exam_gen(sent_e_train, sent_f_train, dict_e2f, dict_f2e,gold=gold_pairs, ratio=5)
    pos_dev, neg_dev, pos_lbl_dev, neg_lbl_dev = neg_exam_gen(sent_e_dev, sent_f_dev, dict_e2f, dict_f2e,gold=gold_pairs)

    pos_test, neg_test, pos_lbl_test, neg_lbl_test = neg_exam_gen(sent_e_test, sent_f_test, dict_e2f, dict_f2e, ratio=0.8, gold=gold_pairs)
    
    #Shuffle data
    train_set_pairs, train_lbls = data_shuffler(pos_train+neg_train,pos_lbl_train+neg_lbl_train)
    dev_set_pairs, dev_lbls = data_shuffler(pos_dev+neg_dev,pos_lbl_dev+neg_lbl_dev)
    test_set_pairs, test_lbls = data_shuffler(pos_test+neg_test, pos_lbl_test+neg_lbl_test)

    print("Writing to file...")

    with open(os.path.join(args.data_dir, "train", "a.toks"), 'w') as fout1, \
         open(os.path.join(args.data_dir, "train", "b.toks"), 'w') as fout2, \
         open(os.path.join(args.data_dir, "train", "sim.txt"), 'w') as fout3:
            for en_sent, fr_sent in train_set_pairs:
                fout1.write(en_sent)
 #               fout1.write("\n")
                fout2.write(fr_sent)
#                fout2.write("\n")
            for label in train_lbls:
                fout3.write(str(label) + '\n')
                    
    with open(os.path.join(args.data_dir, "dev", "a.toks"), 'w') as fout1, \
         open(os.path.join(args.data_dir, "dev", "b.toks"), 'w') as fout2, \
         open(os.path.join(args.data_dir, "dev", "sim.txt"), 'w') as fout3:
            for en_sent, fr_sent in dev_set_pairs:
                fout1.write(en_sent)
#                fout1.write("\n")
                fout2.write(fr_sent)
#                fout2.write("\n")
            for label in dev_lbls:
                fout3.write(str(label) + '\n')
                
    with open(os.path.join(args.data_dir, "test", "a.toks"), 'w') as fout1, \
         open(os.path.join(args.data_dir, "test", "b.toks"), 'w') as fout2, \
         open(os.path.join(args.data_dir, "test", "sim.txt"), 'w') as fout3:
            for en_sent, fr_sent in test_set_pairs:
                fout1.write(en_sent)
#                fout1.write("\n")
                fout2.write(fr_sent)
#                fout2.write("\n")
            for label in test_lbls:
                fout3.write(str(label) + '\n')

    train_id = range(len(train_set_pairs))
    dev_id = range(len(dev_set_pairs))
    test_id = range(len(test_set_pairs))

    with open(os.path.join(args.data_dir, "train", "id.txt"), 'w') as fout1, \
         open(os.path.join(args.data_dir, "dev", "id.txt"), 'w') as fout2, \
         open(os.path.join(args.data_dir, "test", "id.txt"), 'w') as fout3:
	    for index in dev_id:
                fout2.write(str(index)+'\n')
	    for index in test_id:
                fout3.write(str(index)+'\n')
	    for index in train_id:
                fout1.write(str(index)+'\n')

if __name__ == "__main__":
        main()
