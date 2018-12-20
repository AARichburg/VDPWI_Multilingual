import argparse
import os
import re

#Formats test sets for model out of gold-labeled paired files

parser = argparse.ArgumentParser()
parser.add_argument("--english_sent_file", help="English language file")
parser.add_argument("--foreign_sent_file", help="Foreign language file")
parser.add_argument("--test_dir", help="directory containg test file")
args = parser.parse_args()
def main():
    eng_file = os.path.join(args.test_dir, args.english_sent_file)
    for_file = os.path.join(args.test_dir, args.foreign_sent_file)
    sents_e = []
    sents_f = []
    print("Reading files..")
    with open(eng_file, 'r') as f_in_e, open(for_file, 'r') as f_in_f:
        for lines in f_in_e:
            strip_line = re.sub("<.*?>"," ",lines)
            strip_line = strip_line.lstrip()
            sents_e.append(strip_line)
        for lines in f_in_f:
            strip_line = re.sub("<.*?>"," ",lines)
            strip_line = strip_line.lstrip()
            sents_f.append(strip_line)

    ids = range(len(sents_e))
    sim = [1 for k in ids]
    
    print("Writing files...")
    with open(os.path.join(args.test_dir, "a.toks"), 'w') as fout1, \
         open(os.path.join(args.test_dir, "b.toks"), 'w') as fout2, \
         open(os.path.join(args.test_dir, "id.txt"), 'w') as fout3, \
         open(os.path.join(args.test_dir, "sim.txt"), 'w') as fout4:
        for en_sent in sents_e:
            fout1.write(en_sent)
        for fr_sent in sents_f:
            fout2.write(fr_sent)
        for index in ids:
            fout3.write(str(index)+'\n')
        for dummy in sim:
            fout4.write(str(dummy)+'\n')

if __name__ == "__main__":
    main()
