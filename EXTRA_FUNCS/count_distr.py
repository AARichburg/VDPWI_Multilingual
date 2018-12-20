import argparse
import matplotlib.pyplot as plt
import numpy as np

#Calculate distribution of labeled classified pairs
parser = argparse.ArgumentParser()
parser.add_argument('--the_path')
parser.add_argument('--filename')
args = parser.parse_args()

def main():
    with open(args.the_path + args.filename, 'r') as in_file:
        scores = []
        lbls = []
        for line in in_file:
            pieces = line.split('\t')
            ENG = pieces[0]
            FOR = pieces[1]
            LBL = pieces[2]
            SCO = pieces[3]
            scores.append(float(SCO))
            lbls.append(int(LBL))

        #LABEL COUNT
        CONV = 0
        DIV = 0
        for label in lbls:
            if label == 0:
                DIV += 1
            else:
                CONV += 1

        #DISTRIBUTION OF SCORES
        dist = {}
        dist[str(1.0)] = 0
        dist[str(0.9)] = 0
        dist[str(0.8)] = 0
        dist[str(0.7)] = 0
        dist[str(0.6)] = 0
        dist[str(0.5)] = 0
        dist[str(0.4)] = 0
        dist[str(0.3)] = 0
        dist[str(0.2)] = 0
        dist[str(0.1)] = 0
        dist[str(0.0)] = 0
        for num in scores:
            if num < 10:
                dist[str(0.0)] += 1
            elif num < 20 and num >= 10:
                dist[str(0.1)] += 1
            elif num < 30 and num >=20:
                dist[str(0.2)] +=1
            elif num < 40 and num >= 30:
                dist[str(0.3)] +=1
            elif num < 50 and num >= 40:
                dist[str(0.4)] +=1
            elif num < 60 and num >=50:
                dist[str(0.5)] += 1
            elif num < 70 and num >=60:
                dist[str(0.6)] +=1
            elif num < 80 and num >= 70:
                dist[str(0.7)] +=1
            elif num < 90 and num >= 80:
                dist[str(0.8)] +=1
            elif num < 100 and num >= 90:
                dist[str(0.9)] +=1
            else:
                dist[str(1.0)] +=1
                
    total = float(len(scores))
    A = float(dist[str(1.0)])
    B = float(dist[str(0.9)])
    C = float(dist[str(0.8)])
    D = float(dist[str(0.7)])
    E = float(dist[str(0.6)])
    F = float(dist[str(0.5)])
    G = float(dist[str(0.4)])
    H = float(dist[str(0.3)])
    I = float(dist[str(0.2)])
    J = float(dist[str(0.1)])
    K = float(dist[str(0.0)])

    #plt.plot(range(len(scores)), scores, 'r-')
    #plt.yscale('log')
    #plt.title('log')
    #plt.grid(True)
    #plt.show()
    SCORES = ('0-9','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0')
    AMTS = [K, J, I, H, G, F, E, D, C, B, A]
    y_pos = np.arange(len(SCORES))
    plt.bar(y_pos, AMTS)
    plt.xticks(y_pos, SCORES)
    plt.savefig(args.filename + '.png')
    plt.show()
    
    dist[str(1.0)] = (dist[str(1.0)], float(A/total))
    dist[str(0.9)] = (dist[str(0.9)], float(B/total))
    dist[str(0.8)] = (dist[str(0.8)], float(C/total))
    dist[str(0.7)] = (dist[str(0.7)], float(D/total))
    dist[str(0.6)] = (dist[str(0.6)], float(E/total))
    dist[str(0.5)] = (dist[str(0.5)], float(F/total))
    dist[str(0.4)] = (dist[str(0.4)], float(G/total))
    dist[str(0.3)] = (dist[str(0.3)], float(H/total))
    dist[str(0.2)] = (dist[str(0.2)], float(I/total))
    dist[str(0.1)] = (dist[str(0.1)], float(J/total))
    dist[str(0.0)] = (dist[str(0.0)], float(K/total))
    with open(args.the_path + "DISTR_" + args.filename, 'w') as out_file:
        out_file.write("TOTAL: " + str(CONV+DIV) + '\n')
        out_file.write("CONV: " + str(CONV) + '\n')
        out_file.write("DIV: " + str(DIV) + '\n')
        for key in dist:
            out_file.write(str(key) + '\t' + str(dist[key]) + '\n')

if __name__ == "__main__":
        main()
