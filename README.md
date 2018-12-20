# VDPWI_Multilingual
Requirements for VDPWI
- torchtext
- nltk
- pytorch
- numpy

Required arguments
--dataset *the directory containing your dataset with the four files a.toks, b.toks, sim.txt, id.txt
a.toks is your collection of sentences in language a
b.toks is your collection of sentences in language b
sim.txt are the labels for your sentence pairs
id.txt is the enumeration of your dataset examples; a.toks and b.toks need the same number of examples

--word-vectors-dir *the directory containing your word embeddings

--word-vectors-file *the name of the file containing your word embeddings

--castor-dir *the name of the directory to save outputs; also need to contain the directories for your dataset and word embeddings

model_outfile *the name of the file to save intermediate output

Optional arguments
the list of additional optional arguments can be found in __main__.py

Run the code with python __main__.py model_outfile --REQUIRED_ARGUMENTS --OPTIONAL_ARGUMENTS

Saved output is the model saved in pickled format

div_test.py can be used to take an already trained model and generate the list of divergent and non-divergent sentence pairs ordered by score
