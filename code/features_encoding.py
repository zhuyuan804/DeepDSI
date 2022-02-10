import pandas as pd
import numpy as np
import argparse
import os

pd.set_option('mode.chained_assignment', None)      # Close the warning

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../", help="path storing data.")
parser.add_argument('--species', type=str, default="human", help="which species to use.")
args = parser.parse_args()

print("Start processing data from the Uniprot database...")

print("Importing data...")
uniprot_file = os.path.join(args.data_path, args.species,"data/feature/feature_uniprot.csv")
uniprot = pd.read_csv(uniprot_file)

uniprot = uniprot.rename(columns={'Unnamed: 0':'ID'})

print(uniprot.shape)

print("Sift out proteins with fuzzy amino acid letters in their sequences...")
def find_amino_acid(x):
    return ('B' in x) | ('O' in x) | ('J' in x) | ('U' in x) | ('X' in x) | ('Z' in x)

ambiguous_index = uniprot.loc[uniprot['Sequence'].apply(find_amino_acid)].index
uniprot_ambiguous = uniprot.loc[ambiguous_index]


uniprot.drop(ambiguous_index, axis=0, inplace=True)
uniprot.index = range(len(uniprot))
print(uniprot.shape)

print("Protein sequence was encoded by CT method...")
def CT(sequence):
    classMap = {'G':'1','A':'1','V':'1','L':'2','I':'2','F':'2','P':'8',
            'Y':'3','M':'3','T':'3','S':'3','H':'4','N':'4','Q':'4','W':'4',
            'R':'5','K':'5','D':'6','E':'6','C':'7'}
    # print(sequence)
    seq = ''.join([classMap[x] for x in sequence])  # 数字编码
    length = len(seq)
    coding = np.zeros(343,dtype=int)
    for i in range(length-2):       # 减去最后面两个
        if(int(seq[i])==8 or int(seq[i+1])==8 or int(seq[i+2])==8):
            continue
        index = int(seq[i]) + (int(seq[i+1])-1)*7 + (int(seq[i+2])-1)*49 - 1
        coding[index] = coding[index] + 1

    return coding

CT_list = []
for seq in uniprot['Sequence'].values:
    CT_list.append(CT(seq))
uniprot['features_seq'] = CT_list

uniprot.to_pickle(os.path.join(args.data_path, args.species,"processing", "feature.pkl"))
uniprot.to_csv(os.path.join(args.data_path, args.species,"processing", "feature.csv"))



print("Start generating FAS files for building local BLAST")
def write_fasta(sequence):
    filename = os.path.join(args.data_path, args.species, "blast/uniprot_seq.fas")
    with open(filename, "w") as f:
        for i,row in sequence.iterrows():
            f.write(">" + str(i) + "\n")
            f.write(row['Sequence'] + "\n")

write_fasta(uniprot)

print("The end")

