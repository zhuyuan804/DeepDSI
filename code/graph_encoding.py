import pandas as pd
import argparse
import os

pd.set_option('mode.chained_assignment', None)      # Close the warning

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default="../", help="path storing data.")
parser.add_argument('--species', type=str, default="human", help="which species to use.")
args = parser.parse_args()

print("Start processing local BLAST data...")

blast_file = os.path.join(args.data_path, args.species, "blast/ssn.txt")
blast = pd.read_table(blast_file, delimiter=",")


blast = blast.rename(columns={'0':'protein1', '0.1':'protein2', '100.00':'pident', '402':'length', '0.2':'mismatch','0.3':'gapopen',
                              '1':'qstart', '402.1':'qend', '1.1':'sstart', '402.2':'send','0.0':'evalue', '832':'score'})
print("Sift through and compare yourself...")

blast = blast[blast['protein1'] != blast['protein2']]

print("Generating an SSN network file...")
ssnnetwork = blast[['protein1','protein2','evalue']]
ssnnetwork.to_csv(os.path.join(args.data_path, args.species, "networks/sequence_similar_network.txt"), index=False, header=False, sep="\t")

print('end')

