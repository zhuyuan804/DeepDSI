# DeepDSI
Deubiquitinating enzyme-substrate interaction prediction model

1.Human protein and sequence from Uniprot.

2.python features_encoding.py  CT encoding protein sequence and a .fas file that be used to BLAST to generate a sequential similarity network.

3.Run BLAST on a Linux system    input:uniprot_seq.fas  output:ssn.txt

3.dsi_train_test.ipynb The training set(before 2018.1) and test set(after 2018.1) were divided according to the publication time of the data source literature  .

4.python graph_encoding.py input:ssn.txt output:sequence_similar_network.txt  

5.python main.py input:feature.pkl and sequence_similar_network.txt  output:predicted new deubiquitinating enzyme-substrate interactions

ssn.txt ppi.txt and ppi_entry.txt can be download from https://pan.baidu.com/s/1RbpmOHwda5JDRsDsjIHPjQ password:x514
