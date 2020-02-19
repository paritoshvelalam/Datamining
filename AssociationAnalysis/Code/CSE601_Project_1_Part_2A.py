import itertools
import pandas as pd
import numpy as np
import sys
from collections import defaultdict

data_file = sys.argv[1]
with open(data_file, 'r') as f:  # open the file
    contents = f.readlines()
data_genes = []
frequent_1set = set()


# Geneartes all the combinations of given itemset
def combination_gen(itemsets, r):
    combinations = itertools.combinations(itemsets, r)
    return [set(i) for i in list(combinations)]

# Generate all subsets for rules
def subsets(arr):
    l = []
    for index, value in enumerate(arr):
        l.extend(list(itertools.combinations(arr, index + 1)))
    return l

# Union of sets and select only required length sets
def joinSet(itemSet, length):
    union_list = []
    for i in itemSet:
        for j in itemSet:
            if len(i.union(j)) == length:
                union_list.append(i.union(j))

    return set(union_list)

# parse input data
for line in contents:
    i = 1;
    l1 = line.replace("\n", "").split("\t")
    gene = []
    for att in l1:
        a = "G" + str(i) + "_" + att.replace(" ", "_")
        gene.append(a)
        frequent_1set.add(a)
        i = i + 1
    data_genes.append(gene)
support = (int(sys.argv[2]) / 100)
transactionList = list()
itemSet = set()

# Genearte all 1 length itemsets
for record in data_genes:
    transaction = frozenset(record)
    transactionList.append(transaction)
    for item in transaction:
        itemSet.add(frozenset([item]))

a, b = itemSet, transactionList


# Genearte Frequent itemsets with support >= minsupport
def frequentItemSet(itemSet, transactionList, minSupport, freqSet):
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet


freqSet = defaultdict(int)  # Store support count for itemsets
iset = frequentItemSet(a, b, support, freqSet)
print("number of length-" + str(1) + "  frequent itemsets:" + str(len(iset)))
currentLSet = iset
k = 2
AllFrequentItemsets = dict() # Store all length frequent item sets

# loop to genearte all length frequent itemsets
while (currentLSet != set([])):
    AllFrequentItemsets[k - 1] = currentLSet
    currentLSet = joinSet(currentLSet, k) # joinset to generate length k item sets from length k-1 itemsets
    currentCSet = frequentItemSet(currentLSet, b, support, freqSet) # frequentItemSet to generate frequent item sets with support >= minsupport
    print("number of length-" + str(k) + "  frequent itemsets:" + str(len(currentCSet)))
    currentLSet = currentCSet
    k = k + 1
c = 0
for k, v in AllFrequentItemsets.items():
    c = c + len(v)

print("number of all length frequent itemsets:" + str(c))
