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

# minsupport
support = (int(sys.argv[2]) / 100)

# minconfidence
minConfidence = int(sys.argv[3]) / 100
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

# Generate length 1 frequent item sets
freqSet = defaultdict(int)  # Store support count for itemsets
iset = frequentItemSet(a, b, support, freqSet)
#print("number of length-" + str(1) + "  frequent itemsets:" + str(len(iset)))
currentLSet = iset
k = 2
AllFrequentItemsets = dict()  # Store all length frequent item sets

# loop to genearte all length frequent itemsets
while (currentLSet != set([])):
    AllFrequentItemsets[k - 1] = currentLSet
    currentLSet = joinSet(currentLSet, k)       # joinset to generate length k item sets from length k-1 itemsets
    currentCSet = frequentItemSet(currentLSet, b, support, freqSet)     # frequentItemSet to generate frequent item sets with support >= minsupport
    #print("number of length-" + str(k) + "  frequent itemsets:" + str(len(currentCSet)))
    currentLSet = currentCSet
    k = k + 1

c = 0
for k, v in AllFrequentItemsets.items():
    c = c + len(v)

#print("number of all length frequent itemsets:" + str(c))

# Generate Rule with Confidence >= minconfidence
toRetRules = []
c = 0
for key, value in AllFrequentItemsets.items():
    for item in value:
        _subsets = map(frozenset, [x for x in subsets(item)])
        for element in _subsets:
            remain = item.difference(element)                         # Obtain BODY
            if len(remain) > 0:
                f1 = float(freqSet[item]) / len(transactionList)      # Support of RULE
                f2 = float(freqSet[element]) / len(transactionList)   # Support of HEAD
                if (f2 == 0):
                    print(freqSet[element])
                    print(item, element)
                confidence = f1 / f2
                if confidence >= minConfidence:
                    c = c + 1
                    toRetRules.append((str(list(item)), str(list(element)), str(list(remain)),
                                       str(confidence)))
dfObj = pd.DataFrame(toRetRules, columns=['RULE', 'HEAD', 'BODY', 'CONFIDENCE']) # Rules dataframe 
print("total number of rules generated   " + str(len(dfObj)))

# Write rules to a .csv file
dfObj.to_csv('Rules.csv', sep=',')         
print(str(len(dfObj)) + " rules generated.")


def queryInput(query):
    if (query[:19] == 'asso_rule.template1'):
        Result = Template1(eval(query[19:]))    # Obtain Template 1 result
    elif (query[:19] == 'asso_rule.template2'):
        Result = Template2(eval(query[19:]))    # Obtain Template 2 result
    elif (query[:19] == 'asso_rule.template3'):
        Result = Template3(eval(query[19:]))    # Obtain Template 3 result
    return Result.drop_duplicates()

# Template1 Query
def Template1(query):
    Result = pd.DataFrame(data=None, columns=dfObj.columns)

    if (query[0] == "RULE" and query[1] == "ANY"):
        for item in query[2]:
            x = dfObj[dfObj['RULE'].str.contains(item)]
            Result = Result.append(x)
    if (query[0] == "RULE" and query[1] == "NONE"):
        Result = dfObj.copy()
        for item in query[2]:
            x = ~Result['RULE'].str.contains(item)
            Result = Result[x]
    if (query[0] == "RULE" and query[1] == 1):
        for item in query[2]:
            x = dfObj[dfObj['RULE'].str.contains(item)]
            Result = Result.append(x)
        rem = combination_gen(set(query[2]), 2)
        for rem_item in rem:
            x = ~Result['RULE'].str.contains(str(rem_item)[1:-1])
            Result = Result[x]

    if (query[0] == "BODY" and query[1] == "ANY"):
        for item in query[2]:
            x = dfObj[dfObj['BODY'].str.contains(item)]
            Result = Result.append(x)
    if (query[0] == "BODY" and query[1] == "NONE"):
        Result = dfObj.copy()
        for item in query[2]:
            x = ~Result['BODY'].str.contains(item)
            Result = Result[x]
    if (query[0] == "BODY" and query[1] == 1):
        for item in query[2]:
            x = dfObj[dfObj['BODY'].str.contains(item)]
            Result = Result.append(x)
        rem = combination_gen(set(query[2]), 2)
        for rem_item in rem:
            x = ~Result['BODY'].str.contains(str(rem_item)[1:-1])
            Result = Result[x]

    if (query[0] == "HEAD" and query[1] == "ANY"):
        for item in query[2]:
            x = dfObj[dfObj['HEAD'].str.contains(item)]
            Result = Result.append(x)
    if (query[0] == "HEAD" and query[1] == "NONE"):
        Result = dfObj.copy()
        for item in query[2]:
            x = ~Result['HEAD'].str.contains(item)
            Result = Result[x]
    if (query[0] == "HEAD" and query[1] == 1):
        for item in query[2]:
            x = dfObj[dfObj['HEAD'].str.contains(item)]
            Result = Result.append(x)
        rem = combination_gen(set(query[2]), 2)
        for rem_item in rem:
            x = ~Result['HEAD'].str.contains(str(rem_item)[1:-1])
            Result = Result[x]
    return Result.drop_duplicates()

# Template2 Query
def Template2(query):
    Result = pd.DataFrame(data=None, columns=dfObj.columns)
    if (query[0] == "RULE"):
        for i in range(len(dfObj)):
            if ((len(eval(dfObj['BODY'].iloc[i])) + len(eval(dfObj['HEAD'].iloc[i]))) >= query[1]):
                Result = Result.append(dfObj.iloc[i])
    elif (query[0] == "BODY"):
        for i in range(len(dfObj)):
            if ((len(eval(dfObj['BODY'].iloc[i]))) >= query[1]):
                Result = Result.append(dfObj.iloc[i])
    elif (query[0] == "HEAD"):
        for i in range(len(dfObj)):
            if ((len(eval(dfObj['HEAD'].iloc[i]))) >= query[1]):
                Result = Result.append(dfObj.iloc[i])
    return Result.drop_duplicates()

# Template3 Query
def Template3(query):
    print(query)
    Result = pd.DataFrame(data=None, columns=dfObj.columns)
    r1 = pd.DataFrame(data=None, columns=dfObj.columns)
    r2 = pd.DataFrame(data=None, columns=dfObj.columns)
    if (query[0] == "1or1"):
        r1 = r1.append(Template1(query[1:4]))
        r1 = r1.append(Template1(query[4:7]))
        Result = Result.append(r1)
    elif (query[0] == "1and1"):
        r1 = r1.append(Template1(query[1:4]))
        r2 = r2.append(Template1(query[4:7]))
        Result = pd.merge(r1, r2, how='inner',
                               on=['RULE', 'HEAD', 'BODY', 'CONFIDENCE'])
    elif (query[0] == "1or2"):
        r1 = r1.append(Template1(query[1:4]))
        r1 = r1.append(Template2(query[4:6]))
        Result = Result.append(r1)
    elif (query[0] == "1and2"):
        r1 = r1.append(Template1(query[1:4]))
        r2 = r2.append(Template2(query[4:6]))
        Result = pd.merge(r1, r2, how='inner',
                               on=['RULE', 'HEAD', 'BODY', 'CONFIDENCE'])
    elif (query[0] == "2or2"):
        r1 = r1.append(Template2(query[1:3]))
        r1 = r1.append(Template2(query[3:5]))
        Result = Result.append(r1)
    elif (query[0] == "2and2"):
        r1 = r1.append(Template2(query[1:3]))
        r2 = r2.append(Template2(query[3:5]))
        Result = pd.merge(r1, r2, how='inner',
                               on=['RULE', 'HEAD', 'BODY', 'CONFIDENCE'])
    return Result.drop_duplicates()

# Query Input
while (True):
    print("Enter the Query")
    q = input("> ")
    try:
        if (q.lower() == "exit"):
            break
        else:
            r = queryInput(q)
            print(r[['HEAD', 'BODY']])
            print("rows retreived " + str(len(r)) + "\n")
    except:
        print("Enter correct query.\n")



