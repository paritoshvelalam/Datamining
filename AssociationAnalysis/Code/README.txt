README

=========================================

Dependencies:
Python 3.6.1
Pandas 0.20.1
Itertools

For CSE601_Project_1_Part_2A

Execution in command line: 

python3 CSE601_Project_1_Part_2A [filename] [support percent]

Example:

python3 CSE601_Project_1_Part_2A associationruletestdata.txt 50

---------------------------------------------------------------------------------------------

For CSE601_Project_1_Part_2B

Execution in command line: 

python3 CSE601_Project_1_Part_2B [filename] [support percent] [confidence percent]

Example:

python3 CSE601_Project_1_Part_2B associationruletestdata.txt 50 70

For Querying:

Type in query as per given format. Type "exit" to quit program.

Query format examples:

asso_rule.template1("RULE", "ANY", ['G59_Up'])
asso_rule.template1("RULE", "NONE", ['G59_Up'])
asso_rule.template1("RULE", 1, ['G59_Up', 'G10_Down'])
asso_rule.template1("HEAD", "ANY", ['G59_Up'])
asso_rule.template1("HEAD", "NONE", ['G59_Up'])
asso_rule.template1("HEAD", 1, ['G59_Up', 'G10_Down'])
asso_rule.template1("BODY", "ANY", ['G59_Up'])
asso_rule.template1("BODY", "NONE", ['G59_Up'])
asso_rule.template1("BODY", 1, ['G59_Up', 'G10_Down'])
asso_rule.template2("RULE", 3)
asso_rule.template2("HEAD", 2)
asso_rule.template2("BODY", 1)
asso_rule.template3("1or1", "HEAD", "ANY",['G10_Down'], "BODY", 1, ['G59_Up'])
asso_rule.template3("1and1", "HEAD", "ANY",['G10_Down'], "BODY", 1, ['G59_Up'])
asso_rule.template3("1or2", "HEAD", "ANY",['G10_Down'], "BODY", 2)
asso_rule.template3("1and2", "HEAD", "ANY",['G10_Down'], "BODY", 2)
asso_rule.template3("2or2", "HEAD", 1, "BODY", 2)
asso_rule.template3("2and2", "HEAD", 1, "BODY", 2)


Notes:
support and confidence must be given in perecentages
"Up" and "Down" should be entered in sentence case and not in "UpPER" or "lower" case.