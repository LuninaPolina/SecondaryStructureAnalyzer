import sys

import numpy as np

if __name__ == "__main__":
    # parse args to choose type of data
    if sys.argv[-1] == '--balance':
        outType = "balance"
        train_out = open("trainBalancedHuge.csv", "w")
        valid_out = open("validBalancedHuge.csv", "w")
        test_out = open("testBalancedHuge.csv", "w")
        db_out = open("dbBalancedHuge.csv", "w")
    else:
        outType = "all"
        train_out = open("trainAll.csv", "w")
        valid_out = open("validAll.csv", "w")
        test_out = open("testAll.csv", "w")
        db_out = open("dbAll.csv", "w")

    totalProt = 2423
    totalAnim = 3000

    # calc amount of samples in sets
    validAnim = 250
    testAnim = int(0.25 * totalAnim)
    trainAnim = totalAnim - testAnim - validAnim

    validProt = int(0.1 * totalProt)
    testProt = int(0.25 * totalProt)
    trainProt = totalProt - testProt - validProt

    data = open("data.csv", "r")
    db = open("db.csv", "r")

    db_other = open("ref_db.csv", "r")
    train_other = open("train.csv", "r")
    valid_other = open("valid.csv", "r")
    test_other = open("test.csv", "r")

    dataLines = np.array(data.readlines())
    dbLines = db.readlines()

    anim_indices = []
    prot_indices = []
    start_index = int(dbLines[0].split(",")[0])

    for line in dbLines:
        splitted = line.split(",")
        if "prot" in splitted[2]:
            prot_indices.append(int(splitted[0]))
        else:
            anim_indices.append(int(splitted[0]))

    # order in db and data is the same so we can use index as position in dataLines
    prot_indices = np.array(prot_indices)
    anim_indices = np.array(anim_indices)
    prot_indices -= start_index
    anim_indices -= start_index

    np.random.shuffle(prot_indices)
    np.random.shuffle(anim_indices)

    train_lines = []
    valid_lines = []
    test_lines = []

    # divide animals to sets
    anim_indices = anim_indices[:totalAnim]
    trainAnimIndices = anim_indices[:trainAnim]
    validAnimIndices = anim_indices[trainAnim:(trainAnim + validAnim)]
    testAnimIndices = anim_indices[(trainAnim + validAnim):(trainAnim + validAnim + testAnim)]
    train_lines.extend(dataLines[trainAnimIndices])
    valid_lines.extend(dataLines[validAnimIndices])
    test_lines.extend(dataLines[testAnimIndices])

    # divide protists to sets
    prot_indices = prot_indices[:totalProt]
    trainProtIndices = prot_indices[:trainProt]
    validProtIndices = prot_indices[trainProt:(trainProt + validProt)]
    testProtIndices = prot_indices[(trainProt + validProt):(trainProt + validProt + testProt)]
    train_lines.extend(dataLines[trainProtIndices])
    valid_lines.extend(dataLines[validProtIndices])
    test_lines.extend(dataLines[testProtIndices])

    # mark data in db as in set or not
    for i in range(len(dbLines)):
        #  drop line end symbol
        dbLines[i] = dbLines[i][:-1]
        if i in trainAnimIndices or i in trainProtIndices:
            dbLines[i] += ',train\n'
        elif i in validAnimIndices or i in validProtIndices:
            dbLines[i] += ',valid\n'
        elif i in testAnimIndices or i in testProtIndices:
            dbLines[i] += ',test\n'
        else:
            dbLines[i] += ',none\n'

    # duplicate protist data in train for balance
    if outType == "balance":
        extension_lines = []
        repeat_count = trainAnim // trainProt
        if repeat_count > 1:
            extension_lines.extend(np.repeat(dataLines[trainProtIndices], repeat_count - 1))
        delta = trainAnim - trainProt * repeat_count
        if delta > 0:
            extension_lines.extend(np.random.choice(dataLines[trainProtIndices], delta))
        train_lines.extend(extension_lines)

    # add old data
    train_other_lines = train_other.readlines()
    train_other_lines[-1] += '\n'
    train_other_lines.extend(train_lines)
    train_lines = train_other_lines

    valid_other_lines = valid_other.readlines()
    valid_other_lines[-1] += '\n'
    valid_other_lines.extend(valid_lines)
    valid_lines = valid_other_lines

    test_other_lines = test_other.readlines()
    test_other_lines[-1] += '\n'
    test_other_lines.extend(test_lines)
    test_lines = test_other_lines

    db_other_lines = db_other.readlines()
    # all lines has but this don't
    db_other_lines[-1] += '\n'
    db_other_lines.extend(dbLines)
    dbLines = db_other_lines

    # shuffle data
    np.random.shuffle(train_lines)
    np.random.shuffle(valid_lines)
    np.random.shuffle(test_lines)

    # drop las line for csv format
    train_lines[-1] = train_lines[-1][:-1]
    valid_lines[-1] = valid_lines[-1][:-1]
    test_lines[-1] = test_lines[-1][:-1]
    dbLines[-1] = dbLines[-1][:-1]

    # save data
    train_out.writelines(train_lines)
    valid_out.writelines(valid_lines)
    test_out.writelines(test_lines)
    db_out.writelines(dbLines)

    db.close()
    data.close()
    train_out.close()
    valid_out.close()
    test_out.close()
    db_out.close()
    train_other.close()
    valid_other.close()
    test_other.close()




