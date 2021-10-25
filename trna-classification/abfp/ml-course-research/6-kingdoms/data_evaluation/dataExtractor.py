# a - archea
# b- bacteria
# p - plant
# f - fungi
# prot - protist
# anim - animal


# fix length of sequence to 220
def resolve_tRNA(tRNA):
    if len(tRNA) > 220:
        fixed_size_tRNA = tRNA[:220]
    elif len(tRNA) < 220:
        fixed_size_tRNA = tRNA + "D" * (220 - len(tRNA))
    else:
        fixed_size_tRNA = tRNA
    return ",".join([x for x in fixed_size_tRNA])


if __name__ == "__main__":
    input = open("eukaryotic-trnas.fa", "r")
    input2 = open("protist_AND_rna_typetRNA.fasta", "r")
    output_db = open("db.csv", "w")
    output_data = open("data.csv", "w")

    startId = 540189
    id = startId

    tRNA_with_id = ""
    tRNA_set = {}

    skip_list = [
        "Aspergillus_fumigatus",
        "Debaryomyces_hansenii",
        "Kluyveromyces_lactis",
        "Saccharomyces_cerevisiae",
        "Candida_glabrata",
        "Encephalitozoon_cuniculi",
        "Magnaporthe_oryzae",
        "Schizosaccharomyces_pombe",
        "Cryptococcus_neoformans_var",
        "Eremothecium_gossypii",
        "Saccharomyces_cerevisiae",
        "Yarrowia_lipolytica",
        "Brachypodium_distachyon",
        "Vitis_vinifera",
        "Physcomitrella_patens_scaffold",
        "Zea_mays",
        "Sorghum_bicolor",
        "Arabidopsis_thaliana",
        "Medicago_truncatula",
        "Glycine_max",
        "Oryza_sativa",
        "Populus_trichocarpa"
    ]
    protist_list = [
        "Leishmania_major",
        "Plasmodium_falciparum"
    ]

    lines = input.readlines()
    lines_prot = input2.readlines()
    lines += lines_prot
    filtered_lines = []
    index = 0

    # skip plants and fungi
    while index < len(lines):
        line = lines[index]
        for skip_name in skip_list:
            if skip_name in line:
                index += 3
                continue
        filtered_lines.append(line)
        index += 1

    index = 0

    while index < len(filtered_lines):
        line = filtered_lines[index]

        # when line contains metainfo
        if line.startswith(">"):
            # save info about prev sample
            if index != 0:
                tRNA = tRNA_with_id.split(",")[1]
                if tRNA not in tRNA_set:
                    tRNA_set[tRNA] = 1
                    tRNA = resolve_tRNA(tRNA)
                    tRNA_with_id = tRNA_with_id.split(",")[0] + "," + tRNA

                    output_data.write(tRNA_with_id + "\n")
                    output_db.write(db_line + "\n")
                else:
                    # we drop this so id is free
                    id -= 1

            # define type of organism
            orgType = 'anim'  # animal
            for protist in protist_list:
                if protist in line:
                    orgType = 'prot'  # protist
                    break
            if line in lines_prot:
                orgType = 'prot'  # protist

            # format db line and add id to tRNA_with_id
            db_line = "{0},{1},{2}".format(id, line[1:-1], orgType)
            tRNA_with_id = "{},".format(id)

            # update id for next sample
            id += 1
        else:
            # this line contains sequence - just add them
            tRNA_with_id += line[:-1]  # skip line end symbol

        index += 1

    input.close()
    output_data.close()
    output_db.close()

