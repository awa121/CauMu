# for CDH1 in BRCA
pathways = ["KEGG_DNA_REPLICATION",
            "GO_POSITIVE_REGULATION_OF_EPITHELIAL_TO_MESENCHYMAL_TRANSITION",
            "GO_EPITHELIAL_TO_MESENCHYMAL_TRANSITION"]

cancerList = ["BRCA", "KIRC", "THCA", "HNSC", "LUAD", "LIHC", "LUSC", "ESCA", "KIRP", "BLCA"]
from utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gene gain function or loss")
    parser.add_argument('--cancer', type=str, default="THCA")
    parser.add_argument('--gene', type=str, default="HRAS", help="The genes you what to know")

    parser.add_argument('--modelName', type=str, default="CEVAE")
    parser.add_argument('--geneMutationRate', type=float, default=600,
                        help="This is the threshold of mutation rate of treatment genes")
    parser.add_argument('--confounderNumber', type=float, default=0.07,
                        help="if use rate filter confounder: (0,1) ; if use number select confounder: more than 1")
    parser.add_argument('--confounderLowRate', type=float, default=0.01)
    parser.add_argument('--pathway', type=str, default="KEGG_DNA_REPLICATION",
                        help="KEGG_DNA_REPLICATION or GO_EPITHELIAL_TO_MESENCHYMAL_TRANSITION or GO_POSITIVE_REGULATION_OF_EPITHELIAL_TO_MESENCHYMAL_TRANSITION.")
    parser.add_argument('--savePath', type=str,
                        default="./CEBP/result/",
                        help="The file name of result saved in .")

    args = parser.parse_args()
    print(args)

    genesTreatment, geneRateTreatment, mutationDataTreatment, \
    genesConfounder, geneRateConfounder, mutationDataConfounder, \
    sampleName, TCGAExpressionData, normalTCGAExpressionData = getGeneNameForCancer(args, specialGene=args.gene)

    # mutated
    geneTumorExpressionData = TCGAExpressionData[:, [1, np.argwhere(TCGAExpressionData[0] == args.gene)[0, 0]]]
    geneTumorMutation = mutationDataTreatment[:, [0, np.argwhere(mutationDataTreatment[0] == args.gene)[0, 0]]]
    geneTumorExpressionData[:, 0] = np.array([i[:15] for i in geneTumorExpressionData[:, 0]])

    # get pathway label for normal sample, tumor cell with mutated gene, tumor cell with none-mutated gene
    TCGASampleName = np.array([i[:15] for i in TCGAExpressionData[:, 1]])

    # tumor cell with mutated gene
    avMuValue1, tumorMuTCGAExpressionData = 0, TCGAExpressionData[0]
    _n = 0
    for _s in geneTumorMutation:
        if _s[1] == "1":
            avMuValue1 += float(geneTumorExpressionData[np.argwhere(geneTumorExpressionData[:, 0] == _s[0])[0, 0], 1])
            _n += 1
            # tumorMuTCGAExpressionData = np.vstack(
            #     (tumorMuTCGAExpressionData, TCGAExpressionData[np.argwhere(TCGASampleName == _s[0])[0, 0]]))
    avMuValue1 = avMuValue1 / _n
    # tumorMuReplicationVector = getReplicationVector(args, tumorMuTCGAExpressionData)
    # tumorMuReplicationVector = getReplicationVector(args, tumorMuTCGAExpressionData, mode=3)
    # avTumorMuReplicationVector = tumorMuReplicationVector[1].astype(float).sum() / len(tumorMuReplicationVector[0])

    # tumor cell with none-mutated gene
    avMuValue0, tumorNonMuTCGAExpressionData = 0, TCGAExpressionData[0]
    _n = 0
    for _s in geneTumorMutation:
        if _s[1] == "0":
            avMuValue0 += float(geneTumorExpressionData[np.argwhere(geneTumorExpressionData[:, 0] == _s[0])[0, 0], 1])
            _n += 1
            # tumorNonMuTCGAExpressionData = np.vstack(
            #     (tumorNonMuTCGAExpressionData, TCGAExpressionData[np.argwhere(TCGASampleName == _s[0])[0, 0]]))
    avMuValue0 = avMuValue0 / _n
    # tumorNonMuReplicationVector = getReplicationVector(args, tumorNonMuTCGAExpressionData,mode=3)
    # avTumorNonMuReplicationVector = tumorNonMuReplicationVector[1].astype(float).sum() / len(tumorNonMuReplicationVector[0])

    # normal cell with gene
    avNorValue=0
    _e=normalTCGAExpressionData[1:,np.argwhere(normalTCGAExpressionData[0]==args.gene)[0,0]]
    avNorValue=_e.astype(float).sum()/len(_e)
    # normalReplicationVector = getReplicationVector(args, normalTCGAExpressionData)
    # avNormalReplicationVector=normalReplicationVector[1].astype(float).sum()/len(normalReplicationVector[0])

    print("In", args.cancer, ":gene,norExpression,non-mutatedExpression,mutatedExpression")
    print(args.gene,avNorValue,avMuValue0,avMuValue1)
    # print(args.gene, avTumorNonMuReplicationVector, avTumorMuReplicationVector,
    #       "Gain-function" if avTumorNonMuReplicationVector < avTumorMuReplicationVector else "Loss-function")

    with open(args.savePath + "/" + args.gene + args.pathway + "gainOrlossFunction.txt", "a+") as f:
        print(args.gene,args.cancer, avMuValue0, avMuValue1,
                "Gain-function" if avMuValue0 < avMuValue1 else "Loss-function","\n",file=f)
