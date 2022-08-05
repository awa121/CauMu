import argparse
from utils import *
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def as_num(x):
    y = '{:.10f}'.format(x)  # .10f
    return y


def specialGenesForCancer(args, gene):
    genesTreatment, geneRateTreatment, mutationDataTreatment, \
    genesConfounder, geneRateConfounder, mutationDataConfounder, \
    sampleName, TCGAExpressionData, normalTCGAExpressionData = getGeneNameForCancer(
        args, specialGene=gene)

    tumorReplicationVector = getReplicationVector(args, TCGAExpressionData, mode=3)
    _max = np.max(tumorReplicationVector[1].astype(float))
    _min = np.min(tumorReplicationVector[1].astype(float))

    tumorReplicationVector[0] = np.array([i[:15] for i in tumorReplicationVector[0]])
    tumorReplicationVector[1] = (tumorReplicationVector[1].astype(float) - _min) / (_max - _min)

    for i in range(1, len(mutationDataConfounder[:, 0])):
        _index = np.argwhere(tumorReplicationVector[0] == mutationDataConfounder[i, 0])
        mutationDataConfounder[i, 0] = tumorReplicationVector[1, _index[0, 0]]
    mutationDataConfounder[0, 0] = "y_factual"

    gene,ate_est = calCauEff(gene, args, mutationDataTreatment, mutationDataConfounder)
    if isinstance(ate_est, bool):
        _ = [args.cancer, gene, "None","None", "None"]
        return _
    _ = [args.cancer, gene, str(np.round((ate_est[0, 2] + ate_est[1, 2]) / 2, 3)),
         str(np.round((ate_est[0, 3] + ate_est[1, 3]) / 2, 3)),
         geneRateTreatment[np.argwhere(geneRateTreatment[:, 0] == gene)[0, 0], 1]]

    return _


if __name__ == '__main__':
    # Command line
    parser = argparse.ArgumentParser(description="CEVAE for cancer gene mutation estimate.")
    parser.add_argument('--cancer', type=str, default="BRCA", help="The cancer you care about")
    parser.add_argument('--gene', type=str, default="TP53", help="The genes you selectd as the treatment")
    parser.add_argument('--modelName', type=str, default="CEVAE")
    parser.add_argument('--geneMutationRate', type=float, default=0.07,
                        help="This is the threshold of mutation rate of treatment genes")
    parser.add_argument('--confounderNumber', type=float, default=200,
                        help="if use rate filter confounder: (0,1) ; if use number select confounder: more than 1")
    parser.add_argument('--hiddenConfounder', type=int, default=50,help="The hidden confounder number.")

    parser.add_argument('--confounderLowRate', type=float, default=0.01)
    parser.add_argument('--pathway', type=str, default="GO_EPITHELIAL_TO_MESENCHYMAL_TRANSITION",
                        help="KEGG_DNA_REPLICATION or GO_EPITHELIAL_TO_MESENCHYMAL_TRANSITION.")
    parser.add_argument('--savePath', type=str,
                        default="./CEBP/result/testtest",
                        help="The file name of result saved in.")

    args = parser.parse_args()
    print(args)

    # hold one cancer and select few genes as treatment one by one.
    ate_sort = []
    for _gene in args.gene.split("_"):
        print(_gene)
        _ = specialGenesForCancer(args, _gene)
        ate_sort.append(_)
    ate_sort = np.array(ate_sort)
    title = ["cancerName", "geneName", "cf","std","mutationRate"]
    ate_sort = np.vstack((title, ate_sort))
    print(ate_sort)
    savename = args.cancer + "_" + args.gene + "_" + str(args.geneMutationRate) + "_" + str(
        args.confounderNumber) + "_" + str(args.confounderLowRate) + "_" +str(args.hiddenConfounder)+"_"+ args.pathway + ".csv"
    np.savetxt(args.savePath + "/" + savename, ate_sort, delimiter=',', fmt="%s")
