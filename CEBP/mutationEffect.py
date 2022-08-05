
import argparse
from utils import *
from multiprocessing import Process,Queue, Pool


if __name__ == '__main__':
    # Command line
    parser = argparse.ArgumentParser(description="CEVAE for cancer gene mutation estimate.")
    parser.add_argument('--cancer', type=str, default="BRCA")
    parser.add_argument('--gene', type=str, default="None",help="The genes you selectd as the treatment")

    parser.add_argument('--modelName', type=str, default="CEVAE")
    parser.add_argument('--geneMutationRate', type=float, default=0.07,help="This is the threshold of mutation rate of treatment genes")
    parser.add_argument('--confounderNumber', type=float, default=0.07,
                        help="if use rate filter confounder: (0,1) ; if use number select confounder: more than 1")
    parser.add_argument('--confounderLowRate', type=float, default=0.01)
    parser.add_argument('--hiddenConfounder', type=int, default=20,help="The hidden confounder number.")
    parser.add_argument('--pathway', type=str, default="KEGG_DNA_REPLICATION",
                        help="KEGG_DNA_REPLICATION or GO_EPITHELIAL_TO_MESENCHYMAL_TRANSITION or GO_POSITIVE_REGULATION_OF_EPITHELIAL_TO_MESENCHYMAL_TRANSITION.")
    parser.add_argument('--savePath', type=str,
                        default="./CEBP/result/",
                        help="The file name of result saved in .")
    parser.add_argument("--repeat",type=int,default=1,help="Experiments repeat number.")

    args = parser.parse_args()
    print(args)

    genesTreatment,geneRateTreatment,mutationDataTreatment,\
    genesConfounder,geneRateConfounder,mutationDataConfounder,\
    sampleName, TCGAExpressionData, normalTCGAExpressionData= getGeneNameForCancer(args)

    tumorReplicationVector = getReplicationVector(args, TCGAExpressionData,mode=3)
    normalReplicationVector = getReplicationVector(args, normalTCGAExpressionData,mode=3)
    _max = np.max(tumorReplicationVector[1].astype(float))
    _min = np.min(tumorReplicationVector[1].astype(float))

    tumorReplicationVector[0] = np.array([i[:15] for i in tumorReplicationVector[0]])
    tumorReplicationVector[1] = (tumorReplicationVector[1].astype(float) - _min) / (_max - _min)

    for i in range(1, len(mutationDataConfounder[:, 0])):
        _index = np.argwhere(tumorReplicationVector[0] == mutationDataConfounder[i, 0])
        mutationDataConfounder[i, 0] = tumorReplicationVector[1, _index[0, 0]]
    mutationDataConfounder[0, 0] = "y_factual"

    ate_sort = []

    # for i in range(args.repeat):
    #     pool = Pool(processes=len(genesTreatment))
    #     resultPool = []
    #     for _g in genesTreatment:
    #         resultPool.append(pool.apply_async(calCauEff, [_g, args, mutationDataTreatment,mutationDataConfounder]))
    #     pool.close()
    #     pool.join()
    #     for res in resultPool:
    #         _g,ate_est = res.get()
    #         _ = [_g, str(np.round((ate_est[0, 2] + ate_est[1, 2]) / 2, 3)),str(np.round((ate_est[0, 3] + ate_est[1, 3]) / 2, 3))]
    #         ate_sort.append(_)
    for i in range(args.repeat):
        for _g in genesTreatment:

            gene,ate_est = calCauEff(_g, args, mutationDataTreatment,mutationDataConfounder)
            _ = [_g, str(np.round((ate_est[0, 2] + ate_est[1, 2]) / 2, 3)),str(np.round((ate_est[0, 3] + ate_est[1, 3]) / 2, 3))]
            ate_sort.append(_)

    ate_sort = np.array(ate_sort)
    ate_sort = ate_sort[np.lexsort(ate_sort[:, ::-1].T)]
    geneRateTreatment = geneRateTreatment[np.lexsort(geneRateTreatment[:, ::-1].T)]
    result = np.hstack((ate_sort, geneRateTreatment))
    title=["geneName","cf","std","geneName","mutationRate"]
    result=np.vstack((title,result))
    print(result)

    if not os.path.exists(args.savePath):
        os.makedirs(args.savePath)
    savename = args.cancer + "_" + args.gene + "_" + str(args.geneMutationRate) + "_" + str(args.confounderNumber) + "_" + str(args.confounderLowRate) + "_" + args.pathway+".csv"
    np.savetxt(args.savePath + "/" + savename, result, delimiter=',', fmt="%s")
