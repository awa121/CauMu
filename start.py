import os
import sys
from subprocess import Popen, PIPE, STDOUT, DEVNULL
import time


def run(games):
    # curenv = os.environ.copy()
    cmds = []
    filePath = "./CEBP/"
    geneStr = "_".join(games["geneList"])
    for _cancer in games["cancerList"]:
        if geneStr == "":
            for _rate in games["treatRate"]:
                log = games["savePath"] + "/" + _cancer + str(_rate) +"_"+ str(games["confounderRate"])+games["pathway"] + ".log"
                cmd = ["nohup", "python",
                       filePath + games["fileName"],
                       "--cancer=" + _cancer,
                       "--modelName=" + games["modelName"],
                       "--geneMutationRate=" + str(_rate),
                       "--confounderNumber=" + str(games["confounderRate"]),
                       "--pathway=" + games["pathway"],
                       "--savePath=" + games["savePath"]
                       ]
                if "repeatNumber" in games.keys():
                    cmd.append("--repeat="+str(games["repeatNumber"]))

                cmd.append(">")
                cmd.append(log)
                cmd.append("2>&1 &")
                cmds.append(cmd)


        else:
            for _hiCo in games["hiddenConfounder"]:
                log = games["savePath"] + "/" + _cancer + geneStr + str(games["confounderRate"]) + str(_hiCo)+games["pathway"] + ".log"
                cmd = ["nohup", "python",
                       filePath + games["fileName"],
                       "--cancer=" + _cancer,
                       "--gene=" + geneStr,
                       "--modelName=" + games["modelName"],
                       "--confounderNumber=" + str(games["confounderRate"]),
                       "--pathway=" + games["pathway"],
                       "--savePath=" + games["savePath"],
                       "--hiddenConfounder=" + str(_hiCo),
                       ">",log,"2>&1 &"
                       ]

                cmds.append(cmd)
    for i in range(len(cmds)):
        print(cmds[i], 'start',i)

        os.system(" ".join(cmds[i]))
        time.sleep(60)


if __name__ == "__main__":
    cancerList = [ "BRCA","KIRC","THCA", "HNSC", "LUAD","LIHC", "LUSC","ESCA", "KIRP","BLCA"]
    # geneList = ["SETD2", "TP53", "RB1", "CNTNAP3", "CSMD1", "ANK3", "OR4C6", "MYT1L"]
    pathways = ["KEGG_DNA_REPLICATION", "GO_EPITHELIAL_TO_MESENCHYMAL_TRANSITION"]
    ls_date = time.strftime("%Y%m%d%H%M", time.localtime())
    resultPath = "./CEBP/result/" + ls_date
    print(resultPath)
    os.mkdir(resultPath)
    modelName = "CEVAE"

    # ----------start1----------#
    # for 10 cancers in two pathway
    # cancerList =  [ "BRCA","KIRC","THCA", "HNSC", "LUAD","LIHC", "LUSC","ESCA", "KIRP","BLCA"]
    # geneList = []
    # geneMutationRate = [10]
    # confounderNumbers = [200]
    # hiddenConfounder = [20,30]
    # for pathway in pathways:
    #     for confounderNumber in confounderNumbers:
    #         savepath = resultPath
    #         fileName = "mutationEffect.py"
    #         games = {"cancerList": cancerList, "geneList": geneList, "modelName": modelName, "treatRate": geneMutationRate,
    #                      "confounderRate": confounderNumber,
    #                      "pathway": pathway, "savePath": savepath, "fileName": fileName,"hiddenConfounder":hiddenConfounder}
    #         run(games=games)


    # ----------start2----------#
    # for LUAD in more treatment
    # hiddenConfounder=[20]
    # cancerList =  ["BRCA","KIRC","HNSC", "LIHC", "ESCA", "KIRP"]
    # geneList = []
    # geneMutationRate = [200]
    # confounderNumbers = [200]
    # for pathway in pathways:
    #     for confounderNumber in confounderNumbers:
    #         savepath = resultPath
    #         fileName = "mutationEffect.py"
    #         games = {"cancerList": cancerList, "geneList": geneList, "modelName": modelName, "treatRate": geneMutationRate,
    #                      "confounderRate": confounderNumber,"hiddenConfounder":hiddenConfounder,
    #                      "pathway": pathway, "savePath": savepath, "fileName": fileName}
    #         run(games=games)
    #
    # hiddenConfounder = [20]
    # cancerList =  ["LUSC","BLCA","LUAD"]
    # geneList = []
    # geneMutationRate = [0.07]
    # confounderNumbers = [200]
    # for pathway in pathways:
    #     for confounderNumber in confounderNumbers:
    #         savepath = resultPath
    #         fileName = "mutationEffect.py"
    #         games = {"cancerList": cancerList, "geneList": geneList, "modelName": modelName, "treatRate": geneMutationRate,
    #                      "confounderRate": confounderNumber,"hiddenConfounder":hiddenConfounder,
    #                      "pathway": pathway, "savePath": savepath, "fileName": fileName}
    #         run(games=games)


    # hiddenConfounder = [20]
    # cancerList = ["THCA"]
    # geneList = []
    # geneMutationRate = [0.01]
    # confounderNumbers = [200]
    # for pathway in pathways:
    #     for confounderNumber in confounderNumbers:
    #         savepath = resultPath
    #         fileName = "mutationEffect.py"
    #         games = {"cancerList": cancerList, "geneList": geneList, "modelName": modelName,
    #                  "treatRate": geneMutationRate,
    #                  "confounderRate": confounderNumber, "hiddenConfounder": hiddenConfounder,
    #                  "pathway": pathway, "savePath": savepath, "fileName": fileName}
    #         run(games=games)


    # # ----------start3----------#
    # # for gain or loss function for LUAD
    # cancerList = ["THCA"]
    # geneList = ["NARS"]
    # pathways = ["GO_EPITHELIAL_TO_MESENCHYMAL_TRANSITION"]
    #
    # confounderNumber = 0.07
    # for gene in geneList:
    #     savepath = resultPath
    #     fileName="gainOrlossFunction.py"
    #     games = {"cancerList": cancerList, "geneList": [gene], "modelName": modelName, "confounderRate": confounderNumber,
    #                  "pathway": pathways[0], "savePath": savepath, "fileName": fileName}
    #     run(games=games)


