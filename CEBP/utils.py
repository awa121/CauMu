import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"#use CPU, Commenting out means using the GPU
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import sys
from tensorflow.contrib.layers.python.layers import initializers
from sklearn.linear_model import LinearRegression

import pandas as pd
from scipy import stats

import time
import subprocess
def run_cmd(cmd):
    cp=subprocess.run(cmd,shell=True)
    if cp.returncode!=0:
        error= """Something wrong has happend when running R command [{cmd}]: {cp.stderr}"""
        raise Exception(error)
    return cp.stdout,cp.stderr

def fc_net(inp, layers, out_layers, scope, lamba=1e-3, activation=tf.nn.relu, reuse=None,
           weights_initializer=initializers.xavier_initializer(uniform=False)):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=activation,
                        normalizer_fn=None,
                        weights_initializer=weights_initializer,
                        reuse=reuse,
                        weights_regularizer=slim.l2_regularizer(lamba)):

        if layers:
            h = slim.stack(inp, slim.fully_connected, layers, scope=scope)
            if not out_layers:
                return h
        else:
            h = inp
        outputs = []
        for i, (outdim, activation) in enumerate(out_layers):
            o1 = slim.fully_connected(h, outdim, activation_fn=activation, scope=scope + '_{}'.format(i + 1))
            outputs.append(o1)
        return outputs if len(outputs) > 1 else outputs[0]


def get_y0_y1(sess, y, f0, f1, shape=(), L=1, verbose=True):
    y0, y1 = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    ymean = y.mean()
    for l in range(L):
        y0 += sess.run(ymean, feed_dict=f0) / L
        y1 += sess.run(ymean, feed_dict=f1) / L

    if L > 1 and verbose:
        print()
    return y0, y1

def estimate_causal_effect(Xt, y, model=LinearRegression(), treatment_idx=0, regression_coef=False):
    model.fit(Xt, y)
    if regression_coef:
        return model.coef_[treatment_idx]
    else:
        Xt1 = pd.DataFrame.copy(Xt)
        Xt1[Xt.columns[treatment_idx]] = 1
        Xt0 = pd.DataFrame.copy(Xt)
        Xt0[Xt.columns[treatment_idx]] = 0

        return (model.predict(Xt1) - model.predict(Xt0)).mean()
def estimate_causal_effect_CEVAE(mutationData,args, pathFileData,reps=10,earl=10,lr=0.001,opt="adam",epochs=100,print_every=10,true_post=True):
    # original edition is from cevae_ihdp.py
    cancer=args.cancer
    import edward as ed
    import tensorflow as tf

    from edward.models import Bernoulli, Normal
 #   from progressbar import ETA, Bar, Percentage, ProgressBar

    from datasets import CANCER
    from evaluation import EvaluatorCancer
    import numpy as np
    import time
    from scipy.stats import sem
    # mutationData: treatment ,y_factual,x
    dataset = CANCER(mutationData,cancer,pathFileData=pathFileData,replications=reps)
    dimx = len(mutationData[0])-2
    scores = np.zeros((reps, 2))
    scores_test = np.zeros((reps, 2))

    M = None  # batch size during training
    d =args.hiddenConfounder  # latent dimension
    lamba = 1e-4  # weight decay
    nh, h = 3, 200  # number and size of hidden layers

    for i, (train, valid, test, contfeats, binfeats) in enumerate(dataset.get_train_valid_test()):
        print('\nReplication {}/{}'.format(i + 1, reps))
        (xtr, ttr, ytr)= train
        (xva, tva, yva)= valid
        (xte, tte, yte)= test
        evaluator_test = EvaluatorCancer(yte, tte)

        # reorder features with binary first and continuous after
        perm = binfeats + contfeats
        xtr, xva, xte = xtr[:, perm], xva[:, perm], xte[:, perm]

        xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate([ytr, yva], axis=0)
        evaluator_train = EvaluatorCancer(yalltr, talltr)

        # zero mean, unit variance for y during training
        ym, ys = np.mean(ytr), np.std(ytr)
        ytr, yva = (ytr - ym) / ys, (yva - ym) / ys
        best_logpvalid = - np.inf

        with tf.Graph().as_default():
            sess = tf.InteractiveSession()

            ed.set_seed(1)
            np.random.seed(1)
            tf.set_random_seed(1)

            x_ph_bin = tf.placeholder(tf.float32, [M, len(binfeats)], name='x_bin')  # binary inputs
            x_ph_cont = tf.placeholder(tf.float32, [M, len(contfeats)], name='x_cont')  # continuous inputs
            t_ph = tf.placeholder(tf.float32, [M, 1])
            y_ph = tf.placeholder(tf.float32, [M, 1])

            x_ph = tf.concat([x_ph_bin, x_ph_cont], 1)
            activation = tf.nn.elu

            # CEVAE model (decoder)
            # p(z)
            z = Normal(loc=tf.zeros([tf.shape(x_ph)[0], d]), scale=tf.ones([tf.shape(x_ph)[0], d]))

            # p(x|z)
            hx = fc_net(z, (nh - 1) * [h], [], 'px_z_shared', lamba=lamba, activation=activation)
            logits = fc_net(hx, [h], [[len(binfeats), None]], 'px_z_bin'.format(i + 1), lamba=lamba,
                            activation=activation)
            x1 = Bernoulli(logits=logits, dtype=tf.float32, name='bernoulli_px_z')

            mu, sigma = fc_net(hx, [h], [[len(contfeats), None], [len(contfeats), tf.nn.softplus]], 'px_z_cont',
                               lamba=lamba,
                               activation=activation)
            x2 = Normal(loc=mu, scale=sigma, name='gaussian_px_z')

            # p(t|z)
            logits = fc_net(z, [h], [[1, None]], 'pt_z', lamba=lamba, activation=activation)
            t = Bernoulli(logits=logits, dtype=tf.float32)

            # p(y|t,z)
            mu2_t0 = fc_net(z, nh * [h], [[1, None]], 'py_t0z', lamba=lamba, activation=activation)
            mu2_t1 = fc_net(z, nh * [h], [[1, None]], 'py_t1z', lamba=lamba, activation=activation)
            y = Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=tf.ones_like(mu2_t0))

            # CEVAE variational approximation (encoder)
            # q(t|x)
            logits_t = fc_net(x_ph, [d], [[1, None]], 'qt', lamba=lamba, activation=activation)
            qt = Bernoulli(logits=logits_t, dtype=tf.float32)
            # q(y|x,t)
            hqy = fc_net(x_ph, (nh - 1) * [h], [], 'qy_xt_shared', lamba=lamba, activation=activation)
            mu_qy_t0 = fc_net(hqy, [h], [[1, None]], 'qy_xt0', lamba=lamba, activation=activation)
            mu_qy_t1 = fc_net(hqy, [h], [[1, None]], 'qy_xt1', lamba=lamba, activation=activation)
            qy = Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=tf.ones_like(mu_qy_t0))
            # q(z|x,t,y)
            inpt2 = tf.concat([x_ph, qy], 1)
            hqz = fc_net(inpt2, (nh - 1) * [h], [], 'qz_xty_shared', lamba=lamba, activation=activation)
            muq_t0, sigmaq_t0 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt0', lamba=lamba,
                                       activation=activation)
            muq_t1, sigmaq_t1 = fc_net(hqz, [h], [[d, None], [d, tf.nn.softplus]], 'qz_xt1', lamba=lamba,
                                       activation=activation)
            qz = Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0, scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)

            # Create data dictionary for edward
            data = {x1: x_ph_bin, x2: x_ph_cont, y: y_ph, qt: t_ph, t: t_ph, qy: y_ph}

            # sample posterior predictive for p(y|z,t)
            y_post = ed.copy(y, {z: qz, t: t_ph}, scope='y_post')
            # crude approximation of the above
            y_post_mean = ed.copy(y, {z: qz.mean(), t: t_ph}, scope='y_post_mean')
            # construct a deterministic version (i.e. use the mean of the approximate posterior) of the lower bound
            # for early stopping according to a validation set
            y_post_eval = ed.copy(y, {z: qz.mean(), qt: t_ph, qy: y_ph, t: t_ph}, scope='y_post_eval')
            x1_post_eval = ed.copy(x1, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x1_post_eval')
            x2_post_eval = ed.copy(x2, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='x2_post_eval')
            t_post_eval = ed.copy(t, {z: qz.mean(), qt: t_ph, qy: y_ph}, scope='t_post_eval')
            logp_valid = tf.reduce_mean(tf.reduce_sum(y_post_eval.log_prob(y_ph) + t_post_eval.log_prob(t_ph), axis=1) +
                                        tf.reduce_sum(x1_post_eval.log_prob(x_ph_bin), axis=1) +
                                        tf.reduce_sum(x2_post_eval.log_prob(x_ph_cont), axis=1) +
                                        tf.reduce_sum(z.log_prob(qz.mean()) - qz.log_prob(qz.mean()), axis=1))

            inference = ed.KLqp({z: qz}, data)
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            inference.initialize(optimizer=optimizer)

            saver = tf.train.Saver(tf.contrib.slim.get_variables())
            tf.global_variables_initializer().run()

            n_epoch, n_iter_per_epoch, idx = epochs, 10 * max(int(xtr.shape[0] / 100),1), np.arange(xtr.shape[0])

            # dictionaries needed for evaluation
            tr0, tr1 = np.zeros((xalltr.shape[0], 1)), np.ones((xalltr.shape[0], 1))
            tr0t, tr1t = np.zeros((xte.shape[0], 1)), np.ones((xte.shape[0], 1))
            f1 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr1}
            f0 = {x_ph_bin: xalltr[:, 0:len(binfeats)], x_ph_cont: xalltr[:, len(binfeats):], t_ph: tr0}
            f1t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr1t}
            f0t = {x_ph_bin: xte[:, 0:len(binfeats)], x_ph_cont: xte[:, len(binfeats):], t_ph: tr0t}

            for epoch in range(n_epoch):
                avg_loss = 0.0

                t0 = time.time()
                np.random.shuffle(idx)
                for j in range(n_iter_per_epoch):
                    # pbar.update(j)
                    batch = np.random.choice(idx, 100)
                    x_train, y_train, t_train = xtr[batch], ytr[batch], ttr[batch]
                    info_dict = inference.update(feed_dict={x_ph_bin: x_train[:, 0:len(binfeats)],
                                                            x_ph_cont: x_train[:, len(binfeats):],
                                                            t_ph: t_train, y_ph: y_train})
                    avg_loss += info_dict['loss']

                avg_loss = avg_loss / n_iter_per_epoch
                avg_loss = avg_loss / 100

                if epoch % earl == 0 or epoch == (n_epoch - 1):
                    logpvalid = sess.run(logp_valid, feed_dict={x_ph_bin: xva[:, 0:len(binfeats)],
                                                                x_ph_cont: xva[:, len(binfeats):],
                                                                t_ph: tva, y_ph: yva})
                    if logpvalid >= best_logpvalid:
                        print('Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_logpvalid, logpvalid))
                        best_logpvalid = logpvalid
                        args.savePath.split("/")[-1]
                        if not os.path.exists('./CEBP/models/'+cancer+"/"+args.savePath.split("/")[-1]):
                            os.makedirs('./CEBP/models/'+cancer+"/"+args.savePath.split("/")[-1])
                        saver.save(sess, './CEBP/models/'+cancer+"/"+args.savePath.split("/")[-1]+'/'+pathFileData.split("/")[-1])

                if epoch % print_every == 0:
                    y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=1)
                    y0, y1 = y0 * ys + ym, y1 * ys + ym
                    score_train = evaluator_train.calc_stats(y1, y0)
                    rmses_train = evaluator_train.y_errors(y0, y1)

                    y0, y1 = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=1)
                    y0, y1 = y0 * ys + ym, y1 * ys + ym
                    score_test = evaluator_test.calc_stats(y1, y0)

                    print("Epoch: {}/{}, log p(x) >= {:0.3f}, train-ate mean ite: {:0.3f}, train-ate mean pred: {:0.3f} " \
                          "rmse_f_tr: {:0.3f}, test-ate mean ite: {:0.3f}, test-ate mean pred: {:0.3f}, " \
                          "dt: {:0.3f}".format(epoch + 1, n_epoch, avg_loss, score_train[0], score_train[1],
                                               rmses_train, score_test[0], score_test[1],time.time() - t0))
            if not os.path.exists('./CEBP/models/'+cancer+"/"+args.savePath.split("/")[-1]):
                os.makedirs('./CEBP/models/'+cancer+"/"+args.savePath.split("/")[-1])
            saver.restore(sess, './CEBP/models/'+cancer+"/"+args.savePath.split("/")[-1]+'/'+pathFileData.split("/")[-1])
            y0, y1 = get_y0_y1(sess, y_post, f0, f1, shape=yalltr.shape, L=100)
            y0, y1 = y0 * ys + ym, y1 * ys + ym
            score = evaluator_train.calc_stats(y1, y0)
            scores[i, :] = score

            y0t, y1t = get_y0_y1(sess, y_post, f0t, f1t, shape=yte.shape, L=100)
            y0t, y1t = y0t * ys + ym, y1t * ys + ym
            score_test = evaluator_test.calc_stats(y1t, y0t)
            scores_test[i, :] = score_test

            print('Replication: {}/{}, train-ate mean ite: {:0.3f}, train-ate mean pred: {:0.3f}' \
                  ', test-ate mean ite: {:0.3f}, test-ate mean pred: {:0.3f}'.format(i + 1, reps,score[0], score[1],score_test[0], score_test[1]))
            sess.close()
    # result:ndarray, first line (4): average train-ate-mean-ite mean+std , average train-ate-mean-pred mean+std
    #                 second line (4): average test-ate-mean-ite mean+std , average test-ate-mean-pred mean+std

    result=[]
    print('CEVAE model total scores')
    means, stds = np.mean(scores, axis=0), sem(scores, axis=0)
    print('train ITE: {:.3f}+-{:.3f}, train ATE: {:.3f}+-{:.3f}' \
          ''.format(means[0], stds[0], means[1], stds[1]))
    result.append([means[0], stds[0], means[1], stds[1]])

    means, stds = np.mean(scores_test, axis=0), sem(scores_test, axis=0)
    print('test ITE: {:.3f}+-{:.3f}, test ATE: {:.3f}+-{:.3f}' \
          ''.format(means[0], stds[0], means[1], stds[1]))
    result.append([means[0], stds[0], means[1], stds[1]])

    return np.array(result)
def calCauEff(gene,args,mutationDataTreatment,mutationData,queue=None,method="CEVAE",reps=10,earl=10,lr=0.001,opt="adam",epochs=500,print_every=10,true_post=True):
    #mutationData is the matrix of confounder
    if (args.gene!="None") and (gene not in mutationData[0]):
        print("There is no mutation data of "+gene +" in "+args.cancer+" samples")
        return False
    if args.gene!="None":  #for special gene
        _index=np.argwhere(mutationData[0]==gene)
        print(gene,_index[0,0])
        _tVector=mutationData[:,_index[0,0]]

        mutationData=np.delete(mutationData,_index[0,0],axis=1)
    else:
        _index = np.argwhere(mutationDataTreatment[0] == gene)
        print(gene, _index[0, 0])
        _tVector = mutationDataTreatment[:, _index[0, 0]]
        if gene in mutationData[0]:
            mutationData = np.delete(mutationData, np.argwhere(mutationData[0]==gene)[0, 0], axis=1)
    mutationData=np.insert(mutationData,0,values=_tVector,axis=1)
    mutationData[0,0]="treatment"

    savePath=args.savePath.split("/")[-1] # year+month+day+ specialGene(if have)
    dataPath="./CEBP/datasets/"+args.cancer+"/"+savePath+"/"
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    from random import shuffle
    dataPure = mutationData[1:].astype(float)
    print("The number of selected samples is",len(dataPure))

    _filename=gene+"_"+args.cancer+"_"+args.gene+"_"+str(args.geneMutationRate)+"_"+str(args.confounderNumber)+"_"+str(args.confounderLowRate)+"_"+args.pathway+"_"+str(args.hiddenConfounder)+"_"
    for i in range(reps):
        _l = [i for i in range(len(dataPure))]
        shuffle(_l)
        dataPure = dataPure[_l]
        np.savetxt(dataPath+_filename+str(i+1)+".csv",dataPure,delimiter=',',fmt="%s")
    print(gene)
    if method=="CEVAE":
        ate_est4 = estimate_causal_effect_CEVAE(mutationData,args=args,pathFileData=dataPath+_filename,reps=reps,earl=earl,lr=lr,opt=opt,epochs=epochs,print_every=print_every,true_post=true_post)
        print("hello",ate_est4)
    if queue is not None:
        queue.put(ate_est4)
    return gene,ate_est4


def fillterGene(rOrN, mutationData,lowRate=0.01,specialGene=""):
    #filter genes by mutation rate or number we care: if rate (0，1) ; if number >1, and int
    sampleNum=len(mutationData)-1
    _deIndex=[]

    geneRate = []
    if rOrN<1:# use rate filter genes
        for i in range(1,len(mutationData[0])):
            _mSN = mutationData[1:, i].astype(int)
            _rate = round((sum(_mSN) / sampleNum),3)
            if (_rate < max(rOrN,lowRate)) and (specialGene!=mutationData[0, i]):
                _deIndex.append(i)
            else:
                geneRate.append([mutationData[0, i],str(_rate)])
        mutationData = np.delete(mutationData, _deIndex, axis=1)
        geneList=mutationData[0,1:]
        return geneList,np.array(geneRate),mutationData
    elif rOrN>=1:
        rOrN=int(rOrN)
        for i in range(1,len(mutationData[0])):
            _mSN = mutationData[1:, i].astype(int)
            _rate = round((sum(_mSN) / sampleNum),3)
            if (_rate <lowRate) and  (specialGene!=mutationData[0, i]):
                _deIndex.append(i)
            else:
                geneRate.append([mutationData[0, i],str(_rate)])
        mutationData=np.delete(mutationData, _deIndex, axis=1)
        geneRate.sort(key=lambda x:float(x[-1]),reverse=True)

        geneRate = np.array(geneRate[:min(rOrN, len(geneRate))])
        _deIndex = []
        for i in range(1, len(mutationData[0])):
            if (mutationData[0, i] not in geneRate[:, 0]) and (mutationData[0, i]!=specialGene):
                _deIndex.append(i)
        mutationData = np.delete(mutationData, _deIndex, axis=1)
        geneList = mutationData[0, 1:]
        return geneList,np.array(geneRate),mutationData
    else:
        print("Rate or number of mutaion genes wrong")
        return

def getGeneNameForCancer(args, dataBasePath="./data/",specialGene=""):
    # specialGene is for the function of holding the special gene as treatment. default is ""
    cancer=args.cancer
    mutationRate=args.geneMutationRate
    print("#Calculate in " + cancer)
    # l1000GenesFile = dataBasePath+"L1000genes978.csv"
    TCGAmutationFile = dataBasePath + "mutation_14/" + cancer + ".csv"
    TCGAGenesFile = dataBasePath + "gene_symbol_list_exceptSLC35E2"
    with open(TCGAmutationFile, "r", encoding="utf-8") as f:
        mutationData = [i[:-1].split(",") for i in f.readlines()]
    mutationGenes = [i[0] for i in mutationData[1:]]
    with open(TCGAGenesFile, "r", encoding="utf-8") as f:
        TCGAGenes = [i[:-1] for i in f.readlines()]
    # genes without filter by mutation rate
    genes = list(set(TCGAGenes) & set(mutationGenes))

    # get mutation samples' name
    mutationSampleName = mutationData[0][1:]

    # get TCGA samples' name
    TCGAExpressionFile = dataBasePath + "TCGA/"+cancer + "/"
    exfilelist = os.listdir(TCGAExpressionFile)

    TCGAExpressionData = ["stageLabel", "sampleName"] + TCGAGenes
    TCGAExpressionData = np.array(TCGAExpressionData).reshape((1, 2 + len(TCGAGenes)))
    for _l in exfilelist:
        with open(TCGAExpressionFile + _l, "r", encoding="utf-8") as f:
            raw_data = [i[:-1].split("\t") for i in f.readlines()]
        print("{} has {} samples".format(_l, len(raw_data[0])))
        raw_data[0].insert(0, "sampleName")
        raw_data.insert(0, [_l[0:1] for i in range(len(raw_data[0]))])
        raw_data[0][0] = "type"
        raw_data = np.array(raw_data).T
        TCGAExpressionData = np.concatenate((TCGAExpressionData, raw_data[1:]), axis=0)
    TCGAsampleName = TCGAExpressionData[:, 1][1:]

    # get jointly samples' name, save as TCGA sample name
    sampleName = [i for i in TCGAsampleName if i[:15] in mutationSampleName]

    # TCGA expression of normal sample
    _Index = [i for i in range(1, len(TCGAExpressionData)) if TCGAExpressionData[i, 0]=="1"]
    normalTCGAExpressionData = TCGAExpressionData[[0]+_Index]

    _deIndex = [i for i in range(1, len(TCGAExpressionData)) if TCGAExpressionData[i, 1] not in sampleName]
    TCGAExpressionData = np.delete(TCGAExpressionData, _deIndex, axis=0)

    mutationData[0][0] = "sampleName"
    mutationData = np.array(mutationData).T
    muSampleName = [i[:15] for i in sampleName]
    _deIndex = [i for i in range(1, len(mutationData)) if mutationData[i, 0] not in muSampleName]
    mutationData = np.delete(mutationData, _deIndex, axis=0)

    print("# After filter, number of samples is",len(sampleName))
    print("# Confounder genes fit the counfounder rate or number in tumor samples:")
    sampleNum = len(mutationData) - 1
    _deIndex=[]
    for i in range(1, len(mutationData[0])):
        if mutationData[0, i] not in genes:
            _deIndex.append(i)
            continue
        # delete low expression gene
        _=np.argwhere(TCGAExpressionData[0]==mutationData[0, i])[0][0]
        averageEX=sum(TCGAExpressionData[1:,_].astype(float))/len(sampleName)
        if averageEX<10:
            _deIndex.append(i)
    mutationData = np.delete(mutationData, _deIndex, axis=1)



    genesTreatment,geneRateTreatment,mutationDataTreatment=fillterGene(args.geneMutationRate,mutationData,args.confounderLowRate)
    genesConfounder,geneRateConfounder,mutationDataConfounder=fillterGene(args.confounderNumber,mutationData,args.confounderLowRate,specialGene=specialGene)
    if specialGene=="":
        print("# And treatment genes number is",len(genesTreatment))
    else:
        print("# Serve as treatment gene is",specialGene)
    print("# And confounder genes number is",len(genesConfounder))

    return genesTreatment,geneRateTreatment,mutationDataTreatment,genesConfounder,geneRateConfounder,mutationDataConfounder, sampleName, TCGAExpressionData,normalTCGAExpressionData

def getReplicationVector(args,TCGAExpressionData,mode=3,dataBasePath="./data/",pvaluecut=0.001, cutoff=0.5):
    pathwayName=args.pathway
    with open(dataBasePath+"DNA replication_revelvant_pathway.csv", "r", encoding="utf-8") as f:
        pathway = [i[:-1].split(",") for i in f.readlines()]
    print("Select",pathwayName,"as outcome")
    pathDist={}
    for _p in pathway:
        pathDist[_p[0]]=_p[1:]
    pathwayGenes=pathDist[pathwayName]
    _deIndex = [i for i in range(2, len(TCGAExpressionData[0])) if TCGAExpressionData[0, i] not in pathwayGenes]
    subdata = np.delete(TCGAExpressionData, [0]+_deIndex, axis=1).T
    subdata[0,0]=""
    _filename = args.cancer + "_" + args.gene + "_" + str(args.geneMutationRate) + "_" + str(args.confounderNumber) + "_" + str(args.confounderLowRate) + "_" + args.pathway+"_"+str(args.hiddenConfounder)
    subdatafile=_filename+str(mode)+"sampleScorePathway"
    setPath="./CEBP/result/tempForR/"
    np.savetxt(setPath+subdatafile, subdata,delimiter=',',fmt="%s")
    if mode==4:
        print("Rfile type:", mode)
        Rfile="./CEBP/sampleScorePathway.R"
    elif mode==3:
        print("Rfile type:",mode)
        Rfile="./CEBP/sampleScorePathway2.R"

    cutoff= round((len(subdata)-1) / 2)
    cmd='Rscript'+" "+Rfile + ' ' + setPath+" "+subdatafile + ' ' + str(pvaluecut)+" "+str(cutoff)
    run_cmd(cmd=cmd)
    # os.system(cmd)


    with open(setPath+subdatafile+"outcomeVector", "r", encoding="utf-8") as f:
        outcomeVector = [i[:-1].split(",") for i in f.readlines()]

    resultVector=np.array(outcomeVector[2][1:]).astype(float)
    sampleName=TCGAExpressionData[1:,1].T
    print(np.shape(sampleName),np.shape(resultVector))
    result=np.vstack([sampleName,resultVector])

    return result



