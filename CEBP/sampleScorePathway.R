# for mode 4
# Title     : TODO
# Objective : TODO
# Created by: awa123
# Created on: 2021/5/20
recor2<-function(corresult)
{
  corresult[which(corresult>0)]=1
  genesums_row=colSums(corresult)
  genesums_col=rowSums(corresult)

  orderindex1=order(genesums_row,decreasing=TRUE)
  orderindex2=order(genesums_col,decreasing=TRUE)
  new_corresult=corresult[orderindex2,orderindex1]
  return (new_corresult)
}

lm_based_samplescore4 <- function (subdata,pvaluecut,cutoff)
{
  # input: subdata, the pathway genes expression matirx
  #       pvaluecut, the threshold of pvalue
  #       cutoff,
  #ccdata=log(as.matrix(subdata)+1)
  ccdata=as.matrix(subdata) #949*36
  cr.value=cor(t(ccdata)) # pearson
  cr.pvalue=matrix(nrow = nrow(ccdata),ncol = nrow(ccdata))
  for(ci in 1:nrow(ccdata))
  {
    for (cj in 1:nrow(ccdata))
    {
      ct=cor.test(ccdata[ci,],ccdata[cj,])
      cr.pvalue[ci,cj]=ct$p.value
    }
  }
  cr.value[which(cr.pvalue>pvaluecut)]=0
  cr.value[is.na(cr.value)]=0
  newcor=recor2(cr.value)
  mid2=match(row.names(newcor),row.names(subdata))
  candidata=subdata[mid2[1:cutoff],]
  candidata2=candidata
  mean_y=log(rowSums(candidata)/ncol(candidata)+1) # the mean value of gene expression in all samples
  squar_Vector=c()
  coeff=c()
  for (cyi in 1:ncol(candidata))
  {
    rlm=lm(candidata2[,cyi] ~ mean_y+0)# a value
    srr1=summary(rlm)
    srr1_ar2=srr1$adj.r.squared
    coeff=c(coeff,rlm[[1]])
    squar_Vector=c(squar_Vector,srr1_ar2)
  }
  result=rbind(as.vector(squar_Vector),as.vector(coeff))
  row.names(result)=c("squre","coeff")
  return(result)
}

args=commandArgs(T)
setPath=args[1]
setwd(setPath)
subdatafile=args[2]
pvaluecut=as.double(args[3])
cutoff=as.double(args[4])


subdata=read.table(subdatafile,header = T,sep = ",")
rownames(subdata)=subdata[,1]
subdata=subdata[,-1]
#colnames(subdata)
subdata <- log(as.matrix(subdata)+1)
result=lm_based_samplescore4(subdata,pvaluecut,cutoff)
write.csv(result,file=paste(subdatafile,"outcomeVector",sep=""))