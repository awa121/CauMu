# for mode 3
# Title     : TODO
# Objective : TODO
# Created by: awa123
# Created on: 2021/5/20
library(WGCNA)

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

lm_based_samplescore3 <- function (subdata,pvaluecut,cutoff)### could log(base)
{

  #ccdata=log(as.matrix(subdata)+1)
  ccdata=as.matrix(subdata)
  corresult=corAndPvalue(t(ccdata))
  corresult$cor[which(corresult$p>pvaluecut)]=0
  corresult$cor[is.na(corresult$cor)]=0
  newcor=recor2(corresult$cor)

  mid2=match(row.names(newcor),row.names(subdata))
  candidata=subdata[mid2[1:cutoff],]
  candidata2=candidata
  mean_y=rowSums(candidata)/ncol(candidata)
  squar_Vector=c()
  coeff=c()
  for (cyi in 1:ncol(candidata))
  {
    rlm=lm(candidata2[,cyi] ~ mean_y+0)
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
result=lm_based_samplescore3(subdata,pvaluecut,cutoff)
print(result)
write.csv(result,file=paste(subdatafile,"outcomeVector",sep=""))