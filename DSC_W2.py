import pandas as pd
def proportion_of_education():
    df=pd.read_csv('assets/NISPUF17.csv')
    total=len(df['EDUC1'])
    A=(len(df[df['EDUC1']==1]))/total
    B=(len(df[df['EDUC1']==2]))/total
    C=(len(df[df['EDUC1']==3]))/total
    D=(len(df[df['EDUC1']==4]))/total
    E={"less than high school":A,"high school":B,"more than high school but not college":C,"college":D }
    return E
proportion_of_education()
    # your code goes here
    # YOUR CODE HERE
    #raise NotImplementedError()



def average_influenza_doses():
    df=pd.read_csv('assets/NISPUF17.csv')
    dff1=df[df['CBF_01']==1]
    dff=dff1['P_NUMFLU'].dropna()
    AA=sum(dff)/len(dff)
    dft1=df[df['CBF_01']==2]
    dft=dft1['P_NUMFLU'].dropna()
    BB=sum(dft)/len(dft)
    return (AA,BB)
    #print(AA)
average_influenza_doses()
    # YOUR CODE HERE
    #raise NotImplementedError()


import pandas as pd
def chickenpox_by_sex():
    df=pd.read_csv('assets/NISPUF17.csv')
    dff=df[df['SEX']==1]
    dff=dff[['SEX','P_NUMVRC','HAD_CPOX']]
    dff=dff.dropna()
    dff1=dff[(dff['P_NUMVRC']>0) & (dff['HAD_CPOX']==1)]
    dff2=dff[(dff['P_NUMVRC']>0) & (dff['HAD_CPOX']==2)]
    M=len(dff1)/len(dff2)
    dff=df[df['SEX']==2]
    dff=dff[['SEX','P_NUMVRC','HAD_CPOX']]
    dff=dff.dropna()
    dff1=dff[(dff['P_NUMVRC']>0) & (dff['HAD_CPOX']==1)]
    dff2=dff[(dff['P_NUMVRC']>0) & (dff['HAD_CPOX']==2)]
    F=len(dff1)/len(dff2)
    D={'male':M, 'female':F}
    return D
chickenpox_by_sex()
    # YOUR CODE HERE
    #raise NotImplementedError()


def corr_chickenpox():
    import scipy.stats as stats
    import numpy as np
    import pandas as pd

    # this is just an example dataframe
    #df=pd.DataFrame({"had_chickenpox_column":np.random.randint(1,3,size=(100)),
                   #"num_chickenpox_vaccine_column":np.random.randint(0,6,size=(100))})

    # here is some stub code to actually run the correlation
    #corr, pval=stats.pearsonr(df["had_chickenpox_column"],df["num_chickenpox_vaccine_column"])

    # just return the correlation

    #return corr
#corr_chickenpox()
    df=pd.read_csv('assets/NISPUF17.csv')
    dff=df[['P_NUMVRC','HAD_CPOX']]
    dff=dff[(dff['HAD_CPOX']==1) | (dff['HAD_CPOX']==2)]
    dff=dff.dropna()
    corr, pval=stats.pearsonr(dff["HAD_CPOX"],dff["P_NUMVRC"])
    #print(corr, pval)
    return corr
corr_chickenpox()
    #raise NotImplementedError()
