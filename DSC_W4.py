import pandas as pd
import numpy as np
import scipy.stats as stats
import re
nhl_df=pd.read_csv("assets/nhl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]


def nhl_correlation():
    CCCties=cities.copy()
    NHLL=nhl_df.copy()
    CCCties=CCCties[CCCties.columns[[0,1,-1]]].dropna(subset=['NHL'])
    CCCties=CCCties[CCCties!='—'].dropna()
    CCCties.sort_values('Metropolitan area')
    NHLL=NHLL[NHLL['year']==2018][NHLL.columns[[0,2,3]]]#[10:18]
    NHLL=NHLL.drop([0,9,18,26,])
    NHLL['W']=pd.to_numeric(NHLL['W'])
    NHLL['L']=pd.to_numeric(NHLL['L'])
    NHLL['w/l']=(NHLL['W'])/(NHLL['L']+NHLL['W'])
    NHLL.index=range(0,len(NHLL))
    NHLL['CT']=[0]*len(NHLL)
    for i in NHLL.index:
        NHLL['CT'][i]=NHLL['team'][i].split()[0]
    NHLL.sort_values(by='CT')
    NHLL['CT'][18]='Minneapolis'
    NHLL['CT'][13]='Raleigh'
    NHLL['CT'][3]='Miami'
    NHLL['CT'][[14,15,12]]='New York'
    NHLL['CT'][19]='Denver'
    NHLL['CT'][30]='Phoenix'
    NHLL['CT'][24]='Los'
    NHLL['CT'][25]='San Francisco'
    NHLL.sort_values(by='CT')
    CCCties['CTS']=[0]*len(CCCties)
    for i in CCCties.index:
        CCCties['CTS'][i]=CCCties[CCCties.columns[0]][i].split()[0]
    CCCties=CCCties[CCCties.columns[[0,3,1]]]
    CCCties['CTS'][0]='New York'
    CCCties['CTS'][4]='Dallas'
    CCCties['CTS'][2]='San Francisco'
    CCCties['CTS'][8]='Minneapolis'
    CCCties['CTS'][10]='Miami'
    CCCties['CTS'][5]='Washington'
    CCCties['CTS'][43]='Vegas'
    #CCCties=CCCties[CCCties.columns[[1,2]]]
    df=CCCties.set_index(['CTS'])
    DH=NHLL.set_index(['CT'])
    indx=DH.index
    DFF=DH.merge(df,how='left',left_index=True, right_index=True)
    col=DFF.index#.iloc[3]
    #col[3]=
    DFF['column']=range(0,len(DFF))
    DFF['cit']=DFF.index
    DFF=DFF.set_index('column')
    DFF.sort_values(by='cit')
    DFF['w/l'][9]=(DFF['w/l'][9]+DFF['w/l'][10])/2
    DFF['w/l'][15]=(DFF['w/l'][15]+DFF['w/l'][16]+DFF['w/l'][17])/3
    DFF=DFF.drop([10,16,17])
    population_by_region = DFF['Population (2016 est.)[8]'] # pass in metropolitan area population from CCCties
    win_loss_by_region = DFF['w/l'] # pass in win/loss ratio from nba_df in the same order as CCCties["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"
    population_by_region=population_by_region.astype('float64')
    win_loss_by_region=win_loss_by_region.astype('float64')
    stats.pearsonr(population_by_region, win_loss_by_region)[0]

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]


nhl_correlation()



import pandas as pd
import numpy as np
import scipy.stats as stats
import re
nba_df=pd.read_csv("assets/nba.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
# YOUR CODE HERE
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nba_correlation():
    CCCTIES=cities.copy()
    NBADF=nba_df.copy()
    CCCTIES=CCCTIES[CCCTIES.columns[[0,1,4]]].dropna(subset=['NBA'])
    CCCTIES=CCCTIES[CCCTIES!='—'].dropna()
    #CCCTIES
    NBADF=NBADF[NBADF['year']==2018][NBADF.columns[[0,3]]]
    NBADF['CT']=[0]*len(NBADF)
    for i in NBADF.index:
        NBADF['CT'][i]=NBADF['team'][i].split()[0]
    NBADF['CT'][[11,10]]='New York'
    NBADF['CT'][20]='New Orleans'
    NBADF['CT'][21]='San Antonio'
    NBADF['CT'][16]='San Francisco'
    NBADF['CT'][19]='Salt'
    CCCTIES['CTS']=[0]*len(CCCTIES)
    for i in CCCTIES.index:
        CCCTIES['CTS'][i]=CCCTIES[CCCTIES.columns[0]][i].split()[0]
    CCCTIES=CCCTIES[CCCTIES.columns[[0,3,1]]]
    CCCTIES['CTS'][0]='New York'
    CCCTIES['CTS'][28]='New Orleans'
    CCCTIES['CTS'][41]='San Antonio'
    CCCTIES['CTS'][4]='Dallas'
    CCCTIES['CTS'][25]='Indiana'
    CCCTIES['CTS'][10]='Miami'
    CCCTIES['CTS'][8]='Minnesota'
    CCCTIES['CTS'][5]='Washington'
    CCCTIES['CTS'][2]='San Francisco'
    df=CCCTIES.set_index(['CTS'])
    DH=NBADF.set_index(['CT'])
    indx=DH.index
    DFF=DH.merge(df,how='left',left_index=True, right_index=True)
    DFF
    DFF['column']=range(0,len(DFF))
    DFF['cit']=DFF.index
    DFF=DFF.set_index('column')
    DFF['W/L%']=DFF['W/L%'].astype('float64')
    DFF['Population (2016 est.)[8]']=DFF['Population (2016 est.)[8]'].astype('float64')
    DFF['W/L%'][10]=(DFF['W/L%'][10]+DFF['W/L%'][11])/2
    DFF['W/L%'][17]=(DFF['W/L%'][17]+DFF['W/L%'][18])/2
    DFF=DFF.drop([11,18])
    population_by_region = DFF['Population (2016 est.)[8]'] # pass in metropolitan area population from CCCTIES
    win_loss_by_region = DFF['W/L%'] # pass in win/loss ratio from NBADF in the same order as CCCTIES["Metropolitan area"]

    #assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    #assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]


#NBADF
#CCCTIES.sort_values('CTS')






    population_by_region = DFF['Population (2016 est.)[8]'] # pass in metropolitan area population from CCCTIES
    win_loss_by_region = DFF['W/L%'] # pass in win/loss ratio from NBADF in the same order as CCCTIES["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    return stats.pearsonr(population_by_region, win_loss_by_region)


nba_correlation()




import pandas as pd
import numpy as np
import scipy.stats as stats
import re
mlb_df=pd.read_csv("assets/mlb.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]


def mlb_correlation():
    CCCT=cities.copy()
    MLBDF=mlb_df.copy()
    CCCT=CCCT[CCCT.columns[[0,1,3]]].dropna(subset=['MLB'])
    CCCT=CCCT[CCCT!='—'].dropna()
    CCCT['CTS']=[0]*len(CCCT)
    for i in CCCT.index:
        CCCT['CTS'][i]=CCCT[CCCT.columns[0]][i].split()[0]

    CCCT=CCCT[CCCT.columns[[0,3,1]]]

    CCCT['CTS'][2]='San Francisco'
    CCCT['CTS'][40]='San Diego'
    CCCT['CTS'][0]='New York'
    CCCT['CTS'][4]='Dallas'
    CCCT['CTS'][25]='Indiana'
    CCCT['CTS'][10]='Miami'
    CCCT['CTS'][8]='Minnesota'
    CCCT['CTS'][5]='Washington'
    MLBDF=MLBDF[MLBDF['year']==2018][MLBDF.columns[[0,3]]]
    MLBDF['CT']=[0]*len(MLBDF)
    for i in MLBDF.index:
        MLBDF['CT'][i]=MLBDF['team'][i].split()[0]

    MLBDF['CT'][[1,18]]='New York'
    MLBDF['CT'][28]='San Francisco'
    MLBDF['CT'][29]='San Diego'
    MLBDF['CT'][27]='Phoenix'
    MLBDF['CT'][26]='Denver'
    MLBDF['CT'][11]='San Francisco'
    MLBDF['CT'][14]='Dallas'
    df=CCCT.set_index(['CTS'])
    DH=MLBDF.set_index(['CT'])
    indx=DH.index
    DFF=DH.merge(df,how='left',left_index=True, right_index=True)
    DFF['column']=range(0,len(DFF))
    DFF['cit']=DFF.index
    DFF=DFF.set_index('column')
    DFF['W-L%']=DFF['W-L%'].astype('float64')
    DFF['Population (2016 est.)[8]']=DFF['Population (2016 est.)[8]'].astype('float64')
    DFF['W-L%'][12]=(DFF['W-L%'][12]+DFF['W-L%'][13])/2
    DFF['W-L%'][17]=(DFF['W-L%'][17]+DFF['W-L%'][18])/2
    DFF['W-L%'][23]=(DFF['W-L%'][23]+DFF['W-L%'][24])/2
    DFF['W-L%'][3]=(DFF['W-L%'][3]+DFF['W-L%'][4])/2
    DFF=DFF.drop([13,18,24,4])
    CCCT
    MLBDF
    #DFF[DFF.isna().any(axis=1)]
    population_by_region = DFF['Population (2016 est.)[8]'] # pass in metropolitan area population from CCCT
    win_loss_by_region = DFF['W-L%'] # pass in win/loss ratio from MLBDF in the same order as CCCT["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]
mlb_correlation()




import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

def nfl_correlation():
    CCCT=cities.copy()
    NFLDF=nfl_df.copy()
# YOUR CODE HERE
    CCCT=CCCT[CCCT.columns[[0,1,2]]].dropna(subset=['NFL'])
    CCCT=CCCT[CCCT!='—'].dropna()
    CCCT['CTS']=[0]*len(CCCT)
    for i in CCCT.index:
        CCCT['CTS'][i]=CCCT[CCCT.columns[0]][i].split()[0]

    CCCT=CCCT[CCCT.columns[[0,3,1]]]

    CCCT['CTS'][0]='New York'
    CCCT['CTS'][2]='San Francisco'
    CCCT['CTS'][28]='New Orleans'
    CCCT['CTS'][41]='San Antonio'
    CCCT['CTS'][4]='Dallas'
    CCCT['CTS'][25]='Indiana'
    CCCT['CTS'][10]='Miami'
    CCCT['CTS'][8]='Minnesota'
    CCCT['CTS'][5]='Washington'
    CCCT['CTS'][43]='Vegas'
    CCCT['CTS'][40]='San Diego'
    NFLDF=NFLDF[NFLDF['year']==2018][NFLDF.columns[[12,13]]]
    NFLDF['CT']=[0]*len(NFLDF)
    for i in NFLDF.index:
        NFLDF['CT'][i]=NFLDF['team'][i].split()[0]
    NFLDF=NFLDF[NFLDF['CT']!='AFC']
    NFLDF=NFLDF[NFLDF['CT']!='NFC']
    NFLDF['CT'][[4,24]]='New York'
    NFLDF['CT'][1]='Boston'
    NFLDF['CT'][31]='New Orleans'
    NFLDF['CT'][38]='San Francisco'
    NFLDF['CT'][39]='Phoenix'
    NFLDF['CT'][32]='Charlotte'
    NFLDF['CT'][12]='Indiana'
    NFLDF['CT'][19]='San Francisco'
    NFLDF['CT'][13]='Nashville'
    df=CCCT.set_index(['CTS'])
    DH=NFLDF.set_index(['CT'])
    indx=DH.index
    DFF=DH.merge(df,how='left',left_index=True, right_index=True)
    DFF['column']=range(0,len(DFF))
    DFF['cit']=DFF.index
    DFF=DFF.set_index('column')
    DFF['W-L%']=DFF['W-L%'].astype('float64')
    DFF['Population (2016 est.)[8]']=DFF['Population (2016 est.)[8]'].astype('float64')
    DFF['W-L%'][16]=(DFF['W-L%'][16]+DFF['W-L%'][17])/2
    DFF['W-L%'][22]=(DFF['W-L%'][22]+DFF['W-L%'][23])/2
    DFF['W-L%'][27]=(DFF['W-L%'][27]+DFF['W-L%'][28])/2
    DFF=DFF.drop([17,23,28])


#CCCT=CCCT[CCCT.columns[[1,2]]]
    population_by_region = DFF['Population (2016 est.)[8]'] # pass in metropolitan area population from CCCT
    win_loss_by_region = DFF['W-L%'] # pass in win/loss ratio from NFLDF in the same order as CCCT["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]

nfl_correlation()




import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nhl_df=pd.read_csv("assets/nhl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]
nba_df=pd.read_csv("assets/nba.csv")
mlb_df=pd.read_csv("assets/mlb.csv")
nfl_df=pd.read_csv("assets/nfl.csv")



def nhl_correlation():
    CC100=cities.copy()
    NHLDF=nhl_df.copy()
    # YOUR CODE HERE
    CC100=CC100[CC100.columns[[0,1,-1]]].dropna(subset=['NHL'])
    NHLDF=NHLDF[NHLDF['year']==2018][NHLDF.columns[[0,2,3]]]#[10:18]
    CC100=CC100[CC100!='—'].dropna()
    CC100.sort_values('Metropolitan area')

    NHLDF=NHLDF.drop([0,9,18,26,])
    NHLDF['W']=pd.to_numeric(NHLDF['W'])
    NHLDF['L']=pd.to_numeric(NHLDF['L'])
    NHLDF['w/l']=(NHLDF['W'])/(NHLDF['L']+NHLDF['W'])
    NHLDF.index=range(0,len(NHLDF))
    NHLDF['CT']=[0]*len(NHLDF)
    for i in NHLDF.index:
        NHLDF['CT'][i]=NHLDF['team'][i].split()[0]
    NHLDF.sort_values(by='CT')
    NHLDF['CT'][18]='Minneapolis'
    NHLDF['CT'][13]='Raleigh'
    NHLDF['CT'][3]='Miami'
    NHLDF['CT'][[14,15,12]]='New York'
    NHLDF['CT'][19]='Denver'
    NHLDF['CT'][30]='Phoenix'
    NHLDF['CT'][24]='Los'
    NHLDF['CT'][25]='San Francisco'
    NHLDF.sort_values(by='CT')
    CC100['CTS']=[0]*len(CC100)
    for i in CC100.index:
        CC100['CTS'][i]=CC100[CC100.columns[0]][i].split()[0]
    CC100=CC100[CC100.columns[[0,3,1]]]
    CC100['CTS'][0]='New York'
    CC100['CTS'][4]='Dallas'
    CC100['CTS'][2]='San Francisco'
    CC100['CTS'][8]='Minneapolis'
    CC100['CTS'][10]='Miami'
    CC100['CTS'][5]='Washington'
    CC100['CTS'][43]='Vegas'
    #CC100=CC100[CC100.columns[[1,2]]]
    df=CC100.set_index(['CTS'])
    DH=NHLDF.set_index(['CT'])
    indx=DH.index
    DFF=DH.merge(df,how='left',left_index=True, right_index=True)
    col=DFF.index#.iloc[3]
    #col[3]=
    DFF['column']=range(0,len(DFF))
    DFF['cit']=DFF.index
    DFF=DFF.set_index('column')
    DFF.sort_values(by='cit')
    DFF['w/l'][9]=(DFF['w/l'][9]+DFF['w/l'][10])/2
    DFF['w/l'][15]=(DFF['w/l'][15]+DFF['w/l'][16]+DFF['w/l'][17])/3
    DFF=DFF.drop([10,16,17])
    DFF['Population (2016 est.)[8]'] = DFF['Population (2016 est.)[8]'].astype('float64') # pass in metropolitan area population from CC100
    DFF['w/l'] = DFF['w/l'].astype('float64') # pass in win/loss ratio from nba_df in the same order as CC100["Metropolitan area"]


    return DFF

def nba_correlation():
    CC200=cities.copy()
    NBADF=nba_df.copy()
    CC200=CC200[CC200.columns[[0,1,4]]].dropna(subset=['NBA'])
    NBADF=NBADF[NBADF['year']==2018][NBADF.columns[[0,3]]]
    CC200=CC200[CC200!='—'].dropna()
    #CC200

    NBADF['CT']=[0]*len(NBADF)
    for i in NBADF.index:
        NBADF['CT'][i]=NBADF['team'][i].split()[0]
    NBADF['CT'][[11,10]]='New York'
    NBADF['CT'][20]='New Orleans'
    NBADF['CT'][21]='San Antonio'
    NBADF['CT'][16]='San Francisco'
    NBADF['CT'][19]='Salt'
    CC200['CTS']=[0]*len(CC200)
    for i in CC200.index:
        CC200['CTS'][i]=CC200[CC200.columns[0]][i].split()[0]
    CC200=CC200[CC200.columns[[0,3,1]]]
    CC200['CTS'][0]='New York'
    CC200['CTS'][28]='New Orleans'
    CC200['CTS'][41]='San Antonio'
    CC200['CTS'][4]='Dallas'
    CC200['CTS'][25]='Indiana'
    CC200['CTS'][10]='Miami'
    CC200['CTS'][8]='Minnesota'
    CC200['CTS'][5]='Washington'
    CC200['CTS'][2]='San Francisco'
    df=CC200.set_index(['CTS'])
    DH=NBADF.set_index(['CT'])
    indx=DH.index
    DFF=DH.merge(df,how='left',left_index=True, right_index=True)
    DFF
    DFF['column']=range(0,len(DFF))
    DFF['cit']=DFF.index
    DFF=DFF.set_index('column')
    DFF['W/L%']=DFF['W/L%'].astype('float64')
    DFF['Population (2016 est.)[8]']=DFF['Population (2016 est.)[8]'].astype('float64')
    DFF['W/L%'][10]=(DFF['W/L%'][10]+DFF['W/L%'][11])/2
    DFF['W/L%'][17]=(DFF['W/L%'][17]+DFF['W/L%'][18])/2
    DFF2=DFF.drop([11,18])



    return DFF2

def mlb_correlation():
    CC300=cities.copy()
    MLBDF=mlb_df.copy()
    CC300=CC300[CC300.columns[[0,1,3]]].dropna(subset=['MLB'])
    MLBDF=MLBDF[MLBDF['year']==2018][MLBDF.columns[[0,3]]]
    CC300=CC300[CC300!='—'].dropna()
    CC300['CTS']=[0]*len(CC300)
    for i in CC300.index:
        CC300['CTS'][i]=CC300[CC300.columns[0]][i].split()[0]

    CC300=CC300[CC300.columns[[0,3,1]]]

    CC300['CTS'][2]='San Francisco'
    CC300['CTS'][40]='San Diego'
    CC300['CTS'][0]='New York'
    CC300['CTS'][4]='Dallas'
    CC300['CTS'][25]='Indiana'
    CC300['CTS'][10]='Miami'
    CC300['CTS'][8]='Minnesota'
    CC300['CTS'][5]='Washington'

    MLBDF['CT']=[0]*len(MLBDF)
    for i in MLBDF.index:
        MLBDF['CT'][i]=MLBDF['team'][i].split()[0]

    MLBDF['CT'][[1,18]]='New York'
    MLBDF['CT'][28]='San Francisco'
    MLBDF['CT'][29]='San Diego'
    MLBDF['CT'][27]='Phoenix'
    MLBDF['CT'][26]='Denver'
    MLBDF['CT'][11]='San Francisco'
    MLBDF['CT'][14]='Dallas'
    df=CC300.set_index(['CTS'])
    DH=MLBDF.set_index(['CT'])
    indx=DH.index
    DFF3=DH.merge(df,how='left',left_index=True, right_index=True)
    CC300
    MLBDF


    return DFF3

def nfl_correlation():
# YOUR CODE HERE
    CC400=cities.copy()
    NFLDF=nfl_df.copy()
    CC400=CC400[CC400.columns[[0,1,2]]].dropna(subset=['NFL'])
    NFLDF=NFLDF[NFLDF['year']==2018][NFLDF.columns[[12,13]]]
    CC400=CC400[CC400!='—'].dropna()
    CC400['CTS']=[0]*len(CC400)
    for i in CC400.index:
        CC400['CTS'][i]=CC400[CC400.columns[0]][i].split()[0]

    CC400=CC400[CC400.columns[[0,3,1]]]

    CC400['CTS'][0]='New York'
    CC400['CTS'][2]='San Francisco'
    CC400['CTS'][28]='New Orleans'
    CC400['CTS'][41]='San Antonio'
    CC400['CTS'][4]='Dallas'
    CC400['CTS'][25]='Indiana'
    CC400['CTS'][10]='Miami'
    CC400['CTS'][8]='Minnesota'
    CC400['CTS'][5]='Washington'
    CC400['CTS'][43]='Vegas'
    CC400['CTS'][40]='San Diego'

    NFLDF['CT']=[0]*len(NFLDF)
    for i in NFLDF.index:
        NFLDF['CT'][i]=NFLDF['team'][i].split()[0]
    NFLDF=NFLDF[NFLDF['CT']!='AFC']
    NFLDF=NFLDF[NFLDF['CT']!='NFC']
    NFLDF['CT'][[4,24]]='New York'
    NFLDF['CT'][1]='Boston'
    NFLDF['CT'][31]='New Orleans'
    NFLDF['CT'][38]='San Francisco'
    NFLDF['CT'][39]='Phoenix'
    NFLDF['CT'][32]='Charlotte'
    NFLDF['CT'][12]='Indiana'
    NFLDF['CT'][19]='San Francisco'
    NFLDF['CT'][13]='Nashville'
    df=CC400.set_index(['CTS'])
    DH=NFLDF.set_index(['CT'])
    indx=DH.index
    DFF=DH.merge(df,how='left',left_index=True, right_index=True)
    DFF['column']=range(0,len(DFF))
    DFF['cit']=DFF.index
    DFF=DFF.set_index('column')
    DFF['W-L%']=DFF['W-L%'].astype('float64')
    DFF['Population (2016 est.)[8]']=DFF['Population (2016 est.)[8]'].astype('float64')
    DFF['W-L%'][16]=(DFF['W-L%'][16]+DFF['W-L%'][17])/2
    DFF['W-L%'][22]=(DFF['W-L%'][22]+DFF['W-L%'][23])/2
    DFF['W-L%'][27]=(DFF['W-L%'][27]+DFF['W-L%'][28])/2
    DFF4=DFF.drop([17,23,28])


    return DFF4


def sports_team_performance():

    mlb_df=mlb_correlation()
    nhl_df=nhl_correlation()
    nba_df=nba_correlation()
    nfl_df=nfl_correlation()
    nba_nfl = pd.merge(nba_df,nfl_df, on='Metropolitan area')
    pval_nba_nfl = stats.ttest_rel(nba_nfl['W/L%'],nba_nfl['W-L%'])[1]
    nba_nhl = pd.merge(nba_df,nhl_df, on='Metropolitan area')
    pval_nba_nhl = stats.ttest_rel(nba_nhl['W/L%'],nba_nhl['w/l'])[1]
    nba_mlb = pd.merge(nba_df,mlb_df, on='Metropolitan area')
    pval_nba_mlb = stats.ttest_rel(nba_mlb['W/L%'],nba_mlb['W-L%'])[1]
    mlb_nfl = pd.merge(mlb_df,nfl_df, on='Metropolitan area')
    pval_mlb_nfl = stats.ttest_rel(mlb_nfl['W-L%_x'],mlb_nfl['W-L%_y'])[1]
    mlb_nhl = pd.merge(mlb_df,nhl_df, on='Metropolitan area')
    pval_mlb_nhl = stats.ttest_rel(mlb_nhl['W-L%'],mlb_nhl['w/l'])[1]
    nhl_nfl = pd.merge(nhl_df,nfl_df, on='Metropolitan area')
    pval_nhl_nfl = stats.ttest_rel(nhl_nfl['w/l'],nhl_nfl['W-L%'])[1]
    pv = {'NFL': {"NFL": np.nan, 'NBA': pval_nba_nfl, 'NHL': pval_nhl_nfl, 'MLB': pval_mlb_nfl},
    'NBA': {"NFL": pval_nba_nfl, 'NBA': np.nan, 'NHL': pval_nba_nhl, 'MLB': pval_nba_mlb},
    'NHL': {"NFL": pval_nhl_nfl, 'NBA': pval_nba_nhl, 'NHL': np.nan, 'MLB': pval_mlb_nhl},
    'MLB': {"NFL": pval_mlb_nfl, 'NBA': pval_nba_mlb, 'NHL': pval_mlb_nhl, 'MLB': np.nan}
      }

    p_values = pd.DataFrame(pv)
    #assert abs(p_values.loc["NBA", "NHL"] - 0.02) <= 1e-2, "The NBA-NHL p-value should be around 0.02"
    #assert abs(p_values.loc["MLB", "NFL"] - 0.80) <= 1e-2, "The MLB-NFL p-value should be around 0.80"
    return  p_values #nba_df#
sports_team_performance()
