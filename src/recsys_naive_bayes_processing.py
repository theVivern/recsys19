import numpy as np
import pandas as pd


def process_test_naives_bayes(data: pd.DataFrame, metadata: pd.DataFrame, encoders: dict, config: dict):
    
    # retain only rows with reference nan or numeric
    data=data.loc[(data.reference.isnull()) | (data.reference.str.contains('^[0-9]*$'))]
    
    # drop unwanted columns
    # data=data.drop(['is_validation','is_train'],axis=1)
    
    # return sessions with clicout with nan reference
    sessions_with_ref_nan = data.loc[data.reference.isnull()].session_id
    data=data.loc[data.session_id.isin(sessions_with_ref_nan)]
    
    # add number of clickouts/session
    n_clickouts=data.loc[data.reference.isnull()].groupby('session_id',as_index=False)['step'].count().rename(columns={'step':'n_clickouts'})
    data=data.join(n_clickouts.set_index('session_id'), on='session_id')
    
    # get step number of clickouts to predict
    nan_clickout_step_num=data.loc[data.reference.isna(),['session_id','step']].rename(columns={'step':'step_clickout'})
    data=data.join(nan_clickout_step_num.set_index('session_id'), on='session_id')
    
    # make unique key using step_clickout (accounts for sessions with multiple clickouts)
    data['key'] = (data['user_id'] + '_' + data['session_id'] + '_' + data['step_clickout'].astype(str))
    
    # return only steps that are less than clickout to predict
    data = data.loc[data.step <= data.step_clickout]
    
    # add group length (accounts for removal of non-numeric references)
    count_by_key=data.groupby('key',as_index=False)['step'].count().rename(columns={'step':'n_group'})
    data=data.join(count_by_key.set_index('key'), on='key')
    
    # add renumbered step
    data['obs_num'] = 1
    data['obs_num'] = data.groupby('key')['obs_num'].cumsum()
    
    # add inverted step numer
    data['step_inverted']=data['n_group']-data['obs_num']
    # remove all step_inverted greater than session_length
    data = data.loc[data.step_inverted <= config['session_length']]

    # encode
    if list(encoders.keys()) is not None:
        for col in list(encoders.keys()):
            data[col]=encoders[col].transform(data[col])
    
    
    # generate wide format for each session where steps and references are wide
    if config['add_prices']:
        data_wide=data.pivot(index='key',columns='step_inverted',values=['action_type','reference','mean_prices']).copy()
    else:
        data_wide=data.pivot(index='key',columns='step_inverted',values=['action_type','reference']).copy()
    
    # drop actual clickout to predict
    data_wide.drop(0, axis=1, level=1,inplace=True)
    
    # collapse column multi index for ease of indexing
    data_wide.columns=data_wide.columns.map(lambda x: '|'.join([str(i) for i in x]))
    
    # add metadata for each reference 
    for i in range(config['session_length']):
        data_wide=data_wide.join(metadata.set_index('item_id'), on=('reference|' + str(i+1)),rsuffix = ('|' + str(i+1)))
    
    # add platform, device
    platform_device=data.loc[(data.key.isin(data_wide.index)) & (data.step_inverted==0),np.append(config['cols_to_append'],['key','impressions','timestamp','target'])]
    
    data_wide=data_wide.join(platform_device.set_index('key'))

    # fill na 0
    data_wide=data_wide.fillna(0)
    

    
    return data_wide


def process_train_naives_bayes(data: pd.DataFrame, metadata: pd.DataFrame, encoders: dict, config: dict):
    
    # retain only rows with numeric reference
    data=data.loc[data.reference.str.contains('^[0-9]*$')]
    
    # drop unwanted columns
    # data=data.drop(['is_validation','is_train'],axis=1)
    
    # return sessions with clickout item
    sessions_with_clickout = data.loc[data.action_type=='clickout item'].session_id
    data=data.loc[data.session_id.isin(sessions_with_clickout)]
    
    # add number of clickouts/session
    n_clickouts=data.loc[data.action_type=='clickout item'].groupby('session_id',as_index=False)['step'].count().rename(columns={'step':'n_clickouts'})
    data=data.join(n_clickouts.set_index('session_id'), on='session_id')
    
    # get step number of clickouts to predict
    nan_clickout_step_num=data.loc[data.action_type=='clickout item',['session_id','step']].rename(columns={'step':'step_clickout'})
    data=data.join(nan_clickout_step_num.set_index('session_id'), on='session_id')
    
    # make unique key using step_clickout (accounts for sessions with multiple clickouts)
    data['key'] = (data['user_id'] + '_' + data['session_id'] + '_' + data['step_clickout'].astype(str))
    
    # return only steps that are less than clickout to predict
    data = data.loc[data.step <= data.step_clickout]
    
    # add group length (accounts for removal of non-numeric references)
    count_by_key=data.groupby('key',as_index=False)['step'].count().rename(columns={'step':'n_group'})
    data=data.join(count_by_key.set_index('key'), on='key')
    
    # add renumbered step
    data['obs_num'] = 1
    data['obs_num'] = data.groupby('key')['obs_num'].cumsum()
    
    # add inverted step numer
    data['step_inverted']=data['n_group']-data['obs_num']
    # remove all step_inverted greater than session_length
    data = data.loc[data.step_inverted <= config['session_length']]

    # encode
    if list(encoders.keys()) is not None:
        for col in list(encoders.keys()):
            data[col]=encoders[col].transform(data[col])
    
    
    # generate wide format for each session where steps and references are wide
    if config['add_prices']:
        data_wide=data.pivot(index='key',columns='step_inverted',values=['action_type','reference','mean_prices']).copy()
    else:
        data_wide=data.pivot(index='key',columns='step_inverted',values=['action_type','reference']).copy()
    
    # collapse column multi index for ease of indexing
    data_wide.columns=data_wide.columns.map(lambda x: '|'.join([str(i) for i in x]))

    # change object to category
    # data_wide=data_wide[data_wide.columns[data_wide.dtypes=='object']].astype('category')
    
    # Get ground truth
    data_wide['y'] = data_wide['reference|0']
    
    # drop actual clickout to predict
    if config['add_prices']:
        data_wide.drop(['action_type|0','reference|0','mean_prices|0'],axis=1,inplace=True)
    else:
        data_wide.drop(['action_type|0','reference|0'],axis=1,inplace=True)
    
    # add metadata for each reference 
    for i in range(config['session_length']):
        data_wide=data_wide.join(metadata.set_index('item_id'), on=('reference|' + str(i+1)),rsuffix = ('|' + str(i+1)))
    
    # add platform, device
    if config['cols_to_append'] is not None:
        platform_device=data.loc[(data.key.isin(data_wide.index)) & (data.step_inverted==0),np.append(config['cols_to_append'],['key'])]

        data_wide=data_wide.join(platform_device.set_index('key'))
    
    # fill na 0
    data_wide=data_wide.fillna(0)
   
        
    return data_wide

