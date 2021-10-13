import datetime

import boto3
from pandas import DatetimeIndex, Series, DataFrame
from pyathena import connect
import pandas as pd
import numpy as np
import json
from json import JSONEncoder
from sklearn.preprocessing import StandardScaler
from adtk.detector import PersistAD
from adtk.detector import OutlierDetector
from adtk.detector import AutoregressionAD
from sklearn.neighbors import LocalOutlierFactor


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, DatetimeIndex):
            return obj.to_numpy().tolist()
        if isinstance(obj, DataFrame):
            return obj.to_numpy().tolist()
        if isinstance(obj, Series):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def authenticate():

    keyId = "ASIA5ZSVDEHMXE57NHVP"
    sKeyId = "qE1uvhQfhna3d5l56MXhwRHZUrYdy8A0134ClG95"
    sessionToken = "IQoJb3JpZ2luX2VjEKX//////////wEaCWV1LXdlc3QtMSJHMEUCIEJXBP9t8SR6jddp7ZTQwi1ADXhmCiNn1k/AGCP8BEmvAiEAm0qxEUBdNr3GitI5HArC2+oaVcvhgvRj1hNaJjfxge4qlQMILRACGgw5NDgyOTI0OTM3ODUiDOyMZlpfiOb9LnNxQCryAuZZ1hWhUwM51JRyb/9crEJ2RD3FLSZExFlu4mUbnNd9emu2H8oUjqEzURQFF/JerJICAAGIerfAxXVc1MWMrkNQjKAUZFG4vsU4N2nmRkIMyvke2kXdzNQdk0uXFeuwUof7GppgWb4+t2s21mvJuirnNYiDlflT4bmuKr5eJKEaB3KKk+6WDYqQIPpghblFRAvzUssBppRcnnCFH8odCxrkqD4vDPvyUWy1/nOdt624qvv4g/HuCmGghyPjxI6cwjTKKlaWe1Su+VzHxZVd35KMlmVwXDOaSTvzGJC6vj+DrOq4rn7oQlAgX4Sz4JrMKmZgRlwwQAA5N2o4u7mVmDPLZqnNzP+XFket0a1LGHhmct0mpy3lGc9yN4zPfYnuYnKi2QZkNqxbbdbTvWdusP5ylzTUYtCm0AbuWXu2usJwoBTr1CNA6MqQZ06aoAp7VbXsoJbUr+2m1quJoHm3N8FUz+ObNoD2QKYL0hvKiH3sghwwqJ2biwY6pgFU1BkTU/A+jNrECwY7LMMdXzWePDPhx4UUwjiLZxcTPo2aGEwPA3kVQk1sT1MJYPK+WcoJdDnUSdb3zyZ0e7gWYWQRs9X7LrhnjeSwJm6qGC6WGoubh+2yG3waZHeLQsZfwf98n+AWi34xd06a6iLrE5U1nBYaoIrfpkPxB6Ywjy+ZfXPjcFpKWaXG4h9mwLGdqBfQVX7SUyHCiOdy9Gu42BW+YFfP"
    return connect(aws_access_key_id=keyId,
                   aws_secret_access_key=sKeyId,
                   aws_session_token=sessionToken,
                   s3_staging_dir='s3://peruser-athena-result/',
                   region_name='eu-west-1')

def getUnitID():
    spn_default = [100, 110, 183, 190, 247, 30000, 30856]
    conn = authenticate()
    df_unit_id = pd.read_sql(
        'SELECT DISTINCT unit_id, spn FROM tierra_dwh_blend.adq_master_can_v2 WHERE spn IN (100, 110, 183, 190, 247, 30000, 30856)',
        conn)
    df_info = pd.read_sql(
        "SELECT unit_id, unit_name FROM tierra_dwh_blend.bln_master_unit",
        conn)
    df = pd.merge(df_unit_id, df_info)
    df = df.drop(columns=['spn'])
    df = df.drop_duplicates()
    if df.empty:
        return "Record not found", 400
    return df.to_json(orient="records")


def getSPN(unitID):
    conn = authenticate()
    df = pd.read_sql(
        f"SELECT DISTINCT spn , spn_description FROM tierra_dwh_blend.adq_master_can_v2 WHERE unit_id={unitID} and (spn in (100, 110, 183, 190, 247, 30000, 30856))",
        conn)
    if df.empty:
        return "Record not found", 400
    return df.to_json(orient="records")


def getData(spns, unitID):
    conn = authenticate()
    df = pd.DataFrame([],
                      columns=['spn', 'timestamp', 'can_value_converted'])
    data = []
    df_workcycle = getOnOff(unitID)

    for spn in spns:
        df = pd.read_sql(
            f"SELECT DISTINCT spn , timestamp ,can_value_converted FROM tierra_dwh_blend.adq_master_can_v2 WHERE unit_id={unitID} and spn={spn}",
            conn)
        df = df.drop_duplicates('timestamp')
        df = df.reset_index(drop=True)

        df = df.sort_values(by='timestamp')
        timestamp = df['timestamp'].to_numpy().astype('datetime64[s]')

        tmp = pd.merge_asof(df, df_workcycle, left_on='timestamp', right_on='ontime')
        tmp.loc[(tmp.ontime > tmp.timestamp) | (tmp.offtime < tmp.timestamp), 'ontime'] = pd.NaT
        tmp.loc[(tmp.ontime > tmp.timestamp) | (tmp.offtime < tmp.timestamp), 'offtime'] = pd.NaT

        m_on = tmp.groupby(tmp.ontime).apply(lambda x: x.iloc[[0]])
        m_on.index = m_on.index.droplevel(0)

        m_off = tmp.groupby(tmp.ontime).apply(lambda x: x.iloc[[-1]])

        m_off.index = m_off.index.droplevel(0)

        df['engineonoff'] = np.where(df.index.isin(m_on.index), 'ON',
                                       np.where(df.index.isin(m_off.index), 'OFF', ''))

        ids = outlierDetection(df, spn)

        record = {'spn': spn, 'x': timestamp, 'y': df['can_value_converted'].to_numpy(), 'ids': ids, 'workcycle': df['engineonoff']}

        data.append(record)
    if data:
        return json.dumps({'data': data}, cls=NumpyArrayEncoder)
    else:
        return "Record not found", 400


def outlierDetection(df, spn):
    ids = []
    data = df
    data = data.reset_index(drop=True)
    data = data.set_index('timestamp')
    data_v = pd.DataFrame(data.can_value_converted)
    if int(spn) == 110:
        outlier_detector = OutlierDetector(LocalOutlierFactor(contamination='auto'))
        anomaliesLOF = outlier_detector.fit_detect(data_v)
        anomaliesLOF = pd.DataFrame(anomaliesLOF)
        anomaliesLOF.columns = ['can_value_converted']
        ids = anomaliesLOF['can_value_converted'].apply(lambda x: 1 if x == True else 0)

    if int(spn) == 190 or int(spn) == 100:
        autoreg = AutoregressionAD(n_steps=11, step_size=1, regressor=None, c=6.0, side='both')
        anomaliesAR = autoreg.fit_detect(data_v)
        ids = anomaliesAR['can_value_converted'].apply(lambda x: 1 if x == True else 0)
    if int(spn) == 183 or int(spn) == 247 or int(spn) == 30856 or int(spn) == 30000:
        persistAD = PersistAD(window=11, c=6.0, side='both', min_periods=None, agg='median')
        anomaliesP = persistAD.fit_detect(data_v)
        ids = anomaliesP['can_value_converted'].apply(lambda x: 1 if x == True else 0)

    return ids


def getOnOff(unitID):
    conn = authenticate()
    df = pd.read_sql(
        f"SELECT ontime,offtime FROM tierra_dwh_blend.bln_master_engineonoff WHERE unit_id={unitID}", conn)


    df = df.sort_values(by='ontime')
    df = df.drop_duplicates('ontime')
    df = df.reset_index(drop=True)


    return df


def getGeneralInfo(unit_id):
    conn = authenticate()
    df = pd.read_sql(f"SELECT tenant, unit_brand_name, unit_type_name, unit_model_name FROM tierra_dwh_blend.bln_master_unit WHERE unit_id={unit_id}", conn)
    df_pos = pd.read_sql(f"SELECT unlpos_address FROM blend_smart_duplication_dwh.smartdelta_tierra_dbcore_public_unl_pos WHERE unlpos_unit={unit_id}", conn)

    data = {'tenant': df['tenant'].to_numpy()[0], 'unit_brand_name': df['unit_brand_name'].to_numpy()[0], 'unit_type_name': df['unit_type_name'].to_numpy()[0],
            'unit_model_name': df['unit_model_name'].to_numpy()[0], 'unlpos_address': df_pos['unlpos_address'].to_numpy()}
    if df.empty:
        return "Record not found", 400
    return json.dumps({'data': data}, cls=NumpyArrayEncoder)

