import datetime

import boto3
import stumpy
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

    keyId = "ASIA5ZSVDEHMR5VWGXVO"
    sKeyId = "TI65v4vnlkCRIadd3CSI3NrcnPBMevaIAYkWShwp"
    sessionToken = "IQoJb3JpZ2luX2VjENT//////////wEaCWV1LXdlc3QtMSJGMEQCIDDTjkQqFy9c05TQqvJiVXZsFuWj/Iz3VjphdxNhR7VwAiBvGFSMAYzxZINy8yvR1Kx+b76fIxivLJY1Do/yKbrmqCqVAwhdEAIaDDk0ODI5MjQ5Mzc4NSIMj52i2k0I/L6IdlPYKvICPEM0l9wcXFz3o2FfFD+yA0zlhFhdJXHa9w3yxKLXCdTpXdzUGcP1yPxCPnSQC2D5TQBL8p/W99Jr6SDwh28EXi2g6JjHyC0TRkoxh3UJbdygEqrOQmvqhrbs8Ukv8o/ZCe+5LT+ZGy5nev+IzH8PoANtuWa181QTmsfEdHlECDElafuExjuM+mXEqE05K8v/LYjk3I4X6W/FNcmo4A9tnmFgYQr4m5hclfEgDvEaOt/7kDnmCXKq8rUn/bdtOyCIcH8E25JAwY22kq+cLNxuMGVC6nzHEOumL/Hm4uGQ0w3eT8o53ASyAlUemqSpmeIH+NJ6DZxldNFT+wuIIp/R1licI96NkA9Cnm8CB7ht0pV4PJUQQNKApIE8JFr6noF9ALkzH9y5qJV+KggnQwvT3K5ZmQXo+1bKjQwunPhopzo0dgyjeNOeBw4zNFNCv3MbZcVddsdXpVLF+XhgqceEVxkhLAGSv8iM/FNLMIgmG4Wn1jCk26WLBjqnAa1aDnXfljhOleoTQrtq04hSdZWMyOvkGT+fvb0riMg/feE1BwNb/ntar7AJnHbv6oDp7VtkCHF9EHEkHgQYemEdbpFVmxwm3YKoxlfdBZ4cr4QohAW6ysgGk+fJrgCZvdAivjowAPPYRvhSDmDjkdS/5vI+C3WYCkbI4FCJEoP2WfvApLFpo3OKvOv+Sh/nRUOIGCfsaike2SkgDKlaNqvDUXemcBbk"
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

        record = {'spn': spn, 'x': timestamp, 'y': df['can_value_converted'].to_numpy(), 'ids': ids,
                  'workcycle': df['engineonoff']}

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
    df = pd.read_sql(
        f"SELECT tenant, unit_brand_name, unit_type_name, unit_model_name FROM tierra_dwh_blend.bln_master_unit WHERE unit_id={unit_id}",
        conn)
    df_pos = pd.read_sql(
        f"SELECT unlpos_address FROM blend_smart_duplication_dwh.smartdelta_tierra_dbcore_public_unl_pos WHERE unlpos_unit={unit_id}",
        conn)

    data = {'tenant': df['tenant'].to_numpy()[0], 'unit_brand_name': df['unit_brand_name'].to_numpy()[0],
            'unit_type_name': df['unit_type_name'].to_numpy()[0],
            'unit_model_name': df['unit_model_name'].to_numpy()[0],
            'unlpos_address': df_pos['unlpos_address'].to_numpy()}
    if df.empty:
        return "Record not found", 400
    return json.dumps({'data': data}, cls=NumpyArrayEncoder)


def searchPattern(unitID, spn, min_time, max_time):
    conn = authenticate()
    data = []

    df = pd.read_sql(
        f"SELECT DISTINCT timestamp ,can_value_converted FROM tierra_dwh_blend.adq_master_can_v2 WHERE "
        f"unit_id={unitID} and spn={spn}",
        conn)
    df = df.drop_duplicates('timestamp')
    df = df.reset_index(drop=True)

    df = df.sort_values(by='timestamp')

    mask = (df.timestamp.astype('datetime64[s]') >= min_time) & (df['timestamp'].astype('datetime64[s]') <= max_time)
    motif = df.loc[mask]

    distance_profile = stumpy.core.mass(motif['can_value_converted'], df['can_value_converted'])
    k = 4
    idxs = np.argpartition(distance_profile, k)[:k]
    idxs = idxs[np.argsort(distance_profile[idxs])]
    idxs = np.delete(idxs, 0)
    # idxs è il primo indice della serie temporale
    #TODO set treshood for distance an return thant or indexes null
    size = len(motif['can_value_converted'])

    record = {'indexes': idxs, 'size': size}

    data.append(record)
    if data:
        return json.dumps({'data': data}, cls=NumpyArrayEncoder)
    else:
        return "Record not found", 400
