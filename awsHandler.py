import datetime
import stumpy
from pandas import DatetimeIndex, Series, DataFrame
from pyathena import connect
import pandas as pd
import numpy as np
import json
from json import JSONEncoder
from adtk.detector import PersistAD
from adtk.detector import OutlierDetector
from adtk.detector import AutoregressionAD
from sklearn.neighbors import LocalOutlierFactor
from os import environ

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
    return connect(aws_access_key_id = environ.get('AWS_ACCESS_KEY_ID') ,
                   aws_secret_access_key= environ.get('AWS_SECRET_ACCESS_KEY'),
                   aws_session_token= environ.get('AWS_SESSION_TOKEN'),
                   s3_staging_dir='s3://peruser-athena-result/',
                   region_name='eu-west-1')


connection = authenticate()


def getUnitID():
    spn_default = [100, 110, 183, 190, 247, 30000, 30856]

    df_unit_id = pd.read_sql(
        'SELECT DISTINCT unit_id, spn FROM tierra_dwh_blend.adq_master_can_v2 WHERE spn IN (100, 110, 183, 190, 247, 30000, 30856)',
        connection)
    df_info = pd.read_sql(
        "SELECT unit_id, unit_name FROM tierra_dwh_blend.bln_master_unit",
        connection)
    df = pd.merge(df_unit_id, df_info)
    df = df.drop(columns=['spn'])
    df = df.drop_duplicates()

    if df.empty:
        return "Record not found", 400
    return df.to_json(orient="records")


def getSPN(unitID):
    df = pd.read_sql(
        f"SELECT DISTINCT spn , spn_description FROM tierra_dwh_blend.adq_master_can_v2 WHERE unit_id={unitID} and (spn in (100, 110, 183, 190, 247, 30000, 30856))",
        connection)

    df_time = pd.read_sql(
        f"SELECT MAX(timestamp) as max, MIN(timestamp) as min FROM tierra_dwh_blend.adq_master_can_v2 WHERE unit_id={unitID}",
        connection)
    if df.empty:
        return "Record not found", 400
    if df_time.empty:
        return "Record not found", 400

    record = {'spnList': df.to_numpy(), 'info': df['spn_description'].to_numpy(), 'min_time': df_time['min'][0], 'max_time': df_time['max'][0]}
    return json.dumps({'data': record}, cls=NumpyArrayEncoder)


def getData(spns, unitID,start_time,end_time):
    df = pd.DataFrame([],
                      columns=['spn', 'timestamp', 'can_value_converted'])
    data = []
    df_workcycle = getOnOff(unitID)

    for spn in spns:
        df = pd.read_sql(
            f"SELECT DISTINCT spn , timestamp ,can_value_converted FROM tierra_dwh_blend.adq_master_can_v2 WHERE unit_id={unitID} "
            f"and spn={spn}",
            connection)
        df = df.drop_duplicates('timestamp')
        df = df.reset_index(drop=True)

        df = df.sort_values(by='timestamp')
        timestamp = df['timestamp'].to_numpy().astype('datetime64[s]')
        if df.size < 100 : continue


        tmp = pd.merge_asof(df, df_workcycle, left_on='timestamp', right_on='ontime')
        if tmp.empty:
            df['engineonoff'] = []
        else:
            tmp.loc[(tmp.ontime > tmp.timestamp) | (tmp.offtime < tmp.timestamp), 'ontime'] = pd.NaT
            tmp.loc[(tmp.ontime > tmp.timestamp) | (tmp.offtime < tmp.timestamp), 'offtime'] = pd.NaT


            m_on = tmp.groupby(tmp.ontime).apply(lambda x: x.iloc[[0]])
            m_on.index = m_on.index.droplevel(0)

            m_off = tmp.groupby(tmp.ontime).apply(lambda x: x.iloc[[-1]])
            m_off.index = m_off.index.droplevel(0)

            df['engineonoff'] = np.where(df.index.isin(m_on.index), 'ON',
                                         np.where(df.index.isin(m_off.index), 'OFF', ''))

        mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
        df = df.loc[mask]

        ids = outlierDetection(df, spn)

        record = {'spn': spn, 'x': timestamp, 'y': df['can_value_converted'].to_numpy(), 'ids': ids,
                  'workcycle': df['engineonoff']}

        data.append(record)

    if len(data) > 0:
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
    df = pd.read_sql(
        f"SELECT ontime,offtime FROM tierra_dwh_blend.bln_master_engineonoff WHERE unit_id={unitID}", connection)

    df = df.sort_values(by='ontime')
    df = df.drop_duplicates('ontime')
    df = df.reset_index(drop=True)

    return df


def getGeneralInfo(unit_id):

    df = pd.read_sql(
        f"SELECT tenant, unit_brand_name, unit_type_name, unit_model_name FROM tierra_dwh_blend.bln_master_unit WHERE unit_id={unit_id}",
        connection)
    df_pos = pd.read_sql(
        f"SELECT unlpos_address FROM blend_smart_duplication_dwh.smartdelta_tierra_dbcore_public_unl_pos WHERE unlpos_unit={unit_id}",
        connection)

    data = {'tenant': df['tenant'].to_numpy()[0], 'unit_brand_name': df['unit_brand_name'].to_numpy()[0],
            'unit_type_name': df['unit_type_name'].to_numpy()[0],
            'unit_model_name': df['unit_model_name'].to_numpy()[0],
            'unlpos_address': df_pos['unlpos_address'].to_numpy()}
    if df.empty:
        return "Record not found", 400
    return json.dumps({'data': data}, cls=NumpyArrayEncoder)


def searchPattern(unitID, spn, min_time, max_time):

    data = []
    df = pd.read_sql(
        f"SELECT timestamp ,can_value_converted FROM tierra_dwh_blend.adq_master_can_v2 WHERE "
        f"unit_id={unitID} and spn={spn}",
        connection)

    df = df.sort_values(by='timestamp')
    df = df.drop_duplicates('timestamp')
    df = df.reset_index(drop=True)
    df['timestamp'] = df['timestamp'].astype('datetime64[s]')

    mask = (df['timestamp'] >= min_time) & (df['timestamp'] <= max_time)
    motif = df.loc[mask]

    size = motif['can_value_converted'].size


    distance_profile = stumpy.core.mass(motif['can_value_converted'], df['can_value_converted'])

    theshold = 1.6
    idxs = np.array([])
    index = np.argsort(distance_profile)
    f = np.array([])
    for i in index:
        if distance_profile[i] <= theshold:
            idxs = np.append(idxs, i)
            f = np.append(f, i+size)
            if(pd.IntervalIndex.from_arrays(idxs, f).is_overlapping):
                idxs = np.delete(idxs,-1)
                f = np.delete(f,-1)

    #index = pd.IntervalIndex.from_arrays(idxs, f)
     #print(index.is_overlapping)
    idxs = [int(str(i).replace(".0","")) for i in idxs]
    #print(idxs)

    record = {'indexes': idxs, 'size': size}
    data.append(record)
    if data:
        return json.dumps({'data': record}, cls=NumpyArrayEncoder)
    else:
        return "Record not found", 400
