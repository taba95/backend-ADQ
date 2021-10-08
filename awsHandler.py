import datetime

import boto3
from pandas import DatetimeIndex, Series
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
        if isinstance(obj, Series):
            return  obj.tolist()
        return JSONEncoder.default(self, obj)



def authenticate():
    keyId = "ASIA5ZSVDEHM7ZXHYTTY"
    sKeyId = 'sWikpvPCZ0cfuJj1vrOE8nM/sdLwCcY4mUOP5Xc0'
    sessionToken = "IQoJb3JpZ2luX2VjEC0aCWV1LXdlc3QtMSJHMEUCIB97jO4am3VuyV4T28zIROwdQSFyBsqoupdYOkYoKleAAiEA264UrBSG7G/ErDyCwBV4jw14dIwoTHXEowim1ySyJnIqngMIpv//////////ARACGgw5NDgyOTI0OTM3ODUiDE+nPWFmt7+L3nYDvyryAtSDNGVknSUWkGTM1QT5yV4xYrjvQUMrwIyfLFreNjkl/RZIgLszzocKpbtkv19Iq/7/09v5b8t+QWqeVEJ6kMtuhQwiXDboU9pMkrI4r81S5dB5PAFSKe6m1Hw58irqNnYUrIq/Uw7YuAAvM0uIdYLFUwNG3vzTTKP9nlFTwROkRW3dZMcnILvF1+xvfGVmZFeZPODxSGhEwj+BxEe8CAiqj62RmHySgIp4mBlmk6I9QxViwHXb1rBG6ypH34TbnsH7iifqmdSbFXHRyRAmDvYTJF8elvjX8YZPVdoTADtuRL4I+K3LC24weT3uVhZCqvkKmuJI62ry7ojRhYxMODZWCHT0/SZiem4dYUVUVAvR+YebNwuZtTQKT0MOkvelWmAqakMKfpxCyMer4c5akNGfrvbBTjthCs9ao8DEBMhOBaOrkBZklSLWrVeREnL+RHiN1zlUnPZr4dO4jg1Kk1Cq+oNmKsmcGgCNvlfwxKUt2wcwt/6AiwY6pgHKlMsxB5vEpDmaW4ARxUbKxdE4PI6tA/QAzIo5tUIPBTA5u2lKV4FNLCJ+KTN9RmfjA9lkUrh9j9972+yP+TNYRSs0RqquP4IKqHcssUbxeabsCwGGGi5mJRRHgYldbLEvsuW6NByYjaRgbLi2h6W6fOUVy55IWo66LreUs9KLjmTV9GHmElr7Y/t6vQhpg4T303UlOoSmIHsj7W62DUqlW2nL26aY"
    return connect(aws_access_key_id=keyId,
                   aws_secret_access_key=sKeyId,
                   aws_session_token=sessionToken,
                   s3_staging_dir='s3://peruser-athena-result/',
                   region_name='eu-west-1')

    # return boto3.client("s3",
    #                     aws_access_key_id=keyId,
    #                     aws_secret_access_key=sKeyId,
    #                     aws_session_token=sessionToken,
    #                     )


# def getS3File(fileName, spn, unitID):
#     cursor = authenticate()
#     bucketName = "politecnico-data"
#
#     print(fileName)
#
#     response = s3.get_object(Bucket=bucketName, Key=fileName)
#     status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
#
#     if status == 200:
#         print(f"Successful S3 get_object response. Status - {status}")
#
#         df = pd.read_csv(response.get("Body"),
#                          usecols=['timestamp', 'unit_ID', 'spn', 'can_value_converted',
#                                   'engineonoff', 'ontime',
#                                   'offtime', 'deltaT'])
#         df['timestamp'] = df['timestamp'].astype('datetime64[s]')
#         df = df.replace(np.nan, '', regex=True)
#         df = df[df.spn == spn]
#         print(df.size)
#         df = df[df.unit_ID == unitID]
#         print(df.size)
#         print(df.spn.unique())
#         return df.to_json(orient="records")
#
#     else:
#         print(f"Unsuccessful S3 get_object response. Status - {status}")
#
#     return "ciao"

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
    for spn in spns:
        df = pd.read_sql(
            f"SELECT DISTINCT spn , timestamp ,can_value_converted FROM tierra_dwh_blend.adq_master_can_v2 WHERE unit_id={unitID} and spn={spn}",
            conn)
        df = df.drop_duplicates('timestamp')
        df = df.reset_index(drop=True)

        df = df.sort_values(by='timestamp')
        df['timestamp'] = df['timestamp'].astype('datetime64[s]')

        x_o, y_o = outlierDetection(df, spn)
        timestamp = df['timestamp'].to_numpy().astype('datetime64[s]')

        if len(x_o)==0:
            record = {'spn': spn, 'x': timestamp, 'y': df['can_value_converted'].to_numpy(),
                      'x_o': x_o, 'y_o': y_o}
        else:
            x_o = x_o.to_numpy().astype('datetime64[s]')
            y_o = y_o.to_numpy()

            record = {'spn': spn, 'x': timestamp, 'y': df['can_value_converted'].to_numpy(),
                      'x_o': x_o, 'y_o': y_o}

        data.append(record)
    if data:
        return json.dumps({'data': data}, cls=NumpyArrayEncoder)


    else:
        return "Record not found", 400


def outlierDetection(df, spn):
    x_o = []
    y_o = []
    data = df
    data = data.reset_index(drop=True)
    data = data.set_index('timestamp')
    data_v = pd.DataFrame(data.can_value_converted)
    if int(spn) == 110:
        outlier_detector = OutlierDetector(LocalOutlierFactor(contamination='auto'))
        anomaliesLOF = outlier_detector.fit_detect(data_v)

        anomaliesLOF = pd.DataFrame(anomaliesLOF)
        anomaliesLOF.columns = ['can_value_converted']
        indice_l = np.where(anomaliesLOF['can_value_converted'] == True)
        x_o = data.iloc[indice_l].index
        y_o = data.iloc[indice_l].can_value_converted
        # text = data.iloc[indice_l]['engineonoff']
    if int(spn) == 190 or int(spn) == 100:
        autoreg = AutoregressionAD(n_steps=11, step_size=1, regressor=None, c=6.0, side='both')
        anomaliesAR = autoreg.fit_detect(data_v)
        indice_ar = np.where(anomaliesAR['can_value_converted'] == 1)
        x_o = data.iloc[indice_ar].index
        y_o = data.iloc[indice_ar].can_value_converted
        # text = data.iloc[indice_ar]['engineonoff']

    if int(spn) == 183 or int(spn) == 247 or int(spn) == 30856 or int(spn) == 30000:
        persistAD = PersistAD(window=11, c=6.0, side='both', min_periods=None, agg='median')

        anomaliesP = persistAD.fit_detect(data_v)
        indice_p = np.where(anomaliesP['can_value_converted'] == 1)
        x_o = data.iloc[indice_p].index
        y_o = data.iloc[indice_p].can_value_converted
        # text = data.iloc[indice_p]['engineonoff']


    return x_o, y_o
