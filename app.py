from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin

import awsHandler;
from someException import SomeException

app = Flask(__name__)
cors = CORS(app, resources={"/*": {"origins": "*"}})


@app.errorhandler(SomeException)
def handle_error(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/unit_id', methods=['GET'])
def getUnitID():
    return awsHandler.getUnitID()


@app.route('/spn', methods=['GET'])
def getSPN():
    request_data = request.args.get("unit_id")
    print(request_data)
    if not request_data:
        return "Errore"
    else:
        return awsHandler.getSPN(request_data)


@app.route('/values', methods=['GET'])
def getValues():
    unitID = request.args.get("unit_id")
    spns = request.args.getlist('spn')
    # elements = [x for x in array if x["spn"] in request_data_list]
    return awsHandler.getData(spns, unitID)


@app.route('/info', methods=['GET'])
def getInfo():
    unitID = request.args.get("unit_id")
    return awsHandler.getGeneralInfo(unitID)



@app.route('/plotGetData', methods=['POST'])
def plotGetData():
    request_data = request.get_json()
    folder = 'active_data_quality/'
    nameFile = request_data['nameFile']
    path = folder + nameFile
    spn = int(request_data['spn'])
    unitID = int(request_data['unitID'])
    print(nameFile)
    print(spn)
    print(unitID)
    print(type(unitID))
    print(type(spn))
    # TODO VERIFICA TYPE E LUNGEZZA DATI
    return awsHandler.getS3File(path, spn, unitID)


if __name__ == '__main__':
    app.run()
