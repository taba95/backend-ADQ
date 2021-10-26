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


@app.route('/searchPattern', methods=['GET'])
def getPattern():
    unitID = request.args.get("unit_id")
    spn = request.args.get("spn")
    min_time=request.args.get("min_time")
    max_time=request.args.get("max_time")
    return awsHandler.searchPattern(unitID,spn,min_time,max_time)





if __name__ == '__main__':
    app.run()
