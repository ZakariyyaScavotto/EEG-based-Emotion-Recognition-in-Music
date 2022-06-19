from flask import Flask, render_template, jsonify
import prototypingFinalPythonCodeFlaskVersion
import testCodeforScript
import bazgirFinalPythonCode

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/backgroundCodeTest')
def backgroundCodeTest():
        #while True:
    #testReturn = testCodeforScript.testMain()
    # testReturn = prototypingFinalPythonCodeFlaskVersion.mainBackground()
    print('moving into program')
    testReturn = bazgirFinalPythonCode.mainBackground()
    return jsonify({'output':testReturn})

app.run()