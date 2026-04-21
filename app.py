# app.py
from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/ingest', methods=['POST'])
def ingest():
    data = request.json
    result = subprocess.run([
        'python', 'sla_rag_pipeline.py', 'ingest',
        '--pdf', data['pdf'],
        '--chroma-dir', data.get('chroma_dir', './chroma_db'),
        '--output-dir', data.get('output_dir', './output'),
    ], capture_output=True, text=True)
    return jsonify(json.loads(result.stdout))

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    result = subprocess.run([
        'python', 'sla_rag_pipeline.py', 'query',
        '--tier', data['tier'],
        '--incident-start', data['incident_start'],
        '--current-time', data['current_time'],
        '--service', data.get('service', 'NovaSuite Platform'),
        '--chroma-source', data['chroma_source'],
        '--prior-downtime-min', str(data.get('prior_downtime_min', 0.0)),
    ], capture_output=True, text=True)
    return jsonify(json.loads(result.stdout))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
