from flask import Flask, request, jsonify
from flask_cors import CORS
from generate_response import generate_response

app = Flask(__name__)
CORS(app)

@app.route('/conversation', methods=['POST'])
def conversation():
    data = request.json
    prompt = data.get('message', '')
    if not prompt:
        return jsonify({'error': 'No message provided'}), 400

    try:
        print(f"Received prompt: {prompt}")
        response = generate_response(prompt)
        print(f"Generated response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
