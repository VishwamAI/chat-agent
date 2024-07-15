from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts.generate_response import generate_response

app = Flask(__name__)
CORS(app)

@app.route('/generate_response', methods=['POST'])
def handle_generate_response():
    try:
        user_input = request.json.get('input')
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        response = generate_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')