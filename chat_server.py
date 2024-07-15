import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from scripts.generate_response import generate_response, AuthenticationError

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.ERROR)

def sanitize_input(input_text):
    # TODO: Implement proper input sanitization
    return input_text.strip()

@app.route('/generate_response', methods=['POST'])
def handle_generate_response():
    try:
        user_input = request.json.get('input')
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        sanitized_input = sanitize_input(user_input)

        response = generate_response(sanitized_input)
        return jsonify({'response': response})
    except ValueError as ve:
        return jsonify({'error': f'Invalid input: {str(ve)}'}), 400
    except AuthenticationError as ae:
        return jsonify({'error': f'Authentication failed: {str(ae)}'}), 401
    except Exception as e:
        app.logger.error(f'Unexpected error: {str(e)}')
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')