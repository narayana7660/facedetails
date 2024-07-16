from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    data = request.json
 
    return jsonify({'status': 'success', 'message': 'Form submitted successfully'})

if __name__ == '__main__':
    app.run(debug=True)
