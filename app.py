import subprocess, os
from flask import Flask, Response, render_template, url_for, redirect, request

app = Flask(__name__)

@app.route("/home")
def home():
    return render_template("main.html")
    

@app.route("/demo")
def demo():
    def generate():
        load = request.args.get("Load from save? (Y/n) \n")
        training_start = request.args.get("Run training for model? (Y/n) \n")
        if training_start == "Y" or training_start == "y" or training_start == "Yes" or training_start == "yes":
            settings = request.args.get("Use default settings? (Y/n) \n")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(BASE_DIR, 'my_program.py')

        process = subprocess.Popen(
            ['python3', '-u', 'neural_net_v2.py'], 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in iter(process.stdout.readline, ''):
            clean_line = line.strip()
            # Server-Sent Events require data to be prefixed with "data: "
            yield f"data: {clean_line}\n\n"
            
        process.stdout.close()
        process.wait()

        yield "Program: [DONE]."

    return Response(generate(), mimetype='text/event-stream')

    # return render_template("demo.html")


@app.route("/")
def page():
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)