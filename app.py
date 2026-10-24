import subprocess, os, json
from flask import Flask, Response, render_template, render_template_string, url_for, redirect, request

app = Flask(__name__)

@app.route("/home")
def home():
    return render_template("main.html")
    

@app.route("/demo")
def demo():
    settings_list = [0.001, 1, 64] 
    
    load = request.args.get("Load from save? (Y/n) \n")
    training_start = request.args.get("Run  training for model? (Y/n) \n")
    validation = request.args.get("Run validation test for model? (Y/n) \n")

    settings = request.args.get("Use default settings? (Y/n) \n")
    if settings == "y" or settings == "Y" or settings == "Yes" or settings == "yes":
        settings_list = [0.001, 1, 64]
    elif settings == "n" or settings == "N" or settings == "No" or settings == "no":
        learning_rate =   float(request.args.get("Learning rate (float) =    "))
        epochs =            int(request.args.get("Epochs (int) =             "))
        batch_size =        int(request.args.get("Batch size (int) =         "))
        settings_list = [learning_rate, epochs, batch_size]



    json_list = json.dumps(settings_list)

    def generate():
            
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(BASE_DIR, 'my_program.py')

        process = subprocess.Popen(
            ['python3', '-u', '/version 2/neural_net_v2.py', load, training_start, json_list, validation], 
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