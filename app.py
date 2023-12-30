from flask import Flask, render_template, request

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
app = Flask(__name__)

model_name = "google/pegasus-xsum"

tokenizer = PegasusTokenizer.from_pretrained(model_name) 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST']) 
def submit():
    if request.method == 'POST':
        input_text = request.form.get('inputtext_') 
        print("Received input text:", input_text)
        tokenized_text = tokenizer.encode(input_text, return_tensors="pt", max_length=512 , truncation=True).to(device) 
        summary_ = model.generate(tokenized_text, min_length=30, max_length=300)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True) 
        return render_template('output.html', result=summary)

if __name__ == '__main__':
    app.run()
