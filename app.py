from flask import Flask, render_template, request, jsonify 
import openai 
import logging
  
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING,filename='example2.log', encoding='utf-8')
#logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
#logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger(__name__)
  
app = Flask(__name__) 
  
# OpenAI API Key 
openai.api_key = 'HuggingFaceH4/zephyr-7b-beta'
  
def get_completion(prompt): 
    print(prompt) 
    query = openai.Completion.create( 
        engine="text-davinci-003", 
        prompt=prompt, 
        max_tokens=1024, 
        n=1, 
        stop=None, 
        temperature=0.5, 
    ) 
  
    response = query.choices[0].text 
    logger.info(response)

    return response 
  
@app.route("/", methods=['POST', 'GET']) 
def query_view(): 
    if request.method == 'POST': 
        print('step1') 
        prompt = request.form['prompt'] 
        response = get_completion(prompt) 
        print(response) 
        logger.info(response)
        return jsonify({'response': response}) 
    return render_template('index.html') 
  
  
if __name__ == "__main__": 
    app.run(debug=True) 
    