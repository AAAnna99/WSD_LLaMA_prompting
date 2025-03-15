import os
os.environ["HF_HOME"] = "huggingface_home"
os.environ["HF_HUB_CACHE"]= "huggingface_cache"

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
import lxml.etree as ET
import torch
import re
import transformers


# Estrazione significati corretti dal gold file
def parse_gold_dataset(gold_dataset):
    gold_senses = {}
    with open(gold_dataset, 'r') as file:
        for line in file:
            line = line.strip().split()
            instance_id = line[0]
            instance_sense = line[1:]
            gold_senses[instance_id] = instance_sense

    return gold_senses

# Estrazione parole target, pos tag e contesto dal dataset di input
def load_input_dataset(input_dataset):
    pos_dict = {'ADJ': 'a', 'ADV': 'r', 'NOUN': 'n', 'VERB': 'v'}
    tree = ET.iterparse(input_dataset, tag='sentence')
    sentences = []
    pos_set = set()

    for event, sentence in tree:
        # lista parole del contesto
        words_sentence_list = []     
        # parole con pos tag in pos_dict
        main_words_sentence_list = []   
        pos_list = []               
        # lunghezza in parole (token) di ogni elemento della frase
        token_length_list = []      

        for child in sentence:          
            word = child.text
            token_length_list.append(len(word.split()))

            for w in word.split():
                words_sentence_list.append(w)
            pos = child.get('pos')  
            if pos in pos_dict:
                main_words_sentence_list.append(word)
            pos_list.append(pos)
            pos_set.add(pos)

        i = -1
        idx = -1
        for child in sentence:
            if child.tag == 'wf':
                i += 1
                idx += 1
            elif child.tag == 'instance':
                i += 1
                idx += 1
                id = child.get('id')
                lemma = child.get('lemma').lower()
                pos = child.get('pos')
                word = lemma + '#' + pos_dict[pos]

                context = words_sentence_list[:] 

                single_sentence_final = {
                    'id': id,
                    'context': context,
                    'target_word': word,
                    'target_len': token_length_list[i]
                }
                idx += token_length_list[i] - 1
                sentences.append(single_sentence_final)
    return sentences

# Estrazione sensi da WordNet
def get_wn_senses(lemma, pos, min_sense_freq):
    wn_pos = {
        'n': wn.NOUN,
        'v': wn.VERB,
        'a': wn.ADJ,
        'r': wn.ADV
        }

    wn_senses = {}

    for l in wn.lemmas(lemma, wn_pos[pos]):
        if l.count() > min_sense_freq:

            wn_senses[l.key()] = {
                "Word": lemma,
                "Gloss": l.synset().definition(),
                "Frequency": l.count(),
                "Example": l.synset().examples()[0] if l.synset().examples() else "No example available"
            }

    return wn_senses


def zero_shot_prompt(instance, senses):
    context = " ".join(instance['context'])
    target_word = instance['target_word'].split('#')[0]

    possible_meanings = ""
    
    for i, (sense_key, sense_data) in enumerate(senses.items(), 0):
        possible_meanings += (
            f"sense_number {i}: Definition: {sense_data['definition']}\n"
            f"Frequency: {sense_data['frequency']}\n"
            f"Example: {sense_data['example']}\n\n"
        )

    system_prompt = f"""You are a linguistic expert in word sense disambiguation (WSD).
    Your task is to identify the correct sense for the target word based on the context.
    """

    user_prompt = f"""
    Context: "{context}". 
    Target word: "{target_word}".
    Possible senses list for "{target_word}": {possible_meanings}
    Return JSON object that contains the finalized sense_number. Use the following format for the output.
    <JSON Object with ambiguity word and the sense_number>    
    """  
    prompt = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
      ]

    return prompt
    
def zero_shot_COT_prompt(instance, senses):
    context = " ".join(instance['context'])
    target_word = instance['target_word'].split('#')[0]

    possible_meanings = ""
    
    for i, (sense_key, sense_data) in enumerate(senses.items(), 0):
        possible_meanings += (
            f"sense_number {i}: Definition: {sense_data['definition']}\n"
            f"Frequency: {sense_data['frequency']}\n"
            f"Example: {sense_data['example']}\n\n"
        )

    system_prompt = f"""You are a linguistic expert in word sense disambiguation (WSD).
    Your task is to identify the correct sense for the target word based on the context.
    """

    user_prompt = f"""
    Context: "{context}". 
    Target word: "{target_word}".
    Possible senses list for "{target_word}": {possible_meanings}
    Return JSON object that contains the finalized sense_number. Use the following format for the output.
    <JSON Object with ambiguity word and the sense_number>  

    Let’s think step by step.  
    """  
    prompt = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
      ]

    return prompt
    
def few_shot_prompt(instance, senses):
    context = " ".join(instance['context'])
    target_word = instance['target_word'].split('#')[0]

    possible_meanings = ""
    
    for i, (sense_key, sense_data) in enumerate(senses.items(), 1):
        possible_meanings += (
            f"Sense Number {i}:\n"
            f"Definition: {sense_data['definition']}\n"
            f"Frequency: {sense_data['frequency']}\n\n"
            f"Example: {sense_data['example']}\n\n"
        )   

    system_prompt =  f"""
    You are a linguistic expert in word sense disambiguation (WSD). 
    Your task is to identify the correct sense_number for the target word based on the context.
    """

    user_prompt = f"""\n\nExample 1
                    \nContext: 'The bat flew out of the cave at dusk.'
                    \nTarget word: 'bat'
                    \nPossible senses for 'bat': 
                    \nSense Number 1: 'nocturnal mouselike mammal with forelimbs modified to form membranous wings and anatomical adaptations for echolocation by which they navigate'
                    \nFrequency: 2 
                    \nExample: 'No example available'
                    \n\nSense Number 2: '(baseball) a turn trying to get a hit'
                    \nFrequency: 1
                    \nExample: 'he was at bat when it happened'
                    \n\nSense Number 3: 'a small racket with a long handle used for playing squash'
                    \nFrequency: 0 
                    \nExample: 'No example available'
                    \n\nSense Number 4: 'the club used in playing cricket'
                    \nFrequency: 0 
                    \nExample: 'a cricket bat has a narrow handle and a broad flat end for hitting'
                    \n\nSense Number 5: 'a club used for hitting a ball in various games'
                    \nFrequency: 0 
                    \nExample: 'No example available'
                    
                    \n\nResponse: Sense Number: 1

                    \n\nFollowing the same format above from example:
                    \nContext: '{context}'.
                    \nTarget word: '{target_word}'.
                    \nPossible senses for '{target_word}':
                    '{possible_meanings}'
                    \nSelect the Correct Sense Number.                    
                    """  
                         
                    # \n\nExample 2
                    # \nContext: "Employees in this company enjoy several health benefits and a generous retirement plan."
                    # \nTarget word: "enjoy"
                    # \nPossible senses for "enjoy":
                    # \nSense Number 1: "derive or receive pleasure from; get enjoyment from; take pleasure in"
                    # \nFrequency: 52
                    # \nExample: "She relished her fame and basked in her glory."
                    # \n\nSense Number 2: "have benefit from"
                    # \nFrequency: 15
                    # \nExample: "enjoy privileges"
                    # \n\nSense Number 3: "get pleasure from"
                    # \nFrequency: 10
                    # \nExample: "I love cooking."
                    # \n\nSense Number 4: "have for one's benefit"
                    # \nFrequency: 9\n
                    # \nExample: "The industry enjoyed a boom."
                    # \n\nSense Number 5: "take delight in"
                    # \nFrequency: 6
                    # \nExample: "he delights in his granddaughter"
                    
                    # \n\nResponse: Sense Number: 2
                    
    prompt = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
      ]

    return prompt

def cot_prompt(instance, senses):
    context = " ".join(instance['context'])
    possible_meanings = ""
    examples = ""

    for i, (sense_key, sense_data) in enumerate(senses.items(), 0):
        possible_meanings += (
            f"sense_number {i}:"
            f"\nWord:  {sense_data['Word']}"
            f"\nGloss: {sense_data['Gloss']}"
            f"\nFrequency: {sense_data['Frequency']}\n\n"
        )

        examples += (
            f"Sense number {i}:\n"
            f"Example: {sense_data['Example']}\n\n"
        )

    system_prompt = f"""You are a linguistic expert in word sense disambiguation (WSD).
    Your task is to identify the correct sense for the target word based on the context.
    """

    user_prompt = f"""Do the following tasks.
    1. word has different meanings. Below are possible meanings.
    Comprehend the meanings. "{possible_meanings}"
    2. You can learn more on the usage of each word and the meaning through below Examples. Examples are: "{examples}".
    3. Analyze the sentence below using the following techniques and identify the most suitable meaning of the ambiguous word.
    "{context}"
    - Focus on keywords in the sentence surrounding the ambiguous word.
    - Think about the overall topic and intent of the sentence.
    4. Choose the sense_number that makes the most logical sense within the context.
    "{possible_meanings}"
    5. Return JSON object that contains the ambiguity word and the finalized sense_number. Use the following format for the output.
    <JSON Object with ambiguity word and the sense_number>
    """

    prompt = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
    ]

    return prompt


# Generazione della risposta dal modello
def predict_meaning(pipeline, prompt):
    outputs = pipeline(
        prompt,
        max_new_tokens=1024,
        # repetition_penalty = 1.2,
        temperature = 0.6,
        top_p = 0.8,
        top_k = 30
    )

    response = outputs[0]["generated_text"][-1]
    return response


# Estrazione del significato predetto dal modello
def extract_predicted_sense(response, wn_senses):
    response_cleaned = re.sub(r'["\']', '', re.sub(r'\s+', ' ', response)).strip()
    json_match = re.search(r'\{[^}]+\}', response_cleaned)
    
    if json_match:
        json_block = json_match.group(0)

        sense_match = re.search(r'(sense_number|Sense_number|Sense_Number|sense\_number|Sense Number|number|senseNumber|finalized_sense_number|finalizedSenseNumber|final_sense|finalized_sense_id|id|sense_id)[^0-9"]*["]?(\d+)["]?', json_block)
                
        if sense_match:
            selected_number = int(sense_match.group(2))
           
            # ritornato il sense corrispondente tra i wn_senses
            if 0 <= selected_number < len(wn_senses):
                num_response = list(wn_senses.keys())[selected_number]
                return num_response

    return None 

if __name__ == "__main__":

    hf_token = 'insert_hf_token'
    
    #LLama 3.1-8miliardi-instruct
    model_3_8b = "meta-llama/Llama-3.1-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model_3_8b,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    gold_dataset = '../semeval2007_test.gold.key.txt'
    input_dataset = '../semeval2007_test.data.xml'

    gold_senses = parse_gold_dataset(gold_dataset)
    sentences = load_input_dataset(input_dataset)

    # i significati estratti da WordNet vengono aggiunti nel campo 'wn_senses' di ogni frase
    for instance in sentences:
         target_word, pos = instance['target_word'].split('#')
         senses = get_wn_senses(target_word, pos, min_sense_freq=-1)
         instance['wn_senses'] = senses

    output_file = "semeval2007_zeroshot_pred_8b.txt"

    with open(output_file, "w") as f:
        for instance in sentences:
            target_word, pos = instance['target_word'].split('#')
            senses = instance.get('wn_senses', {})
            
            if senses:
                prompt = zero_shot_prompt(instance, senses)
                # prompt = zero_shot_COT_prompt(instance, senses)
                # prompt = few_shot_prompt(instance, senses)
                # prompt = cot_prompt(instance, senses)

                response = predict_meaning(pipeline, prompt)
                print(f"Response: {response}")
                predicted_sense = extract_predicted_sense(response['content'], senses)

                if predicted_sense:
                    f.write(f"{instance['id']} {predicted_sense}\n")
                else:
                    f.write(f"{instance['id']} {'Not found'}\n")
                    print(f"Warning: No sense key found for instance {instance['id']}.")
                torch.cuda.empty_cache()
            