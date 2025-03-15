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
from huggingface_hub import login
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

                #aggiunta di tag <WSD></WSD> agli estremi della perola target
                target_start = idx
                target_end = idx + token_length_list[i]  
                context.insert(target_start, '<WSD>')
                context.insert(target_end + 1, '</WSD>') 

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

    for i, l in enumerate(wn.lemmas(lemma, wn_pos[pos]), 0):
        if l.count() > min_sense_freq:
            wn_senses[l.key()] = {
                "Id": i, 
                "Word": lemma,
                "Gloss": l.synset().definition(),
                "Frequency": l.count(),
                "Example": l.synset().examples()[0] if l.synset().examples() else "No example available"
            }

    return wn_senses



def cot_verified_prompt(instance, senses):
    context = " ".join(instance['context'])
    possible_meanings = ""
    examples = ""

    for (sense_key, sense_data) in senses.items():
        possible_meanings += (
            f"sense_number: {sense_data['Id']} "
            f"\nWord:  {sense_data['Word']}"
            f"\nGloss: {sense_data['Gloss']}"
            f"\nFrequency: {sense_data['Frequency']}\n\n"
        )

        examples += (
            f"sense_number: {sense_data['Id']} "
            f"\nExample: {sense_data['Example']}\n\n"
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


# Prompt per chiedere al modello di rivalutare la scelta del senso precedentemente selezionato
def verification_prompt(instance, target_word, predicted_sense, senses):
    context = " ".join(instance['context'])
    target_word = instance['target_word'].split('#')[0]

    sense_data = senses.get(predicted_sense)
    sense_definition = sense_data['Gloss'] if sense_data else "Not available"

    system_prompt = """
    Verify if the selected sense is correct for the ambiguous word in the given sentence.
    Respond with a JSON object containing the target word and your answer ('yes' or 'no').
    """

    user_prompt = f"""
    Do the following tasks.
    \n1.Examine the sentence below. '{context}'    
    \n2. You are going to identify if the meaning below is suitable for '{target_word}'.
    Proposed sense: '{predicted_sense}' (Meaning: {sense_definition})
    \n3.Is the proposed sense correct for the target word in this sentence? Respond in JSON format with the following structure:
    {{
        "target_word": "{target_word}",
        "answer": "yes" or "no"
    }}
    """

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return prompt


# Estrazione risposta di conferma/rifiuto del senso selezionato
def extract_verification_answer(verify_response):
    response_cleaned = re.sub(r'["\']', '', re.sub(r'\s+', ' ', verify_response)).strip()
    
    answer_match = re.search(r'\b(yes|no)\b', response_cleaned, re.IGNORECASE)

    if answer_match:
        return answer_match.group(1).lower()  

    print("Errore: Nessuna risposta valida trovata.")
    return None


# Generazione della risposta e verifica dei sensi non corretti
# se il modello conferma la propria scelta, viene estratto il senso, seppur errato
# se il modello rifiuta quel senso, viene inviato un nuovo CoT prompt con la lista dei sensi - il senso rifiutato
def predict_and_verify(instance, senses, pipeline, gold_senses_instance):
    sense_id_to_key = {sense["Id"]: key for key, sense in senses.items()}

    # lista id dei sensi disponibili
    remaining_sense_ids = list(sense_id_to_key.keys())  

    while remaining_sense_ids:
        # set dei sensi ancora disponibili
        remaining_senses = {sense_id_to_key[sense_id]: senses[sense_id_to_key[sense_id]] for sense_id in remaining_sense_ids}

        prompt = cot_verified_prompt(instance, remaining_senses)
        response = predict_meaning(pipeline, prompt)
        print(f"Response: {response}")

        predicted_sense_key = extract_predicted_sense(response['content'], remaining_senses)
        print('Predicted sense key:', predicted_sense_key)

        if predicted_sense_key:
            predicted_sense_id = next((sense["Id"] for key, sense in senses.items() if key == predicted_sense_key), None)

            if predicted_sense_id is None:
                print(f"Error: predicted_sense_key {predicted_sense_key} not found in senses mapping.")
                continue
            
            # se il senso scelto è proprio quello corretto la disambiguazione può terminare
            if predicted_sense_key in gold_senses_instance:  
                return predicted_sense_key

            # verifica del senso errato selezionato
            verify_prompt = verification_prompt(instance, instance['target_word'], predicted_sense_key, remaining_senses)
            verify_response = predict_meaning(pipeline, verify_prompt)
            print(f"Verification for instance: {instance['id']}")

            verification_answer = extract_verification_answer(verify_response['content'])
            print(f"Verification answer: {verification_answer}")

            # il modello conferma la scelta, anche se errata
            if verification_answer == 'yes':
                return predicted_sense_key  
            
            # rimosso il senso errato dalla lista dei sensi disponibili
            print(f"Sense {predicted_sense_key} (ID {predicted_sense_id}) rejected, removing from remaining senses")
            remaining_sense_ids.remove(predicted_sense_id)

        else:
            break  

    print(f"Warning: No valid senses found for instance {instance['id']}.")
    
    # salvate le risposte in cui non viene estratto nessun senso (dubbi nel modello, ripetizioni di frasi senza scelta finale...)
    with open("semeval2007_failed_responses.txt", "a") as fail_file:
        fail_file.write(f"Instance ID: {instance['id']}\n")
        fail_file.write(f"Context: {' '.join(instance['context'])}\n")
        fail_file.write(f"Model Response: {response['content']}\n\n")

    return "Not found" 


# Generazione della risposta dal modello
def predict_meaning(pipeline, prompt):
    outputs = pipeline(
        prompt,
        max_new_tokens=1024,
        temperature = 0.6,
        top_p = 0.8,
        top_k = 30,
        #repetition_penalty=1.2
    )

    response = outputs[0]["generated_text"][-1]
    return response


# Estrazione del significato predetto dal modello
def extract_predicted_sense(response, wn_senses):
    response_cleaned = re.sub(r'["\']', '', re.sub(r'\s+', ' ', response)).strip()
    json_match = re.search(r'\{[^}]+\}', response_cleaned)
    
    if json_match:
        json_block = json_match.group(0)

        sense_match = re.search(r'(sense_number|Sense_number|Sense_Number|sense\_number|Sense Number|senseNumber|id|finalized_sense_number|Finalized Sense Number)[^0-9"]*["]?(\d+)["]?', json_block)

        if sense_match:
            try:
                selected_number = int(sense_match.group(2))  # Converte il numero in intero
                for sense_key, sense_data in wn_senses.items():
                    if sense_data["Id"] == selected_number:
                        return sense_key 
                    
            except ValueError:
                return None  # Se per qualche motivo il valore non è convertibile in intero

    return None  # Se non trova nulla di valido


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
    
    gold_dataset = '../semeval2007.gold.key.txt'
    input_dataset = '../semeval2007.data.xml'

    gold_senses = parse_gold_dataset(gold_dataset)
    sentences = load_input_dataset(input_dataset)

    # i significati estratti da WordNet vengono aggiunti nel campo 'wn_senses' di ogni frase
    for instance in sentences:
         target_word, pos = instance['target_word'].split('#')
         senses = get_wn_senses(target_word, pos, min_sense_freq=-1)
         instance['wn_senses'] = senses

    # output_file = "cot_verify_pred_8b.txt"
    output_file = "semeval2007_cot_verified_pred_8b.txt"

    with open(output_file, "w") as f:
         for instance in sentences:
            target_word, pos = instance['target_word'].split('#')
            senses = instance.get('wn_senses', {})
            gold_sense_instance = gold_senses.get(instance['id'], None)
    
            if senses:
                predicted_sense = predict_and_verify(instance, senses, pipeline, gold_sense_instance)
                f.write(f"{instance['id']} {predicted_sense}\n")
    
            torch.cuda.empty_cache() 
