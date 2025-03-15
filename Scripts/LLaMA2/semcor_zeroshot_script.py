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


def load_model_and_tokenizer(model_name, hf_token):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
        cache_dir = "huggingface_cache", use_auth_token = hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = "huggingface_cache", use_auth_token = hf_token)
    return model, tokenizer


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
def load_input_dataset (input_dataset, gold_senses):
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

                #integrazione frasi d'esempio dal gold file di SemCor
                sensekey = gold_senses.get(id, None)

                single_sentence_final = {
                    'id': id,
                    'context': context,
                    'target_word': word,
                    'target_len': token_length_list[i],
                    'sensekey': sensekey, # senso associato alla parola nella frase di esempio da SemCor
                }
                idx += token_length_list[i] - 1
                sentences.append(single_sentence_final)
    return sentences


# Ricerca delle parole in comune (lemma+POS) tra il dataset di input e SemCor
def find_common_target_words(sentences, semcor_sentences):
    semcor_contexts = {}

    # dizionario per SemCor (lemma#pos -> [context] corrispondente)
    for sentence in semcor_sentences:
        target_word = sentence['target_word']
        sensekey = sentence['sensekey'][0] if sentence['sensekey'] else None
        if sensekey:
            if target_word not in semcor_contexts:
                semcor_contexts[target_word] = {}
            # estratta solo una frase per ogni sensekey
            if sensekey not in semcor_contexts[target_word]:
                semcor_contexts[target_word][sensekey] = sentence['context']

    # estratte solo le frasi per parole target comuni
    common_contexts = {}
    for sentence in sentences:
        target_word = sentence['target_word']
        if target_word in semcor_contexts:
            common_contexts[target_word] = [
                {'sensekey': sk, 'context': ctx}
                for sk, ctx in semcor_contexts[target_word].items()
            ]

    return common_contexts

# Estrazione sensi da WordNet
def get_wn_senses(lemma, pos, min_sense_freq, semcor_contexts):
    wn_pos = {
        'n': wn.NOUN,
        'v': wn.VERB,
        'a': wn.ADJ,
        'r': wn.ADV
        }

    wn_senses = {}
    for l in wn.lemmas(lemma, wn_pos[pos]):
        if l.count() > min_sense_freq:
            sensekey = l.key()
            example = None

            # estratto il contesto di SemCor se disponibile 
            if lemma + '#' + pos in semcor_contexts:
                for context in semcor_contexts[lemma + '#' + pos]:
                    if context['sensekey'] == sensekey:
                        example = " ".join(context['context'])
                        break

            # se non disponibile una frase d'esempio da SemCor viene estratta da WordNet
            if not example:
                example = l.synset().examples()[0] if l.synset().examples() else "No example available"

            wn_senses[l.key()] = {
                "frequency": l.count(),
                "definition": l.synset().definition(),
                "example": example
            }

    return wn_senses


def zero_shot_prompt(instance, senses):
    context = " ".join(instance['context'])
    target_word = instance['target_word'].split('#')[0]

    possible_meanings = ""
    
    for i, (sense_key, sense_data) in enumerate(senses.items(), 1):
        possible_meanings += (
            f"Sense number {i}:\n"
            f"Definition: {sense_data['definition']}\n"
            f"Example: {sense_data['example']}\n\n"
        )
    
    # ALPACA TEMPLATE
    # prompt = f"""
    #     ### Instruction:
    #     You are a linguistic expert in word sense disambiguation (WSD).
    #     Your task is to identify the correct sense_number for an ambiguous word based
    #     on the context and the provided list of possible meanings.
    #     Always respond in the following format: "Sense Number: <sense_number>"
    #     If you are unsure, always choose the closest matching sense from the list.
        
    #     ### Input:
    #     Context: "{context}".
    #     Target word: "{target_word}".
    #     Possible senses list for "{target_word}": {possible_meanings}
        
    #     ### Response:
    #     """

    # CHAT TEMPLATE
    prompt = f"""<s>[INST]
        «SYS» You are a linguistic expert in word sense disambiguation (WSD).
        Your task is to identify the correct sense for an ambiguous word based on the
        context and the provided list of possible senses.
        «/SYS»
        
        Below are the possible senses for the target word.
        Each sense includes:
        - A brief description of the sense.
        - The number of times this sense of the target word appears in a large corpus
        of texts. Higher frequency suggests the sense is more commonly used, while
        lower frequency indicates a rarer or specialized use of the word.
        - An example of sentence illustrating the use of the sense.

        This is the context: {context}
        This is the ambiguous word: {target_word}
        Possible senses for {target_word} are: {possible_meanings}
        
        Choose the most appropriate sense_number listed above.
        Always respond in the following format: "Sense Number: <sense_number>"
        [/INST]"""
    
    return prompt
 

# Generazione della risposta dal modello
def predict_meaning(model, tokenizer, prompt):
    input = tokenizer(
              prompt,
              return_tensors="pt"
            ).to("cuda")

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    start_index = input.input_ids.shape[-1]

    with torch.no_grad():
        output = model.generate(
                input.input_ids,
                max_new_tokens = 1024,
                temperature = 0.6,
                top_p = 0.8,
                top_k = 30,
                repetition_penalty=1.2,
                attention_mask=input.attention_mask,
                pad_token_id=pad_token_id
                )

    generation_output = output[0][start_index:]
    response = tokenizer.decode(generation_output, skip_special_tokens=True)
    
    return response


# Estrazione del significato predetto dal modello
def extract_predicted_sense(response, wn_senses):
    response_cleaned = re.sub(r'\s+', ' ', response).strip()
    
    match = re.search(r'(?:Sense\s*Number\s*[:\-]?\s*)?(\d+)', response_cleaned)
    if match:
        selected_number = int(match.group(1)) -1 # Estrae il numero e lo converte in intero
        if 0 <= selected_number < len(wn_senses):
            num_response = list(wn_senses.keys())[selected_number]
            return num_response
    
    return None


if __name__ == "__main__":

    hf_token = 'insert_hf_token'

    # LLaMA 2 con 7 milardi di parametri
    model_2_7b = "meta-llama/Llama-2-7b-chat-hf"

    model, tokenizer = load_model_and_tokenizer(model_2_7b, hf_token)

    gold_dataset = '../semeval2007.gold.key.txt'
    input_dataset = '../semeval2007.data.xml'

    # SemCor dataset per integrare frasi d'esempio alternative a WordNet
    semcor_gold_dataset = '../semcor.gold.key.txt'
    semcor_dataset = '../semcor.data.xml'

    gold_senses = parse_gold_dataset(gold_dataset)
    semcor_gold_senses = parse_gold_dataset(semcor_gold_dataset)

    sentences = load_input_dataset(input_dataset, gold_senses)
    semcor_sentences = load_input_dataset(semcor_dataset, semcor_gold_senses)
    
    common_contexts = find_common_target_words(sentences, semcor_sentences)

    # i significati estratti da WordNet vengono aggiunti nel campo 'wn_senses' di ogni frase
    for instance in sentences:
        target_word, pos = instance['target_word'].split('#')
        senses = get_wn_senses(target_word, pos, min_sense_freq=-1, semcor_contexts=common_contexts)
        instance['wn_senses'] = senses


    output_file = "semcor_semeval2007_zeroshot_pred_7b.txt"

    with open(output_file, "w") as f:
        for instance in sentences:
            target_word, pos = instance['target_word'].split('#')
            senses = instance.get('wn_senses', {})
    
            if senses:
                prompt = zero_shot_prompt(instance, senses)
                response = predict_meaning(model, tokenizer, prompt)
                print(f"Response: {response}")
                predicted_sense = extract_predicted_sense(response, senses)

                if predicted_sense:
                    f.write(f"{instance['id']} {predicted_sense}\n")
                else:
                    f.write(f"{instance['id']} {'Not found'}\n")
                    print(f"Warning: No sense key found for instance {instance['id']}.")
                torch.cuda.empty_cache()

