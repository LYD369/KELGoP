from prompts import *

import argparse
import json
import time
import re
import random

from openai import OpenAI

from SPARQLWrapper import SPARQLWrapper, JSON
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

import model.BLINK.elq.main_dense as elq_main_dense
import qwikidata.linked_data_interface as ldi

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

import numpy as np
from model.GNN_RAG.gnn.train_model import Trainer_KBQA
import os
import requests
import copy


def prepare_data(dataset_name):
    if dataset_name == "webqsp":
        dataset_path = "./data/webqsp/simple-WebQSP-myTopic.test.jsonl"
    elif dataset_name == "cwq":
        dataset_path = "./data/cwq/simple-CWQ-topic_test.jsonl"
    else:
        dataset_path = "./data/grailqa/simple-GrailQA-mytopic.test.jsonl"
    question_ids = []
    questions = []
    topic_entity_names = []
    topic_entity_ids = []
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        lines = dataset_file.readlines()
        for line in lines:
            line_data = json.loads(line)
            question_ids.append(line_data["Id"])
            questions.append(line_data["Question"])
            topic_entity_names.append(line_data["TopicEntityName"])
            topic_entity_ids.append(line_data["TopicEntityId"])
    return question_ids, questions, topic_entity_names, topic_entity_ids


def prepare_entity_linker(threshold):
    entity_linker = {}
    # ELQ
    models_path = "./model/BLINK/models/" # the path where you stored the ELQ models
    config = {
        "interactive": False,
        "biencoder_model": models_path+"elq_webqsp_large.bin",
        "biencoder_config": models_path+"elq_large_params.txt",
        "cand_token_ids_path": models_path+"entity_token_ids_128.t7",
        "entity_catalogue": models_path+"entity.jsonl",
        "entity_encoding": models_path+"all_entities_large.t7",
        "output_path": "ELQ_logs/", # logging directory
        "faiss_index": "hnsw",
        "index_path": models_path+"faiss_hnsw_index.pkl",
        "num_cand_mentions": 10,
        "num_cand_entities": 10,
        "threshold_type": "joint",
        "threshold": threshold,
    }
    entity_linker['args'] = argparse.Namespace(**config)
    entity_linker['models'] = elq_main_dense.load_models(entity_linker['args'], logger=None)
    return entity_linker


def prepare_GNN(dataset_name):
    gnn = {}
    if dataset_name == "webqsp":
        num_epoch = 200
        data_folder = "model/GNN_RAG/gnn/data/webqsp/"
        num_iter = 3
        num_ins = 2
        load_experiment = "ReaRev_webqsp.ckpt"
    elif dataset_name == "cwq":
        num_epoch = 100
        data_folder = "model/GNN_RAG/gnn/data/CWQ/"
        num_iter = 2
        num_ins = 3
        load_experiment = "ReaRev_CWQ.ckpt"
    config = {
        # ReaRev args
        "model_name": "ReaRev",
        "alg": "bfs",
        "num_iter": num_iter,
        "num_ins": num_ins,
        "num_gnn": 3,
        "loss_type": "kl",
        "use_self_loop": True,
        "normalized_gnn": False,
        "norm_rel": False,
        "data_eff": False,
        "pos_emb": False,
        
        # shared args
        "name": dataset_name,
        "data_folder": data_folder,
        "max_train": 200000,
        # embeddings
        "word2id": "vocab.txt",
        "relation2id": "relations.txt",
        "entity2id": "entities.txt",
        "char2id": "chars.txt",
        "entity_emb_file": None,
        "relation_emb_file": None,
        "relation_word_emb": True,
        "word_emb_file": "word_emb.npy",
        "rel_word_ids": "rel_word_idx.npy",
        "kge_frozen": 0,
        "lm": "sbert",
        "lm_frozen": 1,
        # dimensions, layers, dropout
        "entity_dim": 50,
        "kg_dim": 100,
        "word_dim": 300,
        "lm_dropout": 0.3,
        "linear_dropout": 0.2,
        # optimization
        "num_epoch": num_epoch,
        "warmup_epoch": 0,
        "fact_scale": 3,
        "eval_every": 2,
        "batch_size": 8,
        "gradient_clip": 1.0,
        "lr": 0.0005,
        "decay_rate": 0.0,
        "seed": 19960626,
        "lr_schedule": False,
        "label_smooth": 0.1,
        "fact_drop": 0,
        # model options
        "is_eval": True,
        "checkpoint_dir": "model/GNN_RAG/gnn/checkpoint/pretrain/",
        "log_level": "info",
        "experiment_name": "",
        "load_experiment": load_experiment,
        "load_ckpt_file": None,
        "eps": 0.95,
        "test_batch_size": 20,
        "q_type": "seq",
        "use_cuda": torch.cuda.is_available(),
    }
    gnn['args'] = argparse.Namespace(**config)
    np.random.seed(gnn['args'].seed)
    torch.manual_seed(gnn['args'].seed)
    if gnn['args'].experiment_name == None:
        timestamp = str(int(time.time()))
        gnn['args'].experiment_name = "{}-{}-{}".format(
            gnn['args'].dataset,
            gnn['args'].model_name,
            timestamp,
        )

    entityId2GNNId = dict()
    with open(gnn['args'].data_folder + gnn['args'].entity2id, encoding='utf-8') as f:
        for line in f:
            entityId = line.strip()
            entityId2GNNId[entityId] = len(entityId2GNNId)
    
    GNNId2entityId = {GNNId: entityId for entityId, GNNId in entityId2GNNId.items()}
    gnn['dicts'] = [entityId2GNNId, GNNId2entityId]

    gnn['datas'] = []
    with open(gnn['args'].data_folder + "test.json") as f:
        for line in f:
            gnn['datas'].append(json.loads(line))

    gnn['trainer'] = Trainer_KBQA(args=vars(gnn['args']), model_name=gnn['args'].model_name)

    return gnn


def prepare_llm(llm_name):
    llm = {}
    llm['name'] = llm_name
    if llm['name'] in ["llama3.1", "qwen2.5"]:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
        llm['device'] = "cuda"
        if llm['name'] == "llama3.1":
            llm_model_path = os.environ["LLAMA_PATH"]
        else:
            llm_model_path = os.environ["QWEN_PATH"]
        llm['tokenizer'] = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        llm['model'] = AutoModelForCausalLM.from_pretrained(llm_model_path, torch_dtype=torch.float16)
        llm['model'].to(llm['device'])
    else:
        llm['api_key'] = os.environ["OPENAI_API_KEY"]
    return llm


def reason_with_llm(prompt, llm, input_token_num, output_token_num):
    if llm['name'] in ["gpt-3.5-turbo", "gpt-4o-2024-11-20"]:
        client = OpenAI(
            api_key=llm['api_key']
        )
        input_message=[
            {"role": "system", "content": "You are an AI assistant with the ability to provide insightful responses and make informed judgments."},
            {"role": "user", "content": prompt}
        ]
        success = False
        while not success:
            try:
                completion = client.chat.completions.create(
                    model=llm['name'],
                    messages=input_message,
                    max_tokens=200
                )
                if not isinstance(completion.choices[0].message.content, str):
                    continue
                success = True
            except Exception as e:
                print("Exception in LLM:")
                print(e)
                # input_message too long always happen in prompt_answer
                if "Please reduce the length of the messages." in str(e):
                    prompt_lines = prompt.split("\n")
                    for index in range(len(prompt_lines) - 1, 0, -1):
                        if prompt_lines[index].startswith("Knowledge triplets: ") or prompt_lines[index].startswith("Retrieved knowledge: "):
                            break
                    knowledge_lines = prompt_lines[index:-1]
                    prompt = '\n'.join(prompt_lines[:index] + knowledge_lines[:len(knowledge_lines)//2] + prompt_lines[-1:])
                    input_message = [
                        {"role": "system", "content": "You are an AI assistant with the ability to provide insightful responses and make informed judgments."},
                        {"role": "user", "content": prompt}
                    ]
                if "have exceeded token rate limit" in str(e):
                    time.sleep(60 + 60 * random.random())
                else:
                    time.sleep(2 + 2 * random.random())
        try:
            response_message = completion.choices[0].message.content
            input_token_num += completion.usage.prompt_tokens
            output_token_num += completion.usage.completion_tokens
        except:
            response_message = "null"
    else: # llama3.1 or qwen2.5
        success = False
        cleared = False
        max_new_tokens = 200
        while not success:
            try:
                model_input = llm['tokenizer'](prompt, return_tensors="pt")
                if llm['name'] == "llama3.1":
                    llm_pad_token_id = llm['tokenizer'].eos_token_id
                else:
                    llm_pad_token_id = llm['tokenizer'].pad_token_id
                output_ids = llm['model'].generate(
                    model_input["input_ids"].to(llm['device']),
                    attention_mask=model_input["attention_mask"].to(llm['device']),
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    early_stopping=True,
                    eos_token_id=llm['tokenizer'].eos_token_id,
                    pad_token_id=llm_pad_token_id
                )
                success = True
            except torch.cuda.OutOfMemoryError as e:
                print("Exception in LLM:")
                print(e)
                if not cleared:
                    gc.collect()
                    torch.cuda.empty_cache()
                    max_new_tokens = 100
                    cleared = True
                else:
                    gc.collect()
                    torch.cuda.empty_cache()
                    # input_message too long always happen in prompt_answer
                    prompt_lines = prompt.split("\n")
                    for index in range(len(prompt_lines) - 1, 0, -1):
                        if prompt_lines[index].startswith("Knowledge triplets: ") or prompt_lines[index].startswith("Retrieved knowledge: "):
                            break
                    knowledge_lines = prompt_lines[index:-1]
                    prompt = '\n'.join(prompt_lines[:index] + knowledge_lines[:len(knowledge_lines)//2] + prompt_lines[-1:])
        output_ids = output_ids[0][len(model_input["input_ids"][0]):]
        outputs = llm['tokenizer'].decode(output_ids, skip_special_tokens=True)
        response_message = outputs.split("Question: ")[0].split("You are an AI assistant")[0]
        if response_message[0] == " ":
            response_message = response_message[1:]
        input_token_num += model_input["input_ids"].size(1)
        output_token_num += output_ids.size(0)
    return response_message, input_token_num, output_token_num


def determine(question, dataset_name, llm, llm_input_len, llm_output_len):
    if dataset_name == "grailqa":
        prompt = prompt_judge_grailqa.format(question)
    else:
        prompt = prompt_judge.format(question)
    response, llm_input_len, llm_output_len = reason_with_llm(prompt, llm, llm_input_len, llm_output_len)
    
    # extract the {yes} or {no} in llm's response
    answer_begin = response.find('{')
    answer_end = response.find('}')
    if answer_begin == -1 or answer_end == -1:
        answer = "simple"
    else:
        answer = response[answer_begin+1:answer_end]
    
    if answer.lower() == "simple":
        answer_flag = True
    else:
        answer_flag = False
    
    return response, answer_flag, llm_input_len, llm_output_len


def generate_answer(question, dataset_name, KG_retriever, previous_subQAs, reference_triplets, llm, llm_input_len, llm_output_len):
    if KG_retriever == "2hoptriple":
        if dataset_name == "grailqa":
            prompt_answer = prompt_answer_tri_grailqa
        else:
            prompt_answer = prompt_answer_tri
    elif KG_retriever == "2hoprelation":
        if dataset_name == "grailqa":
            prompt_answer = prompt_answer_rel_grailqa
        else:
            prompt_answer = prompt_answer_rel
    elif KG_retriever == "gnn":
        prompt_answer = prompt_answer_gnn
    if len(previous_subQAs) == 0:
        prompt = prompt_answer.format(question, '\n'.join(reference_triplets))
    else:
        prompt = prompt_answer.format(question, '\n'.join(reference_triplets) + "\nReferences: " + previous_subQAs.rstrip("\n"))
    response, llm_input_len, llm_output_len = reason_with_llm(prompt, llm, llm_input_len, llm_output_len)

    # extract the {answer_entity} in llm's response
    answer_pattern = r'\{([^}]*)\}'
    answers = re.findall(answer_pattern, response)
    if (len(answers) == 0) or ((len(answers) == 1) and (answers[0].lower() in ["none", "unknown"])):
        prompt = prompt_answer_extract.format(question, response)
        response_2, llm_input_len, llm_output_len = reason_with_llm(prompt, llm, llm_input_len, llm_output_len)
        answers = re.findall(answer_pattern, response_2)
    answer_entity = "; ".join(list(set(answers)))
    
    return response, answer_entity, llm_input_len, llm_output_len


def subclassify(question, composition_types, llm, llm_input_len, llm_output_len):
    prompt = prompt_subclassify.format(" or ".join(["\"{" + t + "}\"" for t in composition_types]), "\n".join([prompt_subclassify_examples[t] for t in composition_types]), question)
    response, llm_input_len, llm_output_len = reason_with_llm(prompt, llm, llm_input_len, llm_output_len)
    
    answer_begin = response.find('{')
    answer_end = response.find('}')
    if answer_begin == -1 or answer_end == -1:
        list_composition_types = list(composition_types)
        answer = ""
        for i in range(len(list_composition_types)):
            if list_composition_types[i] in response:
                answer = list_composition_types[i]
                break
        if answer == "":
            answer = list_composition_types[0]
    else:
        answer = response[answer_begin+1:answer_end].lower()
    
    if answer not in composition_types:
        answer = list(composition_types)[0]
    
    return answer, llm_input_len, llm_output_len


def decompose(question, composition_type, llm, llm_input_len, llm_output_len):
    prompt = prompt_decompose_group[composition_type].format(question)
    response, llm_input_len, llm_output_len = reason_with_llm(prompt, llm, llm_input_len, llm_output_len)

    # extract the subquestions in llm's response
    subquestions = []
    response = re.sub(r'\{\[\#(\d+)\]\}', r'([#\1])', response) # replace wrong pattern {[#x]} with ([#x])
    lines = response.split('\n')
    for line in lines:
        begini = line.find('{')
        endi = line.find('}')
        if begini != -1 and endi != -1 and begini + 1 < endi:
            subquestion = line[begini+1:endi]
            # modify error format
            subquestion = re.sub(r'in subquestion-(\d+) and subquestion-(\d+)', r'in ([#\1]) and ([#\2])', subquestion)
            subquestion = re.sub(r'in subquestion-(\d+)', r'in ([#\1])', subquestion)
            subquestion = re.sub(r'the answers? to subquestion-(\d+) and subquestion-(\d+)', r'([#\1]) and ([#\2])', subquestion)
            subquestion = re.sub(r'the answers? to subquestion-(\d+)', r'([#\1])', subquestion)
            subquestion = re.sub(r'from subquestion-(\d+)', r'from ([#\1])', subquestion)
            subquestions.append(subquestion)
    
    return response, subquestions, llm_input_len, llm_output_len


def get_entity_dict_with_ua(entity_id, base_url="https://www.wikidata.org/wiki/Special:EntityData"):
    url = "{}/{}.json".format(base_url, entity_id)
    headers = {
        "User-Agent": "MyWikidataBot/1.0 (your_homepage_url; your_email@example.com) bot"
    }
    response = requests.get(url, headers=headers)
    if response.ok:
        entity_dict_full = response.json()
    else:
        raise ldi.LdiResponseNotOk(
            "input entity id: {}, response.headers: {}, response.status_code: {}, response.text: {}".format(
                entity_id, response.headers, response.status_code, response.text
            )
        )
    returned_entity_id = next(iter(entity_dict_full["entities"]))
    entity_dict = entity_dict_full["entities"][returned_entity_id]
    if entity_id != returned_entity_id:
        ldi.logger.warning(
            "Wikidata redirect detected.  Input entity id={}. Returned entity id={}.".format(
                entity_id, returned_entity_id
            )
        )
    return entity_dict
ldi.get_entity_dict_from_api = get_entity_dict_with_ua


def entity_linking(question, entity_linker):
    # entity linker: ELQ
    data_to_link = [{
        "id": 0,
        "text": question.lower()
    }]
    predictions = elq_main_dense.run(entity_linker['args'], None, *entity_linker['models'], test_data=data_to_link)
    pred = predictions[0]
    # id in ELQ results -> wikidata id
    wikiIds = []
    for triple in pred["pred_triples"]:
        Id = triple[0]
        wikiId = entity_linker['elqId2wikiId'].get(Id, 'null')
        wikiIds.append(wikiId)
    # wikidata id -> freebase id
    fbIds = []
    for wikiId in wikiIds:
        if wikiId == 'null' or wikiId[0] != 'Q':
            fbId = 'null'
        else:
            qdict = {}
            success = False
            while not success:
                try:
                    qdict = get_entity_dict_with_ua(wikiId)
                    success = True
                except Exception as e:
                    print("Exception in wikiId -> freebaseId:")
                    print(e)
                    if "Not Found" in str(e):
                        break
                    time.sleep(30 + 30 * random.random())
            if not success:
                fbId = "none"
            elif 'P646' not in qdict['claims']:
                fbId = "none"
            else:
                p646 = qdict["claims"]["P646"]
                p646 = p646[0]
                fbId = p646["mainsnak"]["datavalue"]["value"]
                if fbId.startswith("/m/") or fbId.startswith("/g/"):
                    fbId = fbId[1] + "." + fbId[3:]
                else:
                    fbId = "none"
        fbIds.append(fbId)
    # select topic entity
    topic_entity_name = ""
    topic_entity_id = "null"
    for i in range(len(fbIds)):
        if fbIds[i] != "none" and fbIds[i] != "null":
            topic_entity_id = fbIds[i]
            topic_entity_name = pred["pred_tuples_string"][i][0]
            break
    return topic_entity_name, topic_entity_id


SPARQLPATH = "http://localhost:8890/sparql"
def sparql_search(sparql_txt):
    success = False
    while not success:
        try:
            sparql = SPARQLWrapper(SPARQLPATH)
            sparql.setQuery(sparql_txt)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            success = True
        except Exception as e:
            print("Exception in SPARQLsearch:")
            print(e)
            if ("Error HTTP/1.1 404 File not found" not in str(e)) and ("Virtuoso S1T00 Error SR171: Transaction timed out" not in str(e)):
                break
            time.sleep(2 + 2 * random.random())
    if success:
        return results["results"]["bindings"]
    else:
        return []


def process_relation(sparql_results):
    relations = []
    for rel in sparql_results:
        relations.append(rel["relation"]["value"].replace("http://rdf.freebase.com/ns/",""))
    return relations


sparql_txt_find_name = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?name
WHERE {
    {
        ns:%s ns:type.object.name ?name .
    }
    UNION
    {
        ns:%s <http://www.w3.org/2002/07/owl#sameAs> ?name .
    }
}"""
wikiPrefix = "http://www.wikidata.org/entity/Q"
def entityid2name(entityId):
    entityName = entityId
    if len(entityId) >= 2:
        if entityId[0:2] in ["m.", "g."]:
            results = sparql_search(sparql_txt_find_name % (entityId, entityId))
            if len(results) != 0:
                entityName = results[0]["name"]["value"]
                if wikiPrefix in entityName:
                    if len(results) > 1:
                        entityName = results[1]["name"]["value"]
                    else:
                        wikiId = entityName.replace(wikiPrefix[:-1], "")
                        qdict = {}
                        success = False
                        while not success:
                            try:
                                qdict = get_entity_dict_with_ua(wikiId)
                                success = True
                            except Exception as e:
                                print("Exception in wikiId -> entityName:")
                                print(e)
                                time.sleep(2 + 2 * random.random())
                        if "en" in qdict["labels"]:
                            entityName = qdict["labels"]["en"]["value"]
                        else:
                            entityName = entityId
    entityName = entityName.replace("\n", "\\n")
    return entityName


def process_entity(sparql_results):
    entities = []
    for ent in sparql_results:
        entityId = ent["entity"]["value"].replace("http://rdf.freebase.com/ns/","")
        entityName = entityid2name(entityId)
        entities.append([entityId, entityName])
    return entities


def construct_candidates(candidates, rel2ents, topic_entity, relation, entities, ishead, istwohop, relations1hop=[]):
    if not istwohop:
        entFor2hop = [ent for ent in entities if (ent[0].startswith('m.') or ent[0].startswith('g.')) and (ent[0] == ent[1])]
        if relation in rel2ents:
            rel2ents[relation].extend(entFor2hop)
        else:
            rel2ents[relation] = entFor2hop
        # add to candidates
        for entity in entities:
            if topic_entity != entity[1]:
                if ishead:
                    candidate = "< {} > [SEP] < {} > [SEP] < {} >".format(topic_entity, relation, entity[1])
                else:
                    candidate = "< {} > [SEP] < {} > [SEP] < {} >".format(entity[1], relation, topic_entity)
                candidates.add(candidate)
    else:
        if ishead:
            removeCands = [cand for cand in candidates if (cand.split(" [SEP] ")[1][2:-2] in relations1hop) and (cand.split(" [SEP] ")[2][2:-2] == topic_entity)]
            for cand in removeCands:
                for entity in entities:
                    if cand.split(" [SEP] ")[0][2:-2] != entity[1]:
                        candidates.add("< {} > [SEP] < {} > [SEP] < {} > [SEP] < {} >".format(cand.split(" [SEP] ")[0][2:-2], cand.split(" [SEP] ")[1][2:-2], relation, entity[1]))
                candidates.remove(cand)
        else:
            removeCands = [cand for cand in candidates if (cand.split(" [SEP] ")[1][2:-2] in relations1hop) and (cand.split(" [SEP] ")[0][2:-2] == topic_entity)]
            for cand in removeCands:
                for entity in entities:
                    if entity[1] != cand.split(" [SEP] ")[2][2:-2]:
                        candidates.add("< {} > [SEP] < {} > [SEP] < {} > [SEP] < {} >".format(entity[1], relation, cand.split(" [SEP] ")[1][2:-2], cand.split(" [SEP] ")[2][2:-2]))
                candidates.remove(cand)
    return candidates, rel2ents


def construct_candidates_1hop(candidates, rel2ents, candidate2isTail, topic_entity, relation, entities, ishead):
    entFor2hop = [ent[0] for ent in entities if (ent[0].startswith('m.') or ent[0].startswith('g.')) and (ent[0] == ent[1])]
    if relation in rel2ents:
        rel2ents[relation].extend(entFor2hop)
    else:
        rel2ents[relation] = entFor2hop
    # add to candidates
    for entity in entities:
        if topic_entity != entity[1]:
            if ishead:
                candidate = "< {} > [SEP] < {} >".format(topic_entity, relation)
            else:
                candidate = "< {} > [SEP] < {} >".format(relation, topic_entity)
            candidates.add(candidate)
            if candidate not in candidate2isTail:
                candidate2isTail[candidate] = ishead
    return candidates, rel2ents, candidate2isTail


def construct_candidates_2hop(candidates, candidate2isTail, relation, relations1hop, ishead):
    if ishead:
        changeCands = [cand for cand in candidates if (len(cand.split(" [SEP] ")) == 2) and (cand.split(" [SEP] ")[1][2:-2] in relations1hop)]
        for cand in changeCands:
            candidate = "< {} > [SEP] < {} > [SEP] < {} >".format(cand.split(" [SEP] ")[0][2:-2], cand.split(" [SEP] ")[1][2:-2], relation)
            candidates.add(candidate)
            if candidate not in candidate2isTail:
                candidate2isTail[candidate] = ishead
    else:
        changeCands = [cand for cand in candidates if (len(cand.split(" [SEP] ")) == 2) and (cand.split(" [SEP] ")[0][2:-2] in relations1hop)]
        for cand in changeCands:
            candidate = "< {} > [SEP] < {} > [SEP] < {} >".format(relation, cand.split(" [SEP] ")[0][2:-2], cand.split(" [SEP] ")[1][2:-2])
            candidates.add(candidate)
            if candidate not in candidate2isTail:
                candidate2isTail[candidate] = ishead
    return candidates, candidate2isTail


sparql_txt_find_tail_rel = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
    ns:%s ?relation ?entity .
    FILTER(?relation != ns:common.topic.description)
}"""
sparql_txt_find_head_rel = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?relation
WHERE {
    ?entity ?relation %s .
}"""
sparql_txt_find_tail = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity
WHERE {{
    ns:{0} ns:{1} ?entity .
}}"""
sparql_txt_find_head = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity
WHERE {{
    ?entity ns:{1} {0} .
}}"""
def find_candidates_2hoptriple(question, topic_entity_name, topic_entity_id):
    width_1hop_rel = 10
    rel_for_2hop = 5
    width_2hop_rel = 5

    rel2isTail = dict()

    # 1 hop relations
    if topic_entity_id.endswith('@en') or topic_entity_id.endswith('^^xsd:dateTime') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
        results = []
    else:
        results = sparql_search(sparql_txt_find_tail_rel % topic_entity_id)
    relations_tail = process_relation(results)
    for rel in relations_tail:
        rel2isTail[rel] = "True"
    if topic_entity_id.endswith('@en') or topic_entity_id.endswith('^^xsd:dateTime') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
        results = sparql_search(sparql_txt_find_head_rel % topic_entity_id)
    else:
        results = sparql_search(sparql_txt_find_head_rel % ("ns:" + topic_entity_id))
    relations_head = process_relation(results)
    for rel in relations_head:
        if (rel in rel2isTail) and (rel2isTail[rel] == "True"):
            rel2isTail[rel] = "both"
        else:
            rel2isTail[rel] = "False"
    relations = list(set(relations_tail + relations_head))
    
    # prune 1 hop relations
    if len(relations) > rel_for_2hop:
        relations_1hop = dense_retrieve(list(relations), question, min(width_1hop_rel, len(relations)))
    else:
        relations_1hop = relations
    
    # 1 hop entities
    candidates = set()
    rel2ents = dict()
    for relation in relations_1hop:
        if rel2isTail[relation] == "True":
            results = sparql_search(sparql_txt_find_tail.format(topic_entity_id, relation))
            entities = process_entity(results)
            candidates, rel2ents = construct_candidates(candidates, rel2ents, topic_entity_name, relation, entities, ishead=True, istwohop=False)
        elif rel2isTail[relation] == "False":
            if topic_entity_id.endswith('@en') or topic_entity_id.endswith('^^xsd:dateTime') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
                results = sparql_search(sparql_txt_find_head.format(topic_entity_id, relation))
            else:
                results = sparql_search(sparql_txt_find_head.format("ns:" + topic_entity_id, relation))
            entities = process_entity(results)
            candidates, rel2ents = construct_candidates(candidates, rel2ents, topic_entity_name, relation, entities, ishead=False, istwohop=False)
        else: # "both"
            results = sparql_search(sparql_txt_find_tail.format(topic_entity_id, relation))
            entities = process_entity(results)
            candidates, rel2ents = construct_candidates(candidates, rel2ents, topic_entity_name, relation, entities, ishead=True, istwohop=False)
            if topic_entity_id.endswith('@en') or topic_entity_id.endswith('^^xsd:dateTime') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
                results = sparql_search(sparql_txt_find_head.format(topic_entity_id, relation))
            else:
                results = sparql_search(sparql_txt_find_head.format("ns:" + topic_entity_id, relation))
            entities = process_entity(results)
            candidates, rel2ents = construct_candidates(candidates, rel2ents, topic_entity_name, relation, entities, ishead=False, istwohop=False)

    # entities for 2 hop searching
    entities_1hop = []
    ent2isTail = dict()
    for relation in relations_1hop[:rel_for_2hop]:
        if relation in rel2ents:
            if len(rel2ents[relation]) > 200: # limit number of entity for each 1hop relation under 200
                entities_1hop.extend(random.sample(rel2ents[relation], 200))
            else:
                entities_1hop.extend(rel2ents[relation])
            for ent in rel2ents[relation]:
                if (ent[0] in ent2isTail) and (ent2isTail[ent[0]] != rel2isTail[relation]):
                    ent2isTail[ent[0]] = "both"
                else:
                    ent2isTail[ent[0]] = rel2isTail[relation]
    entities_1hop = set((ent[0],ent[1]) for ent in entities_1hop)

    # 2 hop triples
    for entity in entities_1hop:
        rel2isTail_2hoprel = dict()
        # 2 hop relations
        if ent2isTail[entity[0]] == "True":
            results = sparql_search(sparql_txt_find_tail_rel % entity[0])
            relations_2hop_cand = process_relation(results)
            for rel in relations_2hop_cand:
                rel2isTail_2hoprel[rel] = "True"
        elif ent2isTail[entity[0]] == "False":
            results = sparql_search(sparql_txt_find_head_rel % ("ns:" + entity[0]))
            relations_2hop_cand = process_relation(results)
            for rel in relations_2hop_cand:
                rel2isTail_2hoprel[rel] = "False"
        else: # "both"
            results = sparql_search(sparql_txt_find_tail_rel % entity[0])
            relations_2hop_cand_tail = process_relation(results)
            for rel in relations_2hop_cand_tail:
                rel2isTail_2hoprel[rel] = "True"
            results = sparql_search(sparql_txt_find_head_rel % ("ns:" + entity[0]))
            relations_2hop_cand_head = process_relation(results)
            for rel in relations_2hop_cand_head:
                if (rel in rel2isTail_2hoprel) and (rel2isTail_2hoprel[rel] == "True"):
                    rel2isTail_2hoprel[rel] = "both"
                else:
                    rel2isTail_2hoprel[rel] = "False"
            relations_2hop_cand = list(set(relations_2hop_cand_tail + relations_2hop_cand_head))
            
        # prune 2 hop relations
        if len(relations_2hop_cand) > width_2hop_rel:
            # retrive
            relations_2hop = dense_retrieve(list(relations_2hop_cand), question, min(width_2hop_rel, len(relations_2hop_cand)))
        else:
            relations_2hop = relations_2hop_cand
        # 2 hop entities
        for relation in relations_2hop:
            if rel2isTail_2hoprel[relation] == "True":
                results = sparql_search(sparql_txt_find_tail.format(entity[0], relation))
                entities = process_entity(results)
                candidates, _ = construct_candidates(candidates, dict(), entity[1], relation, entities, ishead=True, istwohop=True, relations1hop=relations_1hop[:rel_for_2hop])
            elif rel2isTail_2hoprel[relation] == "False":
                results = sparql_search(sparql_txt_find_head.format("ns:" + entity[0], relation))
                entities = process_entity(results)
                candidates, _ = construct_candidates(candidates, dict(), entity[1], relation, entities, ishead=False, istwohop=True, relations1hop=relations_1hop[:rel_for_2hop])
            else: # "both"
                results = sparql_search(sparql_txt_find_tail.format(entity[0], relation))
                entities = process_entity(results)
                candidates, _ = construct_candidates(candidates, dict(), entity[1], relation, entities, ishead=True, istwohop=True, relations1hop=relations_1hop[:rel_for_2hop])
                results = sparql_search(sparql_txt_find_head.format("ns:" + entity[0], relation))
                entities = process_entity(results)
                candidates, _ = construct_candidates(candidates, dict(), entity[1], relation, entities, ishead=False, istwohop=True, relations1hop=relations_1hop[:rel_for_2hop])

    # candidates in 2 hop from topic entity
    return list(candidates)


def find_candidates_2hoprelation(question, topic_entity_name, topic_entity_id):
    width_1hop_rel = 10
    rel_for_2hop = 5
    width_2hop_rel = 5

    rel2isTail = dict()

    # 1 hop relations
    if topic_entity_id.endswith('@en') or topic_entity_id.endswith('^^xsd:dateTime') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
        results = []
    else:
        results = sparql_search(sparql_txt_find_tail_rel % topic_entity_id)
    relations_tail = process_relation(results)
    for rel in relations_tail:
        rel2isTail[rel] = "True"
    if topic_entity_id.endswith('@en') or topic_entity_id.endswith('^^xsd:dateTime') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
        results = sparql_search(sparql_txt_find_head_rel % topic_entity_id)
    else:
        results = sparql_search(sparql_txt_find_head_rel % ("ns:" + topic_entity_id))
    relations_head = process_relation(results)
    for rel in relations_head:
        if (rel in rel2isTail) and (rel2isTail[rel] == "True"):
            rel2isTail[rel] = "both"
        else:
            rel2isTail[rel] = "False"
    relations = list(set(relations_tail + relations_head))
    
    # prune 1 hop relations
    if len(relations) > rel_for_2hop:
        relations_1hop = dense_retrieve(list(relations), question, min(width_1hop_rel, len(relations)))
    else:
        relations_1hop = relations
    
    # 1 hop entities
    candidates = set()
    rel2ents = dict()
    candidate2isTail = dict()
    for relation in relations_1hop:
        if rel2isTail[relation] == "True":
            results = sparql_search(sparql_txt_find_tail.format(topic_entity_id, relation))
            entities = process_entity(results)
            candidates, rel2ents, candidate2isTail = construct_candidates_1hop(candidates, rel2ents, candidate2isTail, topic_entity_name, relation, entities, ishead=True)
        elif rel2isTail[relation] == "False":
            if topic_entity_id.endswith('@en') or topic_entity_id.endswith('^^xsd:dateTime') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
                results = sparql_search(sparql_txt_find_head.format(topic_entity_id, relation))
            else:
                results = sparql_search(sparql_txt_find_head.format("ns:" + topic_entity_id, relation))
            entities = process_entity(results)
            candidates, rel2ents, candidate2isTail = construct_candidates_1hop(candidates, rel2ents, candidate2isTail, topic_entity_name, relation, entities, ishead=False)
        else: # "both"
            results = sparql_search(sparql_txt_find_tail.format(topic_entity_id, relation))
            entities = process_entity(results)
            candidates, rel2ents, candidate2isTail = construct_candidates_1hop(candidates, rel2ents, candidate2isTail, topic_entity_name, relation, entities, ishead=True)
            if topic_entity_id.endswith('@en') or topic_entity_id.endswith('^^xsd:dateTime') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or topic_entity_id.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
                results = sparql_search(sparql_txt_find_head.format(topic_entity_id, relation))
            else:
                results = sparql_search(sparql_txt_find_head.format("ns:" + topic_entity_id, relation))
            entities = process_entity(results)
            candidates, rel2ents, candidate2isTail = construct_candidates_1hop(candidates, rel2ents, candidate2isTail, topic_entity_name, relation, entities, ishead=False)
    
    # entities for 2 hop searching
    entities_1hop = []
    ent2isTail = dict()
    ent2rels = dict()
    for relation in relations_1hop[:rel_for_2hop]:
        if relation in rel2ents:
            if len(rel2ents[relation]) > 10: # limit number of entity for each 1hop relation under 10
                entities_1hop.extend(random.sample(rel2ents[relation], 10))
            else:
                entities_1hop.extend(rel2ents[relation])
            for ent in rel2ents[relation]:
                if (ent in ent2isTail) and (ent2isTail[ent] != rel2isTail[relation]):
                    ent2isTail[ent] = "both"
                else:
                    ent2isTail[ent] = rel2isTail[relation]
                if ent in ent2rels:
                    ent2rels[ent].append(relation)
                else:
                    ent2rels[ent] = [relation]
    entities_1hop = set(entities_1hop)

    # 2 hop relations
    for entity in entities_1hop:
        rel2isTail_2hoprel = dict()
        # search 2 hop relations
        if ent2isTail[entity] == "True":
            results = sparql_search(sparql_txt_find_tail_rel % entity)
            relations_2hop_cand = process_relation(results)
            for rel in relations_2hop_cand:
                rel2isTail_2hoprel[rel] = "True"
        elif ent2isTail[entity] == "False":
            results = sparql_search(sparql_txt_find_head_rel % ("ns:" + entity))
            relations_2hop_cand = process_relation(results)
            for rel in relations_2hop_cand:
                rel2isTail_2hoprel[rel] = "False"
        else: # "both"
            results = sparql_search(sparql_txt_find_tail_rel % entity)
            relations_2hop_cand_tail = process_relation(results)
            for rel in relations_2hop_cand_tail:
                rel2isTail_2hoprel[rel] = "True"
            results = sparql_search(sparql_txt_find_head_rel % ("ns:" + entity))
            relations_2hop_cand_head = process_relation(results)
            for rel in relations_2hop_cand_head:
                if (rel in rel2isTail_2hoprel) and (rel2isTail_2hoprel[rel] == "True"):
                    rel2isTail_2hoprel[rel] = "both"
                else:
                    rel2isTail_2hoprel[rel] = "False"
            relations_2hop_cand = list(set(relations_2hop_cand_tail + relations_2hop_cand_head))
            
        # prune 2 hop relations
        if len(relations_2hop_cand) > width_2hop_rel:
            # retrieve
            relations_2hop = dense_retrieve(list(relations_2hop_cand), question, min(width_2hop_rel, len(relations_2hop_cand)))
        else:
            relations_2hop = relations_2hop_cand
        
        # construct candidates with 2 hop relation
        for relation in relations_2hop:
            if rel2isTail_2hoprel[relation] == "True":
                candidates, candidate2isTail = construct_candidates_2hop(candidates, candidate2isTail, relation, ent2rels[entity], ishead=True)
            elif rel2isTail_2hoprel[relation] == "False":
                candidates, candidate2isTail = construct_candidates_2hop(candidates, candidate2isTail, relation, ent2rels[entity], ishead=False)
            else: # "both"
                candidates, candidate2isTail = construct_candidates_2hop(candidates, candidate2isTail, relation, ent2rels[entity], ishead=True)
                candidates, candidate2isTail = construct_candidates_2hop(candidates, candidate2isTail, relation, ent2rels[entity], ishead=False)

    # candidates in 2 hop from topic entity
    return list(candidates), candidate2isTail


def sparse_retrieve(candidates, question, width):
    # BM25
    if width == 0:
        return []
    tokenized_candidates = [cand.split(" ") for cand in candidates]
    bm25 = BM25Okapi(tokenized_candidates)
    tokenized_question = question.split(" ")
    scores = bm25.get_scores(tokenized_question)
    top_triplets = bm25.get_top_n(tokenized_question, candidates, n=width)
    return top_triplets


dense_retriever = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3')
def dense_retrieve(candidates, question, width):
    # Distilbert
    if width == 0:
        return []
    question_embeddings = dense_retriever.encode(question)
    candidates_embeddings = dense_retriever.encode(candidates)
    scores = util.dot_score(question_embeddings, candidates_embeddings)[0].tolist()
    results = [{"triplet": cand, "score": score} for cand, score in zip(candidates, scores)]
    top_results = sorted(results, key=lambda x: x["score"], reverse=True)[:width]
    top_triplets = [res["triplet"] for res in top_results]
    return top_triplets


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def rerank(candidates, question):
    # MiniLM
    if len(candidates) == 0:
        return []
    model_inputs = [[question, candidate] for candidate in candidates]
    scores = reranker.predict(model_inputs)
    results = [{"input": inp, "score": score} for inp, score in zip(model_inputs, scores)]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    reranked_triplets = [res["input"][1] for res in results]
    return reranked_triplets


def prune_entity(origin_entities):
    pruned_entities = []
    nameless = [ent for ent in origin_entities if ent[0] == ent[1] and (ent[0].startswith("m.") or ent[0].startswith("g."))]
    hasName = [ent for ent in origin_entities if ent not in nameless]
    if len(origin_entities) <= 20 and len(nameless) <= 1:
        return origin_entities
    if len(hasName) >= 20:
        pruned_entities.extend(random.sample(hasName, 20))
    else:
        pruned_entities.extend(hasName)
        pruned_entities.extend(random.sample(nameless, 1))
    return pruned_entities


sparql_txt_find_tail_2hop = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity
WHERE {{
    ns:{0} ns:{1} ?c .
    ?c ns:{2} ?entity .
}}"""
sparql_txt_find_head_2hop = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?entity
WHERE {{
    ?entity ns:{2} ?c .
    ?c ns:{1} {0} .
}}"""
def retrieve_triplets_2hop(question, questions4topic, topic_entity_name_gold, topic_entity_id_gold, KG_retriever, entity_linker, width, reference_num, decompose_depth):
    # get topic entity
    topic_entity_name = copy.deepcopy(topic_entity_name_gold)
    topic_entity_id = copy.deepcopy(topic_entity_id_gold)
    if decompose_depth > 0:
        for q4topic in questions4topic:
            topic_entity_name_q, topic_entity_id_q = entity_linking(q4topic, entity_linker)
            if (topic_entity_name_q != "null") and (topic_entity_name_q not in topic_entity_name) and (topic_entity_id_q not in topic_entity_id):
                topic_entity_name.append(topic_entity_name_q)
                topic_entity_id.append(topic_entity_id_q)
    
    # prepare candidate (triples) / (relations or relation chains)
    candidates = []
    if KG_retriever == "2hoprelation":
        candidate2isTail = {}
    for i in range(len(topic_entity_id)):
        if KG_retriever == "2hoptriple":
            candidates_1topic = find_candidates_2hoptriple(question, topic_entity_name[i], topic_entity_id[i])
        elif KG_retriever == "2hoprelation":
            candidates_1topic, candidate2isTail_1topic = find_candidates_2hoprelation(question, topic_entity_name[i], topic_entity_id[i])
            candidate2isTail.update(candidate2isTail_1topic)
        candidates.extend(candidates_1topic)
    candidates = list(set(candidates))

    # retrieve and rerank relations
    # retrieve
    dense_result = dense_retrieve(candidates, question, min(width, len(candidates)))
    result_1 = set(dense_result)
    # rerank
    result = rerank(result_1, question)
    # process triples for being references
    if KG_retriever == "2hoptriple":
        references = result[:min(reference_num, len(result))]
    elif KG_retriever == "2hoprelation":
        references_relation = result[:min(reference_num, len(result))]

        # add tail / head entity to the retrieved relations
        references = []
        for ref in references_relation:
            elements = [e[2:-2] for e in ref.split(" [SEP] ")]
            if len(elements) == 2: # 1hop relation
                if candidate2isTail[ref]:
                    results = sparql_search(sparql_txt_find_tail.format(topic_entity_id[topic_entity_name.index(elements[0])], elements[1]))
                    entities = process_entity(results)
                    entities = prune_entity(entities)
                    references.append("< {} > [SEP] < {} > [SEP] [ {} ]".format(elements[0], elements[1], " ".join(["< " + ent[1] + " >" for ent in entities])))
                else:
                    eid = topic_entity_id[topic_entity_name.index(elements[1])]
                    if eid.endswith('@en') or eid.endswith('^^xsd:dateTime') or eid.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or eid.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or eid.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
                        results = sparql_search(sparql_txt_find_head.format(eid, elements[0]))
                    else:
                        results = sparql_search(sparql_txt_find_head.format("ns:" + eid, elements[0]))
                    entities = process_entity(results)
                    entities = prune_entity(entities)
                    references.append("[ {} ] [SEP] < {} > [SEP] < {} >".format(" ".join(["< " + ent[1] + " >" for ent in entities]), elements[0], elements[1]))
            elif len(elements) == 3: # 1hop & 2hop relations
                if candidate2isTail[ref]:
                    results = sparql_search(sparql_txt_find_tail_2hop.format(topic_entity_id[topic_entity_name.index(elements[0])], elements[1], elements[2]))
                    entities = process_entity(results)
                    entities = prune_entity(entities)
                    references.append("< {} > [SEP] < {} > [SEP] < {} > [SEP] [ {} ]".format(elements[0], elements[1], elements[2], " ".join(["< " + ent[1] + " >" for ent in entities])))
                else:
                    eid = topic_entity_id[topic_entity_name.index(elements[2])]
                    if eid.endswith('@en') or eid.endswith('^^xsd:dateTime') or eid.endswith('^^<http://www.w3.org/2001/XMLSchema#integer>') or eid.endswith('^^<http://www.w3.org/2001/XMLSchema#date>') or eid.endswith('^^<http://www.w3.org/2001/XMLSchema#float>'):
                        results = sparql_search(sparql_txt_find_head_2hop.format(eid, elements[1], elements[0]))
                    else:
                        results = sparql_search(sparql_txt_find_head_2hop.format("ns:" + eid, elements[1], elements[0]))
                    entities = process_entity(results)
                    entities = prune_entity(entities)
                    references.append("[ {} ] [SEP] < {} > [SEP] < {} > [SEP] < {} >".format(" ".join(["< " + ent[1] + " >" for ent in entities]), elements[0], elements[1], elements[2]))

    return [topic_entity_name, topic_entity_id], references


def retrieve_triplets_gnn(question, questions4topic, entity_linker, gnn, GNN_data, width, reference_num, decompose_depth):
    entityId2GNNId, GNNId2entityId = gnn['dicts']
    # get topic entity & prepare single data for GNN-RAG
    GNN_data_copy = copy.deepcopy(GNN_data)
    topic_entity_gid = GNN_data_copy['entities']
    topic_entity_id = [GNNId2entityId[gid] for gid in topic_entity_gid]
    topic_entity_name = [entityid2name(eid) for eid in topic_entity_id]
    if decompose_depth > 0:
        subgraph_entities = GNN_data_copy['subgraph']['entities']
        for q4topic in questions4topic:
            topic_entity_name_q, topic_entity_id_q = entity_linking(q4topic, entity_linker)
            if (topic_entity_name_q != "null") and (topic_entity_name_q not in topic_entity_name) and (topic_entity_id_q not in topic_entity_id) and (topic_entity_id_q in entityId2GNNId) and (entityId2GNNId[topic_entity_id_q] in subgraph_entities):
                topic_entity_name.append(topic_entity_name_q)
                topic_entity_id.append(topic_entity_id_q)
                GNN_data_copy['entities'].append(entityId2GNNId[topic_entity_id_q])
        # change original question to sub-question
        GNN_data_copy['question'] = question
    
    # prepare GNN-RAG
    success = False
    while not success:
        try:
            gnn['trainer'].init_add1(args=vars(gnn['args']), my_data=GNN_data_copy)
            ckpt_path = os.path.join(gnn['args'].checkpoint_dir, gnn['args'].load_experiment)
            checkpoint = torch.load(ckpt_path)
            model_state_dict = checkpoint["model_state_dict"]
            model = gnn['trainer'].model
            model.load_state_dict(model_state_dict, strict=False)
            success = True
        except Exception as e:
            print("Exception in Loading GNN-RAG: ")
            print(e)
            time.sleep(2)

    # retrieve with gnn in GNN-RAG
    # get retrieved entities with gnn in GNN-RAG
    retrieved_entities_ids = gnn['trainer'].my_evaluate(gnn['trainer'].test_data, question)
    # construct triple link from topic entities to retrieved entities
    references_raw = gnn['trainer'].my_find_path(gnn['trainer'].test_data, question, retrieved_entities_ids)
    # [entity_0_id，relation_1_name，entity_1_id，…， relation_n_name, entity_n_id] -> "< entity_0_name > [SEP] <relation_1_name> [SEP] ..."
    references = []
    for reference_raw in references_raw:
        reference = ""
        reference += "< " + entityid2name(reference_raw[0]) + " > [SEP] "
        for i in range(1, len(reference_raw)-1):
            reference += "< " + reference_raw[i] + " > [SEP] "
        reference += "< " + entityid2name(reference_raw[len(reference_raw)-1]) + " >"
        references.append(reference)
    if len(references) > 200:
        references = references[:200]

    return [topic_entity_name, topic_entity_id], references


def find_nums(string):
    matches = re.findall(r'\[#\d+\]', string)  # [#123]
    matches_and_nums = []
    for match in matches:
        match_len = len(match)
        num = int(match[2:match_len-1])
        if (match, num) not in matches_and_nums:
            matches_and_nums.append((match, num))
    return matches_and_nums


def integrate(question, previous_subQAs, llm, llm_input_len, llm_output_len):
    prompt = prompt_integrate_judge.format(question, previous_subQAs.rstrip("\n"))
    response, llm_input_len, llm_output_len = reason_with_llm(prompt, llm, llm_input_len, llm_output_len)

    # extract the judgement: if subQAs are sufficient for answering the question or not
    flag_sufficient = False
    if "[sufficient]" in response:
        flag_sufficient = True

    # extract the {answer_entity} in llm's response
    answer_pattern = r'\{([^}]*)\}'
    answers = re.findall(answer_pattern, response)
    if (len(answers) == 0) or ((len(answers) == 1) and (answers[0].lower() in ["none", "unknown"])):
        prompt = prompt_answer_extract.format(question, response)
        response_2, llm_input_len, llm_output_len = reason_with_llm(prompt, llm, llm_input_len, llm_output_len)
        answers = re.findall(answer_pattern, response_2)
    answer_entity = "; ".join(answers)
        
    return response, flag_sufficient, answer_entity, llm_input_len, llm_output_len


def handle_question(question_id, question, previous_subQAs, questions4topic, depth, topic_entity_name, topic_entity_id, entity_linker, gnn, GNN_data, response_chain, answer_chain, topic_entity_chain, reference_triplets_chain, args, llm_input_len, llm_output_len):
    # determine whether the question can be answered through one-step reasoning    
    if depth != args.max_decompose_depth:
        response_phase1, flag_onestep, llm_input_len, llm_output_len = determine(question, args.dataset, args.llm, llm_input_len, llm_output_len)
    else:
        response_phase1 = "no_need_for_LLM"
        flag_onestep = True
    answer_entity = ""
    if flag_onestep:
        if ("intersection" in question) or ("[sum up]" in question):
            topic_entity, reference_triplets = ["no_need", "no_need"], ["no_need"]
        else:
            if args.KG_retriever == "gnn":
                topic_entity, reference_triplets = retrieve_triplets_gnn(question, questions4topic, entity_linker, gnn, GNN_data, args.retrieve_width, args.reference_num, depth)
            else:
                topic_entity, reference_triplets = retrieve_triplets_2hop(question, questions4topic, topic_entity_name, topic_entity_id, args.KG_retriever, entity_linker, args.retrieve_width, args.reference_num, depth)
        response_phase2, answer_entity, llm_input_len, llm_output_len = generate_answer(question, args.dataset, args.KG_retriever, previous_subQAs, reference_triplets, args.llm, llm_input_len, llm_output_len)
        response_chain.append([response_phase1, response_phase2])
        answer_chain.append([str(flag_onestep), answer_entity])
        topic_entity_chain.append(topic_entity)
        reference_triplets_chain.append(reference_triplets)
    else:
        left_composition_types = {"composition", "conjunction", "comparative", "superlative"}
        answer_entity_types = []
        response_types = []
        flag_sufficient = False
        while len(left_composition_types) >= (5 - args.max_attempts):
            # for complex question, further determine its composition type
            if len(left_composition_types) == 1:
                composition_type = list(left_composition_types)[0]
            else:
                composition_type, llm_input_len, llm_output_len = subclassify(question, left_composition_types, args.llm, llm_input_len, llm_output_len)
            # decompose the complex question according to the template of composition_type
            response_phase3, subquestions, llm_input_len, llm_output_len = decompose(question, composition_type, args.llm, llm_input_len, llm_output_len)
            response_chain.append([response_phase1, response_phase3])
            answer_chain.append([str(flag_onestep) + ":" + composition_type, ';'.join(subquestions)])
            topic_entity_chain.append([])
            reference_triplets_chain.append([])
            
            if len(subquestions) == 0:  # decompose error
                if args.KG_retriever == "gnn":
                    topic_entity, reference_triplets = retrieve_triplets_gnn(question, questions4topic, entity_linker, gnn, GNN_data, args.retrieve_width, args.reference_num, depth)
                else:
                    topic_entity, reference_triplets = retrieve_triplets_2hop(question, questions4topic, topic_entity_name, topic_entity_id, args.KG_retriever, entity_linker, args.retrieve_width, args.reference_num, depth)
                response_phase2, answer_entity, llm_input_len, llm_output_len = generate_answer(question, args.dataset, args.KG_retriever, previous_subQAs, reference_triplets, args.llm, llm_input_len, llm_output_len)
                response_chain.append(["decompose error", response_phase2])
                answer_chain.append(["decompose error", answer_entity])
                topic_entity_chain.append(topic_entity)
                reference_triplets_chain.append(reference_triplets)
            else:
                # solve subquestions
                previous_subQAs = ""
                subanswers = []
                for i in range(len(subquestions)):
                    # replace the [#xxx] in subquestion with its answer entity
                    subquestion = subquestions[i]
                    previous_answers = find_nums(subquestion)
                    subqs4topic = []
                    if (len(previous_answers) == 1) and (previous_answers[0][1] < i + 1) and (previous_answers[0][1] > 0) and (len(subanswers[previous_answers[0][1] - 1].split(";")) > 1):
                        num_str, num = previous_answers[0]
                        subanswer_set = [sa.strip() for sa in subanswers[num-1].split(";")]
                        for sa in subanswer_set:
                            subqs4topic.append(subquestion.replace(num_str, sa))
                        subquestion = subquestion.replace(num_str, subanswers[num-1])
                    else:
                        for num_str, num in previous_answers:
                            if num < i + 1 and num > 0:
                                subquestion = subquestion.replace(num_str, subanswers[num-1])
                        subqs4topic.append(subquestion)
                    # answer the subquestion
                    answer_entity, response_chain, answer_chain, topic_entity_chain, reference_triplets_chain, llm_input_len, llm_output_len = handle_question(question_id, subquestion, previous_subQAs, subqs4topic, depth+1, topic_entity_name, topic_entity_id, entity_linker, gnn, GNN_data, response_chain, answer_chain, topic_entity_chain, reference_triplets_chain, args, llm_input_len, llm_output_len)
                    subanswers.append(answer_entity)
                    if len(answer_entity) == 0:
                        previous_subQAs += subquestion + " -- " + response_chain[-1][1] + "\n"
                    else:
                        previous_subQAs += subquestion + " -- " + answer_entity + "\n"
                # integrate the subQAs to solve the question
                response_phase2, flag_sufficient, answer_entity, llm_input_len, llm_output_len = integrate(question, previous_subQAs, args.llm, llm_input_len, llm_output_len)
                response_chain.append(["integrate", response_phase2])
                answer_chain.append(["integrate", answer_entity])
                topic_entity_chain.append([])
                reference_triplets_chain.append([])
            
            left_composition_types = left_composition_types - {composition_type}
            answer_entity_types.append(answer_entity)
            response_types.append(response_phase2)
            if flag_sufficient:
                break
        if (len(left_composition_types) == 1) and ("[sufficient]" not in response_phase2):
            response_chain.append(["concatenate", "\n".join(response_types)])
            answer_entity = "; ".join(list(set([a.strip() for a_t in answer_entity_types for a in a_t.split(";")])))
            answer_chain.append(["concatenate", answer_entity])
            topic_entity_chain.append([])
            reference_triplets_chain.append([])
    
    return answer_entity, response_chain, answer_chain, topic_entity_chain, reference_triplets_chain, llm_input_len, llm_output_len


def write_to_file(args, batch_index, question_id, question, answer, topic_entity_chain, reference_triplets_chain, answer_chain, response_chain):
    output_path = "./output/test/{}-{}-{}-{}.test.jsonl".format(batch_index, args.dataset, args.llm['name'], args.KG_retriever)
    output_content = {
        "Id": question_id,
        "Question": question,
        "Answer": answer,
        "TopicEntityChain": topic_entity_chain,
        "ReferenceTripletsChain": reference_triplets_chain,
        "AnswerChain": answer_chain,
        "ResponseChain": response_chain
    }
    success = False
    while not success:
        try:
            with open(output_path, "a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(output_content) + "\n")
            success = True
        except Exception as e:
            print("Exception in WriteToFile:")
            print(e)
            if "No space left on device" not in str(e):
                break
            time.sleep(120 + 60 * random.random())
