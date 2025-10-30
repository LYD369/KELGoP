from utils import *

import argparse
from tqdm import tqdm
import time
import threading
import queue
import math

import torch
import gc

import os


queue_lock = threading.Lock()

def process_batch(batch_index, questions, question_ids, topic_entity_names, topic_entity_ids, entity_linker, gnn, GNN_datas, args, llm_io_lens):
    llm_input_len = 0
    llm_output_len = 0

    for i in tqdm(range(len(questions))):
        if args.llm['name'] in ["qwen2.5", "llama3.1"]:
            gc.collect()
            torch.cuda.empty_cache()
        question_id = question_ids[i]
        question = questions[i]
        topic_entity_name = topic_entity_names[i]
        topic_entity_id = topic_entity_ids[i]
        if args.KG_retriever == "gnn":
            GNN_data = GNN_datas[i]
        else:
            GNN_data = None

        response_chain = []
        answer_chain = []
        topic_entity_chain = []
        reference_triplets_chain = []
        answer, response_chain, answer_chain, topic_entity_chain, reference_triplets_chain, llm_input_len, llm_output_len = handle_question(question_id, question, "", [question], 0, topic_entity_name, topic_entity_id, entity_linker, gnn, GNN_data, response_chain, answer_chain, topic_entity_chain, reference_triplets_chain, args, llm_input_len, llm_output_len)

        # write to file
        write_to_file(args, batch_index, question_id, question, answer, topic_entity_chain, reference_triplets_chain, answer_chain, response_chain)
    
    with queue_lock:
        llm_io_lens.put([llm_input_len, llm_output_len])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="webqsp", choices=["webqsp", "cwq", "grailqa"])
    parser.add_argument("--llm_name", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "qwen2.5", "llama3.1", "gpt-4o-2024-11-20"])
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--KG_retriever", type=str, default="2hoptriple", choices=["2hoptriple", "2hoprelation", "gnn"])
    parser.add_argument("--retrieve_width", type=int, default=20)
    parser.add_argument("--reference_num", type=int, default=10)
    parser.add_argument("--max_decompose_depth", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3531)
    parser.add_argument("--elq_threshold", type=float, default=-5)
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = ""  # your OpenAI API key
    os.environ["LLAMA_PATH"] = ""    # your local LLaMA3.1 model path
    os.environ["QWEN_PATH"] = ""     # your local Qwen2.5 model path

    print("Preparing data of {}...".format(args.dataset))
    question_ids, questions, topic_entity_names, topic_entity_ids = prepare_data(args.dataset)

    print("Preparing entity linker...")
    elq_start_time = time.perf_counter()
    entity_linker = prepare_entity_linker(args.elq_threshold)
    entity_linker['elqId2wikiId'] = entity_linker['models'][6]
    print("\telq_load_time = {}s".format(time.perf_counter() - elq_start_time))

    print("Preparing LLM: {}...".format(args.llm_name))
    args.llm = prepare_llm(args.llm_name)

    if args.KG_retriever == "gnn":
        print("Preparing GNN...")
        gnn = prepare_GNN(args.dataset)
    else:
        gnn = None

    print("Start reasoning (KG retriever: {})...".format(args.KG_retriever))

    threads = []
    llm_io_lens = queue.Queue()

    for index in tqdm(range(math.ceil(len(questions) / args.batch_size))):
        start_i = index * args.batch_size
        end_i = min((index + 1) * args.batch_size, len(questions))

        question_ids_batch = question_ids[start_i:end_i]
        questions_batch = questions[start_i:end_i]
        topic_entity_names_batch = topic_entity_names[start_i:end_i]
        topic_entity_ids_batch = topic_entity_ids[start_i:end_i]
        if args.KG_retriever == "gnn":
            GNN_datas_batch = gnn['datas'][start_i:end_i]
        else:
            GNN_datas_batch = None

        thread = threading.Thread(target=process_batch, args=(index, questions_batch, question_ids_batch, topic_entity_names_batch, topic_entity_ids_batch, entity_linker, gnn, GNN_datas_batch, args, llm_io_lens))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    llm_input_len = 0
    llm_output_len = 0
    while not llm_io_lens.empty():
        llm_io_len_batch = llm_io_lens.get()
        llm_input_len += llm_io_len_batch[0]
        llm_output_len += llm_io_len_batch[1]

    print("\nllm_input_tokens_num = {}, llm_output_tokens_num = {}".format(llm_input_len, llm_output_len))
    print("ok")