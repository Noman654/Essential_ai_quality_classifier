#!/usr/bin/env python3
"""
API-BASED PARALLEL VERSION
- Uses CPU Producer Processes to read & prepare prompts
- Sends requests to existing vLLM API server asynchronously
- No model deployment - connects to running API
"""

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
import time
import asyncio
import aiohttp
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from typing import List, Tuple
import logging
from datetime import datetime
import argparse

# Constants
API_TIMEOUT = 600  # 5 minutes timeout per request

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        force=True
    )


def parse_classification(classification_str):
        """
        Parse the classification string into structured format.
        
        Output format:
        {FDC primary},{FDC secondary or skip}
        {Bloom cognitive process primary (1-6)},{Bloom cognitive process secondary (1-6) or skip}
        {Bloom knowledge domain primary (1-4)},{Bloom knowledge domain secondary (1-4) or skip}
        {Document type v1 primary (1-17)},{Document type v1 secondary (1-17) or skip}
        {Extraction artifacts primary (0-4)},{Extraction artifacts secondary (0-4) or skip}
        {Missing content primary (0-6)},{Missing content secondary (0-6) or skip}
        {Document type v2 primary (1-25)},{Document type v2 secondary (1-25) or skip}
        {Reasoning depth primary (1-6)},{Reasoning depth secondary (1-6) or skip}
        {Technical correctness primary (1-6)},{Technical correctness secondary (1-6) or skip}
        {Educational level primary (1-5)},{Educational level secondary (1-5) or skip}
        """
        # Clean the output - remove any remaining conversation artifacts
        clean_str = classification_str.strip()
        if "assistant" in clean_str:
            clean_str = clean_str.split("assistant")[-1].strip()
        
        lines = [line.strip() for line in clean_str.split('\n') if line.strip()]
        
        result = {
            'raw_output': classification_str,
            'fdc_primary': None,
            'fdc_secondary': None,
            'bloom_cognitive_primary': None,
            'bloom_cognitive_secondary': None,
            'bloom_knowledge_primary': None,
            'bloom_knowledge_secondary': None,
            'doc_type_v1_primary': None,
            'doc_type_v1_secondary': None,
            'extraction_artifacts_primary': None,
            'extraction_artifacts_secondary': None,
            'missing_content_primary': None,
            'missing_content_secondary': None,
            'doc_type_v2_primary': None,
            'doc_type_v2_secondary': None,
            'reasoning_depth_primary': None,
            'reasoning_depth_secondary': None,
            'technical_correctness_primary': None,
            'technical_correctness_secondary': None,
            'educational_level_primary': None,
            'educational_level_secondary': None,
        }
        
        if len(lines) >= 10:
            try:
                # FDC
                parts = lines[0].split(',')
                result['fdc_primary'] = parts[0].strip() if len(parts) > 0 else None
                result['fdc_secondary'] = parts[1].strip() if len(parts) > 1 and parts[1].strip() != 'skip' else None
                
                # Bloom Cognitive
                parts = lines[1].split(',')
                result['bloom_cognitive_primary'] = int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else None
                result['bloom_cognitive_secondary'] = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                
                # Bloom Knowledge
                parts = lines[2].split(',')
                result['bloom_knowledge_primary'] = int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else None
                result['bloom_knowledge_secondary'] = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                
                # Document Type v1
                parts = lines[3].split(',')
                result['doc_type_v1_primary'] = int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else None
                result['doc_type_v1_secondary'] = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                
                # Extraction Artifacts
                parts = lines[4].split(',')
                result['extraction_artifacts_primary'] = int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else None
                result['extraction_artifacts_secondary'] = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                
                # Missing Content
                parts = lines[5].split(',')
                result['missing_content_primary'] = int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else None
                result['missing_content_secondary'] = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                
                # Document Type v2
                parts = lines[6].split(',')
                result['doc_type_v2_primary'] = int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else None
                result['doc_type_v2_secondary'] = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                
                # Reasoning Depth
                parts = lines[7].split(',')
                result['reasoning_depth_primary'] = int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else None
                result['reasoning_depth_secondary'] = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                
                # Technical Correctness
                parts = lines[8].split(',')
                result['technical_correctness_primary'] = int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else None
                result['technical_correctness_secondary'] = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                
                # Educational Level
                parts = lines[9].split(',')
                result['educational_level_primary'] = int(parts[0]) if len(parts) > 0 and parts[0].strip().isdigit() else None
                result['educational_level_secondary'] = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else None
                
            except Exception as e:
                print(f"Warning: Error parsing classification: {e}")
        
        return result


# --- CONFIGURATION ---
# API_URL and Paths will be set from args

NUM_PRODUCERS = 224  # Reduced for distributed execution (20 * 10 workers = 200 total)
BATCH_SIZE = 50000  # Prompts per batch
MAX_CONCURRENT_REQUESTS = 300  # Concurrent API batch requests (increased for better GPU utilization)
MAX_RETRIES = 3  # Number of retry attempts for failed requests
API_BATCH_SIZE = 250  # Number of prompts per API call (vLLM supports batch inference)

def chunk_text(text):
    text = str(text)
    if len(text) <= 30000: return text
    third = 10000
    return f"[beginning]\n{text[:third]}\n[middle]\n{text[len(text)//2-third//2:len(text)//2+third//2]}\n[end]\n{text[-third:]}"

def producer_worker_row_groups(file_path, row_group_indices, task_queue, batch_size, log_file):
    setup_logging(log_file)
    """
    Reads specific Row Groups (chunks of the file) from disk.
    """
    parquet_file = pq.ParquetFile(file_path)
    
    batch_counter = 0
    batch_texts = []
    batch_indices = []
    
    # Calculate global row offset based on previous row groups
    current_row_idx = 0
    for rg_idx in range(row_group_indices[0]):
        current_row_idx += parquet_file.metadata.row_group(rg_idx).num_rows
        
    for rg_idx in row_group_indices:
        # Read this specific chunk from disk
        table = parquet_file.read_row_group(rg_idx, columns=['text'])
        texts = table.column('text').to_pylist()
        
        for text in texts:
            if text and str(text).strip():
                # Prepare Prompt Here
                prompt = f"<|im_start|>system\ntaxonomy<|im_end|>\n<|im_start|>user\n{chunk_text(text)}<|im_end|>\n<|im_start|>assistant\n"
                
                batch_texts.append(prompt)
                batch_indices.append(current_row_idx)
                
                if len(batch_texts) >= batch_size:
                    task_queue.put((batch_counter, batch_texts, batch_indices))
                    batch_counter += 1
                    batch_texts = []
                    batch_indices = []
            
            current_row_idx += 1

    # Flush remaining
    if batch_texts:
        task_queue.put((batch_counter, batch_texts, batch_indices))

async def send_batch_api_request(session: aiohttp.ClientSession, prompts: List[str], indices: List[int], api_url: str) -> List[Tuple[int, str]]:
    """Send a batch of prompts to the vLLM API with retry logic.
    
    vLLM supports batch inference - sending multiple prompts in one request.
    This reduces TCP connections by ~100x and improves throughput.
    """
    payload = {
        "prompt": prompts,  # List of prompts for batch inference
        "max_tokens": 100,
        "temperature": 0.0,
        "stop": ["<|im_end|>"]
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(api_url, json=payload, timeout=aiohttp.ClientTimeout(total=API_TIMEOUT)) as response:
                if response.status != 200:
                    error_body = await response.text()
                    logging.error(f"Batch API request failed. Status: {response.status}. Body: {error_body[:500]}")
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    # Return empty results for all indices in batch
                    return [(idx, "") for idx in indices]
                
                try:
                    result = await response.json()
                except Exception as json_err:
                    logging.error(f"Failed to parse JSON for batch. Error: {json_err}.")
                    return [(idx, "") for idx in indices]

                if "choices" not in result or not result["choices"]:
                    logging.error(f"API response missing 'choices'. Response: {str(result)[:500]}")
                    return [(idx, "") for idx in indices]
                
                # Extract texts for each prompt in batch using the 'index' field
                choices = result['choices']
                # Create a map from index to text
                index_to_text = {}
                for choice in choices:
                    choice_idx = choice.get('index', -1)
                    choice_text = choice.get('text', '')
                    if choice_idx >= 0:
                        index_to_text[choice_idx] = choice_text
                
                # Build results in order of original indices
                results = []
                for i, idx in enumerate(indices):
                    text = index_to_text.get(i, '')  # i is the position in this batch
                    results.append((idx, text))
                return results

        except aiohttp.ClientConnectorError as e:
            logging.error(f"Connection Error for batch (attempt {attempt+1}/{MAX_RETRIES}). Error: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return [(idx, "") for idx in indices]
        except asyncio.TimeoutError:
            logging.error(f"Timeout Error for batch (attempt {attempt+1}/{MAX_RETRIES}).")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return [(idx, "") for idx in indices]
        except Exception as e:
            logging.error(f"General Error for batch. Error: {e}")
            return [(idx, "") for idx in indices]
    
    return [(idx, "") for idx in indices]

async def process_batch_async(batch_idx: int, prompts: List[str], indices: List[int], max_concurrent: int, pbar, api_url: str):
    """Process a batch of prompts asynchronously with batched API calls and connection pooling."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Split into sub-batches for API calls (e.g., 100 prompts per API call)
    sub_batches = []
    for i in range(0, len(prompts), API_BATCH_SIZE):
        sub_prompts = prompts[i:i + API_BATCH_SIZE]
        sub_indices = indices[i:i + API_BATCH_SIZE]
        sub_batches.append((sub_prompts, sub_indices))
    
    async def limited_batch_request(session, batch_prompts, batch_indices):
        """Wrapper to ensure semaphore controls actual request execution."""
        async with semaphore:
            return await send_batch_api_request(session, batch_prompts, batch_indices, api_url)
    
    # Use connection pooling with TCPConnector for connection reuse
    connector = aiohttp.TCPConnector(
        limit=max_concurrent,  # Max connections
        limit_per_host=max_concurrent,  # Max per host
        keepalive_timeout=60,  # Reuse connections for 60s
        enable_cleanup_closed=True
    )
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create tasks for sub-batches (much fewer than individual requests)
        tasks = [limited_batch_request(session, bp, bi) for bp, bi in sub_batches]
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results from all sub-batches
        results = []
        for sub_result in batch_results:
            results.extend(sub_result)
        
        # Count successes for this batch
        success_count = sum(1 for r in results if r[1] != "")
        skipped_count = len(indices) - success_count
        api_calls = len(sub_batches)
        msg = f"Batch {batch_idx} finished. Success: {success_count}/{len(indices)}. API calls: {api_calls}. Skipped/Failed: {skipped_count}"
        print(msg)
        logging.info(msg)
        
        pbar.update(1)
        return results

async def api_consumer(task_queue: mp.Queue, result_queue: mp.Queue, total_batches: int, filename: str, api_url: str):
    """Consume tasks from queue and process them via API."""
    
    with tqdm(total=total_batches, desc=f"   {filename}", unit="batch") as pbar:
        while True:
            try:
                # Non-blocking get with timeout
                if task_queue.empty():
                    await asyncio.sleep(0.001)
                    continue
                    
                item = task_queue.get_nowait()
                if item is None:
                    break
                
                batch_idx, prompts, indices = item
                
                # Process batch asynchronously with rate limiting
                results = await process_batch_async(batch_idx, prompts, indices, MAX_CONCURRENT_REQUESTS, pbar, api_url)
                
                # Extract just the results
                sorted_results = sorted(results, key=lambda x: x[0])
                result_texts = [r[1] for r in sorted_results]
                result_indices = [r[0] for r in sorted_results]
                
                result_queue.put((batch_idx, result_indices, result_texts))
                
            except Exception as e:
                logging.error(f"Error in consumer: {e}")
                await asyncio.sleep(0.1)

def run_async_consumer(task_queue, result_queue, total_batches, filename, log_file, api_url):
    """Wrapper to run async consumer in a separate process."""
    setup_logging(log_file)
    asyncio.run(api_consumer(task_queue, result_queue, total_batches, filename, api_url))

def process_file(input_path, output_path, log_file, api_url):
    filename = Path(input_path).name
    print(f"\nProcessing: {filename}")
    
    ctx = mp.get_context('spawn')
    task_queue = ctx.Queue(maxsize=100)
    result_queue = ctx.Queue()
    
    # --- 1. Split Work (Row Groups) ---
    parquet_file = pq.ParquetFile(input_path)
    total_row_groups = parquet_file.num_row_groups
    total_rows = parquet_file.metadata.num_rows
    estimated_batches = (total_rows // BATCH_SIZE) + 1
    
    # Split row groups among producers
    rg_indices = list(range(total_row_groups))
    chunks = [rg_indices[i::NUM_PRODUCERS] for i in range(NUM_PRODUCERS)]
    
    # --- 2. Start Producers (CPU) ---
    producers = []
    for chunk in chunks:
        if chunk:
            p = ctx.Process(target=producer_worker_row_groups, args=(input_path, chunk, task_queue, BATCH_SIZE, log_file))
            p.start()
            producers.append(p)
    
    # --- 3. Start API Consumer ---
    consumer = ctx.Process(target=run_async_consumer, args=(task_queue, result_queue, estimated_batches, filename, log_file, api_url))
    consumer.start()
    
    # --- 4. Collector ---
    all_indices = []
    all_results = []
    
    # Wait for producers to finish
    for p in producers:
        p.join()
    
    # Signal consumer to finish
    task_queue.put(None)
    
    # Collect results
    while consumer.is_alive() or not result_queue.empty():
        try:
            res = result_queue.get(timeout=1)
            batch_idx, indices, results = res
            all_indices.extend(indices)
            all_results.extend(results)
        except:
            pass
    
    consumer.join()
    
    # --- 5. Merge & Write ---
    print("Merging...")
    # from quality_classifier import DocumentClassifier
    # parser = DocumentClassifier()
    
    parsed_data = []
    for res in all_results:
        try: 
            parsed = parse_classification(res.strip())
        except: 
            parsed = {}
        parsed_data.append(parsed)
        
    df_results = pd.DataFrame(parsed_data, index=all_indices)
    df_orig = pq.read_table(input_path).to_pandas()
    df_final = df_orig.join(df_results, how='left')
    
    cols = ['classification_raw', 'fdc_primary', 'fdc_secondary', 'bloom_cognitive_primary', 'bloom_cognitive_secondary', 'bloom_knowledge_primary', 'bloom_knowledge_secondary', 'doc_type_v1_primary', 'doc_type_v1_secondary', 'extraction_artifacts_primary', 'extraction_artifacts_secondary', 'missing_content_primary', 'missing_content_secondary', 'doc_type_v2_primary', 'doc_type_v2_secondary', 'reasoning_depth_primary', 'reasoning_depth_secondary', 'technical_correctness_primary', 'technical_correctness_secondary', 'educational_level_primary', 'educational_level_secondary']
    for c in cols:
        if c not in df_final.columns: 
            df_final[c] = None
            
    pq.write_table(pa.Table.from_pandas(df_final), output_path, compression='snappy')
    
    # Cleanup
    total_docs = len(all_results)
    del all_indices, all_results, df_results, df_orig, df_final, parsed_data
    
    return total_docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Quality Classifier")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output Parquet file")
    parser.add_argument("--api_ip", type=str, required=True, help="API IP Address")
    args = parser.parse_args()

    API_URL = f"http://{args.api_ip}:8000/v1/completions"
    
    # Setup Logging
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.api_ip}.log"
    setup_logging(log_filename)
    
    logging.info(f"Starting processing file: {args.input_file} on IP: {args.api_ip}")
    print(f"Starting processing file: {args.input_file} on IP: {args.api_ip}")
    
    if os.path.exists(args.output_file):
        print(f"Skipping {args.output_file} (Already exists)")
        exit(0)
        
    start = time.time()
    try:
        docs = process_file(args.input_file, args.output_file, log_filename, API_URL)
        t = time.time() - start
        msg = f"✓ Done {Path(args.input_file).name} | Docs: {docs:,} | Time: {int(t//60)}m {int(t%60)}s"
        print(msg)
        logging.info(msg)
    except Exception as e:
        msg = f"✗ Error: {e}"
        print(msg)
        logging.error(msg, exc_info=True)
        import traceback
        traceback.print_exc()
