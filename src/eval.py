import json
import sys
import os

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from core.llm_api.llm import ModelAPI
from collections import defaultdict
import asyncio
from core.utils import setup_environment
from collections import Counter
import numpy as np

from src.exec.executor import batched_execute_code_test_case


def get_message(prompt, item):
    message = [
        {"role": "system", "content": "You are an expert Python programmer."},
        {"role": "user", "content": f"Solve this problem using python code only:\n{prompt['question']}"},
        {"role": "assistant", "content": f"```python\n{prompt['solution']}\n```"},
        {"role": "user", "content": f"Solve this problem using python code only:\n{prompt['question_2']}"},
        {"role": "assistant", "content": f"```python\n{prompt['solution_2']}\n```"},
        {"role": "user", "content": f"Solve this problem using python code only:\n{item['question']}"}
    ]
    return message


def evaluate():
    save_dir = 'results/test_sub/initial_code/gpt-4-0125-preview'

    if not os.path.exists(f"{save_dir}/merge_eval.json"):
        data = []
        for file in os.listdir(save_dir):
            if file == 'merge_eval.json':
                continue
            with open(f"{save_dir}/{file}") as f:
                item = json.load(f)
                question  = item['metadata']['question']
                solution = item['solution']
                data.append({"question": question, "solution": solution, 'test_cases': item['metadata']['test_cases'], "difficulty": item['metadata']['difficulty']})

        print('size = ', len(data))
        print(data[0])

        res = batched_execute_code_test_case(data)
        with open(f"{save_dir}/merge_eval.json", 'w') as f:
            json.dump(res, f, indent=2)
    else:
        with open(f"{save_dir}/merge_eval.json") as f:
            res = json.load(f)

    print('overall acc = ', np.mean([i['overall_flag'] for i in res]))
    for difficulty in ['introductory', 'interview', 'competition']:
        print(f'{difficulty} acc = ', np.mean([i['overall_flag'] for i in res if i['difficulty'] == difficulty]))


async def main():
    # load data
    prompt_file = 'prompts/code_gen.json'
    in_path = '/home/wenjiaxin/control/data/APPS/test_sub.json'

    exp_name = 'test_sub'
    # model = 'gpt-4-0125-preview'
    model = "claude-3-5-sonnet-20240620"
    save_dir = f"results/{exp_name}/initial_code/{model}"
    os.makedirs(save_dir, exist_ok=True)

    message_list = []
    with open(in_path) as f:
        data = json.load(f)
        for i, item in enumerate(data):
            item['uid'] = i
    print(Counter([i['difficulty'] for i in data]))
    
    with open(prompt_file) as f:
        prompt = json.load(f)
    for i in data:
        message = get_message(prompt, i)
        message_list.append(message)

    print(f"total size = {len(data)}")

    def parse_fn(item, save_path):
        gen = item['response']['completion']
        try:
            if "```python" in gen:
                gen = gen[gen.index("```python")+len("```python"):]
            if "\n```" in gen:
                gen = gen[:gen.index("\n```")]
            assert any(gen.strip())

            item['solution'] = gen
            with open(save_path, 'w') as f:
                json.dump(item, f, indent=2)
        except:
            return None

    # send request
    model_api = ModelAPI(anthropic_num_threads=40, openai_fraction_rate_limit=0.99)
    anthropic_requests = [
        model_api(
            model,
            message,
            max_tokens=4096,
            print_prompt_and_response=False,
            temperature=0,
            top_p=1.0,
            metadata=metadata,
            parse_fn=parse_fn,
            save_path=f"{save_dir}/{metadata['uid']}.json",
        ) for metadata, message in zip(data, message_list)
    ]

    answer = await asyncio.gather(
        *anthropic_requests
    )
    
    return answer


if __name__ == '__main__':
    setup_environment()
    asyncio.run(main())
    # evaluate()