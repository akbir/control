import json
import sys
sys.path.append("/home/wenjiaxin/control")

from core.llm_api.llm import ModelAPI
from collections import defaultdict
import asyncio
from core.utils import setup_environment
from collections import Counter
import numpy as np
import os

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
    # save_path = 'results/apps_test.jsonl'
    # save_path = 'results/apps_test_gpt4-0613.jsonl'
    save_path = 'results/test_sub/claude35sonnet.jsonl'
    # save_path = 'results/test_sub/gpt4-0613.jsonl'

    if not os.path.exists(save_path[:-1]):
        data = []
        with open(save_path) as f:
            for line in f:
                item = json.loads(line)
                # question = item['prompt'][-1]['content'][0]['text'][len("Solve this problem using python code only:"):].strip()
                question = item['prompt'][-1]['content'][len("Solve this problem using python code only:"):].strip()
                if isinstance(item['response'], list):
                    gen = item['response'][0]['completion']
                else:
                    gen = item['response']['completion']
                raw_gen = gen
                try:
                    if "```python" in gen:
                        gen = gen[gen.index("```python")+len("```python"):]
                    if "\n```" in gen:
                        gen = gen[:gen.index("\n```")]
                    assert any(gen.strip())
                except:
                    print(raw_gen)
                    print('='*100)
                    print(gen)
                    exit(0)
                gen = gen.strip()
                data.append({"question": question, "solution": gen})

        print('size = ', len(data))
        print(data[0])

        res = batched_execute_code_test_case(data)
        with open(save_path[:-1], 'w') as f:
            json.dump(res, f, indent=2)
    else:
        with open(save_path[:-1]) as f:
            res = json.load(f)

    print('overall acc = ', np.mean([i['overall_flag'] for i in res]))
    for difficulty in ['introductory', 'interview', 'competition']:
        print(f'{difficulty} acc = ', np.mean([i['overall_flag'] for i in res if i['difficulty'] == difficulty]))


async def main():
    # load data
    prompt_file = 'prompts/code_gen.json'
    in_path = '/home/wenjiaxin/control/data/APPS/test_sub.json'

    exp_name = 'test_sub'
    os.makedirs(f"results/{exp_name}", exist_ok=True)

    # model = "claude-3-5-sonnet-20240620"
    # save_path = f'results/{exp_name}/claude35sonnet.jsonl'

    model = 'gpt-4-0613'
    save_path = f'results/{exp_name}/gpt4-0613.jsonl'

    message_list = []
    with open(in_path) as f:
        data = json.load(f)
    print(Counter([i['difficulty'] for i in data]))
    
    with open(prompt_file) as f:
        prompt = json.load(f)
    for i in data:
        message = get_message(prompt, i)
        message_list.append(message)

    # print example
    # for i in message_list[0]:
    #     print('='*50+i['role']+'='*50)
        # print(i['content'])
    # exit(0)


    # send request
    model_api = ModelAPI(anthropic_num_threads=40, openai_fraction_rate_limit=0.99)
    anthropic_requests = [
        model_api(
            model,
            message,
            max_tokens=4096,
            print_prompt_and_response=False,
            save_path=save_path,
            temperature=0,
            top_p=1.0
        ) for message in message_list
    ]

    answer = await asyncio.gather(
        *anthropic_requests
    )
    
    costs = defaultdict(int)
    for responses in answer:
        for response in responses:
            costs[response.model_id] += response.cost

    print("-" * 80)
    print("Costs:")
    for model_id, cost in costs.items():
        print(f"{model_id}: ${cost}")
    return answer


if __name__ == '__main__':
    setup_environment()
    # asyncio.run(main())
    evaluate()