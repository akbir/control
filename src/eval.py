import json
import sys
sys.path.append("/home/wenjiaxin/control")
print(sys.path)

from core.llm_api.llm import ModelAPI
from collections import defaultdict
import asyncio
from core.utils import setup_environment



async def main():
    # load data
    in_path = ''
    save_path = ''

    message_list = []


    # send request
    model_api = ModelAPI(anthropic_num_threads=2, openai_fraction_rate_limit=0.99)
    anthropic_requests = [
        model_api(
            "claude-3-5-sonnet-20240620",
            message,
            max_tokens=1024,
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
    asyncio.run(main())