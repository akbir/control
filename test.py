from core.llm_api.llm import ModelAPI
from collections import defaultdict
import asyncio
from core.utils import setup_environment


async def demo():
    model_api = ModelAPI(anthropic_num_threads=2, openai_fraction_rate_limit=0.99)
    anthropic_requests = [
        model_api(
            "claude-3-5-sonnet-20240620",
            [
                {"role": "system", "content": "You are a comedic pirate."},
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=20,
            print_prompt_and_response=False,
        )
    ]
    oai_chat_messages = [
        [
            {"role": "system", "content": "You are a comedic pirate."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {
                "role": "system",
                "content": "You are a swashbuckling space-faring voyager.",
            },
            {"role": "user", "content": "Hello!"},
        ],
    ]
    oai_chat_models = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"]
    oai_chat_requests = [
        model_api(
            oai_chat_models,
            prompt=message,
            max_tokens=16_000,
            n=1,
            print_prompt_and_response=False,
        )
        for message in oai_chat_messages
    ]
    answer = await asyncio.gather(
        *anthropic_requests, *oai_chat_requests
    )

    for responses in answer:
        for i in responses:
            print(i.completion)
            print('='*100)

    costs = defaultdict(int)
    for responses in answer:
        for response in responses:
            costs[response.model_id] += response.cost

    print("-" * 80)
    print("Costs:")
    for model_id, cost in costs.items():
        print(f"{model_id}: ${cost}")
    return answer


setup_environment()
asyncio.run(demo())