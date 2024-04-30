from data_handler import *
from prompt_creator import *
from models import *
from OdiaGenBengaliLlama import *
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)
#  export $(cat .env | xargs) && env


def generate_inference_data(
    data_handler: DataHandlerBase,
    prompt_creator: PromptCreator,
    model: Model,
    total: int = -1,
    calcualate_cost: bool = False,
):
    datapoints = data_handler.return_data_point(total)
    personas = data_handler.get_personas()
    total_input_tokens = 0
    total_output_tokens = 0
    for data_point in tqdm(datapoints):
        current_index = data_point["ID"]
        logger.info(f"Current index: {current_index}")
        for persona in personas:
            logger.info(f"Current persona: {persona}")
            prompt = prompt_creator.create_prompt(
                prompt=data_point["text"], persona=persona, domain=data_point["Domain"]
            )
            model_response = model.create_response(prompt)
            # model_response = {
            #     "content": "আপনি কি ভালো আছেন?",
            #     "input_tokens": 10,
            #     "output_tokens": 10,
            # }
            response = model_response["content"]

            data_handler.save_generated_data(
                response, persona=persona, index=current_index
            )

            if calcualate_cost:
                total_input_tokens += model_response["input_tokens"]
                total_output_tokens += model_response["output_tokens"]
                cost = model.calculate_cost(
                    model_response["input_tokens"], model_response["output_tokens"]
                )
                cost_till_now = model.calculate_cost(
                    total_input_tokens, total_output_tokens
                )
                logger.info(
                    f"Cost for index {current_index}: {cost}, Total cost: {cost_till_now}"
                )


if __name__ == "__main__":
    logging.basicConfig(
        filename=f"./logs/data_generation_{datetime.now()}.log", level=logging.INFO
    )
    data_handler = DataHandler("config.yaml")
    message_creator = OdiaGenBanglaLlamaMessageCreator()
    logger.info(f"Model name: {data_handler.get_model_name()}")
    model = OdiaGenBengaliLlama(
        model_name=data_handler.get_model_name(), device="cuda:0"
    )
    logger.info("Data generation started")
    generate_inference_data(
        data_handler=data_handler,
        prompt_creator=message_creator,
        model=model,
        total=1,
        calcualate_cost=True,
    )

    logger.info("Data generation finished")
