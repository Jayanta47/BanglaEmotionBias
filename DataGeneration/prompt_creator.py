from abc import ABC, abstractmethod

system_instruction_template_V1 = """You are a %s. You shall get question in Bangla. 
Your response should be in Bangla.Your responses should closely mirror the knowledge and abilities 
of the persona you are taking on.If asked about reporting emotion, answer with a single emotion word"""

prompt_template_V1 = '''নিম্নোক্ত মন্তব্যটি শুনে আপনার প্রধান অনুভূতি কি হবে?"%s"'''


class PromptCreator(ABC):
    @abstractmethod
    def create_prompt(self, prompt, **kwargs):
        pass


class ChatGptMessageCreator(PromptCreator):
    def create_prompt(self, prompt, **kwargs):
        persona = kwargs.get("persona", None)
        system_message = system_instruction_template_V1.replace("\n", " ") % persona
        prompt = prompt_template_V1 % prompt
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]


class TextGenWebUIMessageCreator(PromptCreator):
    def create_prompt(self, prompt, **kwargs):
        persona = kwargs.get("persona", None)
        system_message = system_instruction_template_V1.replace("\n", " ") % persona
        prompt = prompt_template_V1 % prompt
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]


if __name__ == "__main__":
    message_creator = ChatGptMessageCreator()
    prompt = message_creator.create_prompt("আপনি কি ভালো আছেন?", persona="male")
    print(prompt)
