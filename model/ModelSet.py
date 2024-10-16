import ollama
from termcolor import colored
from pyfiglet import Figlet

model_name = 'qwen2:7b-instruct-fp16'
model_file = '../../../model/qwen2_7b'


def print_fancy(text, color='cyan', font='slant'):
    f = Figlet(font=font)
    fancy_text = f.renderText(text)
    print(colored(fancy_text, color))


if __name__ == '__main__':
    f = open(model_file, "r")
    modelfile = f.read()
    ollama.create(model=model_name, modelfile=modelfile)
    print_fancy("Model Set!", color='magenta')