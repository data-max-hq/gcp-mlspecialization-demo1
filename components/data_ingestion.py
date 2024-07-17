from tfx.components import CsvExampleGen

def create_example_gen(data_path: str):
    return CsvExampleGen(input_base=data_path)