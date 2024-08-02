from tfx.components import ExampleValidator, SchemaGen, StatisticsGen


def create_data_validation(example_gen):
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )
    return statistics_gen, schema_gen, example_validator
