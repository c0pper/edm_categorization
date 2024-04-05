from Categorizer import Categorizer
from dotenv import load_dotenv

load_dotenv()


# categorizer = Categorizer(mode="context")
# categorizer.add_csv_dataset("data/bosch-samples.csv", ["questions"], ["answers"])
# categorizer.run_on_dataset("bosch-samples", llm_or_chain_factory=categorizer.ask_chatgpt)
# categorizer.run_on_dataset("bosch-samples", llm_or_chain_factory=categorizer.ask_elmib)


categorizer = Categorizer(mode="nestor_elmi")
# categorizer.add_csv_dataset("data/bosch-samples.csv", ["questions"], ["answers"])
categorizer.run_on_dataset("bosch-samples", llm_or_chain_factory=categorizer.ask_elmib)

categorizer = Categorizer(mode="nestor_cgpt")