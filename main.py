from Categorizer import Categorizer
from dotenv import load_dotenv

load_dotenv()

categorizer = Categorizer()

# categorizer.run_on_dataset("SN_data_subset", categorizer.ask_chatgpt)
# categorizer.run_on_dataset("SN_data_subset", categorizer.ask_elmib)

categorizer.run_on_dataset("SN_data", categorizer.ask_chatgpt)
categorizer.run_on_dataset("SN_data", categorizer.ask_elmib)
